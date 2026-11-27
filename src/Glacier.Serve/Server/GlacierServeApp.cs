namespace Glacier.Serve.Server;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Pipelines;
using System.Net;
using System.Net.Security;
using System.Net.Sockets;
using System.Security.Authentication;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Serve.Core;
using Glacier.Serve.Diagnostics;
using Glacier.Serve.Parsing;
using Glacier.Serve.Routing;
using Glacier.Serve.Serialization;

public sealed class GlacierServeApp : IDisposable
{
    public const long DefaultMaxRequestBodySize = 32L * 1024 * 1024; // 32 MB

    private readonly int _port;
    private readonly IPAddress _address;
    private readonly RadixTreeRouter _router;
    private readonly X509Certificate2? _tlsCertificate;
    private readonly byte[]? _bearerTokenBytes;
    private readonly HashSet<string>? _authExemptPaths;
    private readonly SemaphoreSlim? _connectionGate;
    private readonly TimeSpan? _headerTimeout;
    private long _maxRequestBodySize;
    private TcpListener? _listener;
    private CancellationTokenSource? _cts;
    private Task? _listenTask;

    public int Port => (_listener?.LocalEndpoint as IPEndPoint)?.Port ?? _port;
    public bool IsRunning => _listener != null;

    public long MaxRequestBodySize
    {
        get => Interlocked.Read(ref _maxRequestBodySize);
        set
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(value, 0);
            Interlocked.Exchange(ref _maxRequestBodySize, value);
        }
    }

    internal GlacierServeApp(
        IPAddress address,
        int port,
        RadixTreeRouter router,
        long maxRequestBodySize = DefaultMaxRequestBodySize,
        X509Certificate2? tlsCertificate = null,
        string? bearerToken = null,
        IReadOnlyCollection<string>? authExemptPaths = null,
        int maxConcurrentConnections = 0,
        TimeSpan? headerTimeout = null)
    {
        _address = address;
        _port = port;
        _router = router;
        _maxRequestBodySize = maxRequestBodySize;
        _tlsCertificate = tlsCertificate;
        _bearerTokenBytes = !string.IsNullOrEmpty(bearerToken) ? Encoding.UTF8.GetBytes(bearerToken) : null;
        _authExemptPaths = authExemptPaths is { Count: > 0 }
            ? new HashSet<string>(authExemptPaths, StringComparer.Ordinal)
            : null;
        _connectionGate = maxConcurrentConnections > 0 ? new SemaphoreSlim(maxConcurrentConnections, maxConcurrentConnections) : null;
        _headerTimeout = headerTimeout;
    }

    public static GlacierServeBuilder CreateBuilder(string[]? args = null) => new();

    public void MapGet(string pattern, RequestDelegate handler) => _router.AddRoute(HttpMethod.Get, pattern, handler);
    public void MapPost(string pattern, RequestDelegate handler) => _router.AddRoute(HttpMethod.Post, pattern, handler);
    public void MapPut(string pattern, RequestDelegate handler) => _router.AddRoute(HttpMethod.Put, pattern, handler);
    public void MapDelete(string pattern, RequestDelegate handler) => _router.AddRoute(HttpMethod.Delete, pattern, handler);

    public Task StartAsync(CancellationToken cancellationToken = default)
    {
        _cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        _listener = new TcpListener(_address, _port);
        _listener.Server.NoDelay = true; // disable Nagle algorithm for sub-millisecond response
        _listener.Start();

        _listenTask = Task.Run(() => AcceptLoopAsync(_cts.Token), _cts.Token);
        return Task.CompletedTask;
    }

    internal static int ComputeBackoffMs(int failureCount, int minBackoffMs = 10, int maxBackoffMs = 1000)
    {
        if (failureCount <= 0) return 0;
        int shift = Math.Min(failureCount - 1, 20);
        return Math.Min(minBackoffMs * (1 << shift), maxBackoffMs);
    }

    private async Task AcceptLoopAsync(CancellationToken ct)
    {
        int failureCount = 0;

        while (!ct.IsCancellationRequested)
        {
            try
            {
                var listener = _listener;
                if (listener == null) break;

                var socket = await listener.AcceptSocketAsync(ct).ConfigureAwait(false);
                failureCount = 0;
                socket.NoDelay = true;
                if (_connectionGate != null && !_connectionGate.Wait(0))
                {
                    _ = SendServiceUnavailableAndCloseAsync(socket);
                    continue;
                }
                _ = Task.Run(() => ProcessConnectionAsync(socket, ct), ct);
            }
            catch (OperationCanceledException)
            {
                GlacierDiagnostics.LogDebug("[Glacier.Serve] AcceptLoopAsync cancelled cleanly during shutdown.");
                break;
            }
            catch (ObjectDisposedException odex)
            {
                if (ct.IsCancellationRequested) break;
                GlacierDiagnostics.LogWarning($"[Glacier.Serve] Listener disposed in AcceptLoopAsync: {odex.Message}");
                break;
            }
            catch (SocketException sex)
            {
                if (ct.IsCancellationRequested) break;
                failureCount++;
                GlacierDiagnostics.LogError($"[Glacier.Serve] Socket error in AcceptLoopAsync (Code {sex.SocketErrorCode}): {sex.Message}", sex);

                int backoffMs = ComputeBackoffMs(failureCount);
                try
                {
                    await Task.Delay(backoffMs, ct).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
            }
            catch (Exception ex)
            {
                if (ct.IsCancellationRequested) break;
                failureCount++;
                GlacierDiagnostics.LogError($"[Glacier.Serve] Unexpected listener error in AcceptLoopAsync: {ex.Message}", ex);

                int backoffMs = ComputeBackoffMs(failureCount);
                try
                {
                    await Task.Delay(backoffMs, ct).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
            }
        }
    }

    private async Task ProcessConnectionAsync(Socket socket, CancellationToken ct)
    {
        Stream? stream = null;
        PipeReader? reader = null;
        PipeWriter? writer = null;
        using var connCts = CancellationTokenSource.CreateLinkedTokenSource(ct);

        try
        {
            stream = new NetworkStream(socket, ownsSocket: true);
            if (_tlsCertificate != null)
            {
                var sslStream = new SslStream(stream, leaveInnerStreamOpen: false);
                stream = sslStream;
                using var tlsCts = _headerTimeout is { } ht
                    ? CancellationTokenSource.CreateLinkedTokenSource(connCts.Token)
                    : null;
                if (tlsCts != null) tlsCts.CancelAfter(_headerTimeout!.Value);
                var tlsToken = tlsCts?.Token ?? connCts.Token;
                await sslStream.AuthenticateAsServerAsync(
                    new SslServerAuthenticationOptions
                    {
                        ServerCertificate = _tlsCertificate,
                        EnabledSslProtocols = SslProtocols.Tls12 | SslProtocols.Tls13,
                        CertificateRevocationCheckMode = X509RevocationMode.NoCheck
                    },
                    tlsToken).ConfigureAwait(false);
            }

            reader = PipeReader.Create(stream);
            writer = PipeWriter.Create(stream);

            while (!connCts.IsCancellationRequested)
            {
                using var headerCts = _headerTimeout is { } headerTimeout
                    ? CancellationTokenSource.CreateLinkedTokenSource(connCts.Token)
                    : null;
                if (headerCts != null) headerCts.CancelAfter(_headerTimeout!.Value);
                CancellationToken headerToken = headerCts?.Token ?? connCts.Token;

                ReadResult readResult;
                HeaderParseResult parseResult;
                HttpMethod method;
                string path;
                string queryString;
                Dictionary<string, string> headers;
                bool routeMatched;
                RouteMatch routeMatch;
                int headerBytesConsumed;
                ReadOnlySequence<byte> buffer;

                while (true)
                {
                    try
                    {
                        readResult = await reader.ReadAsync(headerToken).ConfigureAwait(false);
                    }
                    catch (OperationCanceledException) when (headerCts != null && headerCts.IsCancellationRequested && !connCts.IsCancellationRequested)
                    {
                        var timeoutResp = new HttpResponse(writer) { StatusCode = 408 };
                        timeoutResp.Headers["Connection"] = "close";
                        await timeoutResp.WriteUtf8Async("{\"error\":\"Request Timeout\"}", connCts.Token).ConfigureAwait(false);
                        return;
                    }

                    buffer = readResult.Buffer;
                    if (buffer.IsEmpty && readResult.IsCompleted)
                    {
                        reader.AdvanceTo(buffer.End);
                        return;
                    }

                    parseResult = TryParseRequestAndMatchRoute(
                        buffer,
                        _router,
                        out method,
                        out path,
                        out queryString,
                        out headers,
                        out routeMatched,
                        out routeMatch,
                        out headerBytesConsumed);

                    if (parseResult == HeaderParseResult.Incomplete)
                    {
                        if (buffer.Length >= 16384)
                        {
                            reader.AdvanceTo(buffer.End);
                            var errResp = new HttpResponse(writer) { StatusCode = 431 };
                            errResp.Headers["Connection"] = "close";
                            await errResp.WriteUtf8Async("{\"error\":\"Request Header Fields Too Large\"}", connCts.Token).ConfigureAwait(false);
                            return;
                        }
                        reader.AdvanceTo(buffer.Start, buffer.End);
                        if (readResult.IsCompleted) return;
                        continue;
                    }

                    break;
                }

                if (parseResult == HeaderParseResult.Malformed)
                {
                    reader.AdvanceTo(buffer.End);
                    var errResp = new HttpResponse(writer) { StatusCode = 400 };
                    errResp.Headers["Connection"] = "close";
                    await errResp.WriteUtf8Async("{\"error\":\"Bad Request\"}", connCts.Token).ConfigureAwait(false);
                    break;
                }

                var request = new HttpRequest
                {
                    Method = method,
                    Path = path,
                    QueryString = queryString,
                    BodyReader = reader
                };
                foreach (var (k, v) in headers)
                {
                    request.Headers[k] = v;
                }

                var response = new HttpResponse(writer);
                long maxBodyBytes = MaxRequestBodySize;

                // Slice past parsed HTTP request headers locally without advancing PipeReader yet;
                // PipeReader.AdvanceTo must only be called once per ReadAsync.
                buffer = buffer.Slice(headerBytesConsumed);

                if (request.Header("Transfer-Encoding")?.Contains("chunked", StringComparison.OrdinalIgnoreCase) == true)
                {
                    var (chunkBodyStatus, remainingBuffer) = await ReadChunkedBodyAsync(
                        reader,
                        buffer,
                        readResult.IsCompleted,
                        maxBodyBytes,
                        request,
                        connCts.Token).ConfigureAwait(false);

                    reader.AdvanceTo(remainingBuffer.Start);

                    if (chunkBodyStatus == BodyReadStatus.PayloadTooLarge)
                    {
                        response.StatusCode = 413;
                        response.Headers["Connection"] = "close";
                        await response.WriteUtf8Async("{\"error\":\"Payload Too Large\"}", connCts.Token).ConfigureAwait(false);
                        break;
                    }
                    if (chunkBodyStatus == BodyReadStatus.BadRequest)
                    {
                        response.StatusCode = 400;
                        response.Headers["Connection"] = "close";
                        await response.WriteUtf8Async("{\"error\":\"Malformed chunked transfer encoding\"}", connCts.Token).ConfigureAwait(false);
                        break;
                    }
                }
                else if (request.Header("Content-Length") is string clStr)
                {
                    if (!long.TryParse(clStr, NumberStyles.Integer, CultureInfo.InvariantCulture, out long contentLength) || contentLength < 0)
                    {
                        reader.AdvanceTo(buffer.Start);
                        response.StatusCode = 400;
                        response.Headers["Connection"] = "close";
                        await response.WriteUtf8Async("{\"error\":\"Invalid Content-Length header\"}", connCts.Token).ConfigureAwait(false);
                        break;
                    }

                    if (contentLength > maxBodyBytes || contentLength > int.MaxValue)
                    {
                        reader.AdvanceTo(buffer.Start);
                        response.StatusCode = 413;
                        response.Headers["Connection"] = "close";
                        await response.WriteUtf8Async("{\"error\":\"Payload Too Large\"}", connCts.Token).ConfigureAwait(false);
                        break;
                    }

                    int expectedLen = (int)contentLength;
                    while (buffer.Length < expectedLen)
                    {
                        if (readResult.IsCompleted) break;
                        reader.AdvanceTo(buffer.Start, buffer.End);
                        readResult = await reader.ReadAsync(connCts.Token).ConfigureAwait(false);
                        buffer = readResult.Buffer;
                    }

                    if (expectedLen > 0)
                    {
                        if (buffer.Length < expectedLen)
                        {
                            // Truncated socket stream
                            reader.AdvanceTo(buffer.End);
                            break;
                        }

                        var bodySlice = buffer.Slice(0, expectedLen);
                        byte[] bodyBytes = GC.AllocateUninitializedArray<byte>(expectedLen);
                        bodySlice.CopyTo(bodyBytes);
                        request.Body = bodyBytes;
                        reader.AdvanceTo(buffer.GetPosition(expectedLen));
                    }
                    else
                    {
                        reader.AdvanceTo(buffer.Start);
                    }
                }
                else
                {
                    reader.AdvanceTo(buffer.Start);
                }

                var context = new HttpContext(request, response, connCts.Token)
                {
                    ClientDisconnectProbe = () =>
                    {
                        try
                        {
                            return socket.Poll(0, SelectMode.SelectRead) && socket.Available == 0;
                        }
                        catch
                        {
                            return true;
                        }
                    }
                };

                if (!IsBearerAuthorized(request))
                {
                    response.StatusCode = 401;
                    response.ContentType = "application/json; charset=utf-8";
                    response.Headers["WWW-Authenticate"] = "Bearer";
                    response.Headers["Connection"] = "close";
                    await response.WriteUtf8Async("{\"error\":\"Unauthorized\"}", connCts.Token).ConfigureAwait(false);
                    break;
                }

                if (routeMatched)
                {
                    if (routeMatch.Parameters != null)
                    {
                        for (int i = 0; i < routeMatch.Parameters.Length; i++)
                        {
                            var p = routeMatch.Parameters[i];
                            request.RouteParams[p.Name] = p.Value;
                        }
                    }

                    try
                    {
                        await routeMatch.Handler(context).ConfigureAwait(false);
                    }
                    catch (OperationCanceledException) when (connCts.IsCancellationRequested)
                    {
                        throw;
                    }
                    catch (Exception ex)
                    {
                        GlacierDiagnostics.LogError($"[Glacier.Serve] Route handler error on {request.Method} {request.Path}: {ex.Message}", ex);
                        if (!response.HeadersSent)
                        {
                            response.StatusCode = 500;
                            response.Headers["Connection"] = "close";
                            var err = new ApiErrorResponse
                            {
                                Error = new ApiErrorDetail
                                {
                                    Message = ex.Message,
                                    Type = "internal_server_error"
                                }
                            };
                            await response.WriteJsonAsync(err, ServeJsonContext.Default.ApiErrorResponse, connCts.Token).ConfigureAwait(false);
                        }
                        break;
                    }
                }
                else
                {
                    response.StatusCode = 404;
                    await response.WriteUtf8Async("404 Not Found", connCts.Token).ConfigureAwait(false);
                }

                if (request.Header("Connection")?.Equals("close", StringComparison.OrdinalIgnoreCase) == true ||
                    (response.Headers.TryGetValue("Connection", out var respConn) && respConn.Equals("close", StringComparison.OrdinalIgnoreCase)))
                {
                    break;
                }
            }
        }
        catch (OperationCanceledException)
        {
            connCts.Cancel();
            GlacierDiagnostics.LogDebug("[Glacier.Serve] Connection processing cancelled.");
        }
        catch (Exception ex)
        {
            connCts.Cancel();
            GlacierDiagnostics.LogError($"[Glacier.Serve] ProcessConnectionAsync error: {ex.Message}", ex);
        }
        finally
        {
            connCts.Cancel();
            if (reader != null) try { await reader.CompleteAsync().ConfigureAwait(false); } catch { }
            if (writer != null) try { await writer.CompleteAsync().ConfigureAwait(false); } catch { }
            if (stream != null) try { await stream.DisposeAsync().ConfigureAwait(false); } catch { }
            else socket.Dispose();
            _connectionGate?.Release();
        }
    }

    private static async Task SendServiceUnavailableAndCloseAsync(Socket socket)
    {
        try
        {
            byte[] drain = ArrayPool<byte>.Shared.Rent(1024);
            try
            {
                // Drain incoming request headers briefly so Winsock does not emit RST before the client reads 503.
                if (socket.Poll(10_000, SelectMode.SelectRead))
                {
                    await socket.ReceiveAsync(drain, SocketFlags.None).ConfigureAwait(false);
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(drain);
            }

            await socket.SendAsync(
                "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"u8.ToArray(),
                SocketFlags.None).ConfigureAwait(false);
            socket.Shutdown(SocketShutdown.Both);
        }
        catch
        {
        }
        finally
        {
            socket.Dispose();
        }
    }

    private bool IsBearerAuthorized(HttpRequest request)
    {
        if (_bearerTokenBytes == null)
            return true;

        if (_authExemptPaths != null && _authExemptPaths.Contains(request.Path))
            return true;

        string? authHeader = request.Header("Authorization");
        if (string.IsNullOrEmpty(authHeader) || !authHeader.StartsWith("Bearer ", StringComparison.Ordinal))
            return false;

        byte[] supplied = Encoding.UTF8.GetBytes(authHeader.AsSpan(7).ToString());
        if (supplied.Length != _bearerTokenBytes.Length)
            return false;

        return CryptographicOperations.FixedTimeEquals(supplied, _bearerTokenBytes);
    }

    private enum BodyReadStatus
    {
        Ok,
        PayloadTooLarge,
        BadRequest
    }

    private static async ValueTask<(BodyReadStatus Status, ReadOnlySequence<byte> RemainingBuffer)> ReadChunkedBodyAsync(
        PipeReader reader,
        ReadOnlySequence<byte> buffer,
        bool isCompleted,
        long maxBodyBytes,
        HttpRequest request,
        CancellationToken ct)
    {
        var bodyWriter = new ArrayBufferWriter<byte>(1024);
        long totalBytes = 0;

        while (true)
        {
            int headerLineEnd;
            int headerConsumed;
            long chunkSize;

            while (!TryParseChunkSizeLine(buffer, out chunkSize, out headerLineEnd, out headerConsumed))
            {
                if (headerLineEnd == -2)
                    return (BodyReadStatus.BadRequest, buffer);

                if (isCompleted)
                    return (BodyReadStatus.BadRequest, buffer);

                reader.AdvanceTo(buffer.Start, buffer.End);
                var rr = await reader.ReadAsync(ct).ConfigureAwait(false);
                buffer = rr.Buffer;
                isCompleted = rr.IsCompleted;
            }

            if (chunkSize < 0)
                return (BodyReadStatus.BadRequest, buffer);

            if (chunkSize > maxBodyBytes - totalBytes || chunkSize > int.MaxValue - totalBytes)
                return (BodyReadStatus.PayloadTooLarge, buffer);

            buffer = buffer.Slice(headerConsumed);

            if (chunkSize == 0)
            {
                // Consume optional chunked trailer headers up to empty line (\r\n or \n)
                while (true)
                {
                    int trailerCrlf = ConsumeLeadingCrlf(buffer);
                    if (trailerCrlf > 0)
                    {
                        buffer = buffer.Slice(trailerCrlf);
                        break;
                    }

                    long? lfPos = FindByteOffset(buffer, (byte)'\n');
                    if (lfPos.HasValue)
                    {
                        buffer = buffer.Slice(lfPos.Value + 1);
                        continue;
                    }

                    if (isCompleted)
                        break;

                    reader.AdvanceTo(buffer.Start, buffer.End);
                    var rr = await reader.ReadAsync(ct).ConfigureAwait(false);
                    buffer = rr.Buffer;
                    isCompleted = rr.IsCompleted;
                }

                request.Body = bodyWriter.WrittenMemory;
                return (BodyReadStatus.Ok, buffer);
            }

            int chunkInt = (int)chunkSize;
            while (buffer.Length < chunkInt + 2)
            {
                if (buffer.Length >= chunkInt + 1 && ReadByteAt(buffer, chunkInt) == (byte)'\n')
                    break;

                if (isCompleted)
                {
                    if (buffer.Length < chunkInt + 1)
                        return (BodyReadStatus.BadRequest, buffer);
                    break;
                }

                reader.AdvanceTo(buffer.Start, buffer.End);
                var rr = await reader.ReadAsync(ct).ConfigureAwait(false);
                buffer = rr.Buffer;
                isCompleted = rr.IsCompleted;
            }

            Span<byte> dest = bodyWriter.GetSpan(chunkInt);
            buffer.Slice(0, chunkInt).CopyTo(dest);
            bodyWriter.Advance(chunkInt);
            totalBytes += chunkInt;

            buffer = buffer.Slice(chunkInt);
            int crlfConsume = ConsumeLeadingCrlf(buffer);
            if (crlfConsume == 0)
                return (BodyReadStatus.BadRequest, buffer);
            buffer = buffer.Slice(crlfConsume);
        }
    }

    private static byte ReadByteAt(in ReadOnlySequence<byte> buffer, long offset)
    {
        Span<byte> single = stackalloc byte[1];
        buffer.Slice(offset, 1).CopyTo(single);
        return single[0];
    }

    private static long? FindByteOffset(in ReadOnlySequence<byte> buffer, byte target)
    {
        long offset = 0;
        foreach (ReadOnlyMemory<byte> seg in buffer)
        {
            int idx = seg.Span.IndexOf(target);
            if (idx >= 0) return offset + idx;
            offset += seg.Length;
        }
        return null;
    }

    private static int ConsumeLeadingCrlf(in ReadOnlySequence<byte> buffer)
    {
        if (buffer.IsEmpty) return 0;
        Span<byte> head = stackalloc byte[2];
        int copyLen = (int)Math.Min(buffer.Length, 2);
        buffer.Slice(0, copyLen).CopyTo(head);
        if (copyLen >= 2 && head[0] == (byte)'\r' && head[1] == (byte)'\n')
            return 2;
        if (head[0] == (byte)'\n')
            return 1;
        return 0;
    }

    private static bool TryParseChunkSizeLine(
        in ReadOnlySequence<byte> buffer,
        out long chunkSize,
        out int lineEndFlag,
        out int headerConsumed)
    {
        chunkSize = 0;
        lineEndFlag = -1;
        headerConsumed = 0;

        if (buffer.IsEmpty) return false;

        int scanLen = (int)Math.Min(buffer.Length, 128);
        Span<byte> scratch = stackalloc byte[scanLen];
        buffer.Slice(0, scanLen).CopyTo(scratch);

        int crlf = HttpSimdParser.IndexOfCrlfVector256(scratch);
        if (crlf < 0)
        {
            if (buffer.Length >= 128)
            {
                lineEndFlag = -2; // Malformed oversized chunk header line
            }
            return false;
        }

        int lineEndLen;
        if (scratch[crlf] == (byte)'\r')
        {
            if (crlf + 1 >= scanLen)
            {
                // Wait for '\n' byte following '\r'
                return false;
            }
            lineEndLen = scratch[crlf + 1] == (byte)'\n' ? 2 : 1;
        }
        else
        {
            lineEndLen = 1;
        }

        ReadOnlySpan<byte> hexLine = scratch[..crlf];
        // Strip optional chunk extensions (;ext=val)
        int semi = hexLine.IndexOf((byte)';');
        if (semi >= 0) hexLine = hexLine[..semi];
        hexLine = TrimAsciiWhitespace(hexLine);

        if (hexLine.IsEmpty)
        {
            lineEndFlag = -2;
            return false;
        }

        long parsed = 0;
        for (int i = 0; i < hexLine.Length; i++)
        {
            byte b = hexLine[i];
            int digit = b switch
            {
                >= (byte)'0' and <= (byte)'9' => b - '0',
                >= (byte)'a' and <= (byte)'f' => 10 + (b - 'a'),
                >= (byte)'A' and <= (byte)'F' => 10 + (b - 'A'),
                _ => -1
            };
            if (digit < 0 || parsed > (long.MaxValue >> 4))
            {
                lineEndFlag = -2;
                return false;
            }
            parsed = (parsed << 4) | (uint)digit;
        }

        chunkSize = parsed;
        lineEndFlag = crlf;
        headerConsumed = crlf + lineEndLen;
        return true;
    }

    private static ReadOnlySpan<byte> TrimAsciiWhitespace(ReadOnlySpan<byte> span)
    {
        int start = 0;
        while (start < span.Length && (span[start] == (byte)' ' || span[start] == (byte)'\t')) start++;
        int end = span.Length - 1;
        while (end >= start && (span[end] == (byte)' ' || span[end] == (byte)'\t')) end--;
        return start <= end ? span.Slice(start, end - start + 1) : ReadOnlySpan<byte>.Empty;
    }

    internal static bool TryMatchRoute(IRouter router, HttpMethod method, string path, out RouteMatch match)
    {
        int pathMaxBytes = Encoding.UTF8.GetMaxByteCount(path.Length);
        if (pathMaxBytes <= 512)
        {
            Span<byte> pathSpan = stackalloc byte[512];
            int pathBytesWritten = Encoding.UTF8.GetBytes(path, pathSpan);
            return router.TryMatch(method, pathSpan[..pathBytesWritten], out match);
        }

        byte[] rented = ArrayPool<byte>.Shared.Rent(pathMaxBytes);
        try
        {
            int pathBytesWritten = Encoding.UTF8.GetBytes(path, rented);
            return router.TryMatch(method, rented.AsSpan(0, pathBytesWritten), out match);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
        }
    }

    private enum HeaderParseResult
    {
        Incomplete = 0,
        Valid = 1,
        Malformed = 2
    }

    private static bool HasCompleteOrInvalidLineEnding(ReadOnlySpan<byte> span)
    {
        int eol = HttpSimdParser.IndexOfCrlfVector256(span);
        if (eol < 0) return false;
        if (span[eol] == (byte)'\n') return true;
        // Found '\r': only complete/invalid if another byte follows it
        return span.Length > eol + 1;
    }

    private static HeaderParseResult TryParseRequestAndMatchRoute(
        in ReadOnlySequence<byte> buffer,
        IRouter router,
        out HttpMethod method,
        out string path,
        out string queryString,
        out Dictionary<string, string> headers,
        out bool routeMatched,
        out RouteMatch routeMatch,
        out int totalConsumed)
    {
        int maxHeaderScan = (int)Math.Min(buffer.Length, 16384);
        if (buffer.IsSingleSegment)
        {
            return TryParseSpanHeadersAndMatch(
                buffer.FirstSpan,
                router,
                out method,
                out path,
                out queryString,
                out headers,
                out routeMatched,
                out routeMatch,
                out totalConsumed);
        }

        if (maxHeaderScan <= 4096)
        {
            Span<byte> scratch = stackalloc byte[maxHeaderScan];
            buffer.Slice(0, maxHeaderScan).CopyTo(scratch);
            return TryParseSpanHeadersAndMatch(
                scratch,
                router,
                out method,
                out path,
                out queryString,
                out headers,
                out routeMatched,
                out routeMatch,
                out totalConsumed);
        }

        byte[] rented = ArrayPool<byte>.Shared.Rent(maxHeaderScan);
        try
        {
            buffer.Slice(0, maxHeaderScan).CopyTo(rented);
            return TryParseSpanHeadersAndMatch(
                rented.AsSpan(0, maxHeaderScan),
                router,
                out method,
                out path,
                out queryString,
                out headers,
                out routeMatched,
                out routeMatch,
                out totalConsumed);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
        }
    }

    private static HeaderParseResult TryParseSpanHeadersAndMatch(
        ReadOnlySpan<byte> span,
        IRouter router,
        out HttpMethod method,
        out string path,
        out string queryString,
        out Dictionary<string, string> headers,
        out bool routeMatched,
        out RouteMatch routeMatch,
        out int totalConsumed)
    {
        method = HttpMethod.Get;
        path = "/";
        queryString = string.Empty;
        headers = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        routeMatched = false;
        routeMatch = default;
        totalConsumed = 0;

        if (!HttpSimdParser.TryParseRequestLine(span, out var methodSpan, out var targetSpan, out int reqLineConsumed))
        {
            return HasCompleteOrInvalidLineEnding(span) ? HeaderParseResult.Malformed : HeaderParseResult.Incomplete;
        }

        method = HttpSimdParser.ParseMethod(methodSpan);

        int qIdx = targetSpan.IndexOf((byte)'?');
        ReadOnlySpan<byte> rawPathSpan;
        if (qIdx >= 0)
        {
            rawPathSpan = targetSpan[..qIdx];
            queryString = Encoding.UTF8.GetString(targetSpan[qIdx..]);
        }
        else
        {
            rawPathSpan = targetSpan;
        }

        if (rawPathSpan.IsEmpty)
        {
            rawPathSpan = "/"u8;
        }

        path = Encoding.UTF8.GetString(rawPathSpan);
        // Match directly over the raw UTF-8 path bytes without string -> UTF8 round-trip
        routeMatched = router.TryMatch(method, rawPathSpan, out routeMatch);

        var headerSpan = span[reqLineConsumed..];
        totalConsumed = reqLineConsumed;
        bool hasTransferEncoding = false;
        bool hasContentLength = false;
        string? firstContentLength = null;

        while (headerSpan.Length > 0)
        {
            if (headerSpan.StartsWith("\r\n"u8))
            {
                if (hasTransferEncoding && hasContentLength)
                {
                    return HeaderParseResult.Malformed;
                }
                totalConsumed += 2;
                return HeaderParseResult.Valid;
            }
            if (headerSpan.StartsWith("\n"u8))
            {
                return HeaderParseResult.Malformed;
            }
            if (headerSpan.Length == 1 && headerSpan[0] == (byte)'\r')
            {
                return HeaderParseResult.Incomplete;
            }

            if (HttpSimdParser.TryParseHeader(headerSpan, out var hName, out var hVal, out int hConsumed))
            {
                string headerNameStr = Encoding.UTF8.GetString(hName);
                string headerValueStr = Encoding.UTF8.GetString(hVal);

                if (headerNameStr.Equals("Transfer-Encoding", StringComparison.OrdinalIgnoreCase))
                {
                    hasTransferEncoding = true;
                }
                else if (headerNameStr.Equals("Content-Length", StringComparison.OrdinalIgnoreCase))
                {
                    if (firstContentLength is not null && !string.Equals(firstContentLength, headerValueStr, StringComparison.Ordinal))
                    {
                        return HeaderParseResult.Malformed;
                    }
                    firstContentLength ??= headerValueStr;
                    hasContentLength = true;
                }

                if (hasTransferEncoding && hasContentLength)
                {
                    return HeaderParseResult.Malformed;
                }

                headers[headerNameStr] = headerValueStr;
                totalConsumed += hConsumed;
                headerSpan = span[totalConsumed..];
            }
            else
            {
                return HasCompleteOrInvalidLineEnding(headerSpan) ? HeaderParseResult.Malformed : HeaderParseResult.Incomplete;
            }
        }

        return HeaderParseResult.Incomplete;
    }

    public void Stop()
    {
        _cts?.Cancel();
        _listener?.Stop();
        _listener = null;
    }

    public void Dispose() => Stop();
}

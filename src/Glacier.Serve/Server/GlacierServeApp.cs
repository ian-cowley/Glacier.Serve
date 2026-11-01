namespace Glacier.Serve.Server;

using System;
using System.Buffers;
using System.Globalization;
using System.IO.Pipelines;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Serve.Core;
using Glacier.Serve.Diagnostics;
using Glacier.Serve.Parsing;
using Glacier.Serve.Routing;

public sealed class GlacierServeApp : IDisposable
{
    public const long DefaultMaxRequestBodySize = 32L * 1024 * 1024; // 32 MB

    private readonly int _port;
    private readonly IPAddress _address;
    private readonly RadixTreeRouter _router;
    private long _maxRequestBodySize;
    private TcpListener? _listener;
    private CancellationTokenSource? _cts;
    private Task? _listenTask;

    public int Port => _port;
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

    internal GlacierServeApp(IPAddress address, int port, RadixTreeRouter router, long maxRequestBodySize = DefaultMaxRequestBodySize)
    {
        _address = address;
        _port = port;
        _router = router;
        _maxRequestBodySize = maxRequestBodySize;
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
        using var stream = new NetworkStream(socket, ownsSocket: true);
        var reader = PipeReader.Create(stream);
        var writer = PipeWriter.Create(stream);
        using var connCts = CancellationTokenSource.CreateLinkedTokenSource(ct);

        try
        {
            while (!connCts.IsCancellationRequested)
            {
                var readResult = await reader.ReadAsync(connCts.Token).ConfigureAwait(false);
                var buffer = readResult.Buffer;

                if (buffer.IsEmpty && readResult.IsCompleted) break;

                if (!TryParseRequestAndMatchRoute(
                    buffer,
                    _router,
                    out var method,
                    out var path,
                    out var queryString,
                    out var headers,
                    out bool routeMatched,
                    out var routeMatch,
                    out int headerBytesConsumed))
                {
                    if (buffer.Length >= 16384)
                    {
                        var errResp = new HttpResponse(writer) { StatusCode = 431 };
                        errResp.Headers["Connection"] = "close";
                        await errResp.WriteUtf8Async("{\"error\":\"Request Header Fields Too Large\"}", connCts.Token).ConfigureAwait(false);
                        break;
                    }
                    reader.AdvanceTo(buffer.Start, buffer.End);
                    if (readResult.IsCompleted) break;
                    continue;
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

                // Advance past parsed HTTP request headers before reading body
                buffer = buffer.Slice(headerBytesConsumed);
                reader.AdvanceTo(buffer.Start);

                if (request.Header("Transfer-Encoding")?.Contains("chunked", StringComparison.OrdinalIgnoreCase) == true)
                {
                    var chunkBodyStatus = await ReadChunkedBodyAsync(reader, buffer, readResult.IsCompleted, maxBodyBytes, request, connCts.Token).ConfigureAwait(false);
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
                        response.StatusCode = 400;
                        response.Headers["Connection"] = "close";
                        await response.WriteUtf8Async("{\"error\":\"Invalid Content-Length header\"}", connCts.Token).ConfigureAwait(false);
                        break;
                    }

                    if (contentLength > maxBodyBytes || contentLength > int.MaxValue)
                    {
                        response.StatusCode = 413;
                        response.Headers["Connection"] = "close";
                        await response.WriteUtf8Async("{\"error\":\"Payload Too Large\"}", connCts.Token).ConfigureAwait(false);
                        break;
                    }

                    int expectedLen = (int)contentLength;
                    while (buffer.Length < expectedLen)
                    {
                        reader.AdvanceTo(buffer.Start, buffer.End);
                        readResult = await reader.ReadAsync(connCts.Token).ConfigureAwait(false);
                        buffer = readResult.Buffer;
                        if (readResult.IsCompleted && buffer.Length < expectedLen) break;
                    }

                    if (expectedLen > 0)
                    {
                        if (buffer.Length < expectedLen)
                        {
                            // Truncated socket stream
                            break;
                        }

                        var bodySlice = buffer.Slice(0, expectedLen);
                        byte[] bodyBytes = GC.AllocateUninitializedArray<byte>(expectedLen);
                        bodySlice.CopyTo(bodyBytes);
                        request.Body = bodyBytes;
                        reader.AdvanceTo(buffer.GetPosition(expectedLen));
                    }
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

                    await routeMatch.Handler(context).ConfigureAwait(false);
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
            try { await reader.CompleteAsync().ConfigureAwait(false); } catch { }
            try { await writer.CompleteAsync().ConfigureAwait(false); } catch { }
        }
    }

    private enum BodyReadStatus
    {
        Ok,
        PayloadTooLarge,
        BadRequest
    }

    private static async ValueTask<BodyReadStatus> ReadChunkedBodyAsync(
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
                    return BodyReadStatus.BadRequest;

                if (isCompleted)
                    return BodyReadStatus.BadRequest;

                reader.AdvanceTo(buffer.Start, buffer.End);
                var rr = await reader.ReadAsync(ct).ConfigureAwait(false);
                buffer = rr.Buffer;
                isCompleted = rr.IsCompleted;
            }

            if (chunkSize < 0)
                return BodyReadStatus.BadRequest;

            if (totalBytes + chunkSize > maxBodyBytes || totalBytes + chunkSize > int.MaxValue)
                return BodyReadStatus.PayloadTooLarge;

            buffer = buffer.Slice(headerConsumed);
            reader.AdvanceTo(buffer.Start);

            if (chunkSize == 0)
            {
                // Consume optional chunked trailer headers up to empty line (\r\n or \n)
                while (true)
                {
                    int trailerCrlf = ConsumeLeadingCrlf(buffer);
                    if (trailerCrlf > 0)
                    {
                        reader.AdvanceTo(buffer.GetPosition(trailerCrlf));
                        break;
                    }

                    long? lfPos = FindByteOffset(buffer, (byte)'\n');
                    if (lfPos.HasValue)
                    {
                        buffer = buffer.Slice(lfPos.Value + 1);
                        reader.AdvanceTo(buffer.Start);
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
                return BodyReadStatus.Ok;
            }

            int chunkInt = (int)chunkSize;
            while (buffer.Length < chunkInt + 2)
            {
                if (isCompleted)
                {
                    if (buffer.Length < chunkInt + 1)
                        return BodyReadStatus.BadRequest;
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
                return BodyReadStatus.BadRequest;
            buffer = buffer.Slice(crlfConsume);
            reader.AdvanceTo(buffer.Start);
        }
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

    private static bool TryParseRequestAndMatchRoute(
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

    private static bool TryParseSpanHeadersAndMatch(
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
            return false;
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

        while (headerSpan.Length > 0)
        {
            if (headerSpan.StartsWith("\r\n"u8))
            {
                totalConsumed += 2;
                return true;
            }
            if (headerSpan.StartsWith("\n"u8))
            {
                totalConsumed += 1;
                return true;
            }

            if (HttpSimdParser.TryParseHeader(headerSpan, out var hName, out var hVal, out int hConsumed))
            {
                headers[Encoding.UTF8.GetString(hName)] = Encoding.UTF8.GetString(hVal);
                totalConsumed += hConsumed;
                headerSpan = span[totalConsumed..];
            }
            else
            {
                break;
            }
        }

        return false;
    }

    public void Stop()
    {
        _cts?.Cancel();
        _listener?.Stop();
        _listener = null;
    }

    public void Dispose() => Stop();
}

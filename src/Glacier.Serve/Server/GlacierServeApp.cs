namespace Glacier.Serve.Server;

using System;
using System.Buffers;
using System.Buffers.Text;
using System.Collections.Generic;
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
    private const int MaxChunkHeaderLineBytes = 128;
    private const int MaxChunkTrailerLines = 64;
    private const int MaxChunkTrailerBytes = 8192;

    private static readonly byte[] s_serviceUnavailableResponse =
        "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"u8.ToArray();

    private static readonly AsyncLocal<bool> s_inConnectionOrAcceptContext = new();

    private readonly int _port;
    private readonly IPAddress _address;
    private readonly RadixTreeRouter _router;
    private readonly X509Certificate2? _tlsCertificate;
    private readonly byte[]? _bearerTokenSha256;
    private readonly HashSet<string>? _authExemptPaths;
    private readonly SemaphoreSlim? _connectionGate;
    private readonly TimeSpan? _headerTimeout;
    private readonly ManualResetEventSlim _connectionsDrained = new(initialState: true);
    private long _maxRequestBodySize;
    private TcpListener? _listener;
    private CancellationTokenSource? _cts;
    private Task? _listenTask;
    private int _activeConnections;
    private int _disposed;
    private int _drainedDisposed;

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
        _bearerTokenSha256 = !string.IsNullOrEmpty(bearerToken)
            ? SHA256.HashData(Encoding.UTF8.GetBytes(bearerToken))
            : null;
        if (authExemptPaths is { Count: > 0 })
        {
            _authExemptPaths = new HashSet<string>(StringComparer.Ordinal);
            foreach (string p in authExemptPaths)
            {
                if (!string.IsNullOrEmpty(p))
                    _authExemptPaths.Add(NormalizePath(p));
            }
        }
        _connectionGate = maxConcurrentConnections > 0 ? new SemaphoreSlim(maxConcurrentConnections, maxConcurrentConnections) : null;
        _headerTimeout = headerTimeout;
    }

    internal static string NormalizePath(string path)
    {
        if (string.IsNullOrEmpty(path))
            return "/";

        int qIdx = path.IndexOf('?');
        ReadOnlySpan<char> span = qIdx >= 0 ? path.AsSpan(0, qIdx) : path.AsSpan();
        if (span.IsEmpty)
            return "/";

        bool needsLeadingSlash = span[0] != '/';
        while (span.Length > 1 && span[^1] == '/')
            span = span[..^1];

        if (!needsLeadingSlash && span.Length == path.Length)
            return path;

        return needsLeadingSlash ? string.Concat("/", span) : span.ToString();
    }

    public static GlacierServeBuilder CreateBuilder(string[]? args = null) => new();

    public void MapGet(string pattern, RequestDelegate handler) => _router.AddRoute(HttpMethod.Get, pattern, handler);
    public void MapPost(string pattern, RequestDelegate handler) => _router.AddRoute(HttpMethod.Post, pattern, handler);
    public void MapPut(string pattern, RequestDelegate handler) => _router.AddRoute(HttpMethod.Put, pattern, handler);
    public void MapDelete(string pattern, RequestDelegate handler) => _router.AddRoute(HttpMethod.Delete, pattern, handler);

    public Task StartAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
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
        s_inConnectionOrAcceptContext.Value = true;
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

                if (Interlocked.Increment(ref _activeConnections) == 1 && Volatile.Read(ref _drainedDisposed) == 0)
                {
                    try
                    {
                        _connectionsDrained.Reset();
                    }
                    catch (ObjectDisposedException ex)
                    {
                        GlacierDiagnostics.LogDebug($"[Glacier.Serve] Connections drained signal disposed during accept reset: {ex.Message}");
                    }
                }

                try
                {
                    _ = Task.Run(() => ProcessConnectionAsync(socket, ct), CancellationToken.None);
                }
                catch
                {
                    if (Interlocked.Decrement(ref _activeConnections) == 0 && Volatile.Read(ref _drainedDisposed) == 0)
                    {
                        try
                        {
                            _connectionsDrained.Set();
                        }
                        catch (ObjectDisposedException ex)
                        {
                            GlacierDiagnostics.LogDebug($"[Glacier.Serve] Connections drained signal disposed during spawn rollback: {ex.Message}");
                        }
                    }
                    try { _connectionGate?.Release(); }
                    catch (ObjectDisposedException ex)
                    {
                        GlacierDiagnostics.LogDebug($"[Glacier.Serve] Connection gate disposed during spawn rollback: {ex.Message}");
                    }
                    socket.Dispose();
                    throw;
                }
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

    private sealed class RequestIdleTimeoutException : Exception
    {
    }

    private static async ValueTask<ReadResult> ReadWithIdleTimeoutAsync(
        PipeReader reader,
        TimeSpan? timeout,
        CancellationToken connCt)
    {
        if (!timeout.HasValue)
        {
            return await reader.ReadAsync(connCt).ConfigureAwait(false);
        }

        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(connCt);
        timeoutCts.CancelAfter(timeout.Value);
        try
        {
            return await reader.ReadAsync(timeoutCts.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (timeoutCts.IsCancellationRequested && !connCt.IsCancellationRequested)
        {
            throw new RequestIdleTimeoutException();
        }
    }

    private static async Task MonitorClientDisconnectAsync(
        PipeReader reader,
        CancellationTokenSource requestAbortedCts,
        CancellationToken monitorCt)
    {
        try
        {
            if (reader.TryRead(out ReadResult pre))
            {
                reader.AdvanceTo(pre.Buffer.Start);
                if (pre.IsCompleted || pre.IsCanceled || !pre.Buffer.IsEmpty)
                    return;
            }

            while (!monitorCt.IsCancellationRequested)
            {
                ReadResult rr = await reader.ReadAsync(monitorCt).ConfigureAwait(false);
                // Keep consumed = examined = Start so buffered pipelined bytes remain unexamined
                // for the next keep-alive request iteration.
                reader.AdvanceTo(rr.Buffer.Start);
                if (rr.IsCompleted || rr.IsCanceled || !rr.Buffer.IsEmpty)
                {
                    return;
                }
            }
        }
        catch (OperationCanceledException) when (monitorCt.IsCancellationRequested)
        {
            // Handler finished normally before disconnect.
        }
        catch (Exception ex) when (ex is IOException or SocketException or ObjectDisposedException)
        {
            GlacierDiagnostics.LogDebug($"[Glacier.Serve] Client disconnect detected on pipe reader: {ex.Message}");
            try
            {
                requestAbortedCts.Cancel();
            }
            catch (ObjectDisposedException odex)
            {
                GlacierDiagnostics.LogDebug($"[Glacier.Serve] Request CTS already disposed during disconnect cancel: {odex.Message}");
            }
        }
    }

    private async Task ProcessConnectionAsync(Socket socket, CancellationToken ct)
    {
        s_inConnectionOrAcceptContext.Value = true;
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
                using var tlsCts = _headerTimeout.HasValue
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
                using var headerCts = _headerTimeout.HasValue
                    ? CancellationTokenSource.CreateLinkedTokenSource(connCts.Token)
                    : null;
                if (headerCts != null) headerCts.CancelAfter(_headerTimeout!.Value);
                CancellationToken headerToken = headerCts?.Token ?? connCts.Token;

                ReadResult readResult;
                HeaderParseResult parseResult;
                HttpMethod method;
                string path;
                string queryString;
                Dictionary<string, string>? headers = null;
                bool isChunked;
                long? parsedContentLength;
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
                        await WriteTimeoutAndFlushAsync(writer, connCts.Token).ConfigureAwait(false);
                        return;
                    }

                    buffer = readResult.Buffer;
                    if (buffer.IsEmpty && readResult.IsCompleted)
                    {
                        reader.AdvanceTo(buffer.End);
                        return;
                    }

                    headers?.Clear();
                    parseResult = TryParseRequestAndMatchRoute(
                        buffer,
                        _router,
                        ref headers,
                        out method,
                        out path,
                        out queryString,
                        out isChunked,
                        out parsedContentLength,
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

                var request = new HttpRequest(headers!)
                {
                    Method = method,
                    Path = path,
                    QueryString = queryString,
                    BodyReader = reader
                };

                var response = new HttpResponse(writer);
                long maxBodyBytes = MaxRequestBodySize;

                // Slice past parsed HTTP request headers locally without advancing PipeReader yet;
                // PipeReader.AdvanceTo must only be called once per ReadAsync.
                buffer = buffer.Slice(headerBytesConsumed);

                if (isChunked)
                {
                    BodyReadStatus chunkBodyStatus;
                    ReadOnlySequence<byte> remainingBuffer;
                    try
                    {
                        (chunkBodyStatus, remainingBuffer) = await ReadChunkedBodyAsync(
                            reader,
                            buffer,
                            readResult.IsCompleted,
                            maxBodyBytes,
                            request,
                            _headerTimeout,
                            connCts.Token).ConfigureAwait(false);
                    }
                    catch (RequestIdleTimeoutException)
                    {
                        await WriteTimeoutAndFlushAsync(writer, connCts.Token).ConfigureAwait(false);
                        return;
                    }

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
                else if (parsedContentLength.HasValue)
                {
                    long contentLength = parsedContentLength.Value;
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
                        try
                        {
                            readResult = await ReadWithIdleTimeoutAsync(reader, _headerTimeout, connCts.Token).ConfigureAwait(false);
                        }
                        catch (RequestIdleTimeoutException)
                        {
                            await WriteTimeoutAndFlushAsync(writer, connCts.Token).ConfigureAwait(false);
                            return;
                        }
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

                using var requestAbortedCts = CancellationTokenSource.CreateLinkedTokenSource(connCts.Token);
                response.OnWriteAborted = () =>
                {
                    try
                    {
                        requestAbortedCts.Cancel();
                    }
                    catch (ObjectDisposedException odex)
                    {
                        GlacierDiagnostics.LogDebug($"[Glacier.Serve] Request CTS already disposed during write abort: {odex.Message}");
                    }
                };
                var context = new HttpContext(request, response, requestAbortedCts.Token);

                if (!IsBearerAuthorized(request))
                {
                    response.OnWriteAborted = null;
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

                    using var monitorStopCts = CancellationTokenSource.CreateLinkedTokenSource(connCts.Token);
                    Task disconnectMonitorTask = MonitorClientDisconnectAsync(reader, requestAbortedCts, monitorStopCts.Token);
                    try
                    {
                        await routeMatch.Handler(context).ConfigureAwait(false);
                    }
                    catch (OperationCanceledException) when (connCts.IsCancellationRequested)
                    {
                        throw;
                    }
                    catch (Exception ex) when (ex is IOException or SocketException)
                    {
                        try
                        {
                            requestAbortedCts.Cancel();
                        }
                        catch (ObjectDisposedException odex)
                        {
                            GlacierDiagnostics.LogDebug($"[Glacier.Serve] Request CTS already disposed during socket abort: {odex.Message}");
                        }
                        GlacierDiagnostics.LogDebug($"[Glacier.Serve] Client socket aborted during {request.Method} {request.Path}: {ex.Message}");
                        break;
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
                    finally
                    {
                        response.OnWriteAborted = null;
                        monitorStopCts.Cancel();
                        await disconnectMonitorTask.ConfigureAwait(false);
                    }
                }
                else
                {
                    response.OnWriteAborted = null;
                    response.StatusCode = 404;
                    await response.WriteUtf8Async("404 Not Found", connCts.Token).ConfigureAwait(false);
                }

                if (requestAbortedCts.IsCancellationRequested ||
                    request.Header("Connection")?.Equals("close", StringComparison.OrdinalIgnoreCase) == true ||
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
            if (reader != null)
            {
                try { await reader.CompleteAsync().ConfigureAwait(false); }
                catch (Exception ex) when (ex is IOException or SocketException or ObjectDisposedException or InvalidOperationException)
                {
                    GlacierDiagnostics.LogDebug($"[Glacier.Serve] PipeReader completion error during teardown: {ex.Message}");
                }
            }
            if (writer != null)
            {
                try { await writer.CompleteAsync().ConfigureAwait(false); }
                catch (Exception ex) when (ex is IOException or SocketException or ObjectDisposedException or InvalidOperationException)
                {
                    GlacierDiagnostics.LogDebug($"[Glacier.Serve] PipeWriter completion error during teardown: {ex.Message}");
                }
            }
            if (stream != null)
            {
                try { await stream.DisposeAsync().ConfigureAwait(false); }
                catch (Exception ex) when (ex is IOException or SocketException or ObjectDisposedException)
                {
                    GlacierDiagnostics.LogDebug($"[Glacier.Serve] Stream dispose error during teardown: {ex.Message}");
                }
            }
            else
            {
                socket.Dispose();
            }

            if (_connectionGate != null)
            {
                try { _connectionGate.Release(); }
                catch (ObjectDisposedException ex)
                {
                    GlacierDiagnostics.LogDebug($"[Glacier.Serve] Connection gate already disposed during teardown: {ex.Message}");
                }
            }

            if (Interlocked.Decrement(ref _activeConnections) == 0 && Volatile.Read(ref _drainedDisposed) == 0)
            {
                try
                {
                    _connectionsDrained.Set();
                }
                catch (ObjectDisposedException ex)
                {
                    GlacierDiagnostics.LogDebug($"[Glacier.Serve] Connections drained signal disposed during teardown: {ex.Message}");
                }
            }
        }
    }

    private static async Task WriteTimeoutAndFlushAsync(PipeWriter writer, CancellationToken ct)
    {
        var timeoutResp = new HttpResponse(writer) { StatusCode = 408 };
        timeoutResp.Headers["Connection"] = "close";
        await timeoutResp.WriteUtf8Async("{\"error\":\"Request Timeout\"}", ct).ConfigureAwait(false);
    }

    private static async Task SendServiceUnavailableAndCloseAsync(Socket socket)
    {
        try
        {
            byte[] drain = ArrayPool<byte>.Shared.Rent(1024);
            try
            {
                int totalDrained = 0;
                // Drain any already-buffered request bytes before sending 503.
                while (socket.Available > 0 && totalDrained < 16384)
                {
                    int read = await socket.ReceiveAsync(drain, SocketFlags.None).ConfigureAwait(false);
                    if (read == 0)
                        return;
                    totalDrained += read;
                }

                await socket.SendAsync(s_serviceUnavailableResponse, SocketFlags.None).ConfigureAwait(false);
                // Half-close send only (FIN) so Winsock ACKs any in-flight client request bytes
                // instead of emitting RST before the client finishes reading 503.
                socket.Shutdown(SocketShutdown.Send);

                using var drainCts = new CancellationTokenSource(TimeSpan.FromMilliseconds(150));
                while (totalDrained < 16384)
                {
                    int read = await socket.ReceiveAsync(drain, SocketFlags.None, drainCts.Token).ConfigureAwait(false);
                    if (read == 0)
                        break;
                    totalDrained += read;
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(drain);
            }
        }
        catch (Exception ex) when (ex is SocketException or IOException or ObjectDisposedException or OperationCanceledException)
        {
            GlacierDiagnostics.LogDebug($"[Glacier.Serve] 503 overload response write failed: {ex.Message}");
        }
        finally
        {
            socket.Dispose();
        }
    }

    private bool IsBearerAuthorized(HttpRequest request)
    {
        if (_bearerTokenSha256 == null)
            return true;

        if (_authExemptPaths != null && _authExemptPaths.Contains(NormalizePath(request.Path)))
            return true;

        ReadOnlySpan<char> tokenSpan = ReadOnlySpan<char>.Empty;
        string? authHeader = request.Header("Authorization");
        if (!string.IsNullOrEmpty(authHeader) && authHeader.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
        {
            tokenSpan = authHeader.AsSpan(7).Trim();
        }
        else if (request.Header("x-api-key") is { Length: > 0 } apiKeyHeader)
        {
            tokenSpan = apiKeyHeader.AsSpan().Trim();
        }

        if (tokenSpan.IsEmpty)
            return false;

        Span<byte> suppliedHash = stackalloc byte[32];
        int maxBytes = Encoding.UTF8.GetMaxByteCount(tokenSpan.Length);
        if (maxBytes <= 512)
        {
            Span<byte> tokenBytes = stackalloc byte[512];
            int written = Encoding.UTF8.GetBytes(tokenSpan, tokenBytes);
            SHA256.HashData(tokenBytes[..written], suppliedHash);
        }
        else
        {
            byte[] rented = ArrayPool<byte>.Shared.Rent(maxBytes);
            try
            {
                int written = Encoding.UTF8.GetBytes(tokenSpan, rented);
                SHA256.HashData(rented.AsSpan(0, written), suppliedHash);
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(rented);
            }
        }

        return CryptographicOperations.FixedTimeEquals(suppliedHash, _bearerTokenSha256);
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
        TimeSpan? idleTimeout,
        CancellationToken connCt)
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
                var rr = await ReadWithIdleTimeoutAsync(reader, idleTimeout, connCt).ConfigureAwait(false);
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
                int trailerLines = 0;
                int trailerBytes = 0;

                // Consume optional RFC 9112 chunked trailer headers strictly terminated by \r\n up to empty \r\n
                while (true)
                {
                    if (buffer.Length >= 2)
                    {
                        byte b0 = ReadByteAt(buffer, 0);
                        byte b1 = ReadByteAt(buffer, 1);
                        if (b0 == (byte)'\r' && b1 == (byte)'\n')
                        {
                            buffer = buffer.Slice(2);
                            break;
                        }
                        if (b0 == (byte)'\n' || b0 == (byte)'\r')
                        {
                            return (BodyReadStatus.BadRequest, buffer);
                        }
                    }
                    else if (buffer.Length == 1)
                    {
                        byte b0 = ReadByteAt(buffer, 0);
                        if (b0 == (byte)'\n')
                            return (BodyReadStatus.BadRequest, buffer);

                        if (isCompleted)
                            return (BodyReadStatus.BadRequest, buffer);

                        reader.AdvanceTo(buffer.Start, buffer.End);
                        var rrWait = await ReadWithIdleTimeoutAsync(reader, idleTimeout, connCt).ConfigureAwait(false);
                        buffer = rrWait.Buffer;
                        isCompleted = rrWait.IsCompleted;
                        continue;
                    }
                    else if (isCompleted)
                    {
                        return (BodyReadStatus.BadRequest, buffer);
                    }

                    long? crlfPos = FindCrlfOffset(buffer, out byte foundByte);
                    if (crlfPos.HasValue)
                    {
                        if (foundByte == (byte)'\n')
                            return (BodyReadStatus.BadRequest, buffer);

                        long crPos = crlfPos.Value;
                        if (buffer.Length <= crPos + 1)
                        {
                            if (isCompleted)
                                return (BodyReadStatus.BadRequest, buffer);

                            reader.AdvanceTo(buffer.Start, buffer.End);
                            var rrCr = await ReadWithIdleTimeoutAsync(reader, idleTimeout, connCt).ConfigureAwait(false);
                            buffer = rrCr.Buffer;
                            isCompleted = rrCr.IsCompleted;
                            continue;
                        }

                        if (ReadByteAt(buffer, crPos + 1) != (byte)'\n')
                            return (BodyReadStatus.BadRequest, buffer);

                        long lineLenLong = crPos + 2;
                        if (trailerLines + 1 > MaxChunkTrailerLines || lineLenLong > MaxChunkTrailerBytes - trailerBytes)
                            return (BodyReadStatus.BadRequest, buffer);

                        int lineLen = (int)lineLenLong;
                        trailerLines++;
                        trailerBytes += lineLen;

                        if (!ValidateChunkTrailerLine(buffer.Slice(0, lineLen)))
                            return (BodyReadStatus.BadRequest, buffer);

                        buffer = buffer.Slice(lineLen);
                        continue;
                    }

                    if (trailerBytes + buffer.Length > MaxChunkTrailerBytes || isCompleted)
                        return (BodyReadStatus.BadRequest, buffer);

                    reader.AdvanceTo(buffer.Start, buffer.End);
                    var rr = await ReadWithIdleTimeoutAsync(reader, idleTimeout, connCt).ConfigureAwait(false);
                    buffer = rr.Buffer;
                    isCompleted = rr.IsCompleted;
                }

                request.Body = bodyWriter.WrittenMemory;
                return (BodyReadStatus.Ok, buffer);
            }

            int chunkInt = (int)chunkSize;
            while (buffer.Length < chunkInt + 2)
            {
                if (isCompleted)
                    return (BodyReadStatus.BadRequest, buffer);

                reader.AdvanceTo(buffer.Start, buffer.End);
                var rr = await ReadWithIdleTimeoutAsync(reader, idleTimeout, connCt).ConfigureAwait(false);
                buffer = rr.Buffer;
                isCompleted = rr.IsCompleted;
            }

            Span<byte> dest = bodyWriter.GetSpan(chunkInt);
            buffer.Slice(0, chunkInt).CopyTo(dest);
            bodyWriter.Advance(chunkInt);
            totalBytes += chunkInt;

            buffer = buffer.Slice(chunkInt);
            int crlfConsume = ConsumeLeadingCrlf(buffer);
            if (crlfConsume != 2)
                return (BodyReadStatus.BadRequest, buffer);
            buffer = buffer.Slice(2);
        }
    }

    private static bool ValidateChunkTrailerLine(in ReadOnlySequence<byte> lineSequence)
    {
        int len = (int)lineSequence.Length;
        byte[]? rented = null;
        Span<byte> lineSpan = len <= 512
            ? stackalloc byte[len]
            : (rented = ArrayPool<byte>.Shared.Rent(len)).AsSpan(0, len);
        try
        {
            lineSequence.CopyTo(lineSpan);
            if (!HttpSimdParser.TryParseHeader(lineSpan, out var tName, out _, out int consumed) || consumed != len)
                return false;

            if (Ascii.EqualsIgnoreCase(tName, "Host"u8) ||
                Ascii.EqualsIgnoreCase(tName, "Content-Length"u8) ||
                Ascii.EqualsIgnoreCase(tName, "Transfer-Encoding"u8))
            {
                return false;
            }
            return true;
        }
        finally
        {
            if (rented != null)
                ArrayPool<byte>.Shared.Return(rented);
        }
    }

    private static byte ReadByteAt(in ReadOnlySequence<byte> buffer, long offset)
    {
        Span<byte> single = stackalloc byte[1];
        buffer.Slice(offset, 1).CopyTo(single);
        return single[0];
    }

    private static long? FindCrlfOffset(in ReadOnlySequence<byte> buffer, out byte foundByte)
    {
        long offset = 0;
        foreach (ReadOnlyMemory<byte> seg in buffer)
        {
            int idx = HttpSimdParser.IndexOfCrlfVector256(seg.Span);
            if (idx >= 0)
            {
                foundByte = seg.Span[idx];
                return offset + idx;
            }
            offset += seg.Length;
        }
        foundByte = 0;
        return null;
    }

    private static int ConsumeLeadingCrlf(in ReadOnlySequence<byte> buffer)
    {
        if (buffer.Length < 2) return 0;
        Span<byte> head = stackalloc byte[2];
        buffer.Slice(0, 2).CopyTo(head);
        return (head[0] == (byte)'\r' && head[1] == (byte)'\n') ? 2 : 0;
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

        int scanLen = (int)Math.Min(buffer.Length, MaxChunkHeaderLineBytes);
        Span<byte> scratch = stackalloc byte[scanLen];
        buffer.Slice(0, scanLen).CopyTo(scratch);

        int crlf = HttpSimdParser.IndexOfCrlfVector256(scratch);
        if (crlf < 0)
        {
            if (buffer.Length >= MaxChunkHeaderLineBytes)
            {
                lineEndFlag = -2; // Malformed oversized chunk header line
            }
            return false;
        }

        if (scratch[crlf] != (byte)'\r')
        {
            // Bare '\n' is rejected under strict RFC 9112 CRLF framing.
            lineEndFlag = -2;
            return false;
        }

        if (crlf + 1 >= scanLen)
        {
            if (scanLen >= MaxChunkHeaderLineBytes)
            {
                lineEndFlag = -2;
                return false;
            }
            // Wait for '\n' byte following '\r'
            return false;
        }

        if (scratch[crlf + 1] != (byte)'\n')
        {
            // Bare '\r' followed by non-'\n' is rejected.
            lineEndFlag = -2;
            return false;
        }

        ReadOnlySpan<byte> fullLine = scratch[..crlf];
        for (int i = 0; i < fullLine.Length; i++)
        {
            byte b = fullLine[i];
            if ((b < 0x20 && b != (byte)'\t') || b == 0x7F)
            {
                lineEndFlag = -2;
                return false;
            }
        }

        // Strip optional chunk extensions (;ext=val). RFC 9112 §7.1.1: chunk-size = 1*HEXDIG
        // (no leading whitespace; optional BWS only allowed after hex digits before ';').
        int semi = fullLine.IndexOf((byte)';');
        ReadOnlySpan<byte> hexLine = semi >= 0 ? TrimTrailingAsciiWhitespace(fullLine[..semi]) : fullLine;

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
        headerConsumed = crlf + 2;
        return true;
    }

    private static ReadOnlySpan<byte> TrimTrailingAsciiWhitespace(ReadOnlySpan<byte> span)
    {
        int end = span.Length - 1;
        while (end >= 0 && (span[end] == (byte)' ' || span[end] == (byte)'\t')) end--;
        return end >= 0 ? span[..(end + 1)] : ReadOnlySpan<byte>.Empty;
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
        ref Dictionary<string, string>? headers,
        out HttpMethod method,
        out string path,
        out string queryString,
        out bool isChunked,
        out long? parsedContentLength,
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
                ref headers,
                out method,
                out path,
                out queryString,
                out isChunked,
                out parsedContentLength,
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
                ref headers,
                out method,
                out path,
                out queryString,
                out isChunked,
                out parsedContentLength,
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
                ref headers,
                out method,
                out path,
                out queryString,
                out isChunked,
                out parsedContentLength,
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
        ref Dictionary<string, string>? headers,
        out HttpMethod method,
        out string path,
        out string queryString,
        out bool isChunked,
        out long? parsedContentLength,
        out bool routeMatched,
        out RouteMatch routeMatch,
        out int totalConsumed)
    {
        method = HttpMethod.Get;
        path = "/";
        queryString = string.Empty;
        isChunked = false;
        parsedContentLength = null;
        routeMatched = false;
        routeMatch = default;
        totalConsumed = 0;

        if (!HttpSimdParser.TryParseRequestLine(span, out var methodSpan, out var targetSpan, out var httpVersion, out int reqLineConsumed))
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

        headers ??= new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        var headerSpan = span[reqLineConsumed..];
        totalConsumed = reqLineConsumed;
        bool hasTransferEncoding = false;
        bool hasContentLength = false;
        int hostCount = 0;

        while (headerSpan.Length > 0)
        {
            if (headerSpan.StartsWith("\r\n"u8))
            {
                if (hasTransferEncoding && hasContentLength)
                    return HeaderParseResult.Malformed;
                if (httpVersion == HttpProtocolVersion.Http11 && hostCount != 1)
                    return HeaderParseResult.Malformed;
                if (hostCount > 1)
                    return HeaderParseResult.Malformed;

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
                string headerNameStr = InternCommonHeaderName(hName);
                string headerValueStr = Encoding.UTF8.GetString(hVal);

                if (ReferenceEquals(headerNameStr, "Host"))
                {
                    hostCount++;
                    if (hostCount > 1 || !IsValidHostHeader(hVal))
                        return HeaderParseResult.Malformed;
                }
                else if (ReferenceEquals(headerNameStr, "Transfer-Encoding"))
                {
                    if (hasTransferEncoding || !TryParseTransferEncodingChunked(hVal))
                        return HeaderParseResult.Malformed;

                    hasTransferEncoding = true;
                    isChunked = true;
                }
                else if (ReferenceEquals(headerNameStr, "Content-Length"))
                {
                    if (!TryParseStrictNonNegativeInt64(hVal, out long clVal))
                        return HeaderParseResult.Malformed;
                    if (parsedContentLength.HasValue && parsedContentLength.Value != clVal)
                        return HeaderParseResult.Malformed;

                    parsedContentLength = clVal;
                    hasContentLength = true;
                }

                if (hasTransferEncoding && hasContentLength)
                    return HeaderParseResult.Malformed;

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

    private static string InternCommonHeaderName(ReadOnlySpan<byte> hName)
    {
        if (Ascii.EqualsIgnoreCase(hName, "Host"u8)) return "Host";
        if (Ascii.EqualsIgnoreCase(hName, "Content-Type"u8)) return "Content-Type";
        if (Ascii.EqualsIgnoreCase(hName, "Content-Length"u8)) return "Content-Length";
        if (Ascii.EqualsIgnoreCase(hName, "Transfer-Encoding"u8)) return "Transfer-Encoding";
        if (Ascii.EqualsIgnoreCase(hName, "Connection"u8)) return "Connection";
        if (Ascii.EqualsIgnoreCase(hName, "Authorization"u8)) return "Authorization";
        if (Ascii.EqualsIgnoreCase(hName, "x-api-key"u8)) return "x-api-key";
        if (Ascii.EqualsIgnoreCase(hName, "User-Agent"u8)) return "User-Agent";
        if (Ascii.EqualsIgnoreCase(hName, "Accept"u8)) return "Accept";
        if (Ascii.EqualsIgnoreCase(hName, "Accept-Encoding"u8)) return "Accept-Encoding";
        if (Ascii.EqualsIgnoreCase(hName, "Cache-Control"u8)) return "Cache-Control";
        return Encoding.UTF8.GetString(hName);
    }

    private static bool TryParseStrictNonNegativeInt64(ReadOnlySpan<byte> span, out long value)
    {
        value = 0;
        if (span.IsEmpty) return false;

        long acc = 0;
        for (int i = 0; i < span.Length; i++)
        {
            byte b = span[i];
            if (b < (byte)'0' || b > (byte)'9')
                return false;

            int digit = b - '0';
            if (acc > (long.MaxValue - digit) / 10)
                return false;

            acc = acc * 10 + digit;
        }

        value = acc;
        return true;
    }

    private static bool TryParseTransferEncodingChunked(ReadOnlySpan<byte> span)
    {
        span = TrimAsciiWhitespace(span);
        return !span.IsEmpty && Ascii.EqualsIgnoreCase(span, "chunked"u8);
    }

    private static bool IsValidHostHeader(ReadOnlySpan<byte> span)
    {
        if (span.IsEmpty) return false;

        for (int i = 0; i < span.Length; i++)
        {
            byte b = span[i];
            if (b <= 0x20 || b >= 0x7F || b == (byte)'@' || b == (byte)'/' || b == (byte)'?' || b == (byte)'#' || b == (byte)'\\')
                return false;
        }

        if (span[0] == (byte)'[')
        {
            int closeBracket = span.IndexOf((byte)']');
            if (closeBracket <= 1) return false;

            ReadOnlySpan<byte> ipv6Bytes = span[1..closeBracket];
            string ipv6Str = Encoding.ASCII.GetString(ipv6Bytes);
            if (!IPAddress.TryParse(ipv6Str, out var addr) || addr.AddressFamily != AddressFamily.InterNetworkV6)
                return false;

            ReadOnlySpan<byte> rest = span[(closeBracket + 1)..];
            if (rest.IsEmpty) return true;
            if (rest[0] != (byte)':') return false;
            return TryParseValidPort(rest[1..]);
        }

        if (span.IndexOfAny((byte)'[', (byte)']') >= 0)
            return false;

        int colonIdx = span.IndexOf((byte)':');
        if (colonIdx >= 0)
        {
            if (colonIdx == 0) return false;
            ReadOnlySpan<byte> portSpan = span[(colonIdx + 1)..];
            if (portSpan.IndexOf((byte)':') >= 0) return false;
            if (!TryParseValidPort(portSpan)) return false;
        }

        return true;
    }

    private static bool TryParseValidPort(ReadOnlySpan<byte> portSpan)
    {
        if (portSpan.IsEmpty || portSpan.Length > 5) return false;
        int port = 0;
        for (int i = 0; i < portSpan.Length; i++)
        {
            byte b = portSpan[i];
            if (b < (byte)'0' || b > (byte)'9') return false;
            port = port * 10 + (b - '0');
        }
        return port <= 65535;
    }

    public void Stop()
    {
        var cts = Interlocked.Exchange(ref _cts, null);
        var listener = Interlocked.Exchange(ref _listener, null);
        var listenTask = Interlocked.Exchange(ref _listenTask, null);

        if (cts != null)
        {
            try { cts.Cancel(); }
            catch (ObjectDisposedException ex)
            {
                GlacierDiagnostics.LogDebug($"[Glacier.Serve] CTS already disposed in Stop: {ex.Message}");
            }
        }

        if (listener != null)
        {
            try { listener.Stop(); }
            catch (ObjectDisposedException ex)
            {
                GlacierDiagnostics.LogDebug($"[Glacier.Serve] Listener already disposed in Stop: {ex.Message}");
            }
            catch (SocketException ex)
            {
                GlacierDiagnostics.LogDebug($"[Glacier.Serve] Listener stop socket notice: {ex.Message}");
            }
        }

        if (!s_inConnectionOrAcceptContext.Value)
        {
            if (listenTask != null && !listenTask.IsCompleted)
            {
                try
                {
                    listenTask.Wait(TimeSpan.FromSeconds(2));
                }
                catch (AggregateException ex)
                {
                    GlacierDiagnostics.LogDebug($"[Glacier.Serve] AcceptLoop shutdown exception: {ex.InnerException?.Message ?? ex.Message}");
                }
            }

            if (Volatile.Read(ref _activeConnections) > 0)
            {
                _connectionsDrained.Wait(TimeSpan.FromSeconds(2));
            }
        }

        cts?.Dispose();
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        Stop();
        Volatile.Write(ref _drainedDisposed, 1);
        _connectionGate?.Dispose();
        _connectionsDrained.Dispose();
    }
}

namespace Glacier.Serve.Server;

using System;
using System.Buffers;
using System.IO.Pipelines;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Serve.Core;
using Glacier.Serve.Parsing;
using Glacier.Serve.Routing;

public sealed class GlacierServeApp : IDisposable
{
    private readonly int _port;
    private readonly IPAddress _address;
    private readonly RadixTreeRouter _router;
    private TcpListener? _listener;
    private CancellationTokenSource? _cts;
    private Task? _listenTask;

    public int Port => _port;
    public bool IsRunning => _listener != null;

    internal GlacierServeApp(IPAddress address, int port, RadixTreeRouter router)
    {
        _address = address;
        _port = port;
        _router = router;
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

    private async Task AcceptLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                var socket = await _listener!.AcceptSocketAsync(ct).ConfigureAwait(false);
                socket.NoDelay = true;
                _ = Task.Run(() => ProcessConnectionAsync(socket, ct), ct);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch
            {
                if (ct.IsCancellationRequested) break;
            }
        }
    }

    private async Task ProcessConnectionAsync(Socket socket, CancellationToken ct)
    {
        using var stream = new NetworkStream(socket, ownsSocket: true);
        var reader = PipeReader.Create(stream);
        var writer = PipeWriter.Create(stream);

        byte[] pathUtf8 = new byte[1024];

        try
        {
            while (!ct.IsCancellationRequested)
            {
                var readResult = await reader.ReadAsync(ct).ConfigureAwait(false);
                var buffer = readResult.Buffer;

                if (buffer.IsEmpty && readResult.IsCompleted) break;

                // Zero-copy network ingress parsing using ReadOnlySequence<byte> and pooled/stackalloc buffers
                if (!TryParseRequestHeaders(buffer, out var method, out var path, out var headers, out int totalConsumed))
                {
                    reader.AdvanceTo(buffer.Start, buffer.End);
                    if (readResult.IsCompleted) break;
                    continue;
                }

                var request = new HttpRequest
                {
                    Method = method,
                    Path = path,
                    BodyReader = reader
                };
                foreach (var (k, v) in headers)
                {
                    request.Headers[k] = v;
                }

                if (request.Header("Transfer-Encoding")?.Contains("chunked", StringComparison.OrdinalIgnoreCase) == true)
                {
                    using var ms = new MemoryStream();
                    int chunkOffset = totalConsumed;
                    while (true)
                    {
                        var remainingBytes = buffer.Slice(chunkOffset).ToArray();
                        int crlf = HttpSimdParser.IndexOfCrlfVector256(remainingBytes.AsSpan());
                        if (crlf < 0)
                        {
                            reader.AdvanceTo(buffer.Start, buffer.End);
                            readResult = await reader.ReadAsync(ct).ConfigureAwait(false);
                            buffer = readResult.Buffer;
                            remainingBytes = buffer.Slice(chunkOffset).ToArray();
                            crlf = HttpSimdParser.IndexOfCrlfVector256(remainingBytes.AsSpan());
                            if (crlf < 0) break;
                        }

                        string hexStr = Encoding.ASCII.GetString(remainingBytes, 0, crlf).Trim();
                        if (!int.TryParse(hexStr, System.Globalization.NumberStyles.HexNumber, null, out int chunkSize))
                            break;

                        int lineEndLen = (remainingBytes[crlf] == (byte)'\r' && remainingBytes.Length > crlf + 1 && remainingBytes[crlf + 1] == (byte)'\n') ? 2 : 1;
                        chunkOffset += crlf + lineEndLen;

                        if (chunkSize == 0)
                        {
                            if (buffer.Length >= chunkOffset + 2) chunkOffset += 2;
                            break;
                        }

                        while (buffer.Length < chunkOffset + chunkSize)
                        {
                            reader.AdvanceTo(buffer.Start, buffer.End);
                            readResult = await reader.ReadAsync(ct).ConfigureAwait(false);
                            buffer = readResult.Buffer;
                        }

                        var chunkBytes = buffer.Slice(chunkOffset, chunkSize).ToArray();
                        ms.Write(chunkBytes, 0, chunkBytes.Length);
                        chunkOffset += chunkSize;

                        if (buffer.Length >= chunkOffset + 2)
                        {
                            var crlfCheck = buffer.Slice(chunkOffset, 2).ToArray();
                            if (crlfCheck[0] == '\r' && crlfCheck[1] == '\n') chunkOffset += 2;
                            else if (crlfCheck[0] == '\n') chunkOffset += 1;
                        }
                    }

                    request.Body = ms.ToArray();
                    totalConsumed = chunkOffset;
                }
                else if (request.Header("Content-Length") is string clStr && int.TryParse(clStr, out int clVal))
                {
                    int contentLength = clVal;
                    while (buffer.Length < totalConsumed + contentLength)
                    {
                        reader.AdvanceTo(buffer.Start, buffer.End);
                        readResult = await reader.ReadAsync(ct).ConfigureAwait(false);
                        buffer = readResult.Buffer;
                        if (readResult.IsCompleted && buffer.Length < totalConsumed + contentLength) break;
                    }

                    if (contentLength > 0 && buffer.Length >= totalConsumed + contentLength)
                    {
                        var bodySlice = buffer.Slice(totalConsumed, contentLength);
                        if (bodySlice.IsSingleSegment)
                        {
                            request.Body = bodySlice.First;
                        }
                        else
                        {
                            byte[] bodyBytes = GC.AllocateUninitializedArray<byte>(contentLength);
                            bodySlice.CopyTo(bodyBytes);
                            request.Body = bodyBytes;
                        }
                        totalConsumed += contentLength;
                    }
                }

                reader.AdvanceTo(buffer.GetPosition(totalConsumed));

                var response = new HttpResponse(writer);
                var context = new HttpContext(request, response, ct);

                int pathMaxBytes = Encoding.UTF8.GetMaxByteCount(request.Path.Length);
                if (pathMaxBytes > pathUtf8.Length)
                {
                    pathUtf8 = new byte[pathMaxBytes];
                }
                int pathBytesWritten = Encoding.UTF8.GetBytes(request.Path, pathUtf8);
                if (_router.TryMatch(method, pathUtf8.AsSpan(0, pathBytesWritten), out var match))
                {
                    if (match.Parameters != null)
                    {
                        foreach (var p in match.Parameters)
                        {
                            request.RouteParams[p.Name] = p.Value;
                        }
                    }

                    await match.Handler(context).ConfigureAwait(false);
                }
                else
                {
                    response.StatusCode = 404;
                    await response.WriteUtf8Async("404 Not Found").ConfigureAwait(false);
                }

                // Check Connection header from request or response
                if (request.Header("Connection")?.Equals("close", StringComparison.OrdinalIgnoreCase) == true ||
                    response.Headers.TryGetValue("Connection", out var respConn) && respConn.Equals("close", StringComparison.OrdinalIgnoreCase))
                {
                    break;
                }
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ProcessConnectionAsync ERROR]: {ex}");
        }
        finally
        {
            await reader.CompleteAsync().ConfigureAwait(false);
            await writer.CompleteAsync().ConfigureAwait(false);
        }
    }

    private static bool TryParseRequestHeaders(
        in ReadOnlySequence<byte> buffer,
        out HttpMethod method,
        out string path,
        out Dictionary<string, string> headers,
        out int totalConsumed)
    {
        int maxHeaderScan = (int)Math.Min(buffer.Length, 8192);
        if (buffer.IsSingleSegment)
        {
            return TryParseSpanHeaders(buffer.FirstSpan, out method, out path, out headers, out totalConsumed);
        }

        if (maxHeaderScan <= 4096)
        {
            Span<byte> scratch = stackalloc byte[maxHeaderScan];
            buffer.Slice(0, maxHeaderScan).CopyTo(scratch);
            return TryParseSpanHeaders(scratch, out method, out path, out headers, out totalConsumed);
        }

        byte[] rented = ArrayPool<byte>.Shared.Rent(maxHeaderScan);
        try
        {
            buffer.Slice(0, maxHeaderScan).CopyTo(rented);
            return TryParseSpanHeaders(rented.AsSpan(0, maxHeaderScan), out method, out path, out headers, out totalConsumed);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
        }
    }

    private static bool TryParseSpanHeaders(
        ReadOnlySpan<byte> span,
        out HttpMethod method,
        out string path,
        out Dictionary<string, string> headers,
        out int totalConsumed)
    {
        method = HttpMethod.Get;
        path = "/";
        headers = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        totalConsumed = 0;

        if (!HttpSimdParser.TryParseRequestLine(span, out var methodSpan, out var pathSpan, out int reqLineConsumed))
        {
            return false;
        }

        method = HttpSimdParser.ParseMethod(methodSpan);
        path = Encoding.UTF8.GetString(pathSpan);

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

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

        try
        {
            while (!ct.IsCancellationRequested)
            {
                var readResult = await reader.ReadAsync(ct).ConfigureAwait(false);
                var buffer = readResult.Buffer;

                if (buffer.IsEmpty && readResult.IsCompleted) break;

                // Read continuous sequence
                byte[] tempBytes = buffer.ToArray();
                var span = tempBytes.AsSpan();

                if (!HttpSimdParser.TryParseRequestLine(span, out var methodSpan, out var pathSpan, out int reqLineConsumed))
                {
                    reader.AdvanceTo(buffer.Start, buffer.End);
                    if (readResult.IsCompleted) break;
                    continue;
                }

                var method = HttpSimdParser.ParseMethod(methodSpan);
                var request = new HttpRequest
                {
                    Method = method,
                    Path = Encoding.UTF8.GetString(pathSpan),
                    BodyReader = reader
                };

                // Parse headers
                var headerSpan = span[reqLineConsumed..];
                int totalConsumed = reqLineConsumed;

                while (headerSpan.Length > 0)
                {
                    if (headerSpan.StartsWith("\r\n"u8))
                    {
                        totalConsumed += 2;
                        break;
                    }
                    if (headerSpan.StartsWith("\n"u8))
                    {
                        totalConsumed += 1;
                        break;
                    }

                    if (HttpSimdParser.TryParseHeader(headerSpan, out var hName, out var hVal, out int hConsumed))
                    {
                        request.Headers[Encoding.UTF8.GetString(hName)] = Encoding.UTF8.GetString(hVal);
                        totalConsumed += hConsumed;
                        headerSpan = span[totalConsumed..];
                    }
                    else
                    {
                        break;
                    }
                }

                var response = new HttpResponse(writer);
                var context = new HttpContext(request, response, ct);

                if (_router.TryMatch(method, pathSpan, out var match))
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

                reader.AdvanceTo(buffer.GetPosition(totalConsumed));

                // Check Connection header
                if (request.Header("Connection")?.Equals("close", StringComparison.OrdinalIgnoreCase) == true)
                {
                    break;
                }
            }
        }
        catch
        {
            // Client disconnect
        }
        finally
        {
            await reader.CompleteAsync().ConfigureAwait(false);
            await writer.CompleteAsync().ConfigureAwait(false);
        }
    }

    public void Stop()
    {
        _cts?.Cancel();
        _listener?.Stop();
        _listener = null;
    }

    public void Dispose() => Stop();
}

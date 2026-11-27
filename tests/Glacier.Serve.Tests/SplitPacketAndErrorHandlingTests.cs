using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Inference.Bounce;
using Glacier.Serve.Inference.Http;
using Glacier.Serve.Serialization;
using Glacier.Serve.Server;
using Xunit;

namespace Glacier.Serve.Tests;

public sealed class SplitPacketAndErrorHandlingTests
{
    private static int GetAvailablePort()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    private static async Task<(int StatusCode, Dictionary<string, string> Headers, string Body)> ReadRawHttpResponseAsync(NetworkStream stream, CancellationToken ct)
    {
        using var ms = new MemoryStream();
        byte[] buf = new byte[4096];
        int headerEnd = -1;

        while (headerEnd < 0)
        {
            int n = await stream.ReadAsync(buf, ct);
            if (n == 0)
                throw new EndOfStreamException("Server closed connection before sending HTTP response headers.");
            ms.Write(buf, 0, n);
            byte[] arr = ms.ToArray();
            headerEnd = arr.AsSpan().IndexOf("\r\n\r\n"u8);
        }

        byte[] raw = ms.ToArray();
        string headerText = Encoding.ASCII.GetString(raw, 0, headerEnd);
        string[] lines = headerText.Split("\r\n");
        string[] statusParts = lines[0].Split(' ', 3);
        int statusCode = int.Parse(statusParts[1]);

        var headers = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (int i = 1; i < lines.Length; i++)
        {
            int colon = lines[i].IndexOf(':');
            if (colon > 0)
            {
                headers[lines[i][..colon].Trim()] = lines[i][(colon + 1)..].Trim();
            }
        }

        int bodyStart = headerEnd + 4;
        int contentLength = headers.TryGetValue("Content-Length", out var clVal) && int.TryParse(clVal, out int cl) ? cl : -1;

        using var bodyMs = new MemoryStream();
        if (raw.Length > bodyStart)
        {
            bodyMs.Write(raw, bodyStart, raw.Length - bodyStart);
        }

        if (contentLength >= 0)
        {
            while (bodyMs.Length < contentLength)
            {
                int n = await stream.ReadAsync(buf, ct);
                if (n == 0) break;
                bodyMs.Write(buf, 0, n);
            }
            byte[] fullBody = bodyMs.ToArray();
            return (statusCode, headers, Encoding.UTF8.GetString(fullBody, 0, Math.Min(contentLength, fullBody.Length)));
        }

        while (true)
        {
            int n = await stream.ReadAsync(buf, ct);
            if (n == 0) break;
            bodyMs.Write(buf, 0, n);
        }

        return (statusCode, headers, Encoding.UTF8.GetString(bodyMs.ToArray()));
    }

    [Fact]
    public async Task GlacierServeApp_SplitPacketContentLength_AndKeepAlive_Succeeds()
    {
        int port = GetAvailablePort();
        using var app = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .MapPost("/echo", async ctx =>
            {
                await ctx.Response.WriteBytesAsync(ctx.Request.Body, ctx.RequestAborted);
            })
            .Build();

        await app.StartAsync();

        using var tcp = new TcpClient { NoDelay = true };
        await tcp.ConnectAsync(IPAddress.Loopback, port);
        await using var stream = tcp.GetStream();

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));

        // Request 1: headers split across 2 TCP writes, delay 20ms, then body split across 2 TCP writes
        string body1 = "{\"message\":\"split-packet-1\",\"count\":42}";
        byte[] body1Bytes = Encoding.UTF8.GetBytes(body1);
        string hdr1a = "POST /echo HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n";
        string hdr1b = $"Content-Length: {body1Bytes.Length}\r\n\r\n";

        await stream.WriteAsync(Encoding.ASCII.GetBytes(hdr1a), cts.Token);
        await stream.FlushAsync(cts.Token);
        await Task.Delay(20, cts.Token);

        await stream.WriteAsync(Encoding.ASCII.GetBytes(hdr1b), cts.Token);
        await stream.FlushAsync(cts.Token);
        await Task.Delay(20, cts.Token);

        int mid1 = body1Bytes.Length / 2;
        await stream.WriteAsync(body1Bytes.AsMemory(0, mid1), cts.Token);
        await stream.FlushAsync(cts.Token);
        await Task.Delay(20, cts.Token);

        await stream.WriteAsync(body1Bytes.AsMemory(mid1), cts.Token);
        await stream.FlushAsync(cts.Token);

        var resp1 = await ReadRawHttpResponseAsync(stream, cts.Token);
        Assert.Equal(200, resp1.StatusCode);
        Assert.Equal(body1, resp1.Body);

        // Request 2 on the SAME keep-alive connection: headers, delay 20ms, body
        string body2 = "{\"message\":\"keep-alive-split-2\"}";
        byte[] body2Bytes = Encoding.UTF8.GetBytes(body2);
        string hdr2 = $"POST /echo HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: {body2Bytes.Length}\r\nConnection: close\r\n\r\n";

        await stream.WriteAsync(Encoding.ASCII.GetBytes(hdr2), cts.Token);
        await stream.FlushAsync(cts.Token);
        await Task.Delay(20, cts.Token);

        await stream.WriteAsync(body2Bytes, cts.Token);
        await stream.FlushAsync(cts.Token);

        var resp2 = await ReadRawHttpResponseAsync(stream, cts.Token);
        Assert.Equal(200, resp2.StatusCode);
        Assert.Equal(body2, resp2.Body);

        app.Stop();
    }

    [Fact]
    public async Task GlacierServeApp_SplitPacketChunkedTransferEncoding_AndKeepAlive_Succeeds()
    {
        int port = GetAvailablePort();
        using var app = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .MapPost("/echo", async ctx =>
            {
                await ctx.Response.WriteBytesAsync(ctx.Request.Body, ctx.RequestAborted);
            })
            .Build();

        await app.StartAsync();

        using var tcp = new TcpClient { NoDelay = true };
        await tcp.ConnectAsync(IPAddress.Loopback, port);
        await using var stream = tcp.GetStream();

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));

        // Request 1: chunked request split across headers, chunk-size, chunk-data, CRLF, and zero-terminator
        string headers1 = "POST /echo HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: chunked\r\n\r\n";
        string[] segments1 =
        [
            headers1,
            "6\r",
            "\n{\"x\":",
            "1\r",
            "\n7\r\n",
            ",\"y\":2}\r\n",
            "0\r\n",
            "\r\n"
        ];

        foreach (string seg in segments1)
        {
            await stream.WriteAsync(Encoding.ASCII.GetBytes(seg), cts.Token);
            await stream.FlushAsync(cts.Token);
            await Task.Delay(15, cts.Token);
        }

        var resp1 = await ReadRawHttpResponseAsync(stream, cts.Token);
        Assert.Equal(200, resp1.StatusCode);
        Assert.Equal("{\"x\":1,\"y\":2}", resp1.Body);

        // Request 2 on the same keep-alive connection with chunked body + trailer header
        string[] segments2 =
        [
            "POST /echo HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
            "5\r\nhello\r\n",
            "6\r\n world\r\n0\r\nX-Trailer: ok\r\n\r\n"
        ];

        foreach (string seg in segments2)
        {
            await stream.WriteAsync(Encoding.ASCII.GetBytes(seg), cts.Token);
            await stream.FlushAsync(cts.Token);
            await Task.Delay(15, cts.Token);
        }

        var resp2 = await ReadRawHttpResponseAsync(stream, cts.Token);
        Assert.Equal(200, resp2.StatusCode);
        Assert.Equal("hello world", resp2.Body);

        app.Stop();
    }

    [Fact]
    public async Task MapPeerBounce_SplitPacketPost_Returns200ChatCompletion()
    {
        var mockHandler = new StubHttpMessageHandler((req, _) =>
        {
            if (req.Method == HttpMethod.Get && req.RequestUri!.AbsolutePath == "/v1/models")
            {
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent("{\"object\":\"list\",\"data\":[{\"id\":\"mock-primary\"}]}", Encoding.UTF8, "application/json")
                });
            }

            if (req.Method == HttpMethod.Post && req.RequestUri!.AbsolutePath == "/v1/chat/completions")
            {
                var completion = new ChatCompletionResponse
                {
                    Id = "mock-split",
                    Created = 1710000000,
                    Model = "mock-primary",
                    Choices =
                    [
                        new ChatChoice
                        {
                            Index = 0,
                            Message = new ChatMessage { Role = "assistant", Content = "FINAL_ANSWER: 99" },
                            FinishReason = "stop"
                        }
                    ],
                    Usage = new ChatUsage { PromptTokens = 4, CompletionTokens = 3, TotalTokens = 7 }
                };
                string json = JsonSerializer.Serialize(completion, ServeJsonContext.Default.ChatCompletionResponse);
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(json, Encoding.UTF8, "application/json")
                });
            }

            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        });

        using var outboundHttp = new HttpClient(mockHandler);
        var bounceService = new PeerBounceService(outboundHttp, new SandboxedProcessBounceVerifier(null), "http://mock-primary.local:18080");

        int port = GetAvailablePort();
        using var app = GlacierServeApp.CreateBuilder().UsePort(port).Build();
        app.MapPeerBounce(bounceService, mapOpenAiAndHealthRoutes: true, modelAlias: "Glacier-Bounce-Split");
        await app.StartAsync();

        string body = JsonSerializer.Serialize(new BounceChatRequest
        {
            Model = "Glacier-Bounce-Split",
            Messages = [new ChatMessage { Role = "user", Content = "Return FINAL_ANSWER: 99" }],
            Stream = false
        }, ServeJsonContext.Default.BounceChatRequest);
        byte[] bodyBytes = Encoding.UTF8.GetBytes(body);

        using var tcp = new TcpClient { NoDelay = true };
        await tcp.ConnectAsync(IPAddress.Loopback, port);
        await using var stream = tcp.GetStream();
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));

        string headers = $"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {bodyBytes.Length}\r\nConnection: close\r\n\r\n";
        await stream.WriteAsync(Encoding.ASCII.GetBytes(headers), cts.Token);
        await stream.FlushAsync(cts.Token);
        await Task.Delay(20, cts.Token);

        await stream.WriteAsync(bodyBytes, cts.Token);
        await stream.FlushAsync(cts.Token);

        var resp = await ReadRawHttpResponseAsync(stream, cts.Token);
        Assert.Equal(200, resp.StatusCode);
        var chat = JsonSerializer.Deserialize(resp.Body, ServeJsonContext.Default.ChatCompletionResponse);
        Assert.NotNull(chat);
        Assert.Equal("FINAL_ANSWER: 99", chat.Choices[0].Message.Content);

        app.Stop();
    }

    [Fact]
    public async Task MapPeerBounce_UnreachablePrimaryAndTimeout_Returns502And504JsonInsteadOfSocketReset()
    {
        int deadPort = GetAvailablePort();
        using var realHttp = new HttpClient { Timeout = TimeSpan.FromSeconds(3) };
        var unreachableBounce = new PeerBounceService(
            realHttp,
            new SandboxedProcessBounceVerifier(null),
            $"http://127.0.0.1:{deadPort}");

        int bouncePort = GetAvailablePort();
        using var app = GlacierServeApp.CreateBuilder().UsePort(bouncePort).Build();
        app.MapPeerBounce(unreachableBounce, mapOpenAiAndHealthRoutes: true, modelAlias: "Glacier-Bounce-Down");

        // Also test timeout -> 504 on /timeout/v1/chat/completions via a second PeerBounceService
        using var timeoutHttp = new HttpClient(new StubHttpMessageHandler((_, ct) =>
            Task.FromException<HttpResponseMessage>(new TaskCanceledException("Simulated upstream timeout", new TimeoutException(), ct))));
        var timeoutBounce = new PeerBounceService(
            timeoutHttp,
            new SandboxedProcessBounceVerifier(null),
            "http://mock-timeout.local:18080");
        app.MapPeerBounce(
            timeoutBounce,
            new PeerBounceEndpointOptions(
                ModelAlias: "Glacier-Bounce-Timeout",
                MapOpenAiAndHealthRoutes: false,
                BounceChatPath: "/api/bounce-timeout/chat/completions"));

        await app.StartAsync();

        using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{bouncePort}") };
        string reqJson = JsonSerializer.Serialize(new BounceChatRequest
        {
            Model = "Glacier-Bounce-Down",
            Messages = [new ChatMessage { Role = "user", Content = "Hello" }],
            Stream = false
        }, ServeJsonContext.Default.BounceChatRequest);

        // 1. Unreachable --primary returns clean HTTP 502 JSON ApiErrorResponse (no ConnectionResetError / 0 bytes)
        using var resp502 = await client.PostAsync("/v1/chat/completions", new StringContent(reqJson, Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.BadGateway, resp502.StatusCode);
        string body502 = await resp502.Content.ReadAsStringAsync();
        var err502 = JsonSerializer.Deserialize(body502, ServeJsonContext.Default.ApiErrorResponse);
        Assert.NotNull(err502);
        Assert.Equal("bad_gateway", err502.Error.Type);
        Assert.False(string.IsNullOrWhiteSpace(err502.Error.Message));

        // 2. Upstream timeout returns clean HTTP 504 JSON ApiErrorResponse
        using var resp504 = await client.PostAsync("/api/bounce-timeout/chat/completions", new StringContent(reqJson, Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.GatewayTimeout, resp504.StatusCode);
        string body504 = await resp504.Content.ReadAsStringAsync();
        var err504 = JsonSerializer.Deserialize(body504, ServeJsonContext.Default.ApiErrorResponse);
        Assert.NotNull(err504);
        Assert.Equal("gateway_timeout", err504.Error.Type);
        Assert.False(string.IsNullOrWhiteSpace(err504.Error.Message));

        app.Stop();
    }

    [Fact]
    public async Task GlacierServeApp_PreHeaderRouteException_Returns500JsonInsteadOfClosingWithZeroBytes()
    {
        int port = GetAvailablePort();
        using var app = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .MapGet("/boom", _ => throw new InvalidOperationException("simulated pre-header failure"))
            .Build();

        await app.StartAsync();

        using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };
        using var resp = await client.GetAsync("/boom");
        Assert.Equal(HttpStatusCode.InternalServerError, resp.StatusCode);

        string json = await resp.Content.ReadAsStringAsync();
        var err = JsonSerializer.Deserialize(json, ServeJsonContext.Default.ApiErrorResponse);
        Assert.NotNull(err);
        Assert.Equal("internal_server_error", err.Error.Type);
        Assert.Equal("simulated pre-header failure", err.Error.Message);

        app.Stop();
    }

    [Theory]
    [InlineData("POST /echo HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 4\r\nTransfer-Encoding: chunked\r\n\r\n4\r\ntest\r\n0\r\n\r\n")]
    [InlineData("POST /echo HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 4\r\nContent-Length: 8\r\n\r\ntest")]
    [InlineData("GET /echo HTTP/1.1\r\nHost: 127.0.0.1\r\nBad Header : 1\r\n\r\n")]
    [InlineData("GET /echo HTTP/1.1\r\nHost: 127.0.0.1\n\n")]
    public async Task GlacierServeApp_RejectsHttpSmugglingAndMalformedHeaders_With400(string rawRequest)
    {
        int port = GetAvailablePort();
        using var app = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .MapGet("/echo", ctx => ctx.Response.WriteUtf8Async("ok"))
            .MapPost("/echo", ctx => ctx.Response.WriteUtf8Async("ok"))
            .Build();

        await app.StartAsync();

        using var tcp = new TcpClient { NoDelay = true };
        await tcp.ConnectAsync(IPAddress.Loopback, port);
        await using var stream = tcp.GetStream();
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));

        await stream.WriteAsync(Encoding.ASCII.GetBytes(rawRequest), cts.Token);
        await stream.FlushAsync(cts.Token);

        var resp = await ReadRawHttpResponseAsync(stream, cts.Token);
        Assert.Equal(400, resp.StatusCode);

        app.Stop();
    }

    private sealed class StubHttpMessageHandler : HttpMessageHandler
    {
        private readonly Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> _handler;

        public StubHttpMessageHandler(Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> handler)
        {
            _handler = handler;
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            => _handler(request, cancellationToken);
    }
}

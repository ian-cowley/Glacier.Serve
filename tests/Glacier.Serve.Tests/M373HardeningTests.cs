namespace Glacier.Serve.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Pipelines;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Security;
using System.Net.Sockets;
using System.Security.Authentication;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Formats.Arrow;
using Glacier.Polaris;
using Glacier.Polaris.Data;
using Glacier.Serve.Core;
using Glacier.Serve.Daemons;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Inference.Bounce;
using Glacier.Serve.Inference.Http;
using Glacier.Serve.Inference.PagedAttention;
using Glacier.Serve.Parsing;
using Glacier.Serve.Routing;
using Glacier.Serve.Serialization;
using Glacier.Serve.Server;
using Xunit;
using ServeHttpMethod = Glacier.Serve.Routing.HttpMethod;

public sealed class M373HardeningTests
{
    private static int GetAvailablePort()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    private static async Task<(int StatusCode, Dictionary<string, string> Headers, string Body)> SendRawRequestAsync(
        int port,
        string rawRequest,
        CancellationToken ct = default)
    {
        using var tcp = new TcpClient { NoDelay = true };
        await tcp.ConnectAsync(IPAddress.Loopback, port, ct);
        await using var stream = tcp.GetStream();

        byte[] reqBytes = Encoding.ASCII.GetBytes(rawRequest);
        await stream.WriteAsync(reqBytes, ct);
        await stream.FlushAsync(ct);

        return await ReadRawHttpResponseAsync(stream, ct);
    }

    private static async Task<(int StatusCode, Dictionary<string, string> Headers, string Body)> ReadRawHttpResponseAsync(
        NetworkStream stream,
        CancellationToken ct)
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
            headerEnd = ms.ToArray().AsSpan().IndexOf("\r\n\r\n"u8);
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

    [Theory]
    [InlineData("GET /echo HTTP/1.1\r\n\r\n")] // Missing Host on HTTP/1.1
    [InlineData("GET /echo HTTP/1.1\r\nHost: a\r\nHost: b\r\n\r\n")] // Duplicate Host on HTTP/1.1
    [InlineData("GET /echo HTTP/1.0\r\nHost: a\r\nHost: b\r\n\r\n")] // Duplicate Host on HTTP/1.0
    [InlineData("GET /echo HTTP/1.1\r\nHost: \r\n\r\n")] // Empty Host
    [InlineData("GET /echo HTTP/1.1\r\nHost: user@evil.com\r\n\r\n")] // Userinfo in Host
    [InlineData("GET /echo HTTP/1.1\r\nHost: evil.com/path\r\n\r\n")] // Slash in Host
    [InlineData("GET /echo HTTP/1.1\r\nHost: evil.com:99999\r\n\r\n")] // Out-of-range port in Host
    [InlineData("GET /echo HTTP/1.1\r\nHost: bad host\r\n\r\n")] // Whitespace in Host
    [InlineData("GET /echo HTTP/1.1\r\nHost: foo]bar\r\n\r\n")] // Stray ] in Host
    [InlineData("GET /echo HTTP/1.1\r\nHost: foo[bar\r\n\r\n")] // Stray [ in Host
    [InlineData("POST /echo HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n")] // Duplicate TE
    [InlineData("POST /echo HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: notchunked\r\n\r\n0\r\n\r\n")] // Non-token substring
    [InlineData("POST /echo HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked-ext\r\n\r\n0\r\n\r\n")] // Invalid coding
    [InlineData("POST /echo HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked, gzip\r\n\r\n0\r\n\r\n")] // chunked not final
    [InlineData("POST /echo HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: gzip, chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n")] // Un-decoded gzip transfer coding
    [InlineData("POST /echo HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: deflate, chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n")] // Un-decoded deflate transfer coding
    [InlineData("POST /echo HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: compress, chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n")] // Un-decoded compress transfer coding
    [InlineData("POST /echo HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked, chunked\r\n\r\n0\r\n\r\n")] // Duplicate chunked token
    [InlineData("POST /echo HTTP/1.1\r\nHost: localhost\r\nContent-Length: +5\r\n\r\nhello")] // Plus sign in Content-Length
    [InlineData("POST /echo HTTP/1.1\r\nHost: localhost\r\nContent-Length: -1\r\n\r\n")] // Negative Content-Length
    [InlineData("POST /echo HTTP/1.1\r\nHost: localhost\r\nContent-Length: 99999999999999999999999999\r\n\r\n")] // Overflow Content-Length
    [InlineData("POST /echo HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n 5\r\nhello\r\n0\r\n\r\n")] // Leading whitespace in chunk size
    [InlineData("POST /echo HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n5 \r\nhello\r\n0\r\n\r\n")] // Trailing whitespace in chunk size without extension
    [InlineData("POST /echo HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n4\ntest\r\n0\r\n\r\n")] // Bare LF chunk size line
    [InlineData("POST /echo HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n4\rtest\r\n0\r\n\r\n")] // Bare CR chunk size line
    [InlineData("POST /echo HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n4\r\ntest\n0\r\n\r\n")] // Bare LF chunk data terminator
    [InlineData("POST /echo HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n0\r\nX-Trailer: val\n\r\n")] // Bare LF chunk trailer
    [InlineData("POST /echo HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n0\r\nHost: smuggled\r\n\r\n")] // Forbidden Host trailer
    [InlineData("POST /echo HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n0\r\nContent-Length: 5\r\n\r\n")] // Forbidden CL trailer
    [InlineData("POST /echo HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n0\r\nTransfer-Encoding: chunked\r\n\r\n")] // Forbidden TE trailer
    public async Task Rfc9112_RejectsSmugglingAndMalformedFraming_With400(string rawRequest)
    {
        int port = GetAvailablePort();
        using var app = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .MapGet("/echo", ctx => ctx.Response.WriteUtf8Async("ok"))
            .MapPost("/echo", ctx => ctx.Response.WriteBytesAsync(ctx.Request.Body))
            .Build();

        await app.StartAsync();
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        var resp = await SendRawRequestAsync(port, rawRequest, cts.Token);
        Assert.Equal(400, resp.StatusCode);
        app.Stop();
    }

    [Fact]
    public async Task Http11_HalfCloseShutdownSend_DoesNotCancelRequestAborted()
    {
        int port = GetAvailablePort();
        using var app = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .MapGet("/slow", async ctx =>
            {
                await Task.Delay(150);
                await ctx.Response.WriteUtf8Async(ctx.RequestAborted.IsCancellationRequested ? "aborted" : "ok");
            })
            .Build();

        await app.StartAsync();
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        using var tcp = new TcpClient { NoDelay = true };
        await tcp.ConnectAsync(IPAddress.Loopback, port, cts.Token);
        await using var stream = tcp.GetStream();

        await stream.WriteAsync("GET /slow HTTP/1.1\r\nHost: localhost\r\n\r\n"u8.ToArray(), cts.Token);
        await stream.FlushAsync(cts.Token);
        tcp.Client.Shutdown(SocketShutdown.Send);

        var resp = await ReadRawHttpResponseAsync(stream, cts.Token);
        Assert.Equal(200, resp.StatusCode);
        Assert.Equal("ok", resp.Body);

        app.Stop();
    }

    [Fact]
    public async Task GlacierServeApp_DisposeWithSlowHandlerBeyondDrainTimeout_DoesNotThrowObjectDisposed()
    {
        int port = GetAvailablePort();
        var handlerStarted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var handlerExited = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        var app = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .MapGet("/hold", async ctx =>
            {
                handlerStarted.TrySetResult();
                try
                {
                    await Task.Delay(2300);
                    if (!ctx.RequestAborted.IsCancellationRequested)
                    {
                        await ctx.Response.WriteUtf8Async("done");
                    }
                }
                finally
                {
                    handlerExited.TrySetResult();
                }
            })
            .Build();

        await app.StartAsync();
        using var tcp = new TcpClient { NoDelay = true };
        await tcp.ConnectAsync(IPAddress.Loopback, port);
        await using var stream = tcp.GetStream();
        await stream.WriteAsync("GET /hold HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"u8.ToArray());
        await stream.FlushAsync();

        await handlerStarted.Task.WaitAsync(TimeSpan.FromSeconds(5));
        app.Dispose();
        await handlerExited.Task.WaitAsync(TimeSpan.FromSeconds(5));
    }

    [Fact]
    public async Task Rfc9112_AcceptsHttp10WithoutHost_AndHttp11Ipv6Host_AndRejectsTrailerOverflowAndEof()
    {
        int port = GetAvailablePort();
        using var app = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .MapGet("/echo", ctx => ctx.Response.WriteUtf8Async("ok"))
            .MapPost("/echo", ctx => ctx.Response.WriteBytesAsync(ctx.Request.Body))
            .Build();

        await app.StartAsync();
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));

        var http10Resp = await SendRawRequestAsync(port, "GET /echo HTTP/1.0\r\n\r\n", cts.Token);
        Assert.Equal(200, http10Resp.StatusCode);

        var ipv6HostResp = await SendRawRequestAsync(port, "GET /echo HTTP/1.1\r\nHost: [::1]:8080\r\nConnection: close\r\n\r\n", cts.Token);
        Assert.Equal(200, ipv6HostResp.StatusCode);

        // Excess trailer lines (> 64) -> 400
        var sbTrailers = new StringBuilder("POST /echo HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n");
        for (int i = 0; i < 66; i++)
            sbTrailers.Append($"X-T{i}: v\r\n");
        sbTrailers.Append("\r\n");
        var trailerCountResp = await SendRawRequestAsync(port, sbTrailers.ToString(), cts.Token);
        Assert.Equal(400, trailerCountResp.StatusCode);

        // EOF mid-trailer before terminating empty CRLF -> 400
        using (var tcp = new TcpClient { NoDelay = true })
        {
            await tcp.ConnectAsync(IPAddress.Loopback, port, cts.Token);
            await using var stream = tcp.GetStream();
            await stream.WriteAsync("POST /echo HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n0\r\nX-Partial: 1\r\n"u8.ToArray(), cts.Token);
            await stream.FlushAsync(cts.Token);
            tcp.Client.Shutdown(SocketShutdown.Send);
            var eofResp = await ReadRawHttpResponseAsync(stream, cts.Token);
            Assert.Equal(400, eofResp.StatusCode);
        }

        app.Stop();
    }

    [Fact]
    public async Task Slowloris_BodyAndChunkReads_EnforceHeaderTimeout_With408()
    {
        int port = GetAvailablePort();
        using var app = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .WithHeaderTimeout(TimeSpan.FromMilliseconds(150))
            .MapPost("/echo", ctx => ctx.Response.WriteBytesAsync(ctx.Request.Body))
            .Build();

        await app.StartAsync();
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));

        // 1. Stalled Content-Length body
        using (var tcpCl = new TcpClient { NoDelay = true })
        {
            await tcpCl.ConnectAsync(IPAddress.Loopback, port, cts.Token);
            await using var sCl = tcpCl.GetStream();
            await sCl.WriteAsync("POST /echo HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100\r\n\r\nabc"u8.ToArray(), cts.Token);
            await sCl.FlushAsync(cts.Token);

            var resp = await ReadRawHttpResponseAsync(sCl, cts.Token);
            Assert.Equal(408, resp.StatusCode);
        }

        // 2. Stalled chunked body
        using (var tcpChunk = new TcpClient { NoDelay = true })
        {
            await tcpChunk.ConnectAsync(IPAddress.Loopback, port, cts.Token);
            await using var sChunk = tcpChunk.GetStream();
            await sChunk.WriteAsync("POST /echo HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n20\r\nhello"u8.ToArray(), cts.Token);
            await sChunk.FlushAsync(cts.Token);

            var resp = await ReadRawHttpResponseAsync(sChunk, cts.Token);
            Assert.Equal(408, resp.StatusCode);
        }

        app.Stop();
    }

    [Fact]
    public void HttpSimdParser_SimdLengthsAndHeaderValueCtlGuard()
    {
        // Reject CTL / NUL in header value
        Assert.False(HttpSimdParser.TryParseHeader("X-Test: abc\0def\r\n"u8, out _, out _, out _));
        Assert.False(HttpSimdParser.TryParseHeader("X-Test: abc\u0007def\r\n"u8, out _, out _, out _));
        Assert.True(HttpSimdParser.TryParseHeader("X-Test: abc\tdef\r\n"u8, out _, out var val, out _));
        Assert.Equal("abc\tdef", Encoding.ASCII.GetString(val));

        // Request line returns version
        Assert.True(HttpSimdParser.TryParseRequestLine("GET /v1/models HTTP/1.1\r\n"u8, out _, out _, out var v11, out _));
        Assert.Equal(HttpProtocolVersion.Http11, v11);
        Assert.True(HttpSimdParser.TryParseRequestLine("GET /v1/models HTTP/1.0\r\n"u8, out _, out _, out var v10, out _));
        Assert.Equal(HttpProtocolVersion.Http10, v10);

        // Vector128 / Vector256 / Vector512 boundary sweep
        foreach (int len in new[] { 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 130 })
        {
            byte[] buf = new byte[len];
            buf.AsSpan().Fill((byte)'x');
            int targetPos = len - 1;
            buf[targetPos] = (byte)':';
            Assert.Equal(targetPos, HttpSimdParser.IndexOfByteVector256(buf, (byte)':'));

            buf[targetPos] = (byte)'\n';
            Assert.Equal(targetPos, HttpSimdParser.IndexOfCrlfVector256(buf));
        }
    }

    [Fact]
    public async Task BearerAuth_SupportsTrailingSlashExemptPaths_XApiKey_CaseInsensitiveBearer_AndFixedTimeHash()
    {
        int port = GetAvailablePort();
        const string secret = "super-secret-token-with-non-trivial-length";
        using var app = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .UseBearerAuth(secret, "/health/")
            .MapGet("/health", ctx => ctx.Response.WriteUtf8Async("ok"))
            .MapGet("/secure", ctx => ctx.Response.WriteUtf8Async("authorized"))
            .Build();

        await app.StartAsync();
        using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

        // Trailing slash /health/ matches exempt /health
        Assert.Equal(HttpStatusCode.OK, (await client.GetAsync("/health/")).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await client.GetAsync("/health")).StatusCode);

        // x-api-key accepted
        using (var apiKeyReq = new System.Net.Http.HttpRequestMessage(System.Net.Http.HttpMethod.Get, "/secure"))
        {
            apiKeyReq.Headers.TryAddWithoutValidation("x-api-key", $"  {secret} ");
            var apiKeyResp = await client.SendAsync(apiKeyReq);
            Assert.Equal(HttpStatusCode.OK, apiKeyResp.StatusCode);
        }

        // Case-insensitive bearer prefix + trimmed whitespace accepted
        using (var bearerReq = new System.Net.Http.HttpRequestMessage(System.Net.Http.HttpMethod.Get, "/secure"))
        {
            bearerReq.Headers.TryAddWithoutValidation("Authorization", $"bearer {secret} ");
            var bearerResp = await client.SendAsync(bearerReq);
            Assert.Equal(HttpStatusCode.OK, bearerResp.StatusCode);
        }

        // Different length wrong token rejected with 401
        using (var badReq = new System.Net.Http.HttpRequestMessage(System.Net.Http.HttpMethod.Get, "/secure"))
        {
            badReq.Headers.TryAddWithoutValidation("x-api-key", "short");
            var badResp = await client.SendAsync(badReq);
            Assert.Equal(HttpStatusCode.Unauthorized, badResp.StatusCode);
        }

        app.Stop();
    }

    [Fact]
    public async Task RadixTreeRouter_ConcurrentAddRouteAndTryMatch_IsThreadSafe()
    {
        var router = new RadixTreeRouter();
        router.AddRoute(ServeHttpMethod.Get, "/health", _ => ValueTask.CompletedTask);

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(1.5));
        var writer = Task.Run(() =>
        {
            for (int i = 0; i < 200 && !cts.IsCancellationRequested; i++)
            {
                router.AddRoute(ServeHttpMethod.Get, $"/v1/routes/item_{i}/sub", _ => ValueTask.CompletedTask);
                router.AddRoute(ServeHttpMethod.Post, $"/v1/models/{{id}}/run_{i % 10}", _ => ValueTask.CompletedTask);
            }
        });

        var readers = Enumerable.Range(0, 4).Select(_ => Task.Run(() =>
        {
            while (!writer.IsCompleted && !cts.IsCancellationRequested)
            {
                Assert.True(router.TryMatch(ServeHttpMethod.Get, "/health"u8, out var match));
                Assert.NotNull(match.Handler);
                router.TryMatch(ServeHttpMethod.Get, "/v1/routes/item_42/sub"u8, out RouteMatch _);
                router.TryMatch(ServeHttpMethod.Post, "/v1/models/qwen/run_3"u8, out RouteMatch _);
            }
        })).ToArray();

        await Task.WhenAll([writer, .. readers]);
    }

    [Fact]
    public async Task SleepMemoryService_NullOptionsConstructor_InitializesCleanlyWithoutThrowing()
    {
        await using var sleep = new SleepMemoryService(options: null);
        Assert.True(sleep.TokenBudget > 0);
        var status = sleep.GetStatus();
        Assert.True(status.Enabled);
    }

    [Fact]
    public void PagedBlockPool_Rejects64BitHashCollision_AndEvictsLeastRecentlyUsedBlock()
    {
        using var pool = new PagedBlockPool(totalBlocks: 3, layers: 1, headsKv: 1, headDim: 8, blockSize: 4);

        int[] tokensA = [1, 2, 3, 4];
        int[] tokensB = [5, 6, 7, 8];
        int[] tokensC = [9, 10, 11, 12];
        const ulong sharedCollisionHash = 0xCAFEBABE12345678UL;

        Assert.True(pool.TryAllocateBlock(out int bA));
        pool.BindPrefixBlock(sharedCollisionHash, bA, tokensA);

        // Collision check: exact same 64-bit hash with different tokens MUST NOT match!
        Assert.False(pool.TryFindPrefixBlock(sharedCollisionHash, tokensB, out int mismatched));
        Assert.Equal(-1, mismatched);
        Assert.Equal(1, pool.GetRefCount(bA));

        // Matching tokens succeed
        Assert.True(pool.TryFindPrefixBlock(sharedCollisionHash, tokensA, out int matchedA));
        Assert.Equal(bA, matchedA);
        pool.ReleaseBlock(matchedA);
        pool.ReleaseBlock(bA);

        // Clean pool for LRU test with distinct hashes
        Assert.Equal(1, pool.EvictPrefixCache());
        ulong hA = PagedBlockPool.ComputePrefixHash(1UL, tokensA);
        ulong hB = PagedBlockPool.ComputePrefixHash(1UL, tokensB);
        ulong hC = PagedBlockPool.ComputePrefixHash(1UL, tokensC);

        Assert.True(pool.TryAllocateBlock(out int idA));
        Assert.True(pool.TryAllocateBlock(out int idB));
        Assert.True(pool.TryAllocateBlock(out int idC));
        pool.BindPrefixBlock(hA, idA, tokensA);
        pool.BindPrefixBlock(hB, idB, tokensB);
        pool.BindPrefixBlock(hC, idC, tokensC);

        // Release order: A, B, C -> LRU head is A
        pool.ReleaseBlock(idA);
        pool.ReleaseBlock(idB);
        pool.ReleaseBlock(idC);
        Assert.Equal(3, pool.FreeBlocksCount);
        Assert.Equal(3, pool.CachedPrefixBlocksCount);

        // Touch A so B becomes least-recently-used
        Assert.True(pool.TryFindPrefixBlock(hA, tokensA, out int hitA));
        Assert.Equal(idA, hitA);
        pool.ReleaseBlock(hitA);

        // Allocating 1 new block must evict B (LRU), keeping A and C cached
        Assert.True(pool.TryAllocateBlock(out int evictedId));
        Assert.Equal(idB, evictedId);
        Assert.False(pool.TryFindPrefixBlock(hB, tokensB, out _));
        Assert.True(pool.TryFindPrefixBlock(hA, tokensA, out int stillA));
        Assert.True(pool.TryFindPrefixBlock(hC, tokensC, out int stillC));
        pool.ReleaseBlock(stillA);
        pool.ReleaseBlock(stillC);
        pool.ReleaseBlock(evictedId);
    }

    [Fact]
    public async Task DataFrameHttpExtensions_SerializesTypedColumnsAndLargeArrowIpcAbove64Kb()
    {
        const int rows = 2500;
        var names = new string[rows];
        var cats = new string[rows];
        var nums = new int?[rows];
        var floats = new float?[rows];
        for (int i = 0; i < rows; i++)
        {
            names[i] = $"row_{i}_glacier_arrow_payload";
            cats[i] = (i % 3) switch { 0 => "NA", 1 => "EU", _ => "APAC" };
            nums[i] = i % 10 == 0 ? null : i;
            floats[i] = i == 5 ? float.NaN : (i * 0.5f);
        }

        using var sName = new Utf8StringSeries("Name", names);
        sName.ValidityMask.SetNull(2);
        using var sCat = CategoricalSeries.FromStrings("Region", cats);
        using var sNum = Int32Series.FromValues("Count", nums);
        using var sFloat = Float32Series.FromValues("Score", floats);
        using var sBool = BooleanSeries.FromValues("Active", Enumerable.Range(0, rows).Select(i => (bool?)(i % 2 == 0)).ToArray());
        using var df = new DataFrame([sName, sCat, sNum, sFloat, sBool]);

        // 1. JSON typed serialization (Categorical + Utf8 + NaN->null + Nulls)
        // Drain concurrently so payloads > 64 KB Pipe PauseWriterThreshold flow cleanly
        var jsonPipe = new Pipe();
        var jsonResp = new HttpResponse(jsonPipe.Writer);
        Task<byte[]> readJsonTask = ReadAllPipeBytesAsync(jsonPipe.Reader);
        await jsonResp.WriteDataFrameJsonAsync(df);
        await jsonPipe.Writer.CompleteAsync();
        byte[] jsonFull = await readJsonTask;
        int jsonBodyStart = jsonFull.AsSpan().IndexOf("\r\n\r\n"u8) + 4;
        using (var doc = JsonDocument.Parse(jsonFull.AsMemory(jsonBodyStart)))
        {
            var cols = doc.RootElement.GetProperty("columns");
            Assert.Equal(JsonValueKind.Null, cols[0].GetProperty("data")[2].ValueKind);
            Assert.Equal("APAC", cols[1].GetProperty("data")[2].GetString());
            Assert.Equal(JsonValueKind.Null, cols[3].GetProperty("data")[5].ValueKind);
        }

        // 2. Arrow IPC (> 64 KB) streamed without MemoryStream
        var ipcPipe = new Pipe();
        var ipcResp = new HttpResponse(ipcPipe.Writer);
        Task<byte[]> readIpcTask = ReadAllPipeBytesAsync(ipcPipe.Reader);
        await ipcResp.WriteGlacierStorageArrowIpcAsync(df);
        await ipcPipe.Writer.CompleteAsync();
        byte[] ipcFull = await readIpcTask;
        int ipcBodyStart = ipcFull.AsSpan().IndexOf("\r\n\r\n"u8) + 4;
        byte[] ipcBody = ipcFull[ipcBodyStart..];
        Assert.True(ipcBody.Length > 100 * 1024);

        using var reader = new ArrowStreamReader(new MemoryStream(ipcBody));
        ArrowSchema schema = reader.ReadSchema();
        Assert.Equal(5, schema.FieldCount);
        ArrowRecordBatch? batch = reader.ReadNextRecordBatch();
        Assert.NotNull(batch);
        Assert.Equal(rows, batch!.RowCount);
        Assert.Null(reader.ReadNextRecordBatch());
    }

    [Fact]
    public async Task OpenAiEndpoints_ChunkedSseKeepAlive_FinishReasonSanitization_AndQueueBackpressure429()
    {
        Assert.Equal("length", OpenAiEndpoints.SanitizeFinishReason("pool_oom"));
        Assert.Equal("stop", OpenAiEndpoints.SanitizeFinishReason("cancelled"));
        Assert.Equal("stop", OpenAiEndpoints.SanitizeFinishReason(null));

        string tempDir = M4aTestHelpers.CreateTempDirectory("m373_http_backpressure");
        try
        {
            string ggufPath = Path.Combine(tempDir, "mini.gguf");
            string[] vocab = ["<bos>", "<eos>", "a", "b", "c"];
            M4aTestHelpers.WriteSyntheticGguf(ggufPath, vocab, eosTokenId: 1, bosTokenId: 0, dim: 16, ffnDim: 32, nHeads: 2, nHeadsKv: 2, layerCount: 1, nonZeroLayerWeights: true);

            using var engine = new ContinuousBatchEngine(
                ggufPath,
                totalKvBlocks: 16,
                maxBatchSize: 1,
                maxSeqLen: 64,
                seed: 1,
                defaultReasoningLoopMinPeriod: 1,
                useGpu: false,
                prefillChunkSize: 16,
                defaultMaxReasoningTokens: 0,
                kvCachePrecision: Glacier.Inference.Memory.KvCachePrecision.Fp32,
                enableSpeculativeDecoding: false,
                maxDraftTokens: 0,
                speculativeBatchThreshold: 1,
                mtpSidecarPath: null,
                maxWaitingQueueSize: 1);

            int port = GetAvailablePort();
            using var app = GlacierServeApp.CreateBuilder().UsePort(port).Build();
            app.MapInference(engine, "mini-model");
            await app.StartAsync();

            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

            // 1. SSE chunked streaming with keep-alive reuse
            const string streamBody = """{"model":"mini-model","messages":[{"role":"user","content":"a"}],"max_tokens":3,"stream":true}""";
            using (var sseResp = await client.PostAsync("/v1/chat/completions", new StringContent(streamBody, Encoding.UTF8, "application/json")))
            {
                Assert.Equal(HttpStatusCode.OK, sseResp.StatusCode);
                Assert.Contains("chunked", sseResp.Headers.TransferEncoding.ToString(), StringComparison.OrdinalIgnoreCase);
                string sseText = await sseResp.Content.ReadAsStringAsync();
                Assert.Contains("data: [DONE]", sseText, StringComparison.Ordinal);
                Assert.DoesNotContain("pool_oom", sseText, StringComparison.Ordinal);
            }

            // Second request succeeds on keep-alive client
            using var modelsResp = await client.GetAsync("/v1/models");
            Assert.Equal(HttpStatusCode.OK, modelsResp.StatusCode);

            // 2. Bounded queue triggers 429 Too Many Requests with Retry-After when flooded
            const string longReq = """{"model":"mini-model","messages":[{"role":"user","content":"a"}],"max_tokens":30,"stream":false}""";
            var burstTasks = Enumerable.Range(0, 10)
                .Select(_ => client.PostAsync("/v1/chat/completions", new StringContent(longReq, Encoding.UTF8, "application/json")))
                .ToArray();
            var responses = await Task.WhenAll(burstTasks);
            Assert.Contains(responses, r => r.StatusCode == HttpStatusCode.TooManyRequests || r.StatusCode == HttpStatusCode.ServiceUnavailable);
            foreach (var r in responses)
            {
                if (r.StatusCode is HttpStatusCode.TooManyRequests or HttpStatusCode.ServiceUnavailable)
                {
                    Assert.True(r.Headers.RetryAfter != null);
                }
                r.Dispose();
            }

            app.Stop();
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public async Task ClientDisconnect_MidSseStream_ReleasesKvBlocks()
    {
        string tempDir = M4aTestHelpers.CreateTempDirectory("m373_sse_abort");
        try
        {
            string ggufPath = Path.Combine(tempDir, "mini.gguf");
            string[] vocab = ["<bos>", "<eos>", "a", "b", "c"];
            M4aTestHelpers.WriteSyntheticGguf(ggufPath, vocab, eosTokenId: 1, bosTokenId: 0, dim: 16, ffnDim: 32, nHeads: 2, nHeadsKv: 2, layerCount: 1, nonZeroLayerWeights: true);

            using var engine = new ContinuousBatchEngine(
                ggufPath,
                totalKvBlocks: 16,
                maxBatchSize: 2,
                maxSeqLen: 64,
                seed: 2,
                defaultReasoningLoopMinPeriod: 1,
                useGpu: false,
                prefillChunkSize: 16,
                defaultMaxReasoningTokens: 0,
                kvCachePrecision: Glacier.Inference.Memory.KvCachePrecision.Fp32,
                enableSpeculativeDecoding: false);

            int port = GetAvailablePort();
            using var app = GlacierServeApp.CreateBuilder().UsePort(port).Build();
            app.MapInference(engine, "mini-model");
            await app.StartAsync();

            using (var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") })
            {
                const string streamBody = """{"model":"mini-model","messages":[{"role":"user","content":"a"}],"max_tokens":45,"stream":true}""";
                using var req = new HttpRequestMessage(System.Net.Http.HttpMethod.Post, "/v1/chat/completions")
                {
                    Content = new StringContent(streamBody, Encoding.UTF8, "application/json")
                };
                using var resp = await client.SendAsync(req, HttpCompletionOption.ResponseHeadersRead);
                Assert.Equal(HttpStatusCode.OK, resp.StatusCode);

                await using var stream = await resp.Content.ReadAsStreamAsync();
                using var reader = new StreamReader(stream, Encoding.UTF8);
                string? firstLine = await reader.ReadLineAsync();
                Assert.False(string.IsNullOrEmpty(firstLine));
            }

            // Wait for server write-abort / RST cancellation to sweep KV blocks back to full capacity
            var deadline = DateTime.UtcNow.AddSeconds(5);
            while (DateTime.UtcNow < deadline && (engine.BlockPool.FreeBlocksCount != 16 || engine.ActiveBatchSize != 0))
            {
                await Task.Delay(25);
            }

            Assert.Equal(0, engine.ActiveBatchSize);
            Assert.Equal(16, engine.BlockPool.FreeBlocksCount);

            app.Stop();
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void PeerBounceService_PureCSharpPythonSyntaxChecker_AndGeneratedRegexExtractCodeBlock()
    {
        const string validPy = """
            def fib(n):
                '''Compute Fibonacci.'''
                if n <= 1:
                    return n
                return fib(n - 1) + fib(n - 2)
            """;
        Assert.True(SandboxedProcessBounceVerifier.VerifyPythonSyntaxPureCSharp(validPy).Success);

        // Explicit backslash line continuation & blank/comment lines with mixed whitespace
        Assert.True(SandboxedProcessBounceVerifier.VerifyPythonSyntaxPureCSharp("x = 1 + \\\n    2\n").Success);
        Assert.True(SandboxedProcessBounceVerifier.VerifyPythonSyntaxPureCSharp("if True and \\\n        True:\n    y = 1\n").Success);
        Assert.True(SandboxedProcessBounceVerifier.VerifyPythonSyntaxPureCSharp("def f():\n \t\n    return 1\n").Success);
        Assert.True(SandboxedProcessBounceVerifier.VerifyPythonSyntaxPureCSharp("def f():\n \t# comment\n    return 1\n").Success);
        Assert.False(SandboxedProcessBounceVerifier.VerifyPythonSyntaxPureCSharp("def f():\n \treturn 1\n").Success);
        Assert.False(SandboxedProcessBounceVerifier.VerifyPythonSyntaxPureCSharp("x = 1 + \\").Success);

        var badParen = SandboxedProcessBounceVerifier.VerifyPythonSyntaxPureCSharp("def add(a, b:\n    return a + b");
        Assert.False(badParen.Success);
        Assert.Contains("py_compile failed", badParen.ErrorMessage ?? "", StringComparison.OrdinalIgnoreCase);

        var badIndent = SandboxedProcessBounceVerifier.VerifyPythonSyntaxPureCSharp("def f():\nreturn 1");
        Assert.False(badIndent.Success);

        var badString = SandboxedProcessBounceVerifier.VerifyPythonSyntaxPureCSharp("x = 'unclosed");
        Assert.False(badString.Success);

        const string markdown = "```csharp\nclass A {}\n```\n```python\nx = 1\n```";
        Assert.Equal("class A {}", SandboxedProcessBounceVerifier.ExtractCodeBlock(markdown, "csharp", "cs"));
        Assert.Equal("x = 1", SandboxedProcessBounceVerifier.ExtractCodeBlock(markdown, "python", "py"));
    }

    [Theory]
    [InlineData("x = 1 + \\\n", false)]
    [InlineData("x = 1 + \\\\\n", false)]
    [InlineData("x = 1 + \\\\\n2\n", false)]
    [InlineData("x = 1 + \\\\\\\n2\n", false)]
    [InlineData("x = 1 + \\ \n2\n", false)]
    [InlineData("x = 1 \\ + 2\n", false)]
    [InlineData("x = 1 + \\ # c\n2\n", false)]
    [InlineData("x = (1 + \\ \n    2)\n", false)]
    [InlineData("x = 1 \\\n\ny = 2\n", true)]
    [InlineData("s = 'a\\\nb'\n", true)]
    [InlineData("s = 'a\\\\\nb'\n", false)]
    [InlineData("s = r'a\\\nb'\n", true)]
    [InlineData("s = 'a\\\n", false)]
    [InlineData("s = \"a\\\nb\"\n", true)]
    [InlineData("if True: \\\n    x = 1\n", true)]
    public void PeerBounceService_PureCSharpPythonSyntaxChecker_BackslashAndStringContinuations(string code, bool expectedValid)
    {
        var result = SandboxedProcessBounceVerifier.VerifyPythonSyntaxPureCSharp(code);
        Assert.Equal(expectedValid, result.Success);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Http11AndTls_PipeliningWithoutShutdownSend_AndRapidKeepAlive_DoNotStall(bool useTls)
    {
        using X509Certificate2? cert = useTls ? CreateSelfSignedLocalhostCert() : null;
        var slowEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var allowSlowFinish = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        int port = GetAvailablePort();
        var builder = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .WithHeaderTimeout(TimeSpan.FromMilliseconds(250))
            .MapGet("/one", ctx => ctx.Response.WriteUtf8Async("one"))
            .MapGet("/two", ctx => ctx.Response.WriteUtf8Async("two"))
            .MapGet("/slow", async ctx =>
            {
                slowEntered.TrySetResult();
                await allowSlowFinish.Task;
                await ctx.Response.WriteUtf8Async("slow");
            })
            .MapGet("/fast", ctx => ctx.Response.WriteUtf8Async("fast"))
            .MapGet("/ka/{id}", ctx => ctx.Response.WriteUtf8Async($"ka:{ctx.Request.Param("id")}"));
        if (cert != null)
        {
            builder.UseTls(cert);
        }

        using var app = builder.Build();
        await app.StartAsync();
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));

        // 1. Single-packet pipelining without Shutdown(SocketShutdown.Send)
        using (var tcp = new TcpClient { NoDelay = true })
        {
            await tcp.ConnectAsync(IPAddress.Loopback, port, cts.Token);
            await using Stream stream = await OpenClientStreamAsync(tcp, useTls, cts.Token);
            var reader = new RawHttpResponseReader(stream);

            byte[] req = "GET /one HTTP/1.1\r\nHost: localhost\r\n\r\nGET /two HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"u8.ToArray();
            await stream.WriteAsync(req, cts.Token);
            await stream.FlushAsync(cts.Token);

            var r1 = await reader.ReadNextAsync(cts.Token);
            var r2 = await reader.ReadNextAsync(cts.Token);
            Assert.Equal(200, r1.StatusCode);
            Assert.Equal("one", r1.Body);
            Assert.Equal(200, r2.StatusCode);
            Assert.Equal("two", r2.Body);
        }

        // 2. Active-handler pipelining without Shutdown(SocketShutdown.Send)
        using (var tcp = new TcpClient { NoDelay = true })
        {
            await tcp.ConnectAsync(IPAddress.Loopback, port, cts.Token);
            await using Stream stream = await OpenClientStreamAsync(tcp, useTls, cts.Token);
            var reader = new RawHttpResponseReader(stream);

            await stream.WriteAsync("GET /slow HTTP/1.1\r\nHost: localhost\r\n\r\n"u8.ToArray(), cts.Token);
            await stream.FlushAsync(cts.Token);
            await slowEntered.Task.WaitAsync(cts.Token);

            // Allow MonitorClientDisconnectAsync to enter ReadAsync, then pipeline req 2 mid-handler
            await Task.Delay(30, cts.Token);
            await stream.WriteAsync("GET /fast HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"u8.ToArray(), cts.Token);
            await stream.FlushAsync(cts.Token);
            await Task.Delay(30, cts.Token);

            allowSlowFinish.TrySetResult();

            var rSlow = await reader.ReadNextAsync(cts.Token);
            var rFast = await reader.ReadNextAsync(cts.Token);
            Assert.Equal(200, rSlow.StatusCode);
            Assert.Equal("slow", rSlow.Body);
            Assert.Equal(200, rFast.StatusCode);
            Assert.Equal("fast", rFast.Body);
        }

        // 3. Rapid keep-alive reuse across 25 sequential requests on one connection
        using (var tcp = new TcpClient { NoDelay = true })
        {
            await tcp.ConnectAsync(IPAddress.Loopback, port, cts.Token);
            await using Stream stream = await OpenClientStreamAsync(tcp, useTls, cts.Token);
            var reader = new RawHttpResponseReader(stream);

            for (int i = 0; i < 25; i++)
            {
                string connHdr = i == 24 ? "Connection: close\r\n" : "";
                byte[] reqBytes = Encoding.ASCII.GetBytes($"GET /ka/{i} HTTP/1.1\r\nHost: localhost\r\n{connHdr}\r\n");
                await stream.WriteAsync(reqBytes, cts.Token);
                await stream.FlushAsync(cts.Token);

                var resp = await reader.ReadNextAsync(cts.Token);
                Assert.Equal(200, resp.StatusCode);
                Assert.Equal($"ka:{i}", resp.Body);
            }
        }

        app.Stop();
    }

    [Theory]
    [InlineData("zero_bytes")]
    [InlineData("partial_client_hello")]
    public async Task TlsHandshake_SlowlorisTimeout_ClosesFastAndReleasesConnectionGate(string mode)
    {
        using X509Certificate2 cert = CreateSelfSignedLocalhostCert();
        int port = GetAvailablePort();
        using var app = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .UseTls(cert)
            .WithMaxConcurrentConnections(1)
            .WithHeaderTimeout(TimeSpan.FromMilliseconds(150))
            .MapGet("/ok", ctx => ctx.Response.WriteUtf8Async("ok"))
            .Build();

        await app.StartAsync();
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));

        using (var stalledTcp = new TcpClient { NoDelay = true })
        {
            await stalledTcp.ConnectAsync(IPAddress.Loopback, port, cts.Token);
            await using var rawStream = stalledTcp.GetStream();
            if (mode == "partial_client_hello")
            {
                byte[] partialHello = [0x16, 0x03, 0x01, 0x00, 0x20, 0x01, 0x00, 0x00];
                await rawStream.WriteAsync(partialHello, cts.Token);
                await rawStream.FlushAsync(cts.Token);
            }

            byte[] buf = new byte[64];
            var sw = System.Diagnostics.Stopwatch.StartNew();
            int readBytes = 0;
            try
            {
                readBytes = await rawStream.ReadAsync(buf, cts.Token);
            }
            catch (IOException)
            {
                readBytes = 0;
            }
            sw.Stop();

            Assert.Equal(0, readBytes);
            Assert.True(sw.ElapsedMilliseconds < 2000, $"Expected fast TLS handshake timeout close, took {sw.ElapsedMilliseconds}ms");
        }

        // Verify _connectionGate (capacity = 1) was cleanly released via a real TLS request
        using (var validTcp = new TcpClient { NoDelay = true })
        {
            await validTcp.ConnectAsync(IPAddress.Loopback, port, cts.Token);
            await using Stream tlsStream = await OpenClientStreamAsync(validTcp, useTls: true, cts.Token);
            var reader = new RawHttpResponseReader(tlsStream);

            await tlsStream.WriteAsync("GET /ok HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"u8.ToArray(), cts.Token);
            await tlsStream.FlushAsync(cts.Token);

            var resp = await reader.ReadNextAsync(cts.Token);
            Assert.Equal(200, resp.StatusCode);
            Assert.Equal("ok", resp.Body);
        }

        app.Stop();
    }

    private static X509Certificate2 CreateSelfSignedLocalhostCert()
    {
        using var rsa = RSA.Create(2048);
        var req = new CertificateRequest("CN=localhost", rsa, HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
        var san = new SubjectAlternativeNameBuilder();
        san.AddDnsName("localhost");
        san.AddIpAddress(IPAddress.Loopback);
        req.CertificateExtensions.Add(san.Build());
        using var ephemeral = req.CreateSelfSigned(DateTimeOffset.UtcNow.AddDays(-1), DateTimeOffset.UtcNow.AddDays(30));
        byte[] pfx = ephemeral.Export(X509ContentType.Pfx);
        return X509CertificateLoader.LoadPkcs12(pfx, ReadOnlySpan<char>.Empty);
    }

    private static async Task<Stream> OpenClientStreamAsync(TcpClient tcp, bool useTls, CancellationToken ct)
    {
        NetworkStream netStream = tcp.GetStream();
        if (!useTls)
            return netStream;

        var sslStream = new SslStream(netStream, leaveInnerStreamOpen: false, (_, _, _, _) => true);
        await sslStream.AuthenticateAsClientAsync(
            new SslClientAuthenticationOptions
            {
                TargetHost = "localhost",
                EnabledSslProtocols = SslProtocols.Tls12 | SslProtocols.Tls13,
                CertificateRevocationCheckMode = X509RevocationMode.NoCheck
            },
            ct);
        return sslStream;
    }

    private sealed class RawHttpResponseReader
    {
        private readonly Stream _stream;
        private readonly byte[] _recvBuf = new byte[4096];
        private byte[] _pending = [];

        public RawHttpResponseReader(Stream stream) => _stream = stream;

        public async Task<(int StatusCode, Dictionary<string, string> Headers, string Body)> ReadNextAsync(CancellationToken ct)
        {
            int headerEnd = _pending.AsSpan().IndexOf("\r\n\r\n"u8);
            while (headerEnd < 0)
            {
                int n = await _stream.ReadAsync(_recvBuf, ct);
                if (n == 0)
                    throw new EndOfStreamException("Server closed connection before sending HTTP response headers.");
                AppendPending(_recvBuf.AsSpan(0, n));
                headerEnd = _pending.AsSpan().IndexOf("\r\n\r\n"u8);
            }

            string headerText = Encoding.ASCII.GetString(_pending, 0, headerEnd);
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
            int contentLength = headers.TryGetValue("Content-Length", out var clVal) && int.TryParse(clVal, out int cl) ? cl : 0;
            int totalLen = bodyStart + contentLength;

            while (_pending.Length < totalLen)
            {
                int n = await _stream.ReadAsync(_recvBuf, ct);
                if (n == 0)
                    throw new EndOfStreamException("Server closed connection before sending complete HTTP response body.");
                AppendPending(_recvBuf.AsSpan(0, n));
            }

            string body = Encoding.UTF8.GetString(_pending, bodyStart, contentLength);
            _pending = _pending.Length > totalLen ? _pending[totalLen..] : [];
            return (statusCode, headers, body);
        }

        private void AppendPending(ReadOnlySpan<byte> chunk)
        {
            byte[] next = new byte[_pending.Length + chunk.Length];
            _pending.CopyTo(next, 0);
            chunk.CopyTo(next.AsSpan(_pending.Length));
            _pending = next;
        }
    }

    private static async Task<byte[]> ReadAllPipeBytesAsync(PipeReader reader)
    {
        using var ms = new MemoryStream();
        while (true)
        {
            ReadResult res = await reader.ReadAsync();
            foreach (var seg in res.Buffer)
                ms.Write(seg.Span);
            reader.AdvanceTo(res.Buffer.End);
            if (res.IsCompleted) break;
        }
        await reader.CompleteAsync();
        return ms.ToArray();
    }
}

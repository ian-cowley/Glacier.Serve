namespace Glacier.Serve.Tests;

using System;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Threading.Tasks;
using Glacier.Serve.Core;
using Glacier.Serve.Routing;
using Glacier.Serve.Server;
using HttpMethod = Glacier.Serve.Routing.HttpMethod;
using Xunit;

public class IngressZeroAllocationTests
{
    private static int GetAvailablePort()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    [Fact]
    public void TryMatchRoute_ShortPath_MatchesViaStackAlloc()
    {
        var router = new RadixTreeRouter();
        bool handlerInvoked = false;
        router.AddRoute(HttpMethod.Get, "/short/endpoint", ctx =>
        {
            handlerInvoked = true;
            return ValueTask.CompletedTask;
        });

        bool matched = GlacierServeApp.TryMatchRoute(router, HttpMethod.Get, "/short/endpoint", out var match);
        Assert.True(matched);

        var context = new HttpContext(new HttpRequest(), new HttpResponse(null!));
        match.Handler(context);
        Assert.True(handlerInvoked);
    }

    [Fact]
    public void TryMatchRoute_ShortParameterizedPath_ExtractsParameter()
    {
        var router = new RadixTreeRouter();
        router.AddRoute(HttpMethod.Get, "/items/{itemId}/details", ctx => ValueTask.CompletedTask);

        bool matched = GlacierServeApp.TryMatchRoute(router, HttpMethod.Get, "/items/item_42/details", out var match);
        Assert.True(matched);
        Assert.Equal("item_42", match.GetParam("itemId"));
    }

    [Fact]
    public void TryMatchRoute_LongExactPath_MatchesViaArrayPool()
    {
        // Construct a path string longer than 512 bytes (e.g. 750 characters)
        string padding = new string('x', 700);
        string longPath = $"/long/resource/{padding}";

        var router = new RadixTreeRouter();
        bool handlerInvoked = false;
        router.AddRoute(HttpMethod.Get, longPath, ctx =>
        {
            handlerInvoked = true;
            return ValueTask.CompletedTask;
        });

        bool matched = GlacierServeApp.TryMatchRoute(router, HttpMethod.Get, longPath, out var match);
        Assert.True(matched);

        var context = new HttpContext(new HttpRequest(), new HttpResponse(null!));
        match.Handler(context);
        Assert.True(handlerInvoked);
    }

    [Fact]
    public void TryMatchRoute_LongParameterizedPath_ExtractsLongParameterViaArrayPool()
    {
        // Construct a parameter string longer than 512 bytes
        string longParamValue = new string('a', 600);
        string fullPath = $"/query/{longParamValue}/submit";

        var router = new RadixTreeRouter();
        router.AddRoute(HttpMethod.Get, "/query/{token}/submit", ctx => ValueTask.CompletedTask);

        bool matched = GlacierServeApp.TryMatchRoute(router, HttpMethod.Get, fullPath, out var match);
        Assert.True(matched);
        Assert.Equal(longParamValue, match.GetParam("token"));
    }

    [Fact]
    public void TryMatchRoute_UnmatchedPaths_ReturnFalseCleanly()
    {
        var router = new RadixTreeRouter();
        router.AddRoute(HttpMethod.Get, "/api/active", ctx => ValueTask.CompletedTask);

        // Short unmatched
        bool shortMatched = GlacierServeApp.TryMatchRoute(router, HttpMethod.Get, "/api/inactive", out _);
        Assert.False(shortMatched);

        // Long unmatched (> 512 bytes)
        string longUnmatched = "/api/unknown/" + new string('z', 600);
        bool longMatched = GlacierServeApp.TryMatchRoute(router, HttpMethod.Get, longUnmatched, out _);
        Assert.False(longMatched);
    }

    [Fact]
    public async Task Server_EndToEnd_ProcessesShortAndLongPathsAccurately()
    {
        int port = GetAvailablePort();
        string longParam = new string('k', 650);

        using var app = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .MapGet("/short", async ctx =>
            {
                await ctx.Response.WriteUtf8Async("short_ok");
            })
            .MapGet("/data/{id}", async ctx =>
            {
                string id = ctx.Request.Param("id") ?? "missing";
                await ctx.Response.WriteUtf8Async($"received_id_len:{id.Length}");
            })
            .Build();

        await app.StartAsync();

        using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

        // 1. Short path (<= 512 bytes, stackalloc path)
        var shortRes = await client.GetAsync("/short");
        Assert.True(shortRes.IsSuccessStatusCode);
        string shortBody = await shortRes.Content.ReadAsStringAsync();
        Assert.Equal("short_ok", shortBody);

        // 2. Long path (> 512 bytes, ArrayPool fallback path)
        var longRes = await client.GetAsync($"/data/{longParam}");
        Assert.True(longRes.IsSuccessStatusCode);
        string longBody = await longRes.Content.ReadAsStringAsync();
        Assert.Equal($"received_id_len:{longParam.Length}", longBody);

        // 3. Long path 404 (> 512 bytes non-existent)
        var notFoundRes = await client.GetAsync($"/nonexistent/{new string('q', 600)}");
        Assert.Equal(HttpStatusCode.NotFound, notFoundRes.StatusCode);
    }
}

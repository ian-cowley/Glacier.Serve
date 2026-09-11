namespace Glacier.Serve.Tests;

using System;
using System.Net.Http;
using System.Net.Sockets;
using System.Threading.Tasks;
using Glacier.Serve.Server;
using Xunit;

public class ServerIntegrationTests
{
    private static int GetAvailablePort()
    {
        using var listener = new TcpListener(System.Net.IPAddress.Loopback, 0);
        listener.Start();
        int port = ((System.Net.IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    [Fact]
    public async Task Server_HandlesGetPlaintextAndJson()
    {
        int port = GetAvailablePort();
        using var app = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .MapGet("/hello", async ctx =>
            {
                await ctx.Response.WriteUtf8Async("Hello Glacier!");
            })
            .MapGet("/api/json", async ctx =>
            {
                await ctx.Response.WriteJsonAsync(new { status = "operational", version = "1.0.0" });
            })
            .MapGet("/users/{id}", async ctx =>
            {
                string id = ctx.Request.Param("id") ?? "unknown";
                await ctx.Response.WriteUtf8Async($"User: {id}");
            })
            .Build();

        await app.StartAsync();

        using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

        // 1. Plaintext
        var res1 = await client.GetAsync("/hello");
        Assert.True(res1.IsSuccessStatusCode);
        string text1 = await res1.Content.ReadAsStringAsync();
        Assert.Equal("Hello Glacier!", text1);

        // 2. JSON
        var res2 = await client.GetAsync("/api/json");
        Assert.True(res2.IsSuccessStatusCode);
        string text2 = await res2.Content.ReadAsStringAsync();
        Assert.Contains("operational", text2);

        // 3. Parameterized
        var res3 = await client.GetAsync("/users/999");
        Assert.True(res3.IsSuccessStatusCode);
        string text3 = await res3.Content.ReadAsStringAsync();
        Assert.Equal("User: 999", text3);

        // 4. 404
        var res4 = await client.GetAsync("/non-existent");
        Assert.Equal(System.Net.HttpStatusCode.NotFound, res4.StatusCode);
    }
}

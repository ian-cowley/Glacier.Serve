namespace Glacier.Serve.Tests;

using System;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Serve.Server;
using Xunit;

public sealed record TestStatusDto(string status, string version);

[JsonSerializable(typeof(TestStatusDto))]
internal partial class TestServerJsonContext : JsonSerializerContext
{
}

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
                await ctx.Response.WriteJsonAsync(
                    new TestStatusDto("operational", "1.2.0"),
                    TestServerJsonContext.Default.TestStatusDto);
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

    [Fact]
    public async Task Server_BearerAuth_Enforces401AndPermitsExemptPaths()
    {
        int port = GetAvailablePort();
        const string token = "test-bearer-secret-token";
        using var app = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .UseBearerAuth(token, "/health")
            .MapGet("/health", async ctx => await ctx.Response.WriteUtf8Async("ok"))
            .MapGet("/secure", async ctx => await ctx.Response.WriteUtf8Async("secret-ok"))
            .Build();

        await app.StartAsync();
        using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

        var healthRes = await client.GetAsync("/health");
        Assert.Equal(HttpStatusCode.OK, healthRes.StatusCode);

        var unauthRes = await client.GetAsync("/secure");
        Assert.Equal(HttpStatusCode.Unauthorized, unauthRes.StatusCode);

        using var badReq = new HttpRequestMessage(HttpMethod.Get, "/secure");
        badReq.Headers.Authorization = new AuthenticationHeaderValue("Bearer", "wrong-token");
        var badRes = await client.SendAsync(badReq);
        Assert.Equal(HttpStatusCode.Unauthorized, badRes.StatusCode);

        using var okReq = new HttpRequestMessage(HttpMethod.Get, "/secure");
        okReq.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
        var okRes = await client.SendAsync(okReq);
        Assert.Equal(HttpStatusCode.OK, okRes.StatusCode);
        Assert.Equal("secret-ok", await okRes.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task Server_HeaderTimeout_DropsSlowlorisClientAndFreesSlot()
    {
        int port = GetAvailablePort();
        using var app = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .WithMaxConcurrentConnections(1)
            .WithHeaderTimeout(TimeSpan.FromMilliseconds(500))
            .MapGet("/ok", async ctx => await ctx.Response.WriteUtf8Async("ok"))
            .Build();

        await app.StartAsync();

        using var handler = new HttpClientHandler { UseProxy = false };
        using var http = new HttpClient(handler) { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

        using var slowClient = new TcpClient { NoDelay = true };
        await slowClient.ConnectAsync(IPAddress.Loopback, port);
        NetworkStream slowStream = slowClient.GetStream();
        await slowStream.WriteAsync("GET /ok HTTP/1.1\r\n"u8.ToArray());
        await slowStream.FlushAsync();

        // Second connection while first holds the single connection slot should get 503 immediately
        var busyRes = await http.GetAsync("/ok");
        Assert.Equal(HttpStatusCode.ServiceUnavailable, busyRes.StatusCode);

        // Wait deterministically for the server to time out the Slowloris header with 408 and close the connection
        using var readCts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        byte[] buf = new byte[512];
        var sb = new StringBuilder();
        while (true)
        {
            int n = await slowStream.ReadAsync(buf, readCts.Token);
            if (n == 0)
                break;
            sb.Append(Encoding.ASCII.GetString(buf, 0, n));
        }
        Assert.Contains("408 Request Timeout", sb.ToString(), StringComparison.Ordinal);

        // Allow ProcessConnectionAsync finally block to release the connection semaphore right after stream teardown
        await Task.Delay(50);

        var recoveredRes = await http.GetAsync("/ok");
        Assert.Equal(HttpStatusCode.OK, recoveredRes.StatusCode);
    }

    [Fact]
    public async Task Server_OverloadedConnection_DelayedClientRequest_Receives503WithoutReset()
    {
        int port = GetAvailablePort();
        using var app = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .WithMaxConcurrentConnections(1)
            .WithHeaderTimeout(TimeSpan.FromSeconds(2))
            .MapGet("/ok", async ctx => await ctx.Response.WriteUtf8Async("ok"))
            .Build();

        await app.StartAsync();

        using var holder = new TcpClient { NoDelay = true };
        await holder.ConnectAsync(IPAddress.Loopback, port);
        await holder.GetStream().WriteAsync("GET /ok HTTP/1.1\r\n"u8.ToArray());
        await holder.GetStream().FlushAsync();

        // Connect second raw client and wait 40ms (> old 10ms Poll) before writing request bytes
        using var overloaded = new TcpClient { NoDelay = true };
        await overloaded.ConnectAsync(IPAddress.Loopback, port);
        await Task.Delay(40);

        NetworkStream stream = overloaded.GetStream();
        await stream.WriteAsync("GET /ok HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"u8.ToArray());
        await stream.FlushAsync();

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        byte[] buf = new byte[512];
        var sb = new StringBuilder();
        while (true)
        {
            int n = await stream.ReadAsync(buf, cts.Token);
            if (n == 0)
                break;
            sb.Append(Encoding.ASCII.GetString(buf, 0, n));
        }

        Assert.Contains("503 Service Unavailable", sb.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task Server_TlsAndChunkedStreaming_CompletesHandshakeAndDecodesChunks()
    {
        using var rsa = RSA.Create(2048);
        var req = new CertificateRequest("CN=localhost", rsa, HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
        using var ephemeralCert = req.CreateSelfSigned(DateTimeOffset.UtcNow.AddMinutes(-5), DateTimeOffset.UtcNow.AddHours(1));
        byte[] pfxBytes = ephemeralCert.Export(X509ContentType.Pfx);
        using var cert = X509CertificateLoader.LoadPkcs12(pfxBytes, ReadOnlySpan<char>.Empty, X509KeyStorageFlags.Exportable);

        int port = GetAvailablePort();
        using var app = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .UseTls(cert)
            .MapGet("/stream", async ctx =>
            {
                ctx.Response.ContentType = "text/event-stream";
                ctx.Response.UseChunkedTransfer = true;
                await ctx.Response.WriteSseEventAsync("data: one\n\n"u8.ToArray());
                await ctx.Response.WriteSseEventAsync("data: two\n\n"u8.ToArray());
                await ctx.Response.CompleteChunkedAsync();
            })
            .Build();

        await app.StartAsync();

        using var handler = new HttpClientHandler
        {
            ServerCertificateCustomValidationCallback = (_, _, _, _) => true
        };
        using var client = new HttpClient(handler) { BaseAddress = new Uri($"https://127.0.0.1:{port}") };
        string body = await client.GetStringAsync("/stream");
        Assert.Contains("data: one", body);
        Assert.Contains("data: two", body);
    }
}

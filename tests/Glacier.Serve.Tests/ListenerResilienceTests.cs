namespace Glacier.Serve.Tests;

using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using Glacier.Serve.Diagnostics;
using Glacier.Serve.Server;
using Xunit;

public class ListenerResilienceTests
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
    public void ComputeBackoffMs_CalculatesExponentialBackoffCorrectly()
    {
        Assert.Equal(0, GlacierServeApp.ComputeBackoffMs(0));
        Assert.Equal(0, GlacierServeApp.ComputeBackoffMs(-5));

        Assert.Equal(10, GlacierServeApp.ComputeBackoffMs(1, minBackoffMs: 10, maxBackoffMs: 1000));
        Assert.Equal(20, GlacierServeApp.ComputeBackoffMs(2, minBackoffMs: 10, maxBackoffMs: 1000));
        Assert.Equal(40, GlacierServeApp.ComputeBackoffMs(3, minBackoffMs: 10, maxBackoffMs: 1000));
        Assert.Equal(80, GlacierServeApp.ComputeBackoffMs(4, minBackoffMs: 10, maxBackoffMs: 1000));
        Assert.Equal(160, GlacierServeApp.ComputeBackoffMs(5, minBackoffMs: 10, maxBackoffMs: 1000));
        Assert.Equal(320, GlacierServeApp.ComputeBackoffMs(6, minBackoffMs: 10, maxBackoffMs: 1000));
        Assert.Equal(640, GlacierServeApp.ComputeBackoffMs(7, minBackoffMs: 10, maxBackoffMs: 1000));

        // Capped at maxBackoffMs
        Assert.Equal(1000, GlacierServeApp.ComputeBackoffMs(8, minBackoffMs: 10, maxBackoffMs: 1000));
        Assert.Equal(1000, GlacierServeApp.ComputeBackoffMs(20, minBackoffMs: 10, maxBackoffMs: 1000));
    }

    [Fact]
    public async Task Server_StartsAndStopsCleanly_MultipleCycles()
    {
        var logMessages = new List<(LogLevel Level, string Message)>();
        var testLogger = new DelegateGlacierLogger((level, msg, _) =>
        {
            logMessages.Add((level, msg));
        }, LogLevel.Debug);

        GlacierDiagnostics.SetLogger(testLogger);

        try
        {
            int port = GetAvailablePort();

            for (int cycle = 0; cycle < 3; cycle++)
            {
                using var app = GlacierServeApp.CreateBuilder()
                    .UsePort(port)
                    .MapGet("/ping", async ctx =>
                    {
                        await ctx.Response.WriteUtf8Async("pong");
                    })
                    .Build();

                await app.StartAsync();
                Assert.True(app.IsRunning);

                using (var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") })
                {
                    var res = await client.GetAsync("/ping");
                    Assert.True(res.IsSuccessStatusCode);
                    string text = await res.Content.ReadAsStringAsync();
                    Assert.Equal("pong", text);
                }

                app.Stop();
                Assert.False(app.IsRunning);
            }

            // Verify clean cancellation logging occurred
            Assert.Contains(logMessages, m => m.Message.Contains("AcceptLoopAsync cancelled cleanly"));
        }
        finally
        {
            GlacierDiagnostics.Reset();
        }
    }

    [Fact]
    public async Task Server_HandlesMalformedTcpConnections_Resiliently()
    {
        var logMessages = new List<string>();
        var testLogger = new DelegateGlacierLogger(msg => logMessages.Add(msg), LogLevel.Debug);
        GlacierDiagnostics.SetLogger(testLogger);

        try
        {
            int port = GetAvailablePort();

            using var app = GlacierServeApp.CreateBuilder()
                .UsePort(port)
                .MapGet("/status", async ctx =>
                {
                    await ctx.Response.WriteUtf8Async("operational");
                })
                .Build();

            await app.StartAsync();

            // 1. Connect raw socket and send invalid HTTP bytes, then close immediately
            using (var rawSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp))
            {
                await rawSocket.ConnectAsync(IPAddress.Loopback, port);
                await rawSocket.SendAsync(Encoding.UTF8.GetBytes("GARBAGE_NOT_HTTP\r\n\r\n"), SocketFlags.None);
                rawSocket.Shutdown(SocketShutdown.Both);
                rawSocket.Close();
            }

            // Small delay for connection processing
            await Task.Delay(50);

            // 2. Verify server remains operational and processes valid HTTP request immediately
            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };
            var response = await client.GetAsync("/status");
            Assert.True(response.IsSuccessStatusCode);
            string body = await response.Content.ReadAsStringAsync();
            Assert.Equal("operational", body);

            app.Stop();
        }
        finally
        {
            GlacierDiagnostics.Reset();
        }
    }
}

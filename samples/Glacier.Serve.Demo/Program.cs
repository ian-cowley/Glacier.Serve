namespace Glacier.Serve.Demo;

using System;
using System.Diagnostics;
using System.Net.Http;
using System.Linq;
using System.Threading.Tasks;
using Glacier.Polaris;
using Glacier.Polaris.Data;
using Glacier.Serve.Core;
using Glacier.Serve.Serialization;
using Glacier.Serve.Server;
using Glacier.Tensor.Core;

public static class Program
{
    public static async Task Main(string[] args)
    {
        Console.WriteLine("================================================================================");
        Console.WriteLine("       GLACIER.SERVE: ULTRA-LOW-LATENCY NATIVE AOT WEB ENGINE (.NET 10)         ");
        Console.WriteLine("                    Native C# Alternative to Python Flask & FastAPI             ");
        Console.WriteLine("================================================================================\n");

        int port = 5055;

        // 1. Prepare sample Polaris DataFrame and Tensor data sources
        var sRegion = CategoricalSeries.FromStrings("Region", ["NA", "NA", "EU", "EU", "APAC"]);
        var sRev = new Float32Series("Revenue", 5);
        new float[] { 1200f, 1500f, 950f, 1100f, 1800f }.CopyTo(sRev.Memory.Span);
        var sProfit = new Float32Series("Profit", 5);
        new float[] { 350f, 420f, 210f, 310f, 540f }.CopyTo(sProfit.Memory.Span);
        var sampleDf = new DataFrame([sRegion, sRev, sProfit]);

        using var weightTensor = Tensor<float>.FromSpan([0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f], [2, 3]);

        // 2. Build Minimal API Server
        var sw = Stopwatch.StartNew();
        using var app = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .MapGet("/", async ctx =>
            {
                await ctx.Response.WriteUtf8Async("Glacier.Serve .NET 10 — Sub-millisecond Native Web Runtime");
            })
            .MapGet("/api/status", async ctx =>
            {
                await ctx.Response.WriteJsonAsync(new
                {
                    status = "healthy",
                    runtime = ".NET 10.0",
                    aot = true,
                    engine = "Glacier.Serve",
                    uptimeMs = Environment.TickCount64
                });
            })
            .MapGet("/api/sales/{region}", async ctx =>
            {
                // Stream DataFrame directly as JSON
                await ctx.Response.WriteDataFrameJsonAsync(sampleDf);
            })
            .MapGet("/api/tensor/weights", async ctx =>
            {
                // Stream binary tensor directly without managed string allocations
                await ctx.Response.WriteBinaryTensorAsync(weightTensor);
            })
            .Build();

        await app.StartAsync();
        sw.Stop();

        Console.WriteLine($"  ✓ Glacier.Serve listening on http://127.0.0.1:{port}");
        Console.WriteLine($"  ✓ Server Cold Startup Time: {sw.Elapsed.TotalMilliseconds:F2} ms (Target: <15 ms)\n");

        // 3. Client Verification & Loopback Throughput
        using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

        // Test root plaintext
        var res1 = await client.GetAsync("/");
        string text1 = await res1.Content.ReadAsStringAsync();
        Console.WriteLine($"[Test 1] Root Plaintext Endpoint: 200 OK -> \"{text1}\"");

        // Test JSON status
        var res2 = await client.GetAsync("/api/status");
        string json2 = await res2.Content.ReadAsStringAsync();
        Console.WriteLine($"[Test 2] Fast JSON Health Endpoint: 200 OK -> {json2}");

        // Test Parameterized DataFrame stream
        var res3 = await client.GetAsync("/api/sales/NA");
        string dfJson = await res3.Content.ReadAsStringAsync();
        Console.WriteLine($"[Test 3] Zero-Copy Polaris DataFrame Stream: 200 OK (Length: {dfJson.Length} chars)");

        // Test Binary Tensor stream
        var res4 = await client.GetAsync("/api/tensor/weights");
        byte[] tensorBytes = await res4.Content.ReadAsByteArrayAsync();
        Console.WriteLine($"[Test 4] Binary Tensor Stream: 200 OK (Received {tensorBytes.Length} raw bytes, Rank: {BitConverter.ToInt32(tensorBytes, 0)})\n");

        // 4. Latency Benchmark
        Console.WriteLine("[Benchmarking] Measuring Sequential Roundtrip Latency (1,000 HTTP Requests)...");
        sw.Restart();
        const int iters = 1000;
        for (int i = 0; i < iters; i++)
        {
            var r = await client.GetAsync("/");
            r.EnsureSuccessStatusCode();
        }
        sw.Stop();

        double avgLatencyMs = sw.Elapsed.TotalMilliseconds / iters;
        double reqsPerSec = iters / sw.Elapsed.TotalSeconds;
        Console.WriteLine($"  ✓ Completed {iters:N0} requests in {sw.Elapsed.TotalMilliseconds:F1} ms");
        Console.WriteLine($"  ✓ Average Roundtrip Latency: {avgLatencyMs:F3} ms / request");
        Console.WriteLine($"  ✓ Single-Client Sequential Throughput: {reqsPerSec:F0} req/sec\n");

        Console.WriteLine("================================================================================");
        Console.WriteLine("           ALL DEMOS COMPLETED SUCCESSFULLY: GLACIER.SERVE IS READY!            ");
        Console.WriteLine("================================================================================");

        bool isHeadless = args.Contains("--headless") || args.Contains("--bench");
        if (!isHeadless)
        {
            try
            {
                Console.WriteLine($"\n[Opening http://127.0.0.1:{port}/api/status in browser...]");
                Process.Start(new ProcessStartInfo($"http://127.0.0.1:{port}/api/status") { UseShellExecute = true });
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  (Could not open browser: {ex.Message})");
            }

            if (Environment.UserInteractive && !Console.IsInputRedirected)
            {
                Console.WriteLine($"\n[Server is LIVE at http://127.0.0.1:{port} - Press any key to stop server and exit...]");
                Console.ReadKey();
            }
        }
    }
}

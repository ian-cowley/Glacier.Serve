namespace Glacier.Serve.Tests;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Gpu;
using Glacier.Inference.Hardware;
using Glacier.Serve.Daemons;
using Glacier.Serve.Inference.Http;
using Glacier.Serve.Server;
using Xunit;

public sealed class Tier5M7Phase2HardeningTests : IDisposable
{
    private sealed class StubServeHandler : HttpMessageHandler
    {
        private readonly Func<HttpRequestMessage, int, CancellationToken, Task<HttpResponseMessage>> _responder;
        private int _calls;

        public StubServeHandler(Func<HttpRequestMessage, int, CancellationToken, Task<HttpResponseMessage>> responder)
        {
            _responder = responder;
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            int call = Interlocked.Increment(ref _calls);
            return _responder(request, call, cancellationToken);
        }
    }

    public Tier5M7Phase2HardeningTests() => NpuSidecarDiscovery.InvalidateCache();
    public void Dispose() => NpuSidecarDiscovery.InvalidateCache();

    private static int AllocatePort()
    {
        using var l = new TcpListener(IPAddress.Loopback, 0);
        l.Start();
        return ((IPEndPoint)l.LocalEndpoint).Port;
    }

    private static HttpResponseMessage Json(HttpStatusCode status, string body) =>
        new(status) { Content = new StringContent(body, Encoding.UTF8, "application/json") };

    [Fact]
    public async Task MetricsEndpoint_ConcurrentScrapesWithFailingProbe_AndPrometheusLabelEscaping()
    {
        int port = AllocatePort();
        using var app = GlacierServeApp.CreateBuilder()
            .UseHost("127.0.0.1")
            .UsePort(port)
            .Build();

        var evilDevice = new DeviceInfo
        {
            Id = "dev\\1\"id\n\r",
            Index = 0,
            Name = "NPU \"Quote\" \\Slash\nLine\rÜnicode",
            Vendor = GpuVendor.Amd,
            IsNpu = true,
            DedicatedVramBytes = 0,
            IsDisplayDevice = false,
            SharedVramBytes = 16UL * 1024 * 1024 * 1024,
            SupportedEngines = [InferenceEngineType.Cpu],
            RecommendedEngine = InferenceEngineType.Cpu,
            SafetyNotes = "test"
        };

        int probeCalls = 0;
        app.MapMetricsEndpoints(engine: null, new MetricsEndpointOptions
        {
            DeviceProvider = () => [evilDevice],
            SidecarProbe = async ct =>
            {
                Interlocked.Increment(ref probeCalls);
                await Task.Delay(5, ct);
                throw new HttpRequestException("sidecar down");
            }
        });

        await app.StartAsync();
        using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}"), Timeout = TimeSpan.FromSeconds(5) };

        // 20 concurrent /health + /metrics scrapes
        var healthTasks = Enumerable.Range(0, 10).Select(_ => client.GetAsync("/health")).ToArray();
        var metricsTasks = Enumerable.Range(0, 10).Select(_ => client.GetStringAsync("/metrics")).ToArray();
        await Task.WhenAll(healthTasks);
        string[] metricBodies = await Task.WhenAll(metricsTasks);

        Assert.All(healthTasks, t => Assert.Equal(HttpStatusCode.OK, t.Result.StatusCode));
        foreach (string metrics in metricBodies)
        {
            Assert.Contains("glacier_accelerator_device_info", metrics);
            Assert.Contains("\\\\", metrics);
            Assert.Contains("\\\"", metrics);
            Assert.Contains("\\n", metrics);
            Assert.DoesNotContain("\"Quote\" \\Slash\nLine", metrics);
        }
        Assert.True(probeCalls >= 1);
    }

    [Fact]
    public async Task SleepMemoryService_ConcurrentTransition_300dNpu_EmptyPrompts_NullFields_AndCancellation()
    {
        int npuOnline = 0;
        int delayMs = 0;
        using var handler = new StubServeHandler(async (req, _, ct) =>
        {
            string path = req.RequestUri?.AbsolutePath ?? "";
            if (Volatile.Read(ref npuOnline) == 0)
            {
                return Json(HttpStatusCode.ServiceUnavailable, """{"error":"offline"}""");
            }
            if (path.EndsWith("/models", StringComparison.OrdinalIgnoreCase))
            {
                return Json(HttpStatusCode.OK, """{"data":[{"id":"embed-300d"}]}""");
            }
            if (path.EndsWith("/embeddings", StringComparison.OrdinalIgnoreCase))
            {
                int d = Volatile.Read(ref delayMs);
                if (d > 0) await Task.Delay(d, ct);
                float[] vec = new float[300];
                vec[0] = 1.0f;
                vec[299] = 0.25f;
                return Json(HttpStatusCode.OK, $"{{\"data\":[{{\"index\":0,\"embedding\":{JsonSerializer.Serialize(vec)}}}]}}");
            }
            return Json(HttpStatusCode.NotFound, "{}");
        });
        using var http = new HttpClient(handler);

        await using var service = new SleepMemoryService(new SleepMemoryOptions
        {
            TokenBudget = 50,
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = "http://127.0.0.1:52625/v1",
            HttpClient = http,
            ProbeTimeout = TimeSpan.FromSeconds(1),
            ProbeCacheTtl = TimeSpan.Zero,
            EmbeddingTimeout = TimeSpan.FromSeconds(1)
        });

        // Empty/whitespace and null Role/Content must not throw unhandled NullReferenceException
        var weirdMessages = new List<(string Role, string Content)>
        {
            (null!, null!),
            ("user", "   "),
            ("assistant", "Memory baseline entry before NPU comes online.")
        };
        await service.ConsolidateMessagesAsync(weirdMessages);
        await service.RecordTurnCompletionAsync("   ", "recorded turn");

        // Now bring 300d (non-multiple-of-8) NPU online and exercise concurrent operations + TriggerSleepAsync
        Volatile.Write(ref npuOnline, 1);
        await Task.WhenAll(
            service.RecordTurnCompletionAsync("q1", "Golden puppy meadow trace"),
            service.ConsolidateMessagesAsync([("user", "q2"), ("assistant", "response")]),
            service.TriggerSleepAsync());

        Assert.True(service.IsUsingNpuEmbeddings);
        Assert.Equal(300, service.ActiveEmbeddingDimensions);

        // Cancellation mid-embedding propagates OperationCanceledException cleanly
        Volatile.Write(ref delayMs, 500);
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(30));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            service.RecordTurnCompletionAsync("cancel me", "slow embedding target", cts.Token));
    }
}

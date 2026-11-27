using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Gpu;
using Glacier.Inference.Hardware;
using Glacier.Serve.Daemons;
using Glacier.Serve.Inference.Http;
using Glacier.Serve.Serialization;
using Glacier.Serve.Server;
using Xunit;

namespace Glacier.Serve.Tests;

public sealed class NpuTelemetryAndSleepMemoryTests
{
    private static List<DeviceInfo> CreateSyntheticAllAcceleratorDevices()
    {
        return
        [
            new DeviceInfo
            {
                Id = "cuda:0",
                Index = 0,
                Name = "NVIDIA GeForce RTX 4090 \"Lab\"",
                Vendor = GpuVendor.Nvidia,
                DedicatedVramBytes = 24UL * 1024 * 1024 * 1024,
                SharedVramBytes = 16UL * 1024 * 1024 * 1024,
                IsDisplayDevice = true,
                IsNpu = false,
                SupportedEngines = [InferenceEngineType.BareMetal, InferenceEngineType.Vulkan],
                RecommendedEngine = InferenceEngineType.BareMetal,
                SafetyNotes = "dGPU"
            },
            new DeviceInfo
            {
                Id = "gpu:1",
                Index = 1,
                Name = "AMD Radeon 890M Graphics",
                Vendor = GpuVendor.Amd,
                DedicatedVramBytes = 512UL * 1024 * 1024,
                SharedVramBytes = 16642998272UL, // ~15.5 GiB
                IsDisplayDevice = true,
                IsNpu = false,
                SupportedEngines = [InferenceEngineType.Vulkan, InferenceEngineType.Cpu],
                RecommendedEngine = InferenceEngineType.Vulkan,
                SafetyNotes = "iGPU"
            },
            new DeviceInfo
            {
                Id = "amd-xdna2",
                Index = 2,
                Name = "AMD Ryzen AI XDNA 2 NPU\\Dev",
                Vendor = GpuVendor.Amd,
                DedicatedVramBytes = 0UL,
                SharedVramBytes = 16642998272UL, // 15.5 GiB
                IsDisplayDevice = false,
                IsNpu = true,
                SupportedEngines = [InferenceEngineType.NpuSidecar],
                RecommendedEngine = InferenceEngineType.NpuSidecar,
                SafetyNotes = "NPU sidecar"
            },
            new DeviceInfo
            {
                Id = "cpu",
                Index = 3,
                Name = "AMD Ryzen AI 9 HX 370",
                Vendor = GpuVendor.Cpu,
                DedicatedVramBytes = 0UL,
                SharedVramBytes = 32UL * 1024 * 1024 * 1024,
                IsDisplayDevice = false,
                IsNpu = false,
                SupportedEngines = [InferenceEngineType.Cpu],
                RecommendedEngine = InferenceEngineType.Cpu,
                SafetyNotes = "CPU"
            }
        ];
    }

    [Fact]
    public async Task HealthEndpoint_WithNullEngine_ReturnsAllDeviceTypesAndReachableMockNpuSidecar()
    {
        NpuSidecarDiscovery.InvalidateCache();
        using var mockHandler = new MockNpuSidecarHttpHandler(dimensions: 768)
        {
            Models = ["embeddinggemma", "qwen3-4b"]
        };
        using var sidecarClient = new HttpClient(mockHandler);

        var app = GlacierServeApp.CreateBuilder()
            .UseHost("127.0.0.1")
            .UsePort(0)
            .Build();

        app.MapMetricsEndpoints(
            engine: null,
            new MetricsEndpointOptions
            {
                ProbeNpuSidecar = true,
                SidecarCacheTtl = TimeSpan.Zero,
                SidecarHttpClient = sidecarClient,
                SidecarCandidates = [("http://127.0.0.1:52625", NpuSidecarRuntime.FastFlowLM)],
                DeviceProvider = CreateSyntheticAllAcceleratorDevices
            });

        await app.StartAsync();
        try
        {
            Assert.True(app.Port > 0, "Expected GlacierServeApp.Port to resolve ephemeral bound port.");

            using var http = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{app.Port}") };
            using var resp = await http.GetAsync("/health");
            Assert.Equal(HttpStatusCode.OK, resp.StatusCode);

            Assert.True(resp.Headers.TryGetValues("Server", out var serverValues));
            Assert.Contains("Glacier.Serve/1.9.0 (.NET 10)", string.Join(" ", serverValues));

            string json = await resp.Content.ReadAsStringAsync();
            var health = JsonSerializer.Deserialize(json, ServeJsonContext.Default.HealthCheckResponse);
            Assert.NotNull(health);
            Assert.Equal("ok", health!.Status);
            Assert.Equal("None", health.Engine);
            Assert.False(health.PagedAttention);
            Assert.Null(health.Backend);

            Assert.NotNull(health.Devices);
            Assert.Equal(4, health.Devices!.Count);
            Assert.Contains(health.Devices, d => d.Type == "dgpu" && d.Id == "cuda:0");
            Assert.Contains(health.Devices, d => d.Type == "igpu" && d.Id == "gpu:1");
            Assert.Contains(health.Devices, d => d.Type == "npu" && d.Id == "amd-xdna2" && d.IsNpu);
            Assert.Contains(health.Devices, d => d.Type == "cpu" && d.Id == "cpu");

            Assert.NotNull(health.NpuDevices);
            Assert.Single(health.NpuDevices!);
            Assert.Equal("amd-xdna2", health.NpuDevices[0].Id);
            Assert.True(health.NpuDevices[0].SharedVramGb > 15.0);

            Assert.NotNull(health.NpuSidecar);
            Assert.True(health.NpuSidecar!.Available);
            Assert.Equal("http://127.0.0.1:52625", health.NpuSidecar.BaseUrl);
            Assert.Equal("FastFlowLM", health.NpuSidecar.Runtime);
            Assert.Equal(["embeddinggemma", "qwen3-4b"], health.NpuSidecar.Models);
            Assert.NotNull(health.NpuSidecar.LatencyMs);
        }
        finally
        {
            app.Dispose();
        }
    }

    [Fact]
    public async Task HealthAndMetrics_WhenNpuSidecarOffline_DegradeGracefullyHttp200ZeroHang()
    {
        NpuSidecarDiscovery.InvalidateCache();
        using var mockHandler = new MockNpuSidecarHttpHandler(dimensions: 768)
        {
            IsOnline = false
        };
        using var sidecarClient = new HttpClient(mockHandler);

        var app = GlacierServeApp.CreateBuilder()
            .UseHost("127.0.0.1")
            .UsePort(0)
            .Build();

        app.MapMetricsEndpoints(
            engine: null,
            new MetricsEndpointOptions
            {
                ProbeNpuSidecar = true,
                SidecarCacheTtl = TimeSpan.Zero,
                SidecarPerEndpointTimeout = TimeSpan.FromMilliseconds(200),
                SidecarHttpClient = sidecarClient,
                SidecarCandidates = [("http://127.0.0.1:52625", NpuSidecarRuntime.FastFlowLM)],
                DeviceProvider = CreateSyntheticAllAcceleratorDevices
            });

        await app.StartAsync();
        try
        {
            using var http = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{app.Port}") };

            var healthResp = await http.GetAsync("/health");
            Assert.Equal(HttpStatusCode.OK, healthResp.StatusCode);
            var health = JsonSerializer.Deserialize(
                await healthResp.Content.ReadAsStringAsync(),
                ServeJsonContext.Default.HealthCheckResponse);

            Assert.NotNull(health);
            Assert.NotNull(health!.NpuSidecar);
            Assert.False(health.NpuSidecar!.Available);
            Assert.Null(health.NpuSidecar.BaseUrl);
            Assert.Null(health.NpuSidecar.Runtime);
            Assert.Empty(health.NpuSidecar.Models);
            Assert.Null(health.NpuSidecar.LatencyMs);

            var metricsResp = await http.GetAsync("/metrics");
            Assert.Equal(HttpStatusCode.OK, metricsResp.StatusCode);
            string metricsText = await metricsResp.Content.ReadAsStringAsync();

            Assert.Contains("glacier_npu_sidecar_up 0", metricsText);
            Assert.Contains("glacier_npu_sidecar_latency_ms 0.00", metricsText);
            Assert.Contains("glacier_npu_sidecar_models_total 0", metricsText);
            Assert.Contains("glacier_npu_devices_total 1", metricsText);
        }
        finally
        {
            app.Dispose();
        }
    }

    [Fact]
    public async Task MetricsEndpoint_WithNullEngine_EmitsInvariantCultureGaugesAndEscapedLabels()
    {
        NpuSidecarDiscovery.InvalidateCache();
        using var mockHandler = new MockNpuSidecarHttpHandler(dimensions: 768)
        {
            Models = ["embeddinggemma", "qwen3-4b"]
        };
        using var sidecarClient = new HttpClient(mockHandler);

        var app = GlacierServeApp.CreateBuilder()
            .UseHost("127.0.0.1")
            .UsePort(0)
            .Build();

        app.MapMetricsEndpoints(
            engine: null,
            new MetricsEndpointOptions
            {
                ProbeNpuSidecar = true,
                SidecarCacheTtl = TimeSpan.Zero,
                SidecarHttpClient = sidecarClient,
                SidecarProbe = _ => Task.FromResult<NpuSidecarEndpoint?>(
                    new NpuSidecarEndpoint("http://127.0.0.1:52625", NpuSidecarRuntime.FastFlowLM, ["embeddinggemma", "qwen3-4b"], 12.345)),
                DeviceProvider = CreateSyntheticAllAcceleratorDevices
            });

        CultureInfo originalCulture = CultureInfo.CurrentCulture;
        CultureInfo originalUiCulture = CultureInfo.CurrentUICulture;

        await app.StartAsync();
        try
        {
            // Force comma-decimal culture to prove Prometheus formatting remains CultureInfo.InvariantCulture
            CultureInfo.CurrentCulture = new CultureInfo("de-DE");
            CultureInfo.CurrentUICulture = new CultureInfo("de-DE");

            using var http = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{app.Port}") };
            using var resp = await http.GetAsync("/metrics");
            Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
            Assert.Equal("text/plain", resp.Content.Headers.ContentType?.MediaType);

            string body = await resp.Content.ReadAsStringAsync();

            Assert.DoesNotContain("glacier_active_sequences", body);
            Assert.Contains("glacier_accelerator_devices_total 4", body);
            Assert.Contains("glacier_accelerator_devices_by_type{type=\"dgpu\"} 1", body);
            Assert.Contains("glacier_accelerator_devices_by_type{type=\"igpu\"} 1", body);
            Assert.Contains("glacier_accelerator_devices_by_type{type=\"npu\"} 1", body);
            Assert.Contains("glacier_accelerator_devices_by_type{type=\"cpu\"} 1", body);
            Assert.Contains("name=\"NVIDIA GeForce RTX 4090 \\\"Lab\\\"\"", body);
            Assert.Contains("name=\"AMD Ryzen AI XDNA 2 NPU\\\\Dev\"", body);
            Assert.Contains("glacier_accelerator_dedicated_vram_bytes{id=\"cuda:0\",type=\"dgpu\"} 25769803776", body);
            Assert.Contains("glacier_accelerator_shared_vram_bytes{id=\"amd-xdna2\",type=\"npu\"} 16642998272", body);
            Assert.Contains("glacier_npu_devices_total 1", body);
            Assert.Contains("glacier_npu_shared_cap_bytes{id=\"amd-xdna2\",name=\"AMD Ryzen AI XDNA 2 NPU\\\\Dev\"} 16642998272", body);
            Assert.Contains("glacier_npu_shared_cap_bytes_total 16642998272", body);
            Assert.Contains("glacier_npu_sidecar_up 1", body);
            Assert.Contains("glacier_npu_sidecar_latency_ms 12.35", body);
            Assert.DoesNotContain("12,35", body);
            Assert.Contains("glacier_npu_sidecar_models_total 2", body);
        }
        finally
        {
            CultureInfo.CurrentCulture = originalCulture;
            CultureInfo.CurrentUICulture = originalUiCulture;
            app.Dispose();
        }
    }

    [Fact]
    public async Task SleepStatusAndTrigger_WithNullEngine_AndDefaultDeviceManager_ReturnDisabledSafely()
    {
        var app = GlacierServeApp.CreateBuilder()
            .UseHost("127.0.0.1")
            .UsePort(0)
            .Build();

        app.MapMetricsEndpoints(
            engine: null,
            new MetricsEndpointOptions
            {
                ProbeNpuSidecar = false
            });

        await app.StartAsync();
        try
        {
            using var http = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{app.Port}") };

            var healthResp = await http.GetAsync("/health");
            Assert.Equal(HttpStatusCode.OK, healthResp.StatusCode);
            var health = JsonSerializer.Deserialize(await healthResp.Content.ReadAsStringAsync(), ServeJsonContext.Default.HealthCheckResponse);
            Assert.NotNull(health);
            Assert.NotNull(health!.Devices);
            Assert.True(health.Devices!.Count >= 1); // At minimum host CPU is always detected by DeviceManager

            var statusResp = await http.GetAsync("/v1/sleep/status");
            Assert.Equal(HttpStatusCode.OK, statusResp.StatusCode);
            var sleepStatus = JsonSerializer.Deserialize(await statusResp.Content.ReadAsStringAsync(), ServeJsonContext.Default.SleepStatusResponse);
            Assert.NotNull(sleepStatus);
            Assert.False(sleepStatus!.Enabled);
            Assert.Equal("Disabled", sleepStatus.State);

            var triggerResp = await http.PostAsync("/v1/sleep/trigger", content: null);
            Assert.Equal(HttpStatusCode.OK, triggerResp.StatusCode);
            var triggerStatus = JsonSerializer.Deserialize(await triggerResp.Content.ReadAsStringAsync(), ServeJsonContext.Default.SleepTriggerResponse);
            Assert.NotNull(triggerStatus);
            Assert.False(triggerStatus!.Triggered);
            Assert.Equal("Disabled", triggerStatus.State);
        }
        finally
        {
            app.Dispose();
        }
    }

    [Fact]
    public void MapMetricsEndpoints_NullApp_ThrowsArgumentNullException()
    {
        GlacierServeApp? nullApp = null;
        Assert.Throws<ArgumentNullException>(() => nullApp!.MapMetricsEndpoints());
        Assert.Throws<ArgumentNullException>(() => nullApp!.MapMetricsEndpoints(null, new MetricsEndpointOptions()));
    }

    [Fact]
    public async Task SleepMemoryService_WithMockNpuSidecar_Uses768DimSemanticEmbeddingsAndRecallsSynonyms()
    {
        NpuSidecarDiscovery.InvalidateCache();
        using var mockHandler = new MockNpuSidecarHttpHandler(dimensions: 768);
        using var httpClient = new HttpClient(mockHandler);

        await using var service = new SleepMemoryService(new SleepMemoryOptions
        {
            TokenBudget = 50,
            IdleThreshold = TimeSpan.FromMinutes(10),
            PollingInterval = TimeSpan.FromMilliseconds(100),
            MinDreamsToTriggerSleep = 1,
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = "auto-npu",
            HttpClient = httpClient,
            ProbeTimeout = TimeSpan.FromSeconds(1),
            ProbeCacheTtl = TimeSpan.Zero,
            EmbeddingTimeout = TimeSpan.FromSeconds(2)
        });

        bool npuInitialized = await service.InitializeEmbeddingsAsync();
        Assert.True(npuInitialized);
        Assert.True(service.IsUsingNpuEmbeddings);
        Assert.Equal(768, service.ActiveEmbeddingDimensions);
        Assert.Equal("http://127.0.0.1:52625", service.ActiveEmbeddingEndpoint);
        Assert.Equal("embeddinggemma", service.ActiveEmbeddingModelId);

        // Target turn: zero lexical word overlap with query ("What did the young canine do outdoors?")
        await service.RecordTurnCompletionAsync(
            "Log event alpha.",
            "A golden puppy played near the oak tree in the sunny meadow.");

        // Lexical distractor (shares "young" and "outdoors" lexically, but belongs to Botany semantic cluster)
        // Plus orthogonal Astronomy/Maritime/Culinary filler turns to exceed TokenBudget * 2 (100 tokens)
        var conversation = new List<(string Role, string Content)>
        {
            ("user", "Record botanical observation from greenhouse sector seven."),
            ("assistant", "A young botanist studied rare ferns outdoors inside the glasshouse during afternoon inspection."),
            ("user", "Record observatory telemetry log entry number one."),
            ("assistant", "The orbital telescope photographed a distant spiral galaxy and bright nebula across deep space."),
            ("user", "Record maritime shipping log entry number two."),
            ("assistant", "The cargo vessel dropped anchor inside the northern harbor during calm ocean tides at dawn."),
            ("user", "Record culinary kitchen log entry number three."),
            ("assistant", "The pastry chef baked sourdough bread inside a stone oven with fresh rosemary spice today."),
            ("user", "What did the young canine do outdoors?")
        };

        var consolidated = await service.ConsolidateMessagesAsync(conversation);

        Assert.Single(consolidated);
        string packedPrompt = consolidated[0].Content;
        Assert.Contains("Consolidated Context:", packedPrompt);
        Assert.Contains("golden puppy", packedPrompt);
        Assert.True(mockHandler.EmbeddingCallCount >= 8);

        var status = service.GetStatus();
        Assert.True(status.NpuEmbeddingsActive);
        Assert.Equal(768, status.EmbeddingDimensions);
        Assert.Equal("embeddinggemma", status.EmbeddingModel);
        Assert.Equal("http://127.0.0.1:52625", status.EmbeddingEndpoint);

        // Verify TriggerSleepAsync folds 768-dim traces to 64-dim HOPE/MoE without dimension errors
        var sleepResult = await service.TriggerSleepAsync();
        Assert.NotNull(sleepResult);
        Assert.Equal(64, service.ExpansionBank.Dimension);
        Assert.True(service.ExpansionBank.ActiveExpertCount >= 1);
    }

    [Fact]
    public async Task SleepMemoryService_WhenOffline_FallsBackTo64DimFnv1aAndPreservesHippoRagEntityRecall()
    {
        NpuSidecarDiscovery.InvalidateCache();
        using var mockHandler = new MockNpuSidecarHttpHandler(dimensions: 768)
        {
            IsOnline = false
        };
        using var httpClient = new HttpClient(mockHandler);

        await using var service = new SleepMemoryService(new SleepMemoryOptions
        {
            TokenBudget = 50,
            IdleThreshold = TimeSpan.FromMinutes(10),
            PollingInterval = TimeSpan.FromMilliseconds(100),
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = "http://127.0.0.1:52625",
            HttpClient = httpClient,
            ProbeTimeout = TimeSpan.FromMilliseconds(150),
            ProbeCacheTtl = TimeSpan.FromMilliseconds(50),
            EmbeddingTimeout = TimeSpan.FromMilliseconds(200)
        });

        await service.RecordTurnCompletionAsync(
            "Inspect AZURE_VAULT_42.",
            "AZURE_VAULT_42 rotated encryption certificate successfully at node east.");

        var messages = new List<(string Role, string Content)>
        {
            ("user", "Log filler alpha one for routine telemetry archive."),
            ("assistant", "Standard telemetry heartbeat verified across regional gateway nodes without any packet loss or warning flags."),
            ("user", "Log filler beta two for routine storage compaction."),
            ("assistant", "Periodic compaction finished without pending storage anomalies across primary and secondary disk partitions."),
            ("user", "Log filler gamma three for routine thermal sweep."),
            ("assistant", "Routine diagnostic sweep confirmed normal fan speed and thermal headroom across every chassis bay."),
            ("user", "Log filler delta four for routine network health."),
            ("assistant", "Redundant uplink interfaces maintained steady throughput and nominal round-trip latency throughout the hour."),
            ("user", "What is the status of AZURE_VAULT_42?")
        };

        var consolidated = await service.ConsolidateMessagesAsync(messages);

        Assert.False(service.IsUsingNpuEmbeddings);
        Assert.Equal(64, service.ActiveEmbeddingDimensions);
        Assert.Single(consolidated);
        Assert.Contains("AZURE_VAULT_42", consolidated[0].Content);

        var status = service.GetStatus();
        Assert.False(status.NpuEmbeddingsActive);
        Assert.Equal(64, status.EmbeddingDimensions);
        Assert.Null(status.EmbeddingEndpoint);

        var sleepResult = await service.TriggerSleepAsync();
        Assert.NotNull(sleepResult);
    }

    [Fact]
    public async Task SleepMemoryService_MidSessionNpuDropout_MaintainsUniform768DimAndRecallsBothOldAndNew()
    {
        NpuSidecarDiscovery.InvalidateCache();
        using var mockHandler = new MockNpuSidecarHttpHandler(dimensions: 768)
        {
            IsOnline = true
        };
        using var httpClient = new HttpClient(mockHandler);

        await using var service = new SleepMemoryService(new SleepMemoryOptions
        {
            TokenBudget = 60,
            IdleThreshold = TimeSpan.FromMinutes(10),
            PollingInterval = TimeSpan.FromMilliseconds(100),
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = "http://127.0.0.1:52625",
            HttpClient = httpClient,
            ProbeTimeout = TimeSpan.FromSeconds(1),
            EmbeddingTimeout = TimeSpan.FromSeconds(1)
        });

        Assert.True(await service.InitializeEmbeddingsAsync());

        // Ingest Trace A while NPU is online (latches _activeMemExDim = 768)
        await service.RecordTurnCompletionAsync(
            "Record pet log.",
            "The loyal hound guarded the timber cabin overnight.");
        Assert.Equal(768, service.ActiveEmbeddingDimensions);

        // Simulate mid-session NPU crash (HTTP 500)
        mockHandler.IsOnline = false;
        mockHandler.FailureStatusCode = HttpStatusCode.InternalServerError;

        // Ingest Trace B and filler turns during NPU outage -> must latch 768-dim fallback vectors
        await service.RecordTurnCompletionAsync(
            "Check reactor bus.",
            "TARGET_ALPHA_99 backup inverter restored primary power bus.");

        var messages = new List<(string Role, string Content)>
        {
            ("user", "Filler one for auxiliary cooling diagnostics."),
            ("assistant", "General telemetry nominal across auxiliary cooling loops and primary circulation pumps in sector north."),
            ("user", "Filler two for rack voltage inspection."),
            ("assistant", "Scheduled diagnostics confirmed normal voltage rails across every distribution rack and breaker panel."),
            ("user", "Filler three for airflow verification."),
            ("assistant", "Thermal sensors report steady ambient airflow across all bays with zero hot spots detected."),
            ("user", "Filler four for redundant inverter check."),
            ("assistant", "Secondary standby generator and battery bank remain fully charged and synchronized on standby."),
            ("user", "Report status for TARGET_ALPHA_99.")
        };

        var consolidated = await service.ConsolidateMessagesAsync(messages);
        Assert.Equal(768, service.ActiveEmbeddingDimensions);
        Assert.Single(consolidated);
        Assert.Contains("TARGET_ALPHA_99", consolidated[0].Content);

        // Verify Trace A (earlier 768-dim NPU trace) is still queryable without dimension mismatch
        var oldRecall = await service.ConsolidateMessagesAsync(
            new List<(string Role, string Content)>
            {
                ("user", "What did the loyal hound guard at the timber cabin?")
            }.Concat(messages.SkipLast(1)).Concat(new[] { ("user", "What did the loyal hound guard?") }).ToList());
        Assert.Single(oldRecall);
        Assert.Contains("loyal hound", oldRecall[0].Content);
    }

    [Fact]
    public async Task SleepMemoryService_OfflineToOnlineUpgradeOnTriggerSleep_UpgradesRetainedTracesToSemantic()
    {
        NpuSidecarDiscovery.InvalidateCache();
        using var mockHandler = new MockNpuSidecarHttpHandler(dimensions: 768)
        {
            IsOnline = false
        };
        using var httpClient = new HttpClient(mockHandler);

        await using var service = new SleepMemoryService(new SleepMemoryOptions
        {
            TokenBudget = 50,
            IdleThreshold = TimeSpan.FromMinutes(10),
            PollingInterval = TimeSpan.FromMilliseconds(100),
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = "http://127.0.0.1:52625",
            HttpClient = httpClient,
            ProbeTimeout = TimeSpan.FromMilliseconds(200),
            ProbeCacheTtl = TimeSpan.FromMilliseconds(20),
            EmbeddingTimeout = TimeSpan.FromSeconds(1)
        });

        // Ingest while offline -> latched at 64-dim FNV-1a
        await service.RecordTurnCompletionAsync(
            "Event log one.",
            "A golden puppy played near the oak tree in the sunny meadow.");
        await service.RecordTurnCompletionAsync(
            "Botany note.",
            "A young botanist studied rare ferns outdoors inside the glasshouse.");
        await service.RecordTurnCompletionAsync(
            "Astronomy note.",
            "The orbital telescope photographed a distant spiral galaxy and bright nebula.");
        await service.RecordTurnCompletionAsync(
            "Maritime note.",
            "The cargo vessel dropped anchor inside the northern harbor during calm ocean tides.");

        Assert.False(service.IsUsingNpuEmbeddings);
        Assert.Equal(64, service.ActiveEmbeddingDimensions);
        int retainedBefore = service.GetStatus().RetainedTraces;
        Assert.True(retainedBefore >= 4);

        // Wait past short negative probe cooldown, bring NPU sidecar online, and trigger sleep consolidation
        await Task.Delay(40);
        mockHandler.IsOnline = true;
        var result = await service.TriggerSleepAsync();

        Assert.NotNull(result);
        Assert.True(service.IsUsingNpuEmbeddings);
        Assert.Equal(768, service.ActiveEmbeddingDimensions);
        Assert.Equal(retainedBefore, service.GetStatus().RetainedTraces);

        // Query using synonym ("young canine") -> must now semantically recall the upgraded puppy trace
        var consolidated = await service.ConsolidateMessagesAsync(
        [
            ("user", "Summary filler one for culinary kitchen log."),
            ("assistant", "The pastry chef baked sourdough bread inside a stone oven with fresh rosemary spice and sea salt."),
            ("user", "Summary filler two for astronomy observatory log."),
            ("assistant", "Deep sky imaging captured faint emission filaments across the outer spiral arm of the galaxy."),
            ("user", "Summary filler three for maritime shipping log."),
            ("assistant", "Harbor pilots guided the freighter safely past the breakwater before evening tide reversal."),
            ("user", "What did the young canine do outdoors?")
        ]);

        Assert.Single(consolidated);
        Assert.Contains("golden puppy", consolidated[0].Content);
    }

    [Fact]
    public async Task SleepMemoryService_SingleThreadedSynchronizationContext_AndConcurrentDisposal_NoDeadlock()
    {
        NpuSidecarDiscovery.InvalidateCache();
        using var mockHandler = new MockNpuSidecarHttpHandler(dimensions: 768)
        {
            ArtificialDelay = TimeSpan.FromMilliseconds(15)
        };
        using var httpClient = new HttpClient(mockHandler);

        var service = new SleepMemoryService(new SleepMemoryOptions
        {
            TokenBudget = 120,
            IdleThreshold = TimeSpan.FromMinutes(10),
            PollingInterval = TimeSpan.FromMilliseconds(100),
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = "http://127.0.0.1:52625",
            HttpClient = httpClient,
            ProbeTimeout = TimeSpan.FromSeconds(1),
            EmbeddingTimeout = TimeSpan.FromSeconds(2)
        });

        // Run synchronous RecordTurnCompletion on a single-threaded SynchronizationContext
        using var singleThreadCtx = new SingleThreadSynchronizationContext();
        await singleThreadCtx.RunAsync(() =>
        {
            service.RecordTurnCompletion(
                "Sync bridge test.",
                "The loyal puppy rested peacefully beneath the shady tree.");
        });

        Assert.True(service.IsUsingNpuEmbeddings);
        Assert.Equal(768, service.ActiveEmbeddingDimensions);

        // Concurrent ingestion + sleep + disposal must complete cleanly without deadlocks or unhandled exceptions
        var workers = Enumerable.Range(0, 4).Select(i => Task.Run(async () =>
        {
            for (int j = 0; j < 3; j++)
            {
                try
                {
                    await service.RecordTurnCompletionAsync(
                        $"Turn {i}-{j}",
                        $"Astronomical telescope observation {i}-{j} of distant nebula.");
                }
                catch (ObjectDisposedException)
                {
                    break;
                }
                catch (OperationCanceledException)
                {
                    break;
                }
            }
        })).ToArray();

        await Task.Delay(20);
        await service.DisposeAsync();
        await Task.WhenAll(workers);
    }

    [Fact]
    public void FoldAndLiftHelpers_PreserveIdentityAt64Dim_AndPreserveUnitNorm()
    {
        float[] src64 = new float[64];
        for (int i = 0; i < 64; i++) src64[i] = i + 1;

        float[] folded64 = new float[64];
        SleepMemoryService.FoldToHopeDim(src64, folded64);

        float[] lifted64 = new float[64];
        SleepMemoryService.LiftFromHopeDim(folded64, lifted64);

        for (int i = 0; i < 64; i++)
        {
            Assert.Equal(folded64[i], lifted64[i], precision: 5);
        }

        float[] lifted768 = new float[768];
        SleepMemoryService.LiftFromHopeDim(folded64, lifted768);
        double normSq = 0;
        for (int i = 0; i < 768; i++) normSq += lifted768[i] * lifted768[i];
        Assert.InRange(Math.Sqrt(normSq), 0.999, 1.001);
    }

    [Fact]
    public async Task SleepMemoryService_ConcurrentTriggerSleepWithDisposeAsync_DoesNotHangOrLeakObjectDisposedException()
    {
        for (int iter = 0; iter < 8; iter++)
        {
            NpuSidecarDiscovery.InvalidateCache();
            using var mockHandler = new MockNpuSidecarHttpHandler(dimensions: 768)
            {
                IsOnline = false,
                ArtificialDelay = TimeSpan.Zero
            };
            using var httpClient = new HttpClient(mockHandler);

            var service = new SleepMemoryService(new SleepMemoryOptions
            {
                TokenBudget = 60,
                IdleThreshold = TimeSpan.FromMinutes(10),
                PollingInterval = TimeSpan.FromMilliseconds(100),
                EnableNpuEmbeddings = true,
                EmbeddingEndpoint = "http://127.0.0.1:52625",
                HttpClient = httpClient,
                ProbeTimeout = TimeSpan.FromMilliseconds(150),
                ProbeCacheTtl = TimeSpan.FromMilliseconds(5),
                EmbeddingTimeout = TimeSpan.FromSeconds(2)
            });

            // Seed offline traces so TriggerSleepAsync enters TryUpgradeOfflineTracesAsync under _sleepGate
            await service.RecordTurnCompletionAsync("Alpha", "Offline trace alpha about golden puppy.");
            await service.RecordTurnCompletionAsync("Beta", "Offline trace beta about distant spiral galaxy.");
            await Task.Delay(15);

            // Bring mock online with ~60ms latency so s1 holds _sleepGate while s2 awaits _sleepGate
            mockHandler.IsOnline = true;
            mockHandler.ArtificialDelay = TimeSpan.FromMilliseconds(60);

            Task<Glacier.Tune.Sleep.ConsolidationResult?> s1 = service.TriggerSleepAsync();
            Task<Glacier.Tune.Sleep.ConsolidationResult?> s2 = service.TriggerSleepAsync();

            await Task.Delay(15);
            Task d = service.DisposeAsync().AsTask();

            Task all = Task.WhenAll(s1, s2, d);
            Task completed = await Task.WhenAny(all, Task.Delay(TimeSpan.FromSeconds(3)));
            Assert.Same(all, completed);

            foreach (Task t in new Task[] { s1, s2, d })
            {
                Assert.True(t.IsCompleted, $"Iteration {iter}: task stuck in status {t.Status}");
                if (t.IsFaulted)
                {
                    Exception inner = t.Exception!.InnerException!;
                    Assert.Fail($"Iteration {iter}: unexpected fault {inner.GetType().Name}: {inner.Message}");
                }
            }
        }
    }

    [Fact]
    public async Task SleepMemoryService_HighConcurrencyMixWithDisposeAsync_NeverLeaksInternalObjectDisposedException()
    {
        for (int round = 0; round < 30; round++)
        {
            NpuSidecarDiscovery.InvalidateCache();
            using var mockHandler = new MockNpuSidecarHttpHandler(dimensions: 768)
            {
                IsOnline = true,
                ArtificialDelay = TimeSpan.FromMilliseconds(round % 2 == 0 ? 12 : 4)
            };
            using var httpClient = new HttpClient(mockHandler);

            var service = new SleepMemoryService(new SleepMemoryOptions
            {
                TokenBudget = 30,
                IdleThreshold = TimeSpan.FromMinutes(10),
                PollingInterval = TimeSpan.FromMilliseconds(50),
                EnableNpuEmbeddings = true,
                EmbeddingEndpoint = "http://127.0.0.1:52625",
                HttpClient = httpClient,
                ProbeTimeout = TimeSpan.FromSeconds(1),
                ProbeCacheTtl = TimeSpan.Zero,
                EmbeddingTimeout = TimeSpan.FromSeconds(2)
            });

            // Seed some traces first so ConsolidateMessagesAsync and TriggerSleepAsync exercise MemEx + Daemon
            await service.RecordTurnCompletionAsync("Init", "Initial golden puppy observation near oak tree.");

            var bigMessages = new List<(string Role, string Content)>
            {
                ("user", "Log astronomical observation of distant spiral galaxy and bright nebula."),
                ("assistant", "Orbital telescope recorded bright nebula emission lines across deep space."),
                ("user", "Log maritime vessel telemetry in the northern harbor during calm ocean tides."),
                ("assistant", "Cargo vessel anchored smoothly inside the harbor before dawn."),
                ("user", "What did the young canine do outdoors?")
            };

            // Start admitted calls directly on the calling thread before DisposeAsync begins
            var admittedConsolidate = service.ConsolidateMessagesAsync(bigMessages);
            var admittedSleep1 = service.TriggerSleepAsync();
            var admittedSleep2 = service.TriggerSleepAsync();
            var admittedRecord = service.RecordTurnCompletionAsync("Concurrent", "Botanist inspected rare glasshouse ferns.");

            // Plus concurrent Task.Run racers (which may hit pre-entry, mid-flight, or post-dispose)
            var racerConsolidate = Task.Run(() => service.ConsolidateMessagesAsync(bigMessages));
            var racerSleep = Task.Run(() => service.TriggerSleepAsync());
            var syncRecordTask = Task.Run(() =>
            {
                service.RecordTurnCompletion("SyncTurn", "Pastry chef baked sourdough bread inside stone oven.");
            });

            await Task.Delay(round % 5);
            var disposeTask = service.DisposeAsync().AsTask();

            Task all = Task.WhenAll(admittedConsolidate, admittedSleep1, admittedSleep2, admittedRecord, racerConsolidate, racerSleep, syncRecordTask, disposeTask);
            Task completed = await Task.WhenAny(all, Task.Delay(TimeSpan.FromSeconds(4)));
            Assert.Same(all, completed);

            AssertNoInternalObjectDisposedLeak(admittedConsolidate, round, "AdmittedConsolidateMessagesAsync", allowOuterServiceDispose: false);
            AssertNoInternalObjectDisposedLeak(admittedSleep1, round, "AdmittedTriggerSleepAsync_1", allowOuterServiceDispose: false);
            AssertNoInternalObjectDisposedLeak(admittedSleep2, round, "AdmittedTriggerSleepAsync_2", allowOuterServiceDispose: false);
            AssertNoInternalObjectDisposedLeak(admittedRecord, round, "AdmittedRecordTurnCompletionAsync", allowOuterServiceDispose: false);
            AssertNoInternalObjectDisposedLeak(racerConsolidate, round, "RacerConsolidateMessagesAsync", allowOuterServiceDispose: true);
            AssertNoInternalObjectDisposedLeak(racerSleep, round, "RacerTriggerSleepAsync", allowOuterServiceDispose: true);
            AssertNoInternalObjectDisposedLeak(syncRecordTask, round, "RecordTurnCompletion", allowOuterServiceDispose: false);
            AssertNoInternalObjectDisposedLeak(disposeTask, round, "DisposeAsync", allowOuterServiceDispose: false);
        }
    }

    private static void AssertNoInternalObjectDisposedLeak(Task task, int round, string operationName, bool allowOuterServiceDispose)
    {
        Assert.True(task.IsCompleted, $"Round {round} {operationName} did not complete (status={task.Status}).");
        if (!task.IsFaulted)
        {
            return;
        }

        Exception ex = task.Exception!.GetBaseException();
        if (ex is ObjectDisposedException ode)
        {
            if (allowOuterServiceDispose &&
                string.Equals(ode.ObjectName, typeof(SleepMemoryService).FullName, StringComparison.Ordinal))
            {
                return;
            }

            Assert.Fail(
                $"Round {round} {operationName} leaked ObjectDisposedException (ObjectName='{ode.ObjectName}'): {ode.Message}\n{ode.StackTrace}");
        }

        Assert.Fail($"Round {round} {operationName} threw unexpected {ex.GetType().Name}: {ex.Message}");
    }

    [Fact]
    public async Task SleepMemoryEndpoints_ApiSleepStatusAndTrigger_SerializeExpectedNpuWireKeys()
    {
        NpuSidecarDiscovery.InvalidateCache();
        using var mockHandler = new MockNpuSidecarHttpHandler(dimensions: 768)
        {
            Models = ["embeddinggemma"]
        };
        using var sidecarClient = new HttpClient(mockHandler);

        await using var sleepMemory = new SleepMemoryService(new SleepMemoryOptions
        {
            TokenBudget = 80,
            IdleThreshold = TimeSpan.FromMinutes(10),
            PollingInterval = TimeSpan.FromMilliseconds(100),
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = "http://127.0.0.1:52625",
            HttpClient = sidecarClient,
            ProbeTimeout = TimeSpan.FromSeconds(1),
            ProbeCacheTtl = TimeSpan.Zero,
            EmbeddingTimeout = TimeSpan.FromSeconds(2)
        });

        Assert.True(await sleepMemory.InitializeEmbeddingsAsync());
        await sleepMemory.RecordTurnCompletionAsync("Prompt", "A golden puppy played near the sunny meadow.");

        var app = GlacierServeApp.CreateBuilder()
            .UseHost("127.0.0.1")
            .UsePort(0)
            .Build();
        app.MapSleepMemory(sleepMemory);

        await app.StartAsync();
        try
        {
            using var http = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{app.Port}") };

            using var statusResp = await http.GetAsync("/api/sleep/status");
            Assert.Equal(HttpStatusCode.OK, statusResp.StatusCode);

            string rawJson = await statusResp.Content.ReadAsStringAsync();
            using (JsonDocument doc = JsonDocument.Parse(rawJson))
            {
                JsonElement root = doc.RootElement;
                Assert.True(root.GetProperty("enabled").GetBoolean());
                Assert.True(root.GetProperty("npu_embeddings_active").GetBoolean());
                Assert.Equal(768, root.GetProperty("embedding_dimensions").GetInt32());
                Assert.Equal("http://127.0.0.1:52625", root.GetProperty("embedding_endpoint").GetString());
                Assert.Equal("embeddinggemma", root.GetProperty("embedding_model").GetString());
            }

            var typed = JsonSerializer.Deserialize(rawJson, ServeJsonContext.Default.SleepMemoryStatusResponse);
            Assert.NotNull(typed);
            Assert.True(typed!.NpuEmbeddingsActive);
            Assert.Equal(768, typed.EmbeddingDimensions);
            Assert.Equal("http://127.0.0.1:52625", typed.EmbeddingEndpoint);
            Assert.Equal("embeddinggemma", typed.EmbeddingModel);

            using var triggerResp = await http.PostAsync("/api/sleep/trigger", content: null);
            Assert.Equal(HttpStatusCode.OK, triggerResp.StatusCode);
            var triggeredStatus = JsonSerializer.Deserialize(
                await triggerResp.Content.ReadAsStringAsync(),
                ServeJsonContext.Default.SleepMemoryStatusResponse);
            Assert.NotNull(triggeredStatus);
            Assert.True(triggeredStatus!.SleepCycles >= 1);
            Assert.True(triggeredStatus.NpuEmbeddingsActive);
        }
        finally
        {
            app.Dispose();
        }
    }

    [Fact]
    public async Task MetricsEndpoint_CustomSidecarProbeThrowingRawOperationCanceledException_DegradesToHttp200AvailableFalse()
    {
        var app = GlacierServeApp.CreateBuilder()
            .UseHost("127.0.0.1")
            .UsePort(0)
            .Build();

        app.MapMetricsEndpoints(
            engine: null,
            new MetricsEndpointOptions
            {
                ProbeNpuSidecar = true,
                SidecarProbe = _ => throw new OperationCanceledException("Simulated internal probe timeout."),
                DeviceProvider = CreateSyntheticAllAcceleratorDevices
            });

        await app.StartAsync();
        try
        {
            using var http = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{app.Port}") };

            using var healthResp = await http.GetAsync("/health");
            Assert.Equal(HttpStatusCode.OK, healthResp.StatusCode);
            var health = JsonSerializer.Deserialize(
                await healthResp.Content.ReadAsStringAsync(),
                ServeJsonContext.Default.HealthCheckResponse);
            Assert.NotNull(health);
            Assert.NotNull(health!.NpuSidecar);
            Assert.False(health.NpuSidecar!.Available);

            using var metricsResp = await http.GetAsync("/metrics");
            Assert.Equal(HttpStatusCode.OK, metricsResp.StatusCode);
            string metrics = await metricsResp.Content.ReadAsStringAsync();
            Assert.Contains("glacier_npu_sidecar_up 0", metrics);
        }
        finally
        {
            app.Dispose();
        }
    }

    private sealed class MockNpuSidecarHttpHandler : HttpMessageHandler
    {
        private readonly int _dimensions;
        private int _embeddingCallCount;

        public MockNpuSidecarHttpHandler(int dimensions)
        {
            _dimensions = dimensions;
        }

        public bool IsOnline { get; set; } = true;
        public HttpStatusCode FailureStatusCode { get; set; } = HttpStatusCode.ServiceUnavailable;
        public List<string> Models { get; set; } = ["embeddinggemma"];
        public TimeSpan ArtificialDelay { get; set; } = TimeSpan.Zero;
        public int EmbeddingCallCount => Volatile.Read(ref _embeddingCallCount);

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            if (ArtificialDelay > TimeSpan.Zero)
            {
                await Task.Delay(ArtificialDelay, cancellationToken).ConfigureAwait(false);
            }

            if (!IsOnline)
            {
                if (FailureStatusCode == HttpStatusCode.ServiceUnavailable)
                {
                    throw new HttpRequestException("Simulated offline NPU sidecar.");
                }

                return new HttpResponseMessage(FailureStatusCode)
                {
                    Content = new StringContent("{\"error\":\"unavailable\"}", Encoding.UTF8, "application/json")
                };
            }

            string path = request.RequestUri?.AbsolutePath ?? string.Empty;
            if (request.Method == HttpMethod.Get && path.EndsWith("/v1/models", StringComparison.OrdinalIgnoreCase))
            {
                string modelsJson = JsonSerializer.Serialize(new { data = Models.Select(m => new { id = m }).ToArray() });
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(modelsJson, Encoding.UTF8, "application/json")
                };
            }

            if (request.Method == HttpMethod.Post && path.EndsWith("/v1/embeddings", StringComparison.OrdinalIgnoreCase))
            {
                Interlocked.Increment(ref _embeddingCallCount);
                string body = request.Content != null
                    ? await request.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false)
                    : "{}";

                using var doc = JsonDocument.Parse(body);
                string input = doc.RootElement.TryGetProperty("input", out var inputEl)
                    ? inputEl.GetString() ?? string.Empty
                    : string.Empty;

                float[] vec = BuildSemanticClusterVector(input, _dimensions);
                string embJson = JsonSerializer.Serialize(new
                {
                    data = new[]
                    {
                        new { embedding = vec }
                    }
                });

                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(embJson, Encoding.UTF8, "application/json")
                };
            }

            return new HttpResponseMessage(HttpStatusCode.NotFound);
        }

        private static float[] BuildSemanticClusterVector(string text, int dims)
        {
            float[] vec = new float[dims];
            string low = text.ToLowerInvariant();

            if (ContainsAny(low, "puppy", "dog", "canine", "hound", "bark", "meadow"))
            {
                AddClusterSignal(vec, start: 0, width: 32, weight: 1.0f);
            }
            else if (ContainsAny(low, "botanist", "fern", "glasshouse"))
            {
                AddClusterSignal(vec, start: 32, width: 32, weight: 1.0f);
            }
            else if (ContainsAny(low, "telescope", "nebula", "galaxy", "orbital", "astronomy"))
            {
                AddClusterSignal(vec, start: 64, width: 32, weight: 1.0f);
            }
            else if (ContainsAny(low, "vessel", "harbor", "anchor", "ocean", "maritime"))
            {
                AddClusterSignal(vec, start: 96, width: 32, weight: 1.0f);
            }
            else if (ContainsAny(low, "pastry", "bread", "oven", "rosemary", "culinary"))
            {
                AddClusterSignal(vec, start: 128, width: 32, weight: 1.0f);
            }
            else
            {
                int bucket = Math.Abs(StringComparer.Ordinal.GetHashCode(low)) % Math.Max(1, dims - 160);
                vec[160 + bucket] = 1.0f;
            }

            // Tiny deterministic residual so different strings have distinct norms/positions within cluster
            uint h = 2166136261u;
            foreach (char c in low)
            {
                h ^= c;
                h *= 16777619u;
            }
            int tailIdx = dims > 192 ? 192 + (int)(h % (uint)(dims - 192)) : (int)(h % (uint)dims);
            vec[tailIdx] += 0.05f;

            double sumSq = 0;
            for (int i = 0; i < dims; i++) sumSq += vec[i] * vec[i];
            float inv = 1.0f / MathF.Sqrt((float)sumSq);
            for (int i = 0; i < dims; i++) vec[i] *= inv;
            return vec;
        }

        private static void AddClusterSignal(float[] vec, int start, int width, float weight)
        {
            int end = Math.Min(vec.Length, start + width);
            for (int i = start; i < end; i++)
            {
                vec[i] += weight;
            }
        }

        private static bool ContainsAny(string text, params string[] terms)
        {
            foreach (string term in terms)
            {
                if (text.Contains(term, StringComparison.Ordinal))
                    return true;
            }
            return false;
        }
    }

    private sealed class SingleThreadSynchronizationContext : SynchronizationContext, IDisposable
    {
        private readonly BlockingCollection<(SendOrPostCallback Callback, object? State)> _queue = new();
        private readonly Thread _worker;

        public SingleThreadSynchronizationContext()
        {
            _worker = new Thread(RunLoop)
            {
                IsBackground = true,
                Name = "TestSingleThreadSyncCtx"
            };
            _worker.Start();
        }

        private void RunLoop()
        {
            SetSynchronizationContext(this);
            foreach (var (callback, state) in _queue.GetConsumingEnumerable())
            {
                callback(state);
            }
        }

        public override void Post(SendOrPostCallback d, object? state)
        {
            _queue.Add((d, state));
        }

        public override void Send(SendOrPostCallback d, object? state)
        {
            if (Thread.CurrentThread == _worker)
            {
                d(state);
                return;
            }

            using var done = new ManualResetEventSlim();
            Exception? error = null;
            _queue.Add((_ =>
            {
                try { d(state); }
                catch (Exception ex) { error = ex; }
                finally { done.Set(); }
            }, null));
            done.Wait();
            if (error != null) throw error;
        }

        public Task RunAsync(Action action)
        {
            var tcs = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            Post(_ =>
            {
                try
                {
                    action();
                    tcs.SetResult();
                }
                catch (Exception ex)
                {
                    tcs.SetException(ex);
                }
            }, null);
            return tcs.Task;
        }

        public void Dispose()
        {
            _queue.CompleteAdding();
            _worker.Join(TimeSpan.FromSeconds(2));
            _queue.Dispose();
        }
    }
}

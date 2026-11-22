using System;
using System.Collections.Generic;
using System.Globalization;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Hardware;
using Glacier.Serve.Core;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Serialization;
using Glacier.Serve.Server;

namespace Glacier.Serve.Inference.Http;

public sealed class MetricsEndpointOptions
{
    public bool ProbeNpuSidecar { get; init; } = true;
    public TimeSpan SidecarCacheTtl { get; init; } = NpuSidecarDiscovery.DefaultCacheTtl;
    public TimeSpan? SidecarPerEndpointTimeout { get; init; }
    public HttpClient? SidecarHttpClient { get; init; }
    public IReadOnlyList<(string BaseUrl, NpuSidecarRuntime Runtime)>? SidecarCandidates { get; init; }
    public Func<CancellationToken, Task<NpuSidecarEndpoint?>>? SidecarProbe { get; init; }
    public Func<IReadOnlyList<DeviceInfo>>? DeviceProvider { get; init; }
}

public static class MetricsEndpoint
{
    private const ulong OneGiBBytes = 1024UL * 1024UL * 1024UL;

    public static void MapMetricsEndpoints(this GlacierServeApp app, ContinuousBatchEngine? engine = null)
    {
        MapMetricsEndpoints(app, engine, new MetricsEndpointOptions());
    }

    public static void MapMetricsEndpoints(this GlacierServeApp app, ContinuousBatchEngine? engine, MetricsEndpointOptions options)
    {
        ArgumentNullException.ThrowIfNull(app);
        ArgumentNullException.ThrowIfNull(options);

        app.MapGet("/health", async ctx =>
        {
            var (devices, npuDevices) = GetHardwareSnapshot(options);
            var sidecar = await ProbeSidecarSafeAsync(options, ctx.RequestAborted).ConfigureAwait(false);

            var deviceStatuses = new List<AcceleratorDeviceStatus>(devices.Count);
            var npuStatuses = new List<AcceleratorDeviceStatus>(npuDevices.Count);

            for (int i = 0; i < devices.Count; i++)
            {
                deviceStatuses.Add(ToDeviceStatus(devices[i]));
            }
            for (int i = 0; i < npuDevices.Count; i++)
            {
                npuStatuses.Add(ToDeviceStatus(npuDevices[i]));
            }

            var health = new HealthCheckResponse
            {
                Status = "ok",
                Server = "Glacier.Serve",
                Engine = engine != null ? "ContinuousBatchEngine" : "None",
                Backend = engine?.ActiveBackend,
                GpuAccelerated = engine?.IsGpuAccelerated,
                PagedAttention = engine != null,
                ActiveBatches = engine?.ActiveBatchSize ?? 0,
                WaitingQueue = engine?.WaitingQueueLength ?? 0,
                FreeKvBlocks = engine?.BlockPool.FreeBlocksCount ?? 0,
                TotalKvBlocks = engine?.BlockPool.TotalBlocks ?? 0,
                TokensGenerated = engine?.TotalTokensGenerated ?? 0,
                ThroughputTokS = engine?.ThroughputTokensPerSec ?? 0.0,
                SleepState = engine?.SleepDaemon?.CurrentState.ToString(),
                PendingDreams = engine?.SleepDaemon?.PendingDreamCount,
                Devices = deviceStatuses,
                NpuDevices = npuStatuses,
                NpuSidecar = ToSidecarStatus(sidecar)
            };
            await ctx.Response.WriteJsonAsync(health, ServeJsonContext.Default.HealthCheckResponse, ctx.RequestAborted).ConfigureAwait(false);
        });

        app.MapGet("/metrics", async ctx =>
        {
            var (devices, npuDevices) = GetHardwareSnapshot(options);
            var sidecar = await ProbeSidecarSafeAsync(options, ctx.RequestAborted).ConfigureAwait(false);

            var sb = new StringBuilder();
            if (engine != null)
            {
                sb.AppendLine("# HELP glacier_active_sequences Number of active decoding sequences");
                sb.AppendLine("# TYPE glacier_active_sequences gauge");
                sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"glacier_active_sequences {engine.ActiveBatchSize}"));

                sb.AppendLine("# HELP glacier_waiting_requests Number of requests waiting in admission queue");
                sb.AppendLine("# TYPE glacier_waiting_requests gauge");
                sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"glacier_waiting_requests {engine.WaitingQueueLength}"));

                sb.AppendLine("# HELP glacier_free_kv_blocks Free blocks in PagedBlockPool");
                sb.AppendLine("# TYPE glacier_free_kv_blocks gauge");
                sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"glacier_free_kv_blocks {engine.BlockPool.FreeBlocksCount}"));

                sb.AppendLine("# HELP glacier_total_tokens_generated Total generated tokens");
                sb.AppendLine("# TYPE glacier_total_tokens_generated counter");
                sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"glacier_total_tokens_generated {engine.TotalTokensGenerated}"));

                sb.AppendLine("# HELP glacier_step_latency_ms Duration of last iteration batch step in ms");
                sb.AppendLine("# TYPE glacier_step_latency_ms gauge");
                sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"glacier_step_latency_ms {engine.LastIterationMs:F2}"));

                if (engine.SleepDaemon is { } sleepDaemon)
                {
                    sb.AppendLine("# HELP glacier_sleep_state Sleep lifecycle state (0=Awake, 1=EnteringSleep, 2=Sleeping, 3=Waking)");
                    sb.AppendLine("# TYPE glacier_sleep_state gauge");
                    sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"glacier_sleep_state {(int)sleepDaemon.CurrentState}"));

                    sb.AppendLine("# HELP glacier_sleep_cycles_total Completed autonomous sleep consolidation cycles");
                    sb.AppendLine("# TYPE glacier_sleep_cycles_total counter");
                    sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"glacier_sleep_cycles_total {sleepDaemon.CompletedSleepCycles}"));

                    sb.AppendLine("# HELP glacier_sleep_dreams_consolidated_total Total dream candidates consolidated into MoE expansion slots");
                    sb.AppendLine("# TYPE glacier_sleep_dreams_consolidated_total counter");
                    sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"glacier_sleep_dreams_consolidated_total {sleepDaemon.TotalDreamsConsolidated}"));

                    sb.AppendLine("# HELP glacier_sleep_pending_dreams Queued dream candidates awaiting sleep consolidation");
                    sb.AppendLine("# TYPE glacier_sleep_pending_dreams gauge");
                    sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"glacier_sleep_pending_dreams {sleepDaemon.PendingDreamCount}"));
                }
            }

            int dgpuCount = 0;
            int igpuCount = 0;
            int npuCount = 0;
            int cpuCount = 0;

            for (int i = 0; i < devices.Count; i++)
            {
                switch (ClassifyDeviceType(devices[i]))
                {
                    case "dgpu": dgpuCount++; break;
                    case "igpu": igpuCount++; break;
                    case "npu": npuCount++; break;
                    case "cpu": cpuCount++; break;
                }
            }

            sb.AppendLine("# HELP glacier_accelerator_devices_total Total number of detected compute devices (dGPU, iGPU, NPU, CPU)");
            sb.AppendLine("# TYPE glacier_accelerator_devices_total gauge");
            sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"glacier_accelerator_devices_total {devices.Count}"));

            sb.AppendLine("# HELP glacier_accelerator_devices_by_type Number of detected compute devices by accelerator type");
            sb.AppendLine("# TYPE glacier_accelerator_devices_by_type gauge");
            sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"glacier_accelerator_devices_by_type{{type=\"dgpu\"}} {dgpuCount}"));
            sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"glacier_accelerator_devices_by_type{{type=\"igpu\"}} {igpuCount}"));
            sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"glacier_accelerator_devices_by_type{{type=\"npu\"}} {npuCount}"));
            sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"glacier_accelerator_devices_by_type{{type=\"cpu\"}} {cpuCount}"));

            if (devices.Count > 0)
            {
                sb.AppendLine("# HELP glacier_accelerator_device_info Static metadata for each detected accelerator device");
                sb.AppendLine("# TYPE glacier_accelerator_device_info gauge");
                for (int i = 0; i < devices.Count; i++)
                {
                    var d = devices[i];
                    string type = ClassifyDeviceType(d);
                    sb.AppendLine(
                        $"glacier_accelerator_device_info{{id=\"{EscapePrometheusLabel(d.Id)}\",name=\"{EscapePrometheusLabel(d.Name)}\",vendor=\"{EscapePrometheusLabel(d.Vendor.ToString())}\",type=\"{type}\",engine=\"{EscapePrometheusLabel(d.RecommendedEngine.ToString())}\"}} 1");
                }

                sb.AppendLine("# HELP glacier_accelerator_dedicated_vram_bytes Dedicated VRAM capacity in bytes per accelerator device");
                sb.AppendLine("# TYPE glacier_accelerator_dedicated_vram_bytes gauge");
                for (int i = 0; i < devices.Count; i++)
                {
                    var d = devices[i];
                    string type = ClassifyDeviceType(d);
                    sb.AppendLine(string.Create(
                        CultureInfo.InvariantCulture,
                        $"glacier_accelerator_dedicated_vram_bytes{{id=\"{EscapePrometheusLabel(d.Id)}\",type=\"{type}\"}} {d.DedicatedVramBytes}"));
                }

                sb.AppendLine("# HELP glacier_accelerator_shared_vram_bytes Shared/UMA memory capacity in bytes per accelerator device");
                sb.AppendLine("# TYPE glacier_accelerator_shared_vram_bytes gauge");
                for (int i = 0; i < devices.Count; i++)
                {
                    var d = devices[i];
                    string type = ClassifyDeviceType(d);
                    sb.AppendLine(string.Create(
                        CultureInfo.InvariantCulture,
                        $"glacier_accelerator_shared_vram_bytes{{id=\"{EscapePrometheusLabel(d.Id)}\",type=\"{type}\"}} {d.SharedVramBytes}"));
                }
            }

            sb.AppendLine("# HELP glacier_npu_devices_total Total number of detected Neural Processing Units (NPUs)");
            sb.AppendLine("# TYPE glacier_npu_devices_total gauge");
            sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"glacier_npu_devices_total {npuDevices.Count}"));

            ulong totalNpuSharedCapBytes = 0;
            if (npuDevices.Count > 0)
            {
                sb.AppendLine("# HELP glacier_npu_shared_cap_bytes Shared NPU memory budget cap in bytes per NPU device");
                sb.AppendLine("# TYPE glacier_npu_shared_cap_bytes gauge");
                for (int i = 0; i < npuDevices.Count; i++)
                {
                    var npu = npuDevices[i];
                    totalNpuSharedCapBytes += npu.SharedVramBytes;
                    sb.AppendLine(string.Create(
                        CultureInfo.InvariantCulture,
                        $"glacier_npu_shared_cap_bytes{{id=\"{EscapePrometheusLabel(npu.Id)}\",name=\"{EscapePrometheusLabel(npu.Name)}\"}} {npu.SharedVramBytes}"));
                }
            }

            sb.AppendLine("# HELP glacier_npu_shared_cap_bytes_total Aggregate shared NPU memory budget cap in bytes across all NPU devices");
            sb.AppendLine("# TYPE glacier_npu_shared_cap_bytes_total gauge");
            sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"glacier_npu_shared_cap_bytes_total {totalNpuSharedCapBytes}"));

            sb.AppendLine("# HELP glacier_npu_sidecar_up Whether a local OpenAI-compatible NPU sidecar is reachable (1=up, 0=down)");
            sb.AppendLine("# TYPE glacier_npu_sidecar_up gauge");
            sb.AppendLine(sidecar != null ? "glacier_npu_sidecar_up 1" : "glacier_npu_sidecar_up 0");

            sb.AppendLine("# HELP glacier_npu_sidecar_latency_ms Probe latency in ms for the discovered NPU sidecar (0 when offline)");
            sb.AppendLine("# TYPE glacier_npu_sidecar_latency_ms gauge");
            double latencyMs = sidecar?.LatencyMs ?? 0.0;
            sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"glacier_npu_sidecar_latency_ms {latencyMs:F2}"));

            sb.AppendLine("# HELP glacier_npu_sidecar_models_total Number of models reported by the discovered NPU sidecar (0 when offline)");
            sb.AppendLine("# TYPE glacier_npu_sidecar_models_total gauge");
            int modelsCount = sidecar?.Models?.Count ?? 0;
            sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"glacier_npu_sidecar_models_total {modelsCount}"));

            ctx.Response.ContentType = "text/plain; version=0.0.4; charset=utf-8";
            await ctx.Response.WriteUtf8Async(sb.ToString(), ctx.RequestAborted).ConfigureAwait(false);
        });

        app.MapGet("/v1/sleep/status", async ctx =>
        {
            var daemon = engine?.SleepDaemon;
            var status = new SleepStatusResponse
            {
                Enabled = daemon != null,
                State = daemon != null ? daemon.CurrentState.ToString() : "Disabled",
                CompletedCycles = daemon?.CompletedSleepCycles ?? 0,
                TotalDreamsConsolidated = daemon?.TotalDreamsConsolidated ?? 0,
                PendingDreams = daemon?.PendingDreamCount ?? 0
            };
            await ctx.Response.WriteJsonAsync(status, ServeJsonContext.Default.SleepStatusResponse, ctx.RequestAborted).ConfigureAwait(false);
        });

        app.MapPost("/v1/sleep/trigger", async ctx =>
        {
            var daemon = engine?.SleepDaemon;
            if (daemon == null || engine == null)
            {
                await ctx.Response.WriteJsonAsync(
                    new SleepTriggerResponse { Triggered = false, State = "Disabled" },
                    ServeJsonContext.Default.SleepTriggerResponse,
                    ctx.RequestAborted).ConfigureAwait(false);
                return;
            }

            int dim = Math.Max(4, engine.SleepVectorDimension);
            float[] centroid = new float[dim];
            float invSqrt = 1.0f / MathF.Sqrt(dim);
            Array.Fill(centroid, invSqrt);

            var result = await daemon.TriggerImmediateSleepAsync(centroid, ctx.RequestAborted).ConfigureAwait(false);
            var response = result == null
                ? new SleepTriggerResponse
                {
                    Triggered = false,
                    State = daemon.CurrentState.ToString()
                }
                : new SleepTriggerResponse
                {
                    Triggered = true,
                    State = daemon.CurrentState.ToString(),
                    ExpertSlotIndex = result.ExpertSlotIndex,
                    TotalDreams = result.TotalDreamsGenerated,
                    AcceptedDreams = result.TotalDreamsAccepted,
                    AcceptanceRate = result.AcceptanceRate,
                    InitialLoss = result.InitialGkdLoss,
                    FinalLoss = result.FinalGkdLoss,
                    SynapticPruningSucceeded = result.SynapticPruningSucceeded
                };

            await ctx.Response.WriteJsonAsync(response, ServeJsonContext.Default.SleepTriggerResponse, ctx.RequestAborted).ConfigureAwait(false);
        });
    }

    public static void MapInference(this GlacierServeApp app, ContinuousBatchEngine engine, string defaultModelName = "glacier-enterprise-7b")
    {
        OpenAiEndpoints.MapOpenAiEndpoints(app, engine, defaultModelName);
        OllamaEndpoints.MapOllamaEndpoints(app, engine, defaultModelName);
        MapMetricsEndpoints(app, engine);
    }

    public static void MapInference(this GlacierServeApp app, ContinuousBatchEngine engine, string defaultModelName, Glacier.Serve.Daemons.SleepMemoryService? sleepMemory)
    {
        MapInference(app, engine, defaultModelName);
        app.MapSleepMemory(sleepMemory);
    }

    internal static string ClassifyDeviceType(DeviceInfo device)
    {
        if (device.IsNpu) return "npu";
        if (device.Vendor == GpuVendor.Cpu) return "cpu";
        if (device.DedicatedVramBytes < OneGiBBytes) return "igpu";
        return "dgpu";
    }

    private static (IReadOnlyList<DeviceInfo> Devices, IReadOnlyList<DeviceInfo> NpuDevices) GetHardwareSnapshot(MetricsEndpointOptions options)
    {
        IReadOnlyList<DeviceInfo> devices;
        IReadOnlyList<DeviceInfo> npuDevices;

        if (options.DeviceProvider != null)
        {
            devices = options.DeviceProvider() ?? [];
            var npuList = new List<DeviceInfo>();
            for (int i = 0; i < devices.Count; i++)
            {
                if (devices[i].IsNpu)
                {
                    npuList.Add(devices[i]);
                }
            }
            npuDevices = npuList;
        }
        else
        {
            devices = DeviceManager.GetDevices() ?? [];
            npuDevices = DeviceManager.GetNpuDevices() ?? [];
        }

        return (devices, npuDevices);
    }

    private static async Task<NpuSidecarEndpoint?> ProbeSidecarSafeAsync(MetricsEndpointOptions options, CancellationToken ct)
    {
        if (!options.ProbeNpuSidecar)
        {
            return null;
        }

        try
        {
            if (options.SidecarProbe != null)
            {
                return await options.SidecarProbe(ct).ConfigureAwait(false);
            }

            return await NpuSidecarDiscovery.ProbeCachedAsync(
                cacheTtl: options.SidecarCacheTtl,
                httpClient: options.SidecarHttpClient,
                candidates: options.SidecarCandidates,
                perEndpointTimeout: options.SidecarPerEndpointTimeout,
                cancellationToken: ct).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            throw;
        }
        catch (OperationCanceledException)
        {
            return null;
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or TimeoutException or JsonException or UriFormatException or InvalidOperationException or NotSupportedException)
        {
            return null;
        }
    }

    private static AcceleratorDeviceStatus ToDeviceStatus(DeviceInfo device)
    {
        var supported = new List<string>(device.SupportedEngines.Count);
        for (int i = 0; i < device.SupportedEngines.Count; i++)
        {
            supported.Add(device.SupportedEngines[i].ToString());
        }

        return new AcceleratorDeviceStatus
        {
            Id = device.Id,
            Index = device.Index,
            Name = device.Name,
            Vendor = device.Vendor.ToString(),
            Type = ClassifyDeviceType(device),
            IsNpu = device.IsNpu,
            DedicatedVramBytes = device.DedicatedVramBytes,
            SharedVramBytes = device.SharedVramBytes,
            DedicatedVramGb = Math.Round(device.DedicatedVramGb, 3),
            SharedVramGb = Math.Round(device.SharedVramGb, 3),
            RecommendedEngine = device.RecommendedEngine.ToString(),
            SupportedEngines = supported
        };
    }

    private static NpuSidecarStatus ToSidecarStatus(NpuSidecarEndpoint? sidecar)
    {
        if (sidecar == null)
        {
            return new NpuSidecarStatus
            {
                Available = false,
                BaseUrl = null,
                Runtime = null,
                Models = [],
                LatencyMs = null
            };
        }

        return new NpuSidecarStatus
        {
            Available = true,
            BaseUrl = sidecar.BaseUrl,
            Runtime = sidecar.Runtime.ToString(),
            Models = sidecar.Models is { Count: > 0 } ? [.. sidecar.Models] : [],
            LatencyMs = sidecar.LatencyMs
        };
    }

    private static string EscapePrometheusLabel(string value)
    {
        if (string.IsNullOrEmpty(value)) return string.Empty;
        return value
            .Replace("\\", "\\\\", StringComparison.Ordinal)
            .Replace("\"", "\\\"", StringComparison.Ordinal)
            .Replace("\n", "\\n", StringComparison.Ordinal)
            .Replace("\r", "", StringComparison.Ordinal);
    }
}

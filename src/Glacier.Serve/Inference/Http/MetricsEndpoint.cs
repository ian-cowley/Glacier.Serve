using System;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Glacier.Serve.Core;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Server;

namespace Glacier.Serve.Inference.Http;

public static class MetricsEndpoint
{
    public static void MapMetricsEndpoints(this GlacierServeApp app, ContinuousBatchEngine engine)
    {
        app.MapGet("/health", async ctx =>
        {
            var health = new Glacier.Serve.Serialization.HealthCheckResponse
            {
                Status = "ok",
                Server = "Glacier.Serve",
                Engine = "ContinuousBatchEngine",
                Backend = engine.ActiveBackend,
                GpuAccelerated = engine.IsGpuAccelerated,
                PagedAttention = true,
                ActiveBatches = engine.ActiveBatchSize,
                WaitingQueue = engine.WaitingQueueLength,
                FreeKvBlocks = engine.BlockPool.FreeBlocksCount,
                TotalKvBlocks = engine.BlockPool.TotalBlocks,
                TokensGenerated = engine.TotalTokensGenerated,
                ThroughputTokS = engine.ThroughputTokensPerSec,
                SleepState = engine.SleepDaemon?.CurrentState.ToString(),
                PendingDreams = engine.SleepDaemon?.PendingDreamCount
            };
            await ctx.Response.WriteJsonAsync(health, Glacier.Serve.Serialization.ServeJsonContext.Default.HealthCheckResponse);
        });

        app.MapGet("/metrics", async ctx =>
        {
            var sb = new StringBuilder();
            sb.AppendLine("# HELP glacier_active_sequences Number of active decoding sequences");
            sb.AppendLine("# TYPE glacier_active_sequences gauge");
            sb.AppendLine($"glacier_active_sequences {engine.ActiveBatchSize}");

            sb.AppendLine("# HELP glacier_waiting_requests Number of requests waiting in admission queue");
            sb.AppendLine("# TYPE glacier_waiting_requests gauge");
            sb.AppendLine($"glacier_waiting_requests {engine.WaitingQueueLength}");

            sb.AppendLine("# HELP glacier_free_kv_blocks Free blocks in PagedBlockPool");
            sb.AppendLine("# TYPE glacier_free_kv_blocks gauge");
            sb.AppendLine($"glacier_free_kv_blocks {engine.BlockPool.FreeBlocksCount}");

            sb.AppendLine("# HELP glacier_total_tokens_generated Total generated tokens");
            sb.AppendLine("# TYPE glacier_total_tokens_generated counter");
            sb.AppendLine($"glacier_total_tokens_generated {engine.TotalTokensGenerated}");

            sb.AppendLine("# HELP glacier_step_latency_ms Duration of last iteration batch step in ms");
            sb.AppendLine("# TYPE glacier_step_latency_ms gauge");
            sb.AppendLine($"glacier_step_latency_ms {engine.LastIterationMs:F2}");

            if (engine.SleepDaemon is { } sleepDaemon)
            {
                sb.AppendLine("# HELP glacier_sleep_state Sleep lifecycle state (0=Awake, 1=EnteringSleep, 2=Sleeping, 3=Waking)");
                sb.AppendLine("# TYPE glacier_sleep_state gauge");
                sb.AppendLine($"glacier_sleep_state {(int)sleepDaemon.CurrentState}");

                sb.AppendLine("# HELP glacier_sleep_cycles_total Completed autonomous sleep consolidation cycles");
                sb.AppendLine("# TYPE glacier_sleep_cycles_total counter");
                sb.AppendLine($"glacier_sleep_cycles_total {sleepDaemon.CompletedSleepCycles}");

                sb.AppendLine("# HELP glacier_sleep_dreams_consolidated_total Total dream candidates consolidated into MoE expansion slots");
                sb.AppendLine("# TYPE glacier_sleep_dreams_consolidated_total counter");
                sb.AppendLine($"glacier_sleep_dreams_consolidated_total {sleepDaemon.TotalDreamsConsolidated}");

                sb.AppendLine("# HELP glacier_sleep_pending_dreams Queued dream candidates awaiting sleep consolidation");
                sb.AppendLine("# TYPE glacier_sleep_pending_dreams gauge");
                sb.AppendLine($"glacier_sleep_pending_dreams {sleepDaemon.PendingDreamCount}");
            }

            ctx.Response.ContentType = "text/plain; version=0.0.4; charset=utf-8";
            await ctx.Response.WriteUtf8Async(sb.ToString());
        });

        app.MapGet("/v1/sleep/status", async ctx =>
        {
            var daemon = engine.SleepDaemon;
            var status = new Glacier.Serve.Serialization.SleepStatusResponse
            {
                Enabled = daemon != null,
                State = daemon != null ? daemon.CurrentState.ToString() : "Disabled",
                CompletedCycles = daemon?.CompletedSleepCycles ?? 0,
                TotalDreamsConsolidated = daemon?.TotalDreamsConsolidated ?? 0,
                PendingDreams = daemon?.PendingDreamCount ?? 0
            };
            await ctx.Response.WriteJsonAsync(status, Glacier.Serve.Serialization.ServeJsonContext.Default.SleepStatusResponse);
        });

        app.MapPost("/v1/sleep/trigger", async ctx =>
        {
            var daemon = engine.SleepDaemon;
            if (daemon == null)
            {
                await ctx.Response.WriteJsonAsync(
                    new Glacier.Serve.Serialization.SleepTriggerResponse { Triggered = false, State = "Disabled" },
                    Glacier.Serve.Serialization.ServeJsonContext.Default.SleepTriggerResponse);
                return;
            }

            int dim = Math.Max(4, engine.SleepVectorDimension);
            float[] centroid = new float[dim];
            float invSqrt = 1.0f / MathF.Sqrt(dim);
            Array.Fill(centroid, invSqrt);

            var result = await daemon.TriggerImmediateSleepAsync(centroid, ctx.RequestAborted);
            var response = result == null
                ? new Glacier.Serve.Serialization.SleepTriggerResponse
                {
                    Triggered = false,
                    State = daemon.CurrentState.ToString()
                }
                : new Glacier.Serve.Serialization.SleepTriggerResponse
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

            await ctx.Response.WriteJsonAsync(response, Glacier.Serve.Serialization.ServeJsonContext.Default.SleepTriggerResponse);
        });
    }

    public static void MapInference(this GlacierServeApp app, ContinuousBatchEngine engine, string defaultModelName = "glacier-enterprise-7b")
    {
        OpenAiEndpoints.MapOpenAiEndpoints(app, engine, defaultModelName);
        OllamaEndpoints.MapOllamaEndpoints(app, engine, defaultModelName);
        MapMetricsEndpoints(app, engine);
    }
}

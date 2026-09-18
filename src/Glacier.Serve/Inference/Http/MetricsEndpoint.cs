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
                PagedAttention = true,
                ActiveBatches = engine.ActiveBatchSize,
                WaitingQueue = engine.WaitingQueueLength,
                FreeKvBlocks = engine.BlockPool.FreeBlocksCount,
                TotalKvBlocks = engine.BlockPool.TotalBlocks,
                TokensGenerated = engine.TotalTokensGenerated,
                ThroughputTokS = engine.ThroughputTokensPerSec
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

            ctx.Response.ContentType = "text/plain; version=0.0.4; charset=utf-8";
            await ctx.Response.WriteUtf8Async(sb.ToString());
        });
    }

    public static void MapInference(this GlacierServeApp app, ContinuousBatchEngine engine, string defaultModelName = "glacier-enterprise-7b")
    {
        OpenAiEndpoints.MapOpenAiEndpoints(app, engine, defaultModelName);
        OllamaEndpoints.MapOllamaEndpoints(app, engine, defaultModelName);
        MapMetricsEndpoints(app, engine);
    }
}

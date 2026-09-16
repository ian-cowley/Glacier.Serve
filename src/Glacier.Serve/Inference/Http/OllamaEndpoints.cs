using System;
using System.Buffers;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Glacier.Serve.Core;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Server;

namespace Glacier.Serve.Inference.Http;

public static class OllamaEndpoints
{
    public static void MapOllamaEndpoints(this GlacierServeApp app, ContinuousBatchEngine engine, string defaultModelName = "glacier-enterprise-7b")
    {
        // POST /api/chat
        app.MapPost("/api/chat", async ctx =>
        {
            var req = ParseRequest(ctx.Request);
            if (req == null)
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteUtf8Async("{\"error\":\"Invalid JSON request\"}");
                return;
            }

            var (state, tokenStream) = await engine.EnqueueAsync(req);

            ctx.Response.ContentType = "application/x-ndjson; charset=utf-8";
            ctx.Response.Headers["Connection"] = "close";
            await ctx.Response.EnsureHeadersSentAsync(null);

            await foreach (var token in tokenStream)
            {
                string chunk = JsonSerializer.Serialize(new
                {
                    model = defaultModelName,
                    message = new { role = "assistant", content = token },
                    done = false
                }) + "\n";

                byte[] bytes = Encoding.UTF8.GetBytes(chunk);
                await ctx.Response.BodyWriter.WriteAsync(bytes);
                await ctx.Response.BodyWriter.FlushAsync();
            }

            string finalChunk = JsonSerializer.Serialize(new
            {
                model = defaultModelName,
                done = true,
                prompt_eval_count = state.PromptTokens.Length,
                eval_count = state.GeneratedTokens.Count
            }) + "\n";

            byte[] finalBytes = Encoding.UTF8.GetBytes(finalChunk);
            await ctx.Response.BodyWriter.WriteAsync(finalBytes);
            await ctx.Response.BodyWriter.FlushAsync();
        });

        // POST /api/generate
        app.MapPost("/api/generate", async ctx =>
        {
            var req = ParseRequest(ctx.Request);
            if (req == null)
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteUtf8Async("{\"error\":\"Invalid JSON request\"}");
                return;
            }

            var (state, tokenStream) = await engine.EnqueueAsync(req);

            ctx.Response.ContentType = "application/x-ndjson; charset=utf-8";
            ctx.Response.Headers["Connection"] = "close";
            await ctx.Response.EnsureHeadersSentAsync(null);

            await foreach (var token in tokenStream)
            {
                string chunk = JsonSerializer.Serialize(new
                {
                    model = defaultModelName,
                    response = token,
                    done = false
                }) + "\n";

                byte[] bytes = Encoding.UTF8.GetBytes(chunk);
                await ctx.Response.BodyWriter.WriteAsync(bytes);
                await ctx.Response.BodyWriter.FlushAsync();
            }

            string finalChunk = JsonSerializer.Serialize(new
            {
                model = defaultModelName,
                done = true,
                prompt_eval_count = state.PromptTokens.Length,
                eval_count = state.GeneratedTokens.Count
            }) + "\n";

            byte[] finalBytes = Encoding.UTF8.GetBytes(finalChunk);
            await ctx.Response.BodyWriter.WriteAsync(finalBytes);
            await ctx.Response.BodyWriter.FlushAsync();
        });

        // GET /api/tags
        app.MapGet("/api/tags", async ctx =>
        {
            var res = new
            {
                models = new[]
                {
                    new
                    {
                        name = defaultModelName,
                        model = defaultModelName,
                        modified_at = DateTimeOffset.UtcNow.ToString("o"),
                        size = 4680000000L
                    }
                }
            };
            await ctx.Response.WriteJsonAsync(res);
        });
    }

    private static InferenceRequest? ParseRequest(HttpRequest request)
    {
        if (request.Body.IsEmpty) return null;
        try
        {
            return JsonSerializer.Deserialize<InferenceRequest>(request.Body.Span, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });
        }
        catch
        {
            return null;
        }
    }
}

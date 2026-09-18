using System;
using System.Buffers;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Glacier.Serve.Core;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Serialization;
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
                var chunkDto = new OllamaChatChunk
                {
                    Model = defaultModelName,
                    Message = new ChatMessage { Role = "assistant", Content = token },
                    Done = false
                };

                byte[] bytes = JsonSerializer.SerializeToUtf8Bytes(chunkDto, ServeJsonContext.Default.OllamaChatChunk);
                await ctx.Response.BodyWriter.WriteAsync(bytes);
                await ctx.Response.BodyWriter.WriteAsync("\n"u8.ToArray());
                await ctx.Response.BodyWriter.FlushAsync();
            }

            var finalChunkDto = new OllamaChatChunk
            {
                Model = defaultModelName,
                Done = true,
                PromptEvalCount = state.PromptTokens.Length,
                EvalCount = state.GeneratedTokens.Count
            };

            byte[] finalBytes = JsonSerializer.SerializeToUtf8Bytes(finalChunkDto, ServeJsonContext.Default.OllamaChatChunk);
            await ctx.Response.BodyWriter.WriteAsync(finalBytes);
            await ctx.Response.BodyWriter.WriteAsync("\n"u8.ToArray());
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
                var chunkDto = new OllamaGenerateChunk
                {
                    Model = defaultModelName,
                    Response = token,
                    Done = false
                };

                byte[] bytes = JsonSerializer.SerializeToUtf8Bytes(chunkDto, ServeJsonContext.Default.OllamaGenerateChunk);
                await ctx.Response.BodyWriter.WriteAsync(bytes);
                await ctx.Response.BodyWriter.WriteAsync("\n"u8.ToArray());
                await ctx.Response.BodyWriter.FlushAsync();
            }

            var finalChunkDto = new OllamaGenerateChunk
            {
                Model = defaultModelName,
                Done = true,
                PromptEvalCount = state.PromptTokens.Length,
                EvalCount = state.GeneratedTokens.Count
            };

            byte[] finalBytes = JsonSerializer.SerializeToUtf8Bytes(finalChunkDto, ServeJsonContext.Default.OllamaGenerateChunk);
            await ctx.Response.BodyWriter.WriteAsync(finalBytes);
            await ctx.Response.BodyWriter.WriteAsync("\n"u8.ToArray());
            await ctx.Response.BodyWriter.FlushAsync();
        });

        // GET /api/tags
        app.MapGet("/api/tags", async ctx =>
        {
            var res = new OllamaTagsResponse
            {
                Models =
                [
                    new OllamaModelItem
                    {
                        Name = defaultModelName,
                        Model = defaultModelName,
                        ModifiedAt = DateTimeOffset.UtcNow.ToString("o"),
                        Size = 4680000000L
                    }
                ]
            };
            await ctx.Response.WriteJsonAsync(res, ServeJsonContext.Default.OllamaTagsResponse);
        });
    }

    private static InferenceRequest? ParseRequest(HttpRequest request)
    {
        if (request.Body.IsEmpty) return null;
        try
        {
            return JsonSerializer.Deserialize(request.Body.Span, ServeJsonContext.Default.InferenceRequest);
        }
        catch
        {
            return null;
        }
    }
}

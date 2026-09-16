using System;
using System.Buffers;
using System.IO.Pipelines;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Glacier.Serve.Core;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Server;

namespace Glacier.Serve.Inference.Http;

public static class OpenAiEndpoints
{
    public static void MapOpenAiEndpoints(this GlacierServeApp app, ContinuousBatchEngine engine, string defaultModelName = "glacier-enterprise-7b")
    {
        // POST /v1/chat/completions
        app.MapPost("/v1/chat/completions", async ctx =>
        {
            var req = ParseRequest(ctx.Request);
            if (req == null)
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteUtf8Async("{\"error\":\"Invalid JSON request body\"}");
                return;
            }

            var (state, tokenStream) = await engine.EnqueueAsync(req);

            if (req.Stream)
            {
                ctx.Response.ContentType = "text/event-stream";
                ctx.Response.Headers["Cache-Control"] = "no-cache";
                ctx.Response.Headers["Connection"] = "close";
                await ctx.Response.EnsureHeadersSentAsync(null);

                await foreach (var token in tokenStream)
                {
                    string chunkJson = JsonSerializer.Serialize(new
                    {
                        id = state.RequestId,
                        @object = "chat.completion.chunk",
                        created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                        model = string.IsNullOrEmpty(req.Model) ? defaultModelName : req.Model,
                        choices = new[]
                        {
                            new
                            {
                                index = 0,
                                delta = new { content = token },
                                finish_reason = (string?)null
                            }
                        }
                    });

                    byte[] chunkBytes = Encoding.UTF8.GetBytes($"data: {chunkJson}\n\n");
                    await ctx.Response.BodyWriter.WriteAsync(chunkBytes);
                    await ctx.Response.BodyWriter.FlushAsync();
                }

                byte[] doneBytes = Encoding.UTF8.GetBytes("data: [DONE]\n\n");
                await ctx.Response.BodyWriter.WriteAsync(doneBytes);
                await ctx.Response.BodyWriter.FlushAsync();
            }
            else
            {
                string fullResponse = await state.Completion.Task;
                var res = new
                {
                    id = state.RequestId,
                    @object = "chat.completion",
                    created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                    model = string.IsNullOrEmpty(req.Model) ? defaultModelName : req.Model,
                    choices = new[]
                    {
                        new
                        {
                            index = 0,
                            message = new { role = "assistant", content = fullResponse },
                            finish_reason = state.FinishReason
                        }
                    },
                    usage = new
                    {
                        prompt_tokens = state.PromptTokens.Length,
                        completion_tokens = state.GeneratedTokens.Count,
                        total_tokens = state.PromptTokens.Length + state.GeneratedTokens.Count
                    }
                };

                await ctx.Response.WriteJsonAsync(res);
            }
        });

        // GET /v1/models
        app.MapGet("/v1/models", async ctx =>
        {
            var res = new
            {
                @object = "list",
                data = new[]
                {
                    new
                    {
                        id = defaultModelName,
                        @object = "model",
                        created = 1710000000,
                        owned_by = "glacier"
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

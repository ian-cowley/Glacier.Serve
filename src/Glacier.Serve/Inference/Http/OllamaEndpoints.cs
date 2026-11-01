using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Serve.Core;
using Glacier.Serve.Diagnostics;
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
                await WriteBadRequestAsync(ctx.Response, "Invalid JSON request.");
                return;
            }

            // Ollama defaults stream to true when omitted, and returns a single JSON object when stream: false
            bool stream = req.IsStreamExplicitlySet ? req.Stream : true;
            string modelName = string.IsNullOrEmpty(req.Model) ? defaultModelName : req.Model;

            using var requestCts = CancellationTokenSource.CreateLinkedTokenSource(ctx.RequestAborted);
            using var disconnectTimer = OpenAiEndpoints.StartDisconnectMonitor(ctx, requestCts);

            try
            {
                (SequenceState state, IAsyncEnumerable<string> tokenStream) enqueued;
                try
                {
                    enqueued = await engine.EnqueueAsync(req, requestCts.Token);
                }
                catch (ArgumentOutOfRangeException ex)
                {
                    await WriteBadRequestAsync(ctx.Response, ex.Message);
                    return;
                }

                var (state, tokenStream) = enqueued;

                if (!stream)
                {
                    string fullText = await state.Completion.Task.WaitAsync(requestCts.Token);
                    var singleRes = new OllamaChatChunk
                    {
                        Model = modelName,
                        Message = new ChatMessage { Role = "assistant", Content = fullText },
                        Done = true,
                        DoneReason = string.IsNullOrEmpty(state.FinishReason) ? "stop" : state.FinishReason,
                        PromptEvalCount = state.PromptTokens.Length,
                        EvalCount = state.GeneratedTokens.Count
                    };
                    await ctx.Response.WriteJsonAsync(singleRes, ServeJsonContext.Default.OllamaChatChunk, requestCts.Token);
                    return;
                }

                ctx.Response.ContentType = "application/x-ndjson; charset=utf-8";
                ctx.Response.Headers["Connection"] = "close";
                await ctx.Response.EnsureHeadersSentAsync(null, requestCts.Token);

                await foreach (var token in tokenStream.WithCancellation(requestCts.Token))
                {
                    if (ctx.IsClientDisconnected)
                    {
                        requestCts.Cancel();
                        return;
                    }

                    var chunkDto = new OllamaChatChunk
                    {
                        Model = modelName,
                        Message = new ChatMessage { Role = "assistant", Content = token },
                        Done = false
                    };

                    await ctx.Response.WriteNdjsonLineAsync(chunkDto, ServeJsonContext.Default.OllamaChatChunk, requestCts.Token);
                }

                var finalChunkDto = new OllamaChatChunk
                {
                    Model = modelName,
                    Message = new ChatMessage { Role = "assistant", Content = string.Empty },
                    Done = true,
                    DoneReason = string.IsNullOrEmpty(state.FinishReason) ? "stop" : state.FinishReason,
                    PromptEvalCount = state.PromptTokens.Length,
                    EvalCount = state.GeneratedTokens.Count
                };

                await ctx.Response.WriteNdjsonLineAsync(finalChunkDto, ServeJsonContext.Default.OllamaChatChunk, requestCts.Token);
            }
            catch (OperationCanceledException)
            {
                requestCts.Cancel();
            }
            finally
            {
                requestCts.Cancel();
            }
        });

        // POST /api/generate
        app.MapPost("/api/generate", async ctx =>
        {
            var req = ParseRequest(ctx.Request);
            if (req == null)
            {
                await WriteBadRequestAsync(ctx.Response, "Invalid JSON request.");
                return;
            }

            bool stream = req.IsStreamExplicitlySet ? req.Stream : true;
            string modelName = string.IsNullOrEmpty(req.Model) ? defaultModelName : req.Model;

            using var requestCts = CancellationTokenSource.CreateLinkedTokenSource(ctx.RequestAborted);
            using var disconnectTimer = OpenAiEndpoints.StartDisconnectMonitor(ctx, requestCts);

            try
            {
                (SequenceState state, IAsyncEnumerable<string> tokenStream) enqueued;
                try
                {
                    enqueued = await engine.EnqueueAsync(req, requestCts.Token);
                }
                catch (ArgumentOutOfRangeException ex)
                {
                    await WriteBadRequestAsync(ctx.Response, ex.Message);
                    return;
                }

                var (state, tokenStream) = enqueued;

                if (!stream)
                {
                    string fullText = await state.Completion.Task.WaitAsync(requestCts.Token);
                    var singleRes = new OllamaGenerateChunk
                    {
                        Model = modelName,
                        Response = fullText,
                        Done = true,
                        DoneReason = string.IsNullOrEmpty(state.FinishReason) ? "stop" : state.FinishReason,
                        PromptEvalCount = state.PromptTokens.Length,
                        EvalCount = state.GeneratedTokens.Count
                    };
                    await ctx.Response.WriteJsonAsync(singleRes, ServeJsonContext.Default.OllamaGenerateChunk, requestCts.Token);
                    return;
                }

                ctx.Response.ContentType = "application/x-ndjson; charset=utf-8";
                ctx.Response.Headers["Connection"] = "close";
                await ctx.Response.EnsureHeadersSentAsync(null, requestCts.Token);

                await foreach (var token in tokenStream.WithCancellation(requestCts.Token))
                {
                    if (ctx.IsClientDisconnected)
                    {
                        requestCts.Cancel();
                        return;
                    }

                    var chunkDto = new OllamaGenerateChunk
                    {
                        Model = modelName,
                        Response = token,
                        Done = false
                    };

                    await ctx.Response.WriteNdjsonLineAsync(chunkDto, ServeJsonContext.Default.OllamaGenerateChunk, requestCts.Token);
                }

                var finalChunkDto = new OllamaGenerateChunk
                {
                    Model = modelName,
                    Response = string.Empty,
                    Done = true,
                    DoneReason = string.IsNullOrEmpty(state.FinishReason) ? "stop" : state.FinishReason,
                    PromptEvalCount = state.PromptTokens.Length,
                    EvalCount = state.GeneratedTokens.Count
                };

                await ctx.Response.WriteNdjsonLineAsync(finalChunkDto, ServeJsonContext.Default.OllamaGenerateChunk, requestCts.Token);
            }
            catch (OperationCanceledException)
            {
                requestCts.Cancel();
            }
            finally
            {
                requestCts.Cancel();
            }
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
            await ctx.Response.WriteJsonAsync(res, ServeJsonContext.Default.OllamaTagsResponse, ctx.RequestAborted);
        });
    }

    private static ValueTask WriteBadRequestAsync(HttpResponse response, string message)
    {
        response.StatusCode = 400;
        var err = new ApiErrorResponse
        {
            Error = new ApiErrorDetail
            {
                Message = message,
                Type = "invalid_request_error"
            }
        };
        return response.WriteJsonAsync(err, ServeJsonContext.Default.ApiErrorResponse);
    }

    private static InferenceRequest? ParseRequest(HttpRequest request)
    {
        if (request.Body.IsEmpty) return null;
        try
        {
            return JsonSerializer.Deserialize(request.Body.Span, ServeJsonContext.Default.InferenceRequest);
        }
        catch (Exception ex)
        {
            GlacierDiagnostics.LogDebug($"[Glacier.Serve] Failed to deserialize Ollama request body: {ex.Message}");
            return null;
        }
    }
}

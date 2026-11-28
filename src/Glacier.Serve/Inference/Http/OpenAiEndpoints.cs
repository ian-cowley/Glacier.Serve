using System;
using System.Buffers;
using System.Buffers.Text;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Serve.Core;
using Glacier.Serve.Diagnostics;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Serialization;
using Glacier.Serve.Server;

namespace Glacier.Serve.Inference.Http;

public static class OpenAiEndpoints
{
    private static readonly ReadOnlyMemory<byte> s_sseDoneBytes = "data: [DONE]\n\n"u8.ToArray();

    public static void MapOpenAiEndpoints(this GlacierServeApp app, ContinuousBatchEngine engine, string defaultModelName = "glacier-enterprise-7b")
    {
        // POST /v1/chat/completions
        app.MapPost("/v1/chat/completions", async ctx =>
        {
            var req = ParseRequest(ctx.Request);
            if (req == null)
            {
                await WriteBadRequestAsync(ctx.Response, "Invalid JSON request body.");
                return;
            }

            using var requestCts = CancellationTokenSource.CreateLinkedTokenSource(ctx.RequestAborted);

            try
            {
                (SequenceState state, IAsyncEnumerable<string> tokenStream) enqueued;
                try
                {
                    enqueued = await engine.EnqueueAsync(req, defaultModelName, requestCts.Token);
                }
                catch (InferenceOverloadedException ex)
                {
                    await WriteOverloadedAsync(ctx.Response, ex);
                    return;
                }
                catch (ArgumentOutOfRangeException ex)
                {
                    await WriteBadRequestAsync(ctx.Response, ex.Message);
                    return;
                }
                catch (ArgumentException ex)
                {
                    await WriteBadRequestAsync(ctx.Response, ex.Message);
                    return;
                }
                catch (KeyNotFoundException ex)
                {
                    await WriteNotFoundAsync(ctx.Response, ex.Message);
                    return;
                }

                var (state, tokenStream) = enqueued;
                string modelName = !string.IsNullOrEmpty(req.Model)
                    ? req.Model
                    : (!string.IsNullOrEmpty(state.LoraAdapterName) ? $"{defaultModelName}:{state.LoraAdapterName}" : defaultModelName);

                if (req.Stream)
                {
                    ctx.Response.ContentType = "text/event-stream";
                    ctx.Response.Headers["Cache-Control"] = "no-cache";
                    ctx.Response.UseChunkedTransfer = true;
                    await ctx.Response.EnsureHeadersSentAsync(null, requestCts.Token);

                    long nowSec = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
                    var sseBuf = HttpResponse.RentJsonBuffer();
                    try
                    {
                        await foreach (var token in tokenStream.WithCancellation(requestCts.Token))
                        {
                            if (ctx.IsClientDisconnected)
                            {
                                requestCts.Cancel();
                                return;
                            }

                            sseBuf.Clear();
                            WriteChatSseChunk(sseBuf, state.RequestId, nowSec, modelName, token);
                            if (!await ctx.Response.WriteChunkAsync(sseBuf.WrittenMemory, requestCts.Token))
                            {
                                requestCts.Cancel();
                                return;
                            }
                        }

                        string finishReason = SanitizeFinishReason(state.FinishReason);
                        sseBuf.Clear();
                        WriteChatSseFinishChunk(sseBuf, state.RequestId, nowSec, modelName, finishReason);
                        if (!await ctx.Response.WriteChunkAsync(sseBuf.WrittenMemory, requestCts.Token))
                        {
                            requestCts.Cancel();
                            return;
                        }

                        if (req.StreamOptions?.IncludeUsage == true)
                        {
                            int promptTokens = state.PromptTokens.Length;
                            int completionTokens = state.GeneratedTokens.Count;
                            sseBuf.Clear();
                            WriteChatSseUsageChunk(
                                sseBuf,
                                state.RequestId,
                                nowSec,
                                modelName,
                                promptTokens,
                                completionTokens,
                                promptTokens + completionTokens);
                            if (!await ctx.Response.WriteChunkAsync(sseBuf.WrittenMemory, requestCts.Token))
                            {
                                requestCts.Cancel();
                                return;
                            }
                        }

                        if (!await ctx.Response.WriteChunkAsync(s_sseDoneBytes, requestCts.Token))
                        {
                            requestCts.Cancel();
                            return;
                        }
                        await ctx.Response.CompleteChunkedAsync(requestCts.Token);
                    }
                    finally
                    {
                        HttpResponse.ReturnJsonBuffer(sseBuf);
                    }
                }
                else
                {
                    string fullResponse = await state.Completion.Task.WaitAsync(requestCts.Token);
                    var res = new ChatCompletionResponse
                    {
                        Id = state.RequestId,
                        Object = "chat.completion",
                        Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                        Model = modelName,
                        Choices =
                        [
                            new ChatChoice
                            {
                                Index = 0,
                                Message = new ChatMessage { Role = "assistant", Content = fullResponse },
                                FinishReason = SanitizeFinishReason(state.FinishReason)
                            }
                        ],
                        Usage = new ChatUsage
                        {
                            PromptTokens = state.PromptTokens.Length,
                            CompletionTokens = state.GeneratedTokens.Count,
                            TotalTokens = state.PromptTokens.Length + state.GeneratedTokens.Count
                        }
                    };

                    await ctx.Response.WriteJsonAsync(res, ServeJsonContext.Default.ChatCompletionResponse, requestCts.Token);
                }
            }
            catch (Exception ex) when (ex is OperationCanceledException or IOException or SocketException)
            {
                requestCts.Cancel();
            }
            finally
            {
                requestCts.Cancel();
            }
        });

        // POST /v1/completions (Legacy/Text OpenAI Completions Contract)
        app.MapPost("/v1/completions", async ctx =>
        {
            var req = ParseRequest(ctx.Request);
            if (req == null)
            {
                await WriteBadRequestAsync(ctx.Response, "Invalid JSON request body.");
                return;
            }

            using var requestCts = CancellationTokenSource.CreateLinkedTokenSource(ctx.RequestAborted);

            try
            {
                (SequenceState state, IAsyncEnumerable<string> tokenStream) enqueued;
                try
                {
                    enqueued = await engine.EnqueueAsync(req, defaultModelName, requestCts.Token);
                }
                catch (InferenceOverloadedException ex)
                {
                    await WriteOverloadedAsync(ctx.Response, ex);
                    return;
                }
                catch (ArgumentOutOfRangeException ex)
                {
                    await WriteBadRequestAsync(ctx.Response, ex.Message);
                    return;
                }
                catch (ArgumentException ex)
                {
                    await WriteBadRequestAsync(ctx.Response, ex.Message);
                    return;
                }
                catch (KeyNotFoundException ex)
                {
                    await WriteNotFoundAsync(ctx.Response, ex.Message);
                    return;
                }

                var (state, tokenStream) = enqueued;
                string modelName = !string.IsNullOrEmpty(req.Model)
                    ? req.Model
                    : (!string.IsNullOrEmpty(state.LoraAdapterName) ? $"{defaultModelName}:{state.LoraAdapterName}" : defaultModelName);

                if (req.Stream)
                {
                    ctx.Response.ContentType = "text/event-stream";
                    ctx.Response.Headers["Cache-Control"] = "no-cache";
                    ctx.Response.UseChunkedTransfer = true;
                    await ctx.Response.EnsureHeadersSentAsync(null, requestCts.Token);

                    long nowSec = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
                    var sseBuf = HttpResponse.RentJsonBuffer();
                    try
                    {
                        await foreach (var token in tokenStream.WithCancellation(requestCts.Token))
                        {
                            if (ctx.IsClientDisconnected)
                            {
                                requestCts.Cancel();
                                return;
                            }

                            sseBuf.Clear();
                            WriteTextCompletionSseChunk(sseBuf, state.RequestId, nowSec, modelName, token);
                            if (!await ctx.Response.WriteChunkAsync(sseBuf.WrittenMemory, requestCts.Token))
                            {
                                requestCts.Cancel();
                                return;
                            }
                        }

                        string finishReason = SanitizeFinishReason(state.FinishReason);
                        sseBuf.Clear();
                        WriteTextCompletionSseFinishChunk(sseBuf, state.RequestId, nowSec, modelName, finishReason);
                        if (!await ctx.Response.WriteChunkAsync(sseBuf.WrittenMemory, requestCts.Token))
                        {
                            requestCts.Cancel();
                            return;
                        }

                        if (req.StreamOptions?.IncludeUsage == true)
                        {
                            int promptTokens = state.PromptTokens.Length;
                            int completionTokens = state.GeneratedTokens.Count;
                            sseBuf.Clear();
                            WriteTextCompletionSseUsageChunk(
                                sseBuf,
                                state.RequestId,
                                nowSec,
                                modelName,
                                promptTokens,
                                completionTokens,
                                promptTokens + completionTokens);
                            if (!await ctx.Response.WriteChunkAsync(sseBuf.WrittenMemory, requestCts.Token))
                            {
                                requestCts.Cancel();
                                return;
                            }
                        }

                        if (!await ctx.Response.WriteChunkAsync(s_sseDoneBytes, requestCts.Token))
                        {
                            requestCts.Cancel();
                            return;
                        }
                        await ctx.Response.CompleteChunkedAsync(requestCts.Token);
                    }
                    finally
                    {
                        HttpResponse.ReturnJsonBuffer(sseBuf);
                    }
                }
                else
                {
                    string fullResponse = await state.Completion.Task.WaitAsync(requestCts.Token);
                    var res = new CompletionResponse
                    {
                        Id = state.RequestId,
                        Object = "text_completion",
                        Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                        Model = modelName,
                        Choices =
                        [
                            new CompletionChoice
                            {
                                Text = fullResponse,
                                Index = 0,
                                FinishReason = SanitizeFinishReason(state.FinishReason)
                            }
                        ],
                        Usage = new ChatUsage
                        {
                            PromptTokens = state.PromptTokens.Length,
                            CompletionTokens = state.GeneratedTokens.Count,
                            TotalTokens = state.PromptTokens.Length + state.GeneratedTokens.Count
                        }
                    };

                    await ctx.Response.WriteJsonAsync(res, ServeJsonContext.Default.CompletionResponse, requestCts.Token);
                }
            }
            catch (Exception ex) when (ex is OperationCanceledException or IOException or SocketException)
            {
                requestCts.Cancel();
            }
            finally
            {
                requestCts.Cancel();
            }
        });

        // GET /v1/models
        app.MapGet("/v1/models", async ctx =>
        {
            var adapters = engine.GetLoadedLoraAdapters();
            var cards = new List<ModelCard>(1 + adapters.Count)
            {
                new ModelCard
                {
                    Id = defaultModelName,
                    Object = "model",
                    Created = 1710000000,
                    OwnedBy = "glacier"
                }
            };
            for (int i = 0; i < adapters.Count; i++)
            {
                cards.Add(new ModelCard
                {
                    Id = $"{defaultModelName}:{adapters[i]}",
                    Object = "model",
                    Created = 1710000000,
                    OwnedBy = "glacier-lora"
                });
            }

            var res = new ModelListResponse
            {
                Object = "list",
                Data = cards
            };
            await ctx.Response.WriteJsonAsync(res, ServeJsonContext.Default.ModelListResponse, ctx.RequestAborted);
        });
    }

    internal static string SanitizeFinishReason(string? finishReason)
    {
        if (string.IsNullOrEmpty(finishReason))
            return "stop";
        if (string.Equals(finishReason, "pool_oom", StringComparison.OrdinalIgnoreCase))
            return "length";
        if (string.Equals(finishReason, "cancelled", StringComparison.OrdinalIgnoreCase))
            return "stop";
        return finishReason;
    }

    internal static ValueTask WriteOverloadedAsync(HttpResponse response, InferenceOverloadedException ex)
    {
        response.StatusCode = ex.StatusCode;
        response.Headers["Retry-After"] = ex.RetryAfterSeconds.ToString(CultureInfo.InvariantCulture);
        var err = new ApiErrorResponse
        {
            Error = new ApiErrorDetail
            {
                Message = ex.Message,
                Type = ex.StatusCode == 429 ? "rate_limit_exceeded" : "server_overloaded"
            }
        };
        return response.WriteJsonAsync(err, ServeJsonContext.Default.ApiErrorResponse);
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

    private static ValueTask WriteNotFoundAsync(HttpResponse response, string message)
    {
        response.StatusCode = 404;
        var err = new ApiErrorResponse
        {
            Error = new ApiErrorDetail
            {
                Message = message,
                Type = "model_not_found"
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
            GlacierDiagnostics.LogDebug($"[Glacier.Serve] Failed to deserialize OpenAI request body: {ex.Message}");
            return null;
        }
    }

    private static int WriteJsonEscapedUtf8(string s, Span<byte> dest)
    {
        int written = 0;
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            switch (c)
            {
                case '"':
                    dest[written++] = (byte)'\\';
                    dest[written++] = (byte)'"';
                    break;
                case '\\':
                    dest[written++] = (byte)'\\';
                    dest[written++] = (byte)'\\';
                    break;
                case '\n':
                    dest[written++] = (byte)'\\';
                    dest[written++] = (byte)'n';
                    break;
                case '\r':
                    dest[written++] = (byte)'\\';
                    dest[written++] = (byte)'r';
                    break;
                case '\t':
                    dest[written++] = (byte)'\\';
                    dest[written++] = (byte)'t';
                    break;
                default:
                    if (c < 32)
                    {
                        dest[written++] = (byte)'\\';
                        dest[written++] = (byte)'u';
                        dest[written++] = (byte)'0';
                        dest[written++] = (byte)'0';
                        int b0 = (c >> 4) & 0xF;
                        int b1 = c & 0xF;
                        dest[written++] = (byte)(b0 < 10 ? '0' + b0 : 'a' + (b0 - 10));
                        dest[written++] = (byte)(b1 < 10 ? '0' + b1 : 'a' + (b1 - 10));
                    }
                    else if (c <= 0x7F)
                    {
                        dest[written++] = (byte)c;
                    }
                    else
                    {
                        int charCount = (char.IsHighSurrogate(c) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1])) ? 2 : 1;
                        written += Encoding.UTF8.GetBytes(s.AsSpan(i, charCount), dest[written..]);
                        if (charCount == 2) i++;
                    }
                    break;
            }
        }
        return written;
    }

    private static ReadOnlySpan<byte> SsePrefix => "data: {\"id\":\""u8;
    private static ReadOnlySpan<byte> SseChatObject => "\",\"object\":\"chat.completion.chunk\",\"created\":"u8;
    private static ReadOnlySpan<byte> SseTextObject => "\",\"object\":\"text_completion\",\"created\":"u8;
    private static ReadOnlySpan<byte> SseModel => ",\"model\":\""u8;
    private static ReadOnlySpan<byte> SseChatChoices => "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\""u8;
    private static ReadOnlySpan<byte> SseChatSuffix => "\"},\"finish_reason\":null}]}\n\n"u8;
    private static ReadOnlySpan<byte> SseChatFinishChoices => "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\""u8;
    private static ReadOnlySpan<byte> SseTextChoices => "\",\"choices\":[{\"text\":\""u8;
    private static ReadOnlySpan<byte> SseTextSuffix => "\",\"index\":0,\"finish_reason\":null}]}\n\n"u8;
    private static ReadOnlySpan<byte> SseTextFinishChoices => "\",\"choices\":[{\"text\":\"\",\"index\":0,\"finish_reason\":\""u8;
    private static ReadOnlySpan<byte> SseFinishSuffix => "\"}]}\n\n"u8;
    private static ReadOnlySpan<byte> SseUsageChoicesPrefix => "\",\"choices\":[],\"usage\":{\"prompt_tokens\":"u8;
    private static ReadOnlySpan<byte> SseUsageCompletionPrefix => ",\"completion_tokens\":"u8;
    private static ReadOnlySpan<byte> SseUsageTotalPrefix => ",\"total_tokens\":"u8;
    private static ReadOnlySpan<byte> SseUsageSuffix => "}}\n\n"u8;

    private static int GetMaxJsonEscapedByteCount(string s) => checked(s.Length * 6 + 4);

    private static void WriteChatSseChunk(IBufferWriter<byte> writer, string requestId, long nowSec, string modelName, string token)
    {
        int maxBytes = GetMaxJsonEscapedByteCount(token) + GetMaxJsonEscapedByteCount(modelName) + Encoding.UTF8.GetMaxByteCount(requestId.Length) + 512;
        var span = writer.GetSpan(maxBytes);
        int written = 0;

        SsePrefix.CopyTo(span[written..]); written += SsePrefix.Length;
        written += Encoding.UTF8.GetBytes(requestId, span[written..]);
        SseChatObject.CopyTo(span[written..]); written += SseChatObject.Length;
        Utf8Formatter.TryFormat(nowSec, span[written..], out int secLen); written += secLen;
        SseModel.CopyTo(span[written..]); written += SseModel.Length;
        written += WriteJsonEscapedUtf8(modelName, span[written..]);
        SseChatChoices.CopyTo(span[written..]); written += SseChatChoices.Length;
        written += WriteJsonEscapedUtf8(token, span[written..]);
        SseChatSuffix.CopyTo(span[written..]); written += SseChatSuffix.Length;

        writer.Advance(written);
    }

    private static void WriteChatSseFinishChunk(IBufferWriter<byte> writer, string requestId, long nowSec, string modelName, string finishReason)
    {
        int maxBytes = GetMaxJsonEscapedByteCount(modelName) + GetMaxJsonEscapedByteCount(finishReason) + Encoding.UTF8.GetMaxByteCount(requestId.Length) + 512;
        var span = writer.GetSpan(maxBytes);
        int written = 0;

        SsePrefix.CopyTo(span[written..]); written += SsePrefix.Length;
        written += Encoding.UTF8.GetBytes(requestId, span[written..]);
        SseChatObject.CopyTo(span[written..]); written += SseChatObject.Length;
        Utf8Formatter.TryFormat(nowSec, span[written..], out int secLen); written += secLen;
        SseModel.CopyTo(span[written..]); written += SseModel.Length;
        written += WriteJsonEscapedUtf8(modelName, span[written..]);
        SseChatFinishChoices.CopyTo(span[written..]); written += SseChatFinishChoices.Length;
        written += WriteJsonEscapedUtf8(finishReason, span[written..]);
        SseFinishSuffix.CopyTo(span[written..]); written += SseFinishSuffix.Length;

        writer.Advance(written);
    }

    private static void WriteTextCompletionSseChunk(IBufferWriter<byte> writer, string requestId, long nowSec, string modelName, string token)
    {
        int maxBytes = GetMaxJsonEscapedByteCount(token) + GetMaxJsonEscapedByteCount(modelName) + Encoding.UTF8.GetMaxByteCount(requestId.Length) + 512;
        var span = writer.GetSpan(maxBytes);
        int written = 0;

        SsePrefix.CopyTo(span[written..]); written += SsePrefix.Length;
        written += Encoding.UTF8.GetBytes(requestId, span[written..]);
        SseTextObject.CopyTo(span[written..]); written += SseTextObject.Length;
        Utf8Formatter.TryFormat(nowSec, span[written..], out int secLen); written += secLen;
        SseModel.CopyTo(span[written..]); written += SseModel.Length;
        written += WriteJsonEscapedUtf8(modelName, span[written..]);
        SseTextChoices.CopyTo(span[written..]); written += SseTextChoices.Length;
        written += WriteJsonEscapedUtf8(token, span[written..]);
        SseTextSuffix.CopyTo(span[written..]); written += SseTextSuffix.Length;

        writer.Advance(written);
    }

    private static void WriteTextCompletionSseFinishChunk(IBufferWriter<byte> writer, string requestId, long nowSec, string modelName, string finishReason)
    {
        int maxBytes = GetMaxJsonEscapedByteCount(modelName) + GetMaxJsonEscapedByteCount(finishReason) + Encoding.UTF8.GetMaxByteCount(requestId.Length) + 512;
        var span = writer.GetSpan(maxBytes);
        int written = 0;

        SsePrefix.CopyTo(span[written..]); written += SsePrefix.Length;
        written += Encoding.UTF8.GetBytes(requestId, span[written..]);
        SseTextObject.CopyTo(span[written..]); written += SseTextObject.Length;
        Utf8Formatter.TryFormat(nowSec, span[written..], out int secLen); written += secLen;
        SseModel.CopyTo(span[written..]); written += SseModel.Length;
        written += WriteJsonEscapedUtf8(modelName, span[written..]);
        SseTextFinishChoices.CopyTo(span[written..]); written += SseTextFinishChoices.Length;
        written += WriteJsonEscapedUtf8(finishReason, span[written..]);
        SseFinishSuffix.CopyTo(span[written..]); written += SseFinishSuffix.Length;

        writer.Advance(written);
    }

    private static void WriteChatSseUsageChunk(
        IBufferWriter<byte> writer,
        string requestId,
        long nowSec,
        string modelName,
        int promptTokens,
        int completionTokens,
        int totalTokens)
    {
        WriteSseUsageChunk(writer, SseChatObject, requestId, nowSec, modelName, promptTokens, completionTokens, totalTokens);
    }

    private static void WriteTextCompletionSseUsageChunk(
        IBufferWriter<byte> writer,
        string requestId,
        long nowSec,
        string modelName,
        int promptTokens,
        int completionTokens,
        int totalTokens)
    {
        WriteSseUsageChunk(writer, SseTextObject, requestId, nowSec, modelName, promptTokens, completionTokens, totalTokens);
    }

    private static void WriteSseUsageChunk(
        IBufferWriter<byte> writer,
        ReadOnlySpan<byte> objectSpan,
        string requestId,
        long nowSec,
        string modelName,
        int promptTokens,
        int completionTokens,
        int totalTokens)
    {
        int maxBytes = GetMaxJsonEscapedByteCount(modelName) + Encoding.UTF8.GetMaxByteCount(requestId.Length) + 512;
        var span = writer.GetSpan(maxBytes);
        int written = 0;

        SsePrefix.CopyTo(span[written..]); written += SsePrefix.Length;
        written += Encoding.UTF8.GetBytes(requestId, span[written..]);
        objectSpan.CopyTo(span[written..]); written += objectSpan.Length;
        Utf8Formatter.TryFormat(nowSec, span[written..], out int secLen); written += secLen;
        SseModel.CopyTo(span[written..]); written += SseModel.Length;
        written += WriteJsonEscapedUtf8(modelName, span[written..]);
        SseUsageChoicesPrefix.CopyTo(span[written..]); written += SseUsageChoicesPrefix.Length;
        Utf8Formatter.TryFormat(promptTokens, span[written..], out int ptLen); written += ptLen;
        SseUsageCompletionPrefix.CopyTo(span[written..]); written += SseUsageCompletionPrefix.Length;
        Utf8Formatter.TryFormat(completionTokens, span[written..], out int ctLen); written += ctLen;
        SseUsageTotalPrefix.CopyTo(span[written..]); written += SseUsageTotalPrefix.Length;
        Utf8Formatter.TryFormat(totalTokens, span[written..], out int ttLen); written += ttLen;
        SseUsageSuffix.CopyTo(span[written..]); written += SseUsageSuffix.Length;

        writer.Advance(written);
    }
}

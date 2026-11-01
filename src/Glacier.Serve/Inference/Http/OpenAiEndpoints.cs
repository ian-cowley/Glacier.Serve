using System;
using System.Buffers.Text;
using System.Collections.Generic;
using System.IO.Pipelines;
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
            using var disconnectTimer = StartDisconnectMonitor(ctx, requestCts);

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
                string modelName = string.IsNullOrEmpty(req.Model) ? defaultModelName : req.Model;

                if (req.Stream)
                {
                    ctx.Response.ContentType = "text/event-stream";
                    ctx.Response.Headers["Cache-Control"] = "no-cache";
                    ctx.Response.Headers["Connection"] = "close";
                    await ctx.Response.EnsureHeadersSentAsync(null, requestCts.Token);

                    long nowSec = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

                    await foreach (var token in tokenStream.WithCancellation(requestCts.Token))
                    {
                        if (ctx.IsClientDisconnected)
                        {
                            requestCts.Cancel();
                            return;
                        }

                        WriteChatSseChunk(ctx.Response.BodyWriter, state.RequestId, nowSec, modelName, token);
                        var flushRes = await ctx.Response.BodyWriter.FlushAsync(requestCts.Token);
                        if (flushRes.IsCanceled || flushRes.IsCompleted)
                        {
                            requestCts.Cancel();
                            return;
                        }
                    }

                    string finishReason = string.IsNullOrEmpty(state.FinishReason) ? "stop" : state.FinishReason;
                    WriteChatSseFinishChunk(ctx.Response.BodyWriter, state.RequestId, nowSec, modelName, finishReason);
                    WriteSseDone(ctx.Response.BodyWriter);
                    await ctx.Response.BodyWriter.FlushAsync(requestCts.Token);
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
                                FinishReason = string.IsNullOrEmpty(state.FinishReason) ? "stop" : state.FinishReason
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
            catch (OperationCanceledException)
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
            using var disconnectTimer = StartDisconnectMonitor(ctx, requestCts);

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
                string modelName = string.IsNullOrEmpty(req.Model) ? defaultModelName : req.Model;

                if (req.Stream)
                {
                    ctx.Response.ContentType = "text/event-stream";
                    ctx.Response.Headers["Cache-Control"] = "no-cache";
                    ctx.Response.Headers["Connection"] = "close";
                    await ctx.Response.EnsureHeadersSentAsync(null, requestCts.Token);

                    long nowSec = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

                    await foreach (var token in tokenStream.WithCancellation(requestCts.Token))
                    {
                        if (ctx.IsClientDisconnected)
                        {
                            requestCts.Cancel();
                            return;
                        }

                        WriteTextCompletionSseChunk(ctx.Response.BodyWriter, state.RequestId, nowSec, modelName, token);
                        var flushRes = await ctx.Response.BodyWriter.FlushAsync(requestCts.Token);
                        if (flushRes.IsCanceled || flushRes.IsCompleted)
                        {
                            requestCts.Cancel();
                            return;
                        }
                    }

                    string finishReason = string.IsNullOrEmpty(state.FinishReason) ? "stop" : state.FinishReason;
                    WriteTextCompletionSseFinishChunk(ctx.Response.BodyWriter, state.RequestId, nowSec, modelName, finishReason);
                    WriteSseDone(ctx.Response.BodyWriter);
                    await ctx.Response.BodyWriter.FlushAsync(requestCts.Token);
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
                                FinishReason = string.IsNullOrEmpty(state.FinishReason) ? "stop" : state.FinishReason
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
            catch (OperationCanceledException)
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
            var res = new ModelListResponse
            {
                Object = "list",
                Data =
                [
                    new ModelCard
                    {
                        Id = defaultModelName,
                        Object = "model",
                        Created = 1710000000,
                        OwnedBy = "glacier"
                    }
                ]
            };
            await ctx.Response.WriteJsonAsync(res, ServeJsonContext.Default.ModelListResponse, ctx.RequestAborted);
        });
    }

    internal static Timer? StartDisconnectMonitor(HttpContext ctx, CancellationTokenSource requestCts)
    {
        if (ctx.ClientDisconnectProbe == null) return null;
        return new Timer(_ =>
        {
            if (!requestCts.IsCancellationRequested && ctx.IsClientDisconnected)
            {
                try { requestCts.Cancel(); } catch { }
            }
        }, null, dueTime: 10, period: 10);
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
    private static ReadOnlySpan<byte> SseDone => "data: [DONE]\n\n"u8;

    private static int GetMaxJsonEscapedByteCount(string s) => checked(s.Length * 6 + 4);

    private static void WriteChatSseChunk(PipeWriter writer, string requestId, long nowSec, string modelName, string token)
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

    private static void WriteChatSseFinishChunk(PipeWriter writer, string requestId, long nowSec, string modelName, string finishReason)
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

    private static void WriteTextCompletionSseChunk(PipeWriter writer, string requestId, long nowSec, string modelName, string token)
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

    private static void WriteTextCompletionSseFinishChunk(PipeWriter writer, string requestId, long nowSec, string modelName, string finishReason)
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

    private static void WriteSseDone(PipeWriter writer)
    {
        var doneSpan = writer.GetSpan(SseDone.Length);
        SseDone.CopyTo(doneSpan);
        writer.Advance(SseDone.Length);
    }
}

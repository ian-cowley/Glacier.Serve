using System;
using System.Buffers;
using System.Buffers.Text;
using System.IO.Pipelines;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Glacier.Serve.Core;
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

                string modelName = string.IsNullOrEmpty(req.Model) ? defaultModelName : req.Model;
                long nowSec = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

                await foreach (var token in tokenStream)
                {
                    WriteSseChunk(ctx.Response.BodyWriter, state.RequestId, nowSec, modelName, token);
                    await ctx.Response.BodyWriter.FlushAsync();
                }

                WriteSseDone(ctx.Response.BodyWriter);
                await ctx.Response.BodyWriter.FlushAsync();
            }
            else
            {
                string fullResponse = await state.Completion.Task;
                var res = new ChatCompletionResponse
                {
                    Id = state.RequestId,
                    Object = "chat.completion",
                    Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                    Model = string.IsNullOrEmpty(req.Model) ? defaultModelName : req.Model,
                    Choices =
                    [
                        new ChatChoice
                        {
                            Index = 0,
                            Message = new ChatMessage { Role = "assistant", Content = fullResponse },
                            FinishReason = state.FinishReason
                        }
                    ],
                    Usage = new ChatUsage
                    {
                        PromptTokens = state.PromptTokens.Length,
                        CompletionTokens = state.GeneratedTokens.Count,
                        TotalTokens = state.PromptTokens.Length + state.GeneratedTokens.Count
                    }
                };

                await ctx.Response.WriteJsonAsync(res, ServeJsonContext.Default.ChatCompletionResponse);
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
            await ctx.Response.WriteJsonAsync(res, ServeJsonContext.Default.ModelListResponse);
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
                    else
                    {
                        if (c <= 0x7F)
                        {
                            dest[written++] = (byte)c;
                        }
                        else
                        {
                            int charCount = (char.IsHighSurrogate(c) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1])) ? 2 : 1;
                            written += Encoding.UTF8.GetBytes(s.AsSpan(i, charCount), dest[written..]);
                            if (charCount == 2) i++;
                        }
                    }
                    break;
            }
        }
        return written;
    }

    private static ReadOnlySpan<byte> SsePrefix => "data: {\"id\":\""u8;
    private static ReadOnlySpan<byte> SseObject => "\",\"object\":\"chat.completion.chunk\",\"created\":"u8;
    private static ReadOnlySpan<byte> SseModel => ",\"model\":\""u8;
    private static ReadOnlySpan<byte> SseChoices => "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\""u8;
    private static ReadOnlySpan<byte> SseSuffix => "\"},\"finish_reason\":null}]}\n\n"u8;
    private static ReadOnlySpan<byte> SseDone => "data: [DONE]\n\n"u8;

    private static void WriteSseChunk(PipeWriter writer, string requestId, long nowSec, string modelName, string token)
    {
        int maxTokenBytes = Encoding.UTF8.GetMaxByteCount(token.Length) * 2 + 512;
        var span = writer.GetSpan(maxTokenBytes);
        int written = 0;

        SsePrefix.CopyTo(span[written..]); written += SsePrefix.Length;
        written += Encoding.UTF8.GetBytes(requestId, span[written..]);
        SseObject.CopyTo(span[written..]); written += SseObject.Length;
        Utf8Formatter.TryFormat(nowSec, span[written..], out int secLen); written += secLen;
        SseModel.CopyTo(span[written..]); written += SseModel.Length;
        written += Encoding.UTF8.GetBytes(modelName, span[written..]);
        SseChoices.CopyTo(span[written..]); written += SseChoices.Length;
        written += WriteJsonEscapedUtf8(token, span[written..]);
        SseSuffix.CopyTo(span[written..]); written += SseSuffix.Length;

        writer.Advance(written);
    }

    private static void WriteSseDone(PipeWriter writer)
    {
        var doneSpan = writer.GetSpan(SseDone.Length);
        SseDone.CopyTo(doneSpan);
        writer.Advance(SseDone.Length);
    }
}

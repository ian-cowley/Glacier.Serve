namespace Glacier.Serve.Inference.Http;

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Serve.Core;
using Glacier.Serve.Diagnostics;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Inference.Bounce;
using Glacier.Serve.Serialization;
using Glacier.Serve.Server;

public sealed record PeerBounceEndpointOptions(
    string ModelAlias = "Glacier-3Machine-Bounce",
    bool MapOpenAiAndHealthRoutes = true,
    string BounceChatPath = "/api/bounce/chat/completions");

/// <summary>
/// First-class HTTP endpoints for <see cref="PeerBounceService"/>
/// (<c>GET /api/bounce/status</c>, <c>POST /api/bounce/chat/completions</c>,
/// and optional standalone OpenAI/health routes).
/// </summary>
public static class PeerBounceEndpoints
{
    private static readonly ReadOnlyMemory<byte> s_sseDoneBytes = "data: [DONE]\n\n"u8.ToArray();

    public static void MapPeerBounce(
        this GlacierServeApp app,
        PeerBounceService bounceService,
        bool mapOpenAiAndHealthRoutes = true,
        string modelAlias = "Glacier-3Machine-Bounce")
    {
        MapPeerBounce(
            app,
            bounceService,
            new PeerBounceEndpointOptions(
                ModelAlias: string.IsNullOrWhiteSpace(modelAlias) ? "Glacier-3Machine-Bounce" : modelAlias,
                MapOpenAiAndHealthRoutes: mapOpenAiAndHealthRoutes));
    }

    public static void MapPeerBounce(
        this GlacierServeApp app,
        PeerBounceService bounceService,
        PeerBounceEndpointOptions? options)
    {
        ArgumentNullException.ThrowIfNull(app);
        ArgumentNullException.ThrowIfNull(bounceService);

        options ??= new PeerBounceEndpointOptions();
        string modelAlias = string.IsNullOrWhiteSpace(options.ModelAlias) ? "Glacier-3Machine-Bounce" : options.ModelAlias;
        string bounceChatPath = string.IsNullOrWhiteSpace(options.BounceChatPath) ? "/api/bounce/chat/completions" : options.BounceChatPath;

        app.MapGet("/api/bounce/status", async ctx =>
        {
            await ctx.Response.WriteJsonAsync(
                bounceService.GetStatus(),
                ServeJsonContext.Default.BounceStatusResponse,
                ctx.RequestAborted).ConfigureAwait(false);
        });

        app.MapPost(bounceChatPath, ctx => HandleBounceChatAsync(ctx, bounceService, modelAlias));

        if (options.MapOpenAiAndHealthRoutes)
        {
            app.MapGet("/health", async ctx =>
            {
                ctx.Response.ContentType = "application/json; charset=utf-8";
                await ctx.Response.WriteUtf8Async("{\"status\":\"ok\"}", ctx.RequestAborted).ConfigureAwait(false);
            });

            app.MapGet("/v1/models", async ctx =>
            {
                var resp = new ModelListResponse
                {
                    Object = "list",
                    Data =
                    [
                        new ModelCard
                        {
                            Id = modelAlias,
                            Object = "model",
                            Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                            OwnedBy = "glacier"
                        }
                    ]
                };
                await ctx.Response.WriteJsonAsync(
                    resp,
                    ServeJsonContext.Default.ModelListResponse,
                    ctx.RequestAborted).ConfigureAwait(false);
            });

            app.MapPost("/v1/chat/completions", ctx => HandleBounceChatAsync(ctx, bounceService, modelAlias));
        }
    }

    private static async ValueTask HandleBounceChatAsync(HttpContext ctx, PeerBounceService bounceService, string modelAlias)
    {
        var req = ParseBounceRequest(ctx.Request);
        if (req == null || req.Messages == null || req.Messages.Count == 0)
        {
            await WriteBadRequestAsync(ctx.Response, "Invalid JSON request body or missing non-empty 'messages' array.").ConfigureAwait(false);
            return;
        }

        string requestedModel = string.IsNullOrWhiteSpace(req.Model) ? modelAlias : req.Model;
        int maxTokens = req.MaxTokens > 0 ? req.MaxTokens : 1024;
        BounceCompletionResult result;
        try
        {
            result = await bounceService.CompleteAsync(
                req.Messages,
                requestedModel,
                req.Temperature,
                maxTokens,
                ctx.RequestAborted).ConfigureAwait(false);
        }
        catch (HttpRequestException ex)
        {
            GlacierDiagnostics.LogWarning($"[Glacier.Serve] PeerBounce upstream HTTP failure: {ex.Message}");
            if (!ctx.Response.HeadersSent)
            {
                await WriteErrorAsync(
                    ctx.Response,
                    502,
                    "bad_gateway",
                    $"Upstream primary endpoint failed: {ex.Message}",
                    ctx.RequestAborted).ConfigureAwait(false);
            }
            return;
        }
        catch (JsonException ex)
        {
            GlacierDiagnostics.LogWarning($"[Glacier.Serve] PeerBounce upstream returned invalid JSON: {ex.Message}");
            if (!ctx.Response.HeadersSent)
            {
                await WriteErrorAsync(
                    ctx.Response,
                    502,
                    "bad_gateway",
                    $"Upstream primary endpoint returned invalid JSON: {ex.Message}",
                    ctx.RequestAborted).ConfigureAwait(false);
            }
            return;
        }
        catch (OperationCanceledException ex) when (!ctx.RequestAborted.IsCancellationRequested)
        {
            GlacierDiagnostics.LogWarning($"[Glacier.Serve] PeerBounce upstream request timed out: {ex.Message}");
            if (!ctx.Response.HeadersSent)
            {
                await WriteErrorAsync(
                    ctx.Response,
                    504,
                    "gateway_timeout",
                    $"Upstream primary endpoint timed out: {ex.Message}",
                    ctx.RequestAborted).ConfigureAwait(false);
            }
            return;
        }
        catch (Win32Exception ex)
        {
            GlacierDiagnostics.LogWarning($"[Glacier.Serve] PeerBounce verifier process launch failure: {ex.Message}");
            if (!ctx.Response.HeadersSent)
            {
                await WriteErrorAsync(
                    ctx.Response,
                    502,
                    "verifier_unavailable",
                    $"PeerBounce verifier failed to launch: {ex.Message}",
                    ctx.RequestAborted).ConfigureAwait(false);
            }
            return;
        }

        long now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        string id = $"chatcmpl-bounce-{Guid.NewGuid():N}";

        if (req.Stream)
        {
            ctx.Response.ContentType = "text/event-stream";
            ctx.Response.Headers["Cache-Control"] = "no-cache";
            ctx.Response.UseChunkedTransfer = true;
            await ctx.Response.EnsureHeadersSentAsync(null, ctx.RequestAborted).ConfigureAwait(false);

            var chunk = new ChatCompletionChunkResponse
            {
                Id = id,
                Created = now,
                Model = modelAlias,
                Choices =
                [
                    new ChatChunkChoice
                    {
                        Index = 0,
                        Delta = new ChatChunkDelta { Role = "assistant", Content = result.Content },
                        FinishReason = result.FinishReason
                    }
                ]
            };

            var sseBuf = HttpResponse.RentJsonBuffer();
            try
            {
                Span<byte> prefix = sseBuf.GetSpan(6);
                "data: "u8.CopyTo(prefix);
                sseBuf.Advance(6);

                using (var jsonWriter = new Utf8JsonWriter(sseBuf))
                {
                    JsonSerializer.Serialize(jsonWriter, chunk, ServeJsonContext.Default.ChatCompletionChunkResponse);
                }

                Span<byte> suffix = sseBuf.GetSpan(2);
                "\n\n"u8.CopyTo(suffix);
                sseBuf.Advance(2);

                await ctx.Response.WriteSseEventAsync(sseBuf.WrittenMemory, ctx.RequestAborted).ConfigureAwait(false);
                await ctx.Response.WriteSseEventAsync(s_sseDoneBytes, ctx.RequestAborted).ConfigureAwait(false);
                await ctx.Response.CompleteChunkedAsync(ctx.RequestAborted).ConfigureAwait(false);
            }
            finally
            {
                HttpResponse.ReturnJsonBuffer(sseBuf);
            }
            return;
        }

        var completion = new ChatCompletionResponse
        {
            Id = id,
            Object = "chat.completion",
            Created = now,
            Model = modelAlias,
            Choices =
            [
                new ChatChoice
                {
                    Index = 0,
                    Message = new ChatMessage { Role = "assistant", Content = result.Content },
                    FinishReason = result.FinishReason
                }
            ],
            Usage = new ChatUsage
            {
                PromptTokens = result.PromptTokens,
                CompletionTokens = result.CompletionTokens,
                TotalTokens = result.PromptTokens + result.CompletionTokens
            }
        };

        await ctx.Response.WriteJsonAsync(
            completion,
            ServeJsonContext.Default.ChatCompletionResponse,
            ctx.RequestAborted).ConfigureAwait(false);
    }

    private static BounceChatRequest? ParseBounceRequest(HttpRequest request)
    {
        if (request.Body.IsEmpty) return null;
        try
        {
            return JsonSerializer.Deserialize(request.Body.Span, ServeJsonContext.Default.BounceChatRequest);
        }
        catch (Exception ex)
        {
            GlacierDiagnostics.LogDebug($"[Glacier.Serve] Failed to deserialize PeerBounce request body: {ex.Message}");
            return null;
        }
    }

    private static ValueTask WriteBadRequestAsync(HttpResponse response, string message)
    {
        return WriteErrorAsync(response, 400, "invalid_request_error", message, CancellationToken.None);
    }

    private static ValueTask WriteErrorAsync(
        HttpResponse response,
        int statusCode,
        string errorType,
        string message,
        CancellationToken ct)
    {
        response.StatusCode = statusCode;
        var err = new ApiErrorResponse
        {
            Error = new ApiErrorDetail
            {
                Message = message,
                Type = errorType
            }
        };
        return response.WriteJsonAsync(err, ServeJsonContext.Default.ApiErrorResponse, ct);
    }
}

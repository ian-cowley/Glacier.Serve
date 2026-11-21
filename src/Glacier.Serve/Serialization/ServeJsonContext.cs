namespace Glacier.Serve.Serialization;

using System.Collections.Generic;
using System.Text.Json.Serialization;
using Glacier.Serve.Inference.Batching;

public sealed class ChatCompletionResponse
{
    [JsonPropertyName("id")] public string Id { get; set; } = string.Empty;
    [JsonPropertyName("object")] public string Object { get; set; } = "chat.completion";
    [JsonPropertyName("created")] public long Created { get; set; }
    [JsonPropertyName("model")] public string Model { get; set; } = string.Empty;
    [JsonPropertyName("choices")] public List<ChatChoice> Choices { get; set; } = [];
    [JsonPropertyName("usage")] public ChatUsage? Usage { get; set; }
}

public sealed class ChatChoice
{
    [JsonPropertyName("index")] public int Index { get; set; }
    [JsonPropertyName("message")] public ChatMessage Message { get; set; } = new();
    [JsonPropertyName("finish_reason")] public string? FinishReason { get; set; }
}

public sealed class CompletionResponse
{
    [JsonPropertyName("id")] public string Id { get; set; } = string.Empty;
    [JsonPropertyName("object")] public string Object { get; set; } = "text_completion";
    [JsonPropertyName("created")] public long Created { get; set; }
    [JsonPropertyName("model")] public string Model { get; set; } = string.Empty;
    [JsonPropertyName("choices")] public List<CompletionChoice> Choices { get; set; } = [];
    [JsonPropertyName("usage")] public ChatUsage? Usage { get; set; }
}

public sealed class CompletionChoice
{
    [JsonPropertyName("text")] public string Text { get; set; } = string.Empty;
    [JsonPropertyName("index")] public int Index { get; set; }
    [JsonPropertyName("finish_reason")] public string? FinishReason { get; set; }
}

public sealed class ChatUsage
{
    [JsonPropertyName("prompt_tokens")] public int PromptTokens { get; set; }
    [JsonPropertyName("completion_tokens")] public int CompletionTokens { get; set; }
    [JsonPropertyName("total_tokens")] public int TotalTokens { get; set; }
}

public sealed class ApiErrorResponse
{
    [JsonPropertyName("error")] public ApiErrorDetail Error { get; set; } = new();
}

public sealed class ApiErrorDetail
{
    [JsonPropertyName("message")] public string Message { get; set; } = string.Empty;
    [JsonPropertyName("type")] public string Type { get; set; } = "invalid_request_error";
}

public sealed class ModelListResponse
{
    [JsonPropertyName("object")] public string Object { get; set; } = "list";
    [JsonPropertyName("data")] public List<ModelCard> Data { get; set; } = [];
}

public sealed class ModelCard
{
    [JsonPropertyName("id")] public string Id { get; set; } = string.Empty;
    [JsonPropertyName("object")] public string Object { get; set; } = "model";
    [JsonPropertyName("created")] public long Created { get; set; }
    [JsonPropertyName("owned_by")] public string OwnedBy { get; set; } = "glacier";
}

public sealed class OllamaChatChunk
{
    [JsonPropertyName("model")] public string Model { get; set; } = string.Empty;
    [JsonPropertyName("message")] public ChatMessage? Message { get; set; }
    [JsonPropertyName("done")] public bool Done { get; set; }
    [JsonPropertyName("done_reason")] public string? DoneReason { get; set; }
    [JsonPropertyName("prompt_eval_count")] public int? PromptEvalCount { get; set; }
    [JsonPropertyName("eval_count")] public int? EvalCount { get; set; }
}

public sealed class OllamaGenerateChunk
{
    [JsonPropertyName("model")] public string Model { get; set; } = string.Empty;
    [JsonPropertyName("response")] public string Response { get; set; } = string.Empty;
    [JsonPropertyName("done")] public bool Done { get; set; }
    [JsonPropertyName("done_reason")] public string? DoneReason { get; set; }
    [JsonPropertyName("prompt_eval_count")] public int? PromptEvalCount { get; set; }
    [JsonPropertyName("eval_count")] public int? EvalCount { get; set; }
}

public sealed class OllamaTagsResponse
{
    [JsonPropertyName("models")] public List<OllamaModelItem> Models { get; set; } = [];
}

public sealed class OllamaModelItem
{
    [JsonPropertyName("name")] public string Name { get; set; } = string.Empty;
    [JsonPropertyName("model")] public string Model { get; set; } = string.Empty;
    [JsonPropertyName("modified_at")] public string ModifiedAt { get; set; } = string.Empty;
    [JsonPropertyName("size")] public long Size { get; set; }
}

public sealed class HealthCheckResponse
{
    [JsonPropertyName("status")] public string Status { get; set; } = "ok";
    [JsonPropertyName("server")] public string Server { get; set; } = "Glacier.Serve";
    [JsonPropertyName("engine")] public string Engine { get; set; } = "ContinuousBatchEngine";
    [JsonPropertyName("backend")] public string? Backend { get; set; }
    [JsonPropertyName("gpu_accelerated")] public bool? GpuAccelerated { get; set; }
    [JsonPropertyName("paged_attention")] public bool PagedAttention { get; set; } = true;
    [JsonPropertyName("active_batches")] public int ActiveBatches { get; set; }
    [JsonPropertyName("waiting_queue")] public int WaitingQueue { get; set; }
    [JsonPropertyName("free_kv_blocks")] public int FreeKvBlocks { get; set; }
    [JsonPropertyName("total_kv_blocks")] public int TotalKvBlocks { get; set; }
    [JsonPropertyName("tokens_generated")] public long TokensGenerated { get; set; }
    [JsonPropertyName("throughput_tok_s")] public double ThroughputTokS { get; set; }
    [JsonPropertyName("sleep_state")] public string? SleepState { get; set; }
    [JsonPropertyName("pending_dreams")] public int? PendingDreams { get; set; }
}

public sealed class SleepStatusResponse
{
    [JsonPropertyName("enabled")] public bool Enabled { get; set; }
    [JsonPropertyName("state")] public string State { get; set; } = "Disabled";
    [JsonPropertyName("completed_cycles")] public int CompletedCycles { get; set; }
    [JsonPropertyName("total_dreams_consolidated")] public int TotalDreamsConsolidated { get; set; }
    [JsonPropertyName("pending_dreams")] public int PendingDreams { get; set; }
}

public sealed class SleepTriggerResponse
{
    [JsonPropertyName("triggered")] public bool Triggered { get; set; }
    [JsonPropertyName("state")] public string State { get; set; } = "Disabled";
    [JsonPropertyName("expert_slot_index")] public int? ExpertSlotIndex { get; set; }
    [JsonPropertyName("total_dreams")] public int TotalDreams { get; set; }
    [JsonPropertyName("accepted_dreams")] public int AcceptedDreams { get; set; }
    [JsonPropertyName("acceptance_rate")] public float AcceptanceRate { get; set; }
    [JsonPropertyName("initial_loss")] public float InitialLoss { get; set; }
    [JsonPropertyName("final_loss")] public float FinalLoss { get; set; }
    [JsonPropertyName("synaptic_pruning_succeeded")] public bool SynapticPruningSucceeded { get; set; }
}

/// <summary>
/// AOT-safe status payload for <c>GET /api/sleep/status</c> and <c>POST /api/sleep/trigger</c>.
/// </summary>
public sealed class SleepMemoryStatusResponse
{
    [JsonPropertyName("enabled")] public bool Enabled { get; init; }
    [JsonPropertyName("token_budget")] public int TokenBudget { get; init; }
    [JsonPropertyName("state")] public required string State { get; init; }
    [JsonPropertyName("raw_chunks")] public int RawChunks { get; init; }
    [JsonPropertyName("retained_traces")] public int RetainedTraces { get; init; }
    [JsonPropertyName("pruned_chunks")] public int PrunedChunks { get; init; }
    [JsonPropertyName("sleep_cycles")] public int SleepCycles { get; init; }
    [JsonPropertyName("dreams_consolidated")] public int DreamsConsolidated { get; init; }
    [JsonPropertyName("pending_dreams")] public int PendingDreams { get; init; }
    [JsonPropertyName("last_selected_traces")] public int LastSelectedTraces { get; init; }
    [JsonPropertyName("last_packed_tokens")] public int LastPackedTokens { get; init; }
    [JsonPropertyName("last_elapsed_ms")] public double LastElapsedMs { get; init; }
}

/// <summary>
/// AOT-safe status payload for <c>GET /api/bounce/status</c>.
/// </summary>
public sealed class BounceStatusResponse
{
    [JsonPropertyName("primary_endpoint")] public required string PrimaryEndpoint { get; init; }
    [JsonPropertyName("critic_endpoint")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? CriticEndpoint { get; init; }
    [JsonPropertyName("compiler_bounce_enabled")] public bool CompilerBounceEnabled { get; init; }
    [JsonPropertyName("max_repairs")] public int MaxRepairs { get; init; }
    [JsonPropertyName("total_requests")] public long TotalRequests { get; init; }
    [JsonPropertyName("bypassed_requests")] public long BypassedRequests { get; init; }
    [JsonPropertyName("critic_assisted_requests")] public long CriticAssistedRequests { get; init; }
    [JsonPropertyName("critic_fallbacks")] public long CriticFallbacks { get; init; }
    [JsonPropertyName("verifier_checks")] public long VerifierChecks { get; init; }
    [JsonPropertyName("verifier_first_pass_ok")] public long VerifierFirstPassOk { get; init; }
    [JsonPropertyName("repair_attempts")] public long RepairAttempts { get; init; }
    [JsonPropertyName("repairs_succeeded")] public long RepairsSucceeded { get; init; }
}

/// <summary>
/// Outbound AOT-safe chat payload used by <c>PeerBounceService</c>.
/// </summary>
public sealed class BounceChatPayload
{
    [JsonPropertyName("model")] public required string Model { get; init; }
    [JsonPropertyName("messages")] public required ChatMessage[] Messages { get; init; }
    [JsonPropertyName("temperature")] public double Temperature { get; init; }
    [JsonPropertyName("max_tokens")] public int MaxTokens { get; init; }
    [JsonPropertyName("stream")] public bool Stream { get; init; }
    [JsonPropertyName("chat_template_kwargs")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ChatTemplateKwargs? ChatTemplateKwargs { get; init; }
}

/// <summary>
/// Inbound AOT-safe request payload for <c>MapPeerBounce</c> chat endpoints with bounce-appropriate defaults.
/// </summary>
public sealed class BounceChatRequest
{
    [JsonPropertyName("model")] public string? Model { get; set; }
    [JsonPropertyName("messages")] public List<ChatMessage>? Messages { get; set; }
    [JsonPropertyName("temperature")] public double Temperature { get; set; } = 0.0;
    [JsonPropertyName("max_tokens")] public int MaxTokens { get; set; } = 1024;
    [JsonPropertyName("stream")] public bool Stream { get; set; }
}

public sealed class ChatCompletionChunkResponse
{
    [JsonPropertyName("id")] public required string Id { get; init; }
    [JsonPropertyName("object")] public string Object { get; init; } = "chat.completion.chunk";
    [JsonPropertyName("created")] public long Created { get; init; }
    [JsonPropertyName("model")] public required string Model { get; init; }
    [JsonPropertyName("choices")] public required ChatChunkChoice[] Choices { get; init; }
}

public sealed class ChatChunkChoice
{
    [JsonPropertyName("index")] public int Index { get; init; }
    [JsonPropertyName("delta")] public required ChatChunkDelta Delta { get; init; }
    [JsonPropertyName("finish_reason")] public string? FinishReason { get; init; }
}

public sealed class ChatChunkDelta
{
    [JsonPropertyName("role")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Role { get; init; }

    [JsonPropertyName("content")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Content { get; init; }
}

public sealed class TensorDto
{
    [JsonPropertyName("rank")] public int Rank { get; set; }
    [JsonPropertyName("shape")] public int[] Shape { get; set; } = [];
    [JsonPropertyName("length")] public int Length { get; set; }
    [JsonPropertyName("data")] public float[] Data { get; set; } = [];
}

[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    PropertyNameCaseInsensitive = true,
    NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(InferenceRequest))]
[JsonSerializable(typeof(OllamaOptions))]
[JsonSerializable(typeof(ChatMessage))]
[JsonSerializable(typeof(ChatTemplateKwargs))]
[JsonSerializable(typeof(StreamOptions))]
[JsonSerializable(typeof(ChatCompletionResponse))]
[JsonSerializable(typeof(ChatChoice))]
[JsonSerializable(typeof(CompletionResponse))]
[JsonSerializable(typeof(CompletionChoice))]
[JsonSerializable(typeof(ChatUsage))]
[JsonSerializable(typeof(ApiErrorResponse))]
[JsonSerializable(typeof(ApiErrorDetail))]
[JsonSerializable(typeof(ModelListResponse))]
[JsonSerializable(typeof(ModelCard))]
[JsonSerializable(typeof(OllamaChatChunk))]
[JsonSerializable(typeof(OllamaGenerateChunk))]
[JsonSerializable(typeof(OllamaTagsResponse))]
[JsonSerializable(typeof(OllamaModelItem))]
[JsonSerializable(typeof(TensorDto))]
[JsonSerializable(typeof(HealthCheckResponse))]
[JsonSerializable(typeof(SleepStatusResponse))]
[JsonSerializable(typeof(SleepTriggerResponse))]
[JsonSerializable(typeof(SleepMemoryStatusResponse))]
[JsonSerializable(typeof(BounceStatusResponse))]
[JsonSerializable(typeof(BounceChatPayload))]
[JsonSerializable(typeof(BounceChatRequest))]
[JsonSerializable(typeof(ChatCompletionChunkResponse))]
[JsonSerializable(typeof(ChatChunkChoice))]
[JsonSerializable(typeof(ChatChunkDelta))]
public partial class ServeJsonContext : JsonSerializerContext
{
}

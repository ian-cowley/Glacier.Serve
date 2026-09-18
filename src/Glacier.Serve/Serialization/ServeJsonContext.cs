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

public sealed class ChatUsage
{
    [JsonPropertyName("prompt_tokens")] public int PromptTokens { get; set; }
    [JsonPropertyName("completion_tokens")] public int CompletionTokens { get; set; }
    [JsonPropertyName("total_tokens")] public int TotalTokens { get; set; }
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
    [JsonPropertyName("prompt_eval_count")] public int? PromptEvalCount { get; set; }
    [JsonPropertyName("eval_count")] public int? EvalCount { get; set; }
}

public sealed class OllamaGenerateChunk
{
    [JsonPropertyName("model")] public string Model { get; set; } = string.Empty;
    [JsonPropertyName("response")] public string Response { get; set; } = string.Empty;
    [JsonPropertyName("done")] public bool Done { get; set; }
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
    [JsonPropertyName("paged_attention")] public bool PagedAttention { get; set; } = true;
    [JsonPropertyName("active_batches")] public int ActiveBatches { get; set; }
    [JsonPropertyName("waiting_queue")] public int WaitingQueue { get; set; }
    [JsonPropertyName("free_kv_blocks")] public int FreeKvBlocks { get; set; }
    [JsonPropertyName("total_kv_blocks")] public int TotalKvBlocks { get; set; }
    [JsonPropertyName("tokens_generated")] public long TokensGenerated { get; set; }
    [JsonPropertyName("throughput_tok_s")] public double ThroughputTokS { get; set; }
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
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(InferenceRequest))]
[JsonSerializable(typeof(ChatMessage))]
[JsonSerializable(typeof(ChatCompletionResponse))]
[JsonSerializable(typeof(ModelListResponse))]
[JsonSerializable(typeof(OllamaChatChunk))]
[JsonSerializable(typeof(OllamaGenerateChunk))]
[JsonSerializable(typeof(OllamaTagsResponse))]
[JsonSerializable(typeof(TensorDto))]
[JsonSerializable(typeof(HealthCheckResponse))]
public partial class ServeJsonContext : JsonSerializerContext
{
}

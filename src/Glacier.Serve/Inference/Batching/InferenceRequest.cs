using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Glacier.Serve.Inference.Batching;

public sealed class ChatMessage
{
    [JsonPropertyName("role")]
    public string Role { get; set; } = "user";

    [JsonPropertyName("content")]
    public string Content { get; set; } = string.Empty;
}

/// <summary>
/// Nested options object for Ollama /api/generate and /api/chat wire compatibility.
/// </summary>
public sealed class OllamaOptions
{
    [JsonPropertyName("num_predict")]
    public int? NumPredict { get; set; }

    [JsonPropertyName("temperature")]
    public float? Temperature { get; set; }

    [JsonPropertyName("top_p")]
    public float? TopP { get; set; }

    [JsonPropertyName("top_k")]
    public int? TopK { get; set; }

    [JsonPropertyName("repeat_penalty")]
    public float? RepeatPenalty { get; set; }

    [JsonPropertyName("seed")]
    public int? Seed { get; set; }

    [JsonPropertyName("stop")]
    [JsonConverter(typeof(PolymorphicStopSequencesConverter))]
    public string[]? Stop { get; set; }
}

/// <summary>
/// AOT-safe JSON converter accepting either a single string (<c>"stop": "\n"</c>)
/// or an array of strings (<c>"stop": ["\n", "END"]</c>) per the OpenAI API specification.
/// </summary>
public sealed class PolymorphicStopSequencesConverter : JsonConverter<string[]?>
{
    public override string[]? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType == JsonTokenType.Null)
            return null;

        if (reader.TokenType == JsonTokenType.String)
        {
            string? single = reader.GetString();
            return string.IsNullOrEmpty(single) ? [] : [single];
        }

        if (reader.TokenType == JsonTokenType.StartArray)
        {
            var list = new List<string>(4);
            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndArray)
                    return list.ToArray();

                if (reader.TokenType == JsonTokenType.String)
                {
                    string? item = reader.GetString();
                    if (!string.IsNullOrEmpty(item))
                        list.Add(item);
                }
                else
                {
                    reader.Skip();
                }
            }
            throw new JsonException("Unexpected end of JSON while reading 'stop' array.");
        }

        throw new JsonException($"Unsupported JSON token '{reader.TokenType}' for 'stop' parameter.");
    }

    public override void Write(Utf8JsonWriter writer, string[]? value, JsonSerializerOptions options)
    {
        if (value == null)
        {
            writer.WriteNullValue();
            return;
        }

        writer.WriteStartArray();
        for (int i = 0; i < value.Length; i++)
        {
            writer.WriteStringValue(value[i]);
        }
        writer.WriteEndArray();
    }
}

/// <summary>
/// User inference request parameters supporting OpenAI (/v1/chat/completions, /v1/completions)
/// and Ollama (/api/chat, /api/generate) wire contracts under Native AOT source generation.
/// </summary>
public sealed class InferenceRequest
{
    private bool _stream;

    [JsonPropertyName("prompt")]
    public string? Prompt { get; set; }

    [JsonPropertyName("messages")]
    public List<ChatMessage>? Messages { get; set; }

    [JsonPropertyName("model")]
    public string Model { get; set; } = string.Empty;

    [JsonPropertyName("max_tokens")]
    public int MaxTokens { get; set; } = 256;

    [JsonPropertyName("max_completion_tokens")]
    public int? MaxCompletionTokens { get; set; }

    [JsonPropertyName("temperature")]
    public float Temperature { get; set; } = 0.7f;

    [JsonPropertyName("top_p")]
    public float TopP { get; set; } = 0.9f;

    [JsonPropertyName("top_k")]
    public int TopK { get; set; } = 0;

    [JsonPropertyName("repetition_penalty")]
    public float RepetitionPenalty { get; set; } = 1.0f;

    [JsonPropertyName("repeat_penalty")]
    public float? RepeatPenalty { get; set; }

    [JsonPropertyName("seed")]
    public int? Seed { get; set; }

    [JsonPropertyName("stream")]
    public bool Stream
    {
        get => _stream;
        set
        {
            _stream = value;
            IsStreamExplicitlySet = true;
        }
    }

    [JsonIgnore]
    public bool IsStreamExplicitlySet { get; private set; }

    [JsonPropertyName("stop")]
    [JsonConverter(typeof(PolymorphicStopSequencesConverter))]
    public string[]? Stop { get; set; }

    [JsonPropertyName("max_reasoning_tokens")]
    public int MaxReasoningTokens { get; set; } = 0;

    [JsonPropertyName("enable_reasoning_loop_guard")]
    public bool? EnableReasoningLoopGuard { get; set; }

    [JsonPropertyName("reasoning_loop_min_period")]
    public int? ReasoningLoopMinPeriod { get; set; }

    [JsonPropertyName("enable_thinking")]
    public bool? EnableThinking { get; set; }

    [JsonPropertyName("options")]
    public OllamaOptions? Options { get; set; }

    [JsonIgnore]
    public int EffectiveMaxTokens =>
        Options?.NumPredict is > 0 ? Options.NumPredict.Value :
        MaxCompletionTokens is > 0 ? MaxCompletionTokens.Value :
        MaxTokens;

    [JsonIgnore]
    public float EffectiveTemperature => Options?.Temperature ?? Temperature;

    [JsonIgnore]
    public float EffectiveTopP => Options?.TopP ?? TopP;

    [JsonIgnore]
    public int EffectiveTopK => Options?.TopK ?? TopK;

    [JsonIgnore]
    public float EffectiveRepetitionPenalty =>
        Options?.RepeatPenalty ?? RepeatPenalty ?? RepetitionPenalty;

    [JsonIgnore]
    public int? EffectiveSeed => Options?.Seed ?? Seed;

    [JsonIgnore]
    public string[]? EffectiveStop => Options?.Stop ?? Stop;
}

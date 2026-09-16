using System.Collections.Generic;

namespace Glacier.Serve.Inference.Batching;

public sealed class ChatMessage
{
    public string Role { get; set; } = "user";
    public string Content { get; set; } = string.Empty;
}

/// <summary>
/// User inference request parameters.
/// </summary>
public sealed class InferenceRequest
{
    public string? Prompt { get; set; }
    public List<ChatMessage>? Messages { get; set; }
    public string Model { get; set; } = string.Empty;
    public int MaxTokens { get; set; } = 256;
    public float Temperature { get; set; } = 0.7f;
    public float TopP { get; set; } = 0.9f;
    public bool Stream { get; set; } = false;
    public string[]? Stop { get; set; }
}

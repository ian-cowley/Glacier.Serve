using System;
using System.Collections.Generic;
using System.Threading.Channels;
using System.Threading.Tasks;
using Glacier.Inference.Sampling;
using Glacier.Serve.Inference.PagedAttention;

namespace Glacier.Serve.Inference.Batching;

/// <summary>
/// Execution state for an individual sequence inside the continuous batch engine.
/// Tracks allocated virtual pages in PagedBlockPool, generated tokens, and streaming channel.
/// </summary>
public sealed class SequenceState
{
    public string RequestId { get; }
    public int[] PromptTokens { get; }
    public List<int> GeneratedTokens { get; } = new(128);
    public BlockTable BlockTable { get; }
    public int CurrentPos { get; set; }
    public int LastToken { get; set; }
    public int MaxNewTokens { get; set; }
    public SamplingOptions SamplingOptions { get; }
    public string[]? StopSequences { get; set; }

    public Channel<string> TokenChannel { get; } = Channel.CreateUnbounded<string>(new UnboundedChannelOptions
    {
        SingleWriter = true,
        SingleReader = false
    });

    public TaskCompletionSource<string> Completion { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);

    public bool IsPrefillDone { get; set; }
    public bool IsFinished { get; set; }
    public string FinishReason { get; set; } = string.Empty;

    public DateTimeOffset CreatedAt { get; } = DateTimeOffset.UtcNow;
    public DateTimeOffset? FirstTokenTime { get; set; }
    public double TimeToFirstTokenMs => FirstTokenTime.HasValue ? (FirstTokenTime.Value - CreatedAt).TotalMilliseconds : 0.0;

    public SequenceState(string requestId, int[] promptTokens, BlockTable blockTable, int maxNewTokens = 256, float temperature = 0.7f, float topP = 0.9f, string[]? stopSequences = null)
    {
        RequestId = requestId;
        PromptTokens = promptTokens;
        BlockTable = blockTable;
        MaxNewTokens = maxNewTokens;
        SamplingOptions = new SamplingOptions
        {
            Temperature = temperature,
            TopP = topP,
            MaxTokens = maxNewTokens
        };
        StopSequences = stopSequences;
        CurrentPos = 0;
        LastToken = promptTokens.Length > 0 ? promptTokens[^1] : 0;
    }
}

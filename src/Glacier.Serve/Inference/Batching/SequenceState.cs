using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Glacier.Inference.Sampling;
using Glacier.Serve.Inference.PagedAttention;

namespace Glacier.Serve.Inference.Batching;

/// <summary>
/// Execution state for an individual sequence inside the continuous batch engine.
/// Tracks allocated virtual pages in PagedBlockPool, chunked prefill progress,
/// incremental stateful UTF-8 decoding, stop-sequence holdback, and streaming channel.
/// </summary>
public sealed class SequenceState
{
    public string RequestId { get; }
    public int[] PromptTokens { get; }
    public List<int> GeneratedTokens { get; } = new(128);
    public BlockTable BlockTable { get; }
    public int PrefilledTokens { get; set; }
    public int CurrentPos { get; set; }
    public int BoundPrefixBlocks { get; set; }
    public ulong LastPrefixHash { get; set; }
    public int LastToken { get; set; }
    public int MaxNewTokens { get; set; }
    public SamplingOptions SamplingOptions { get; }
    public Sampler? SequenceSampler { get; }
    public string[]? StopSequences { get; set; }
    public CancellationToken CancellationToken { get; }

    public Decoder Utf8Decoder { get; } = Encoding.UTF8.GetDecoder();
    public StringBuilder GeneratedTextBuilder { get; } = new(256);
    public int EmittedTextLength { get; set; }

    public int CurrentReasoningTokens { get; set; }
    public bool IsInReasoningBlock { get; set; }
    public int LoopRecoveryTriggers { get; set; }
    public int PeriodicLoopTriggers { get; set; }
    public int BudgetClampTriggers { get; set; }

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

    public SequenceState(
        string requestId,
        int[] promptTokens,
        BlockTable blockTable,
        int maxNewTokens = 256,
        float temperature = 0.7f,
        float topP = 0.9f,
        string[]? stopSequences = null,
        float repetitionPenalty = 1.0f,
        int topK = 0,
        int? seed = null,
        CancellationToken cancellationToken = default)
    {
        RequestId = requestId;
        PromptTokens = promptTokens;
        BlockTable = blockTable;
        MaxNewTokens = maxNewTokens;
        SamplingOptions = new SamplingOptions
        {
            Temperature = temperature,
            TopP = topP,
            TopK = topK,
            RepetitionPenalty = repetitionPenalty,
            MaxTokens = maxNewTokens
        };
        SequenceSampler = seed.HasValue ? new Sampler(seed.Value) : null;
        StopSequences = stopSequences;
        CancellationToken = cancellationToken;
        CurrentPos = 0;
        LastToken = promptTokens.Length > 0 ? promptTokens[^1] : 0;
    }
}

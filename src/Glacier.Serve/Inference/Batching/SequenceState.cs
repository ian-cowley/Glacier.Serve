using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Glacier.Inference.Engine;
using Glacier.Inference.Model;
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
    public bool ForcedThinkClose { get; set; }

    public string? LoraAdapterName { get; init; }
    public LoraAdapterWeights? LoraAdapter { get; init; }
    public ulong LoraAdapterInstanceId { get; init; }

    internal string? RawUserQueryForSleep { get; init; }
    internal float[]? HiddenStateHistory { get; set; }
    internal MtpDraftProvider? MtpDrafter { get; set; }

    private ReasoningBudgetTracker _thinkTracker;
    private int _unbudgetedScannedTextLen;
    private int _unbudgetedMinTagSearchPos;
    private int[]? _pendingForcedTokens;
    private int _pendingForcedTokenIndex;

    public bool HasPendingForcedTokens => _pendingForcedTokens != null && _pendingForcedTokenIndex < _pendingForcedTokens.Length;

    internal void InitializeReasoningTracker(
        int thinkId,
        int endThinkId,
        int maxReasoningTokens,
        ReadOnlySpan<int> promptTokens,
        string? promptText)
    {
        _thinkTracker = new ReasoningBudgetTracker(thinkId, endThinkId, maxReasoningTokens, promptTokens, promptText);
        IsInReasoningBlock = _thinkTracker.InThink;
        CurrentReasoningTokens = _thinkTracker.ReasoningTokens;
        ForcedThinkClose = _thinkTracker.ForcedThinkClose;
        _unbudgetedScannedTextLen = 0;
        _unbudgetedMinTagSearchPos = 0;
    }

    internal bool ObserveReasoningTokenAndShouldForceClose(int tokenId, bool useSingleTokenDelimiters)
    {
        bool shouldForceClose = useSingleTokenDelimiters
            ? _thinkTracker.ObserveTokenAndShouldForceClose(tokenId)
            : _thinkTracker.ObserveDecodedTextAndShouldForceClose(GeneratedTextBuilder);

        IsInReasoningBlock = _thinkTracker.InThink;
        CurrentReasoningTokens = _thinkTracker.ReasoningTokens;
        ForcedThinkClose = _thinkTracker.ForcedThinkClose;
        return shouldForceClose;
    }

    internal void ObserveUnbudgetedReasoningToken(int tokenId, int thinkId, int endThinkId, bool useSingleTokenDelimiters)
    {
        if (useSingleTokenDelimiters)
        {
            if (tokenId == thinkId)
            {
                IsInReasoningBlock = true;
                CurrentReasoningTokens = 0;
            }
            else if (tokenId == endThinkId)
            {
                IsInReasoningBlock = false;
            }
            else if (IsInReasoningBlock)
            {
                CurrentReasoningTokens++;
            }
            return;
        }

        const string openTag = "<think>";
        const string closeTag = "</think>";
        int curLen = GeneratedTextBuilder.Length;
        int scanStart = Math.Max(_unbudgetedMinTagSearchPos, _unbudgetedScannedTextLen - (closeTag.Length - 1));
        int windowLen = curLen - scanStart;
        bool sawTag = false;

        if (windowLen >= openTag.Length)
        {
            Span<char> stackBuf = windowLen <= 256 ? stackalloc char[windowLen] : default;
            char[]? rented = windowLen > 256 ? new char[windowLen] : null;
            Span<char> window = rented is null ? stackBuf : rented.AsSpan();
            GeneratedTextBuilder.CopyTo(scanStart, window, windowLen);

            int cursor = 0;
            while (cursor <= window.Length - openTag.Length)
            {
                ReadOnlySpan<char> rem = window.Slice(cursor);
                int openRel = rem.IndexOf(openTag.AsSpan(), StringComparison.Ordinal);
                int closeRel = rem.IndexOf(closeTag.AsSpan(), StringComparison.Ordinal);

                if (openRel < 0 && closeRel < 0)
                    break;

                sawTag = true;
                if (openRel >= 0 && (closeRel < 0 || openRel < closeRel))
                {
                    IsInReasoningBlock = true;
                    CurrentReasoningTokens = 0;
                    cursor += openRel + openTag.Length;
                }
                else
                {
                    IsInReasoningBlock = false;
                    cursor += closeRel + closeTag.Length;
                }

                _unbudgetedMinTagSearchPos = scanStart + cursor;
            }
        }

        _unbudgetedScannedTextLen = curLen;
        if (!sawTag && IsInReasoningBlock)
        {
            CurrentReasoningTokens++;
        }
    }

    internal void MarkForcedThinkClose()
    {
        _thinkTracker.MarkForcedClose();
        IsInReasoningBlock = _thinkTracker.InThink;
        ForcedThinkClose = _thinkTracker.ForcedThinkClose;
    }

    internal void EnqueueForcedTokens(int[] tokens)
    {
        _pendingForcedTokens = tokens;
        _pendingForcedTokenIndex = 0;
    }

    internal bool TryPopForcedToken(out int tokenId)
    {
        if (_pendingForcedTokens != null && _pendingForcedTokenIndex < _pendingForcedTokens.Length)
        {
            tokenId = _pendingForcedTokens[_pendingForcedTokenIndex++];
            if (_pendingForcedTokenIndex >= _pendingForcedTokens.Length)
            {
                _pendingForcedTokens = null;
                _pendingForcedTokenIndex = 0;
            }
            return true;
        }

        tokenId = 0;
        return false;
    }

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

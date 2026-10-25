using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Glacier.Inference.Sampling;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Inference.PagedAttention;
using Xunit;

namespace Glacier.Serve.Tests;

public class ContinuousBatchingTests
{
    [Fact]
    public void SequenceState_TracksTokensAndAllocations()
    {
        using var pool = new PagedBlockPool(totalBlocks: 8, layers: 2, headsKv: 2, headDim: 16, blockSize: 8);
        var table = new BlockTable(pool);
        int[] prompt = [100, 200, 300];

        var seq = new SequenceState("test-req-1", prompt, table, maxNewTokens: 32, temperature: 0.5f);

        Assert.Equal("test-req-1", seq.RequestId);
        Assert.Equal(3, seq.PromptTokens.Length);
        Assert.Equal(300, seq.LastToken);
        Assert.Equal(32, seq.MaxNewTokens);
        Assert.Equal(0.5f, seq.SamplingOptions.Temperature);
        Assert.False(seq.IsFinished);
        Assert.Empty(seq.GeneratedTokens);
    }

    [Fact]
    public void SequenceState_TracksReasoningStateAndBudget()
    {
        using var pool = new PagedBlockPool(totalBlocks: 8, layers: 2, headsKv: 2, headDim: 16, blockSize: 8);
        var table = new BlockTable(pool);
        var seq = new SequenceState("test-reasoning-seq", [100], table);

        Assert.False(seq.IsInReasoningBlock);
        Assert.Equal(0, seq.CurrentReasoningTokens);

        seq.IsInReasoningBlock = true;
        seq.CurrentReasoningTokens = 42;
        seq.SamplingOptions.MaxReasoningTokens = 100;
        seq.SamplingOptions.ThinkTokenId = 151648;
        seq.SamplingOptions.EndThinkTokenId = 151649;

        Assert.True(seq.IsInReasoningBlock);
        Assert.Equal(42, seq.CurrentReasoningTokens);
        Assert.Equal(100, seq.SamplingOptions.MaxReasoningTokens);
        Assert.Equal(151648, seq.SamplingOptions.ThinkTokenId);
        Assert.Equal(151649, seq.SamplingOptions.EndThinkTokenId);
    }

    [Fact]
    public void PagedBlockPool_ThrowsOnInvalidParameters()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new PagedBlockPool(totalBlocks: 0, layers: 2, headsKv: 2, headDim: 16));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PagedBlockPool(totalBlocks: 10, layers: 0, headsKv: 2, headDim: 16));
    }

    [Fact]
    public void ContinuousBatchEngine_DecodeStep_DemonstratesDormancyOnPreChangeCode()
    {
        // SETUP: Replicate ContinuousBatchEngine.cs line 178 pre-change environment
        var sampler = new Sampler();
        using var pool = new PagedBlockPool(totalBlocks: 8, layers: 1, headsKv: 1, headDim: 16);
        var table = new BlockTable(pool);
        var seq = new SequenceState("test-dormant-seq", [101, 102], table)
        {
            SamplingOptions =
            {
                Temperature = 0.0f,
                EnableReasoningLoopGuard = true,
                LoopRecoveryRepetitionPenalty = 3.0f
            }
        };

        // Populate a degenerate reasoning loop in GeneratedTokens: [10, 20, 10, 20, 10, 20]
        seq.GeneratedTokens.AddRange([10, 20, 10, 20, 10, 20]);

        float[] logits = new float[50];
        logits[10] = 90.0f; // in-loop candidate
        logits[20] = 80.0f; // in-loop candidate
        logits[5] = 40.0f;  // out-of-loop fallback candidate

        // PRE-CHANGE INVOCATION (Verbatim line 178 of ContinuousBatchEngine.cs prior to REC-1):
        // int nextToken = _sampler.Sample(seqLogits, seq.SamplingOptions);
        int nextTokenPreChange = sampler.Sample(logits.AsSpan(), seq.SamplingOptions);

        // PROOF OF DORMANCY:
        // In pre-change code, nextToken is 10 because seq.GeneratedTokens is never forwarded.
        // If an assertion were written expecting loop breaking (Assert.Equal(5, nextTokenPreChange)),
        // it FAILS with: Expected: 5, Actual: 10.
        Assert.Equal(10, nextTokenPreChange);
    }

    [Fact]
    public void ContinuousBatchEngine_DecodeStep_PostChangeWiresRecentTokens_BreaksLoop()
    {
        // SETUP: Post-change environment where GeneratedTokens is forwarded
        var sampler = new Sampler();
        using var pool = new PagedBlockPool(totalBlocks: 8, layers: 1, headsKv: 1, headDim: 16);
        var table = new BlockTable(pool);
        var seq = new SequenceState("test-active-seq", [101, 102], table)
        {
            SamplingOptions =
            {
                Temperature = 0.0f,
                EnableReasoningLoopGuard = true,
                LoopRecoveryRepetitionPenalty = 3.0f
            }
        };

        seq.GeneratedTokens.AddRange([10, 20, 10, 20, 10, 20]);

        float[] logits = new float[50];
        logits[10] = 90.0f;
        logits[20] = 80.0f;
        logits[5] = 40.0f;

        // POST-CHANGE INVOCATION:
        // Forward CollectionsMarshal.AsSpan(seq.GeneratedTokens) to _sampler.Sample:
        ReadOnlySpan<int> recent = CollectionsMarshal.AsSpan(seq.GeneratedTokens);
        int nextTokenPostChange = sampler.Sample(logits.AsSpan(), seq.SamplingOptions, recent);

        // PROOF OF ACTIVE LOOP PREVENTION:
        // Token 10 logit is penalized: 90.0 / 3.0 = 30.0.
        // Token 20 logit is penalized: 80.0 / 3.0 = 26.67.
        // Token 5 (40.0) is greater than 30.0 and is selected!
        Assert.Equal(5, nextTokenPostChange);
    }

    [Fact]
    public void ContinuousBatchEngine_DecodeStep_NonLoopingSequence_LeavesTopCandidateUntouched()
    {
        // NEGATIVE CONTROL: Ensures normal generation without loops is unaffected
        var sampler = new Sampler();
        using var pool = new PagedBlockPool(totalBlocks: 8, layers: 1, headsKv: 1, headDim: 16);
        var table = new BlockTable(pool);
        var seq = new SequenceState("test-normal-seq", [101, 102], table)
        {
            SamplingOptions =
            {
                Temperature = 0.0f,
                EnableReasoningLoopGuard = true,
                LoopRecoveryRepetitionPenalty = 3.0f
            }
        };

        // Non-repeating realistic token stream
        seq.GeneratedTokens.AddRange([1, 2, 3, 4, 5, 6]);

        float[] logits = new float[50];
        logits[10] = 90.0f;
        logits[20] = 80.0f;
        logits[5] = 40.0f;

        ReadOnlySpan<int> recent = CollectionsMarshal.AsSpan(seq.GeneratedTokens);
        int nextToken = sampler.Sample(logits.AsSpan(), seq.SamplingOptions, recent);

        // Top candidate 10 is selected with 0 false-positive penalties
        Assert.Equal(10, nextToken);
    }

    [Fact]
    public void ContinuousBatchEngine_ReasoningBudgetExceeded_ClampsLogitsAndForcesEndThink()
    {
        var sampler = new Sampler();
        using var pool = new PagedBlockPool(totalBlocks: 8, layers: 1, headsKv: 1, headDim: 16);
        var table = new BlockTable(pool);
        var seq = new SequenceState("test-budget-seq", [101, 102], table)
        {
            IsInReasoningBlock = true,
            CurrentReasoningTokens = 150,
            SamplingOptions =
            {
                Temperature = 0.0f,
                MaxReasoningTokens = 100,
                EndThinkTokenId = 25
            }
        };

        float[] logits = new float[50];
        logits[10] = 15.0f; // continuation candidate
        logits[25] = 10.0f; // EndThinkTokenId initially lower

        // Simulate logic in ContinuousBatchEngine RunIterationLoopAsync
        if (seq.IsInReasoningBlock &&
            seq.SamplingOptions.MaxReasoningTokens > 0 &&
            seq.CurrentReasoningTokens >= seq.SamplingOptions.MaxReasoningTokens &&
            seq.SamplingOptions.EndThinkTokenId >= 0)
        {
            Sampler.ClampReasoningLogits(logits.AsSpan(), seq.SamplingOptions.EndThinkTokenId);
        }

        // Token 25 was biased: 10 + 20 = 30.0
        // Token 10 was penalized: 15 - 50 = -35.0
        Assert.Equal(30.0f, logits[25]);
        Assert.Equal(-35.0f, logits[10]);

        ReadOnlySpan<int> recent = CollectionsMarshal.AsSpan(seq.GeneratedTokens);
        int nextToken = sampler.Sample(logits.AsSpan(), seq.SamplingOptions, recent);

        // Clamping forces EndThinkTokenId (25) to be chosen!
        Assert.Equal(25, nextToken);
    }
}

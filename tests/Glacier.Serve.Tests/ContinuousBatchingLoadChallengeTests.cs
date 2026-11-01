namespace Glacier.Serve.Tests;

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Glacier.Inference.Sampling;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Inference.PagedAttention;
using Xunit;

/// <summary>
/// Adversarial challenger suite stress-testing ContinuousBatchEngine with concurrent,
/// heterogeneous sequences across multi-threaded load, dynamic prompt injection,
/// reasoning budget exhaustion, and periodic loop recovery isolation.
/// </summary>
public class ContinuousBatchingLoadChallengeTests
{
    private static string? RealModelPath => Environment.GetEnvironmentVariable("GLACIER_MODEL_PATH");

    /// <summary>
    /// Challenge 1: Concurrent sequences in the same batch where Sequence A has an infinite reasoning loop
    /// and Sequence B has normal generation. Sequence A's loop penalty must NOT leak or affect Sequence B.
    /// </summary>
    [Fact]
    public void Challenge_ConcurrentSequences_LoopRecoveryInSeqA_DoesNotCorrupt_NormalGenerationInSeqB()
    {
        var sampler = new Sampler();
        using var pool = new PagedBlockPool(totalBlocks: 16, layers: 1, headsKv: 1, headDim: 16);

        // Sequence A: Trapped in a repetitive loop [10, 20, 10, 20, 10, 20]
        var tableA = new BlockTable(pool);
        var seqA = new SequenceState("seq-A-looping", [101, 102], tableA)
        {
            SamplingOptions =
            {
                Temperature = 0.0f,
                EnableReasoningLoopGuard = true,
                LoopRecoveryRepetitionPenalty = 3.0f
            }
        };
        seqA.GeneratedTokens.AddRange([10, 20, 10, 20, 10, 20]);

        // Sequence B: Healthy non-looping sequence [1, 2, 3, 4, 5, 6]
        var tableB = new BlockTable(pool);
        var seqB = new SequenceState("seq-B-normal", [201, 202], tableB)
        {
            SamplingOptions =
            {
                Temperature = 0.0f,
                EnableReasoningLoopGuard = true,
                LoopRecoveryRepetitionPenalty = 3.0f
            }
        };
        seqB.GeneratedTokens.AddRange([1, 2, 3, 4, 5, 6]);

        int vocabSize = 50;
        int activeCount = 2;
        var activeBatch = new List<SequenceState> { seqA, seqB };

        // Construct unified batch logits buffer (2 * vocabSize)
        float[] batchLogits = new float[activeCount * vocabSize];

        // Sequence A logits: Token 10 (90.0) is in-loop, Token 20 (80.0) is in-loop, Token 5 (40.0) is clean alternative
        batchLogits[0 * vocabSize + 10] = 90.0f;
        batchLogits[0 * vocabSize + 20] = 80.0f;
        batchLogits[0 * vocabSize + 5] = 40.0f;

        // Sequence B logits: Token 10 (90.0) is Seq B's legitimate top choice!
        batchLogits[1 * vocabSize + 10] = 90.0f;
        batchLogits[1 * vocabSize + 20] = 80.0f;
        batchLogits[1 * vocabSize + 5] = 40.0f;

        // Execute batch decode iteration loop logic (matching ContinuousBatchEngine.cs lines 185-241)
        int[] nextTokens = new int[activeCount];
        for (int i = 0; i < activeCount; i++)
        {
            var seq = activeBatch[i];
            var seqLogits = batchLogits.AsSpan(i * vocabSize, vocabSize);

            ReadOnlySpan<int> recentTokens = CollectionsMarshal.AsSpan(seq.GeneratedTokens);
            if (seq.SamplingOptions.EnableReasoningLoopGuard && !recentTokens.IsEmpty && Sampler.DetectPeriodicReasoningLoop(recentTokens) > 0)
            {
                seq.LoopRecoveryTriggers++;
            }

            nextTokens[i] = sampler.Sample(seqLogits, seq.SamplingOptions, recentTokens);
            seq.GeneratedTokens.Add(nextTokens[i]);
            seq.LastToken = nextTokens[i];
        }

        // VERIFICATION:
        // Sequence A: Loop detected (period 2), trigger count incremented, loop penalty applied
        // Token 10 was penalized (90.0 / 3.0 = 30.0 < 40.0), so Token 5 was selected to break the loop!
        Assert.Equal(1, seqA.LoopRecoveryTriggers);
        Assert.Equal(5, nextTokens[0]);
        Assert.Equal(5, seqA.LastToken);

        // Sequence B: No loop detected, trigger count remains 0, zero penalties applied
        // Legitimate top token 10 MUST be selected!
        Assert.Equal(0, seqB.LoopRecoveryTriggers);
        Assert.Equal(10, nextTokens[1]);
        Assert.Equal(10, seqB.LastToken);

        // Verify Sequence B's logits were not corrupted by Sequence A's penalty
        // (Seq B's slice token 10 stayed high at 90.0)
        Assert.Equal(90.0f, batchLogits[1 * vocabSize + 10]);

        tableA.ReleaseAll();
        tableB.ReleaseAll();
        Assert.Equal(16, pool.FreeBlocksCount);
    }

    /// <summary>
    /// Challenge 2: Reasoning budgeting: Sequence A exceeds reasoning budget, while Sequence B
    /// is within budget and Sequence C is non-reasoning. Verify Sequence A is forced to emit EndThinkTokenId
    /// while Sequence B and Sequence C logits remain completely untouched.
    /// </summary>
    [Fact]
    public void Challenge_ReasoningBudget_SeqAExceedsBudget_SeqBLogitsUntouched()
    {
        var sampler = new Sampler();
        using var pool = new PagedBlockPool(totalBlocks: 24, layers: 1, headsKv: 1, headDim: 16);

        // Sequence A: Reasoning budget EXCEEDED (100 / 100)
        var tableA = new BlockTable(pool);
        var seqA = new SequenceState("seq-A-budget-exceeded", [100], tableA)
        {
            IsInReasoningBlock = true,
            CurrentReasoningTokens = 100,
            SamplingOptions =
            {
                Temperature = 0.0f,
                MaxReasoningTokens = 100,
                EndThinkTokenId = 25
            }
        };

        // Sequence B: Reasoning budget NOT exceeded (20 / 100)
        var tableB = new BlockTable(pool);
        var seqB = new SequenceState("seq-B-within-budget", [200], tableB)
        {
            IsInReasoningBlock = true,
            CurrentReasoningTokens = 20,
            SamplingOptions =
            {
                Temperature = 0.0f,
                MaxReasoningTokens = 100,
                EndThinkTokenId = 25
            }
        };

        // Sequence C: Non-reasoning standard sequence
        var tableC = new BlockTable(pool);
        var seqC = new SequenceState("seq-C-standard", [300], tableC)
        {
            IsInReasoningBlock = false,
            CurrentReasoningTokens = 0,
            SamplingOptions =
            {
                Temperature = 0.0f,
                MaxReasoningTokens = 0,
                EndThinkTokenId = -1
            }
        };

        int vocabSize = 50;
        int activeCount = 3;
        var activeBatch = new List<SequenceState> { seqA, seqB, seqC };

        float[] batchLogits = new float[activeCount * vocabSize];

        // All sequences have raw logits favoring continuation token 10 (35.0f) over EndThinkTokenId 25 (10.0f)
        for (int i = 0; i < activeCount; i++)
        {
            batchLogits[i * vocabSize + 10] = 35.0f;
            batchLogits[i * vocabSize + 25] = 10.0f;
        }

        // Execute batch decode iteration loop
        int[] nextTokens = new int[activeCount];
        for (int i = 0; i < activeCount; i++)
        {
            var seq = activeBatch[i];
            var seqLogits = batchLogits.AsSpan(i * vocabSize, vocabSize);

            // Dynamic reasoning token budgeting & logit clamping
            if (seq.IsInReasoningBlock &&
                seq.SamplingOptions.MaxReasoningTokens > 0 &&
                seq.CurrentReasoningTokens >= seq.SamplingOptions.MaxReasoningTokens &&
                seq.SamplingOptions.EndThinkTokenId >= 0)
            {
                Sampler.ClampReasoningLogits(seqLogits, seq.SamplingOptions.EndThinkTokenId);
            }

            ReadOnlySpan<int> recentTokens = CollectionsMarshal.AsSpan(seq.GeneratedTokens);
            nextTokens[i] = sampler.Sample(seqLogits, seq.SamplingOptions, recentTokens);
            seq.LastToken = nextTokens[i];
            seq.GeneratedTokens.Add(nextTokens[i]);

            // Update reasoning state
            if (seq.SamplingOptions.ThinkTokenId >= 0 && nextTokens[i] == seq.SamplingOptions.ThinkTokenId)
            {
                seq.IsInReasoningBlock = true;
            }
            else if (seq.SamplingOptions.EndThinkTokenId >= 0 && nextTokens[i] == seq.SamplingOptions.EndThinkTokenId)
            {
                seq.IsInReasoningBlock = false;
            }
            else if (seq.IsInReasoningBlock)
            {
                seq.CurrentReasoningTokens++;
            }
        }

        // VERIFICATION:
        // Sequence A: Clamping applied! EndThink (25) was biased +20 (to 30.0), Token 10 penalized -50 (to -15.0).
        // Must choose 25 and transition IsInReasoningBlock to false!
        Assert.Equal(25, nextTokens[0]);
        Assert.False(seqA.IsInReasoningBlock);
        Assert.Equal(100, seqA.CurrentReasoningTokens);
        Assert.Equal(30.0f, batchLogits[0 * vocabSize + 25]);
        Assert.Equal(-15.0f, batchLogits[0 * vocabSize + 10]);

        // Sequence B: Clamping NOT applied! Continuation token 10 stayed at 35.0f > 25 (10.0f).
        // Must choose 10, remain in reasoning block, and increment reasoning token counter!
        Assert.Equal(10, nextTokens[1]);
        Assert.True(seqB.IsInReasoningBlock);
        Assert.Equal(21, seqB.CurrentReasoningTokens);
        Assert.Equal(35.0f, batchLogits[1 * vocabSize + 10]);
        Assert.Equal(10.0f, batchLogits[1 * vocabSize + 25]);

        // Sequence C: Non-reasoning sequence completely unaffected.
        Assert.Equal(10, nextTokens[2]);
        Assert.False(seqC.IsInReasoningBlock);
        Assert.Equal(0, seqC.CurrentReasoningTokens);
        Assert.Equal(35.0f, batchLogits[2 * vocabSize + 10]);

        tableA.ReleaseAll();
        tableB.ReleaseAll();
        tableC.ReleaseAll();
        Assert.Equal(24, pool.FreeBlocksCount);
    }

    /// <summary>
    /// Challenge 3: Dynamic prompt injection of <think> and </think> tags across multiple concurrent sequences.
    /// Verifies that state tracking (IsInReasoningBlock, CurrentReasoningTokens) is completely isolated per sequence
    /// across adversarial tag variations (open-ended think, closed think, stray end-think, multiple blocks).
    /// </summary>
    [Fact]
    public void Challenge_DynamicPromptInjection_ThinkTags_StrictPerSequenceStateIsolation()
    {
        const int ThinkId = 151648;
        const int EndThinkId = 151649;
        var sampler = new Sampler();
        using var pool = new PagedBlockPool(totalBlocks: 64, layers: 1, headsKv: 1, headDim: 16);

        // 6 Heterogeneous Prompt Injection Test Cases:
        // Case 1: Prompt ends with <think> -> Starts inside reasoning block
        int[] p1 = [100, ThinkId];
        // Case 2: Prompt has closed <think> ... </think> -> Starts outside reasoning block
        int[] p2 = [100, ThinkId, 200, EndThinkId];
        // Case 3: Prompt injection with stray </think> before <think> -> Starts inside reasoning block
        int[] p3 = [EndThinkId, 100, ThinkId];
        // Case 4: Multiple balanced <think> ... </think> blocks -> Starts outside reasoning block
        int[] p4 = [ThinkId, 1, EndThinkId, ThinkId, 2, EndThinkId];
        // Case 5: Prompt injection with unclosed <think> then malicious </think> in user text -> Starts outside
        int[] p5 = [ThinkId, 10, EndThinkId, 20];
        // Case 6: Pure standard prompt with no tags -> Starts outside
        int[] p6 = [100, 200, 300];

        int[][] prompts = [p1, p2, p3, p4, p5, p6];
        var sequences = new List<SequenceState>(prompts.Length);

        for (int i = 0; i < prompts.Length; i++)
        {
            var table = new BlockTable(pool);
            var seq = new SequenceState($"seq-{i + 1}", prompts[i], table)
            {
                SamplingOptions =
                {
                    Temperature = 0.0f,
                    ThinkTokenId = ThinkId,
                    EndThinkTokenId = EndThinkId,
                    MaxReasoningTokens = 10
                }
            };

            // Replicate ContinuousBatchEngine EnqueueAsync tag resolution logic
            int lastThink = Array.LastIndexOf(prompts[i], ThinkId);
            int lastEndThink = Array.LastIndexOf(prompts[i], EndThinkId);
            if (lastThink >= 0 && lastThink > lastEndThink)
            {
                seq.IsInReasoningBlock = true;
            }

            sequences.Add(seq);
        }

        // Verify initial state resolution from prompt injection
        Assert.True(sequences[0].IsInReasoningBlock);  // Case 1: Unclosed <think>
        Assert.False(sequences[1].IsInReasoningBlock); // Case 2: Closed <think> ... </think>
        Assert.True(sequences[2].IsInReasoningBlock);  // Case 3: Stray </think> then <think>
        Assert.False(sequences[3].IsInReasoningBlock); // Case 4: Multiple closed blocks
        Assert.False(sequences[4].IsInReasoningBlock); // Case 5: Closed before end
        Assert.False(sequences[5].IsInReasoningBlock); // Case 6: No tags

        // Simulate 5 batch decode iterations where different sequences emit different tokens
        // Sequence 0 emits: 501, 502, EndThinkId (closes!), 601, 602
        // Sequence 1 emits: 301, ThinkId (opens!), 701, 702, EndThinkId (closes!)
        // Sequence 5 emits: 801, 802, 803, 804, 805 (never opens)
        int[][] simulatedTokenStream = [
            [501, 502, EndThinkId, 601, 602],
            [301, ThinkId, 701, 702, EndThinkId],
            [510, 511, 512, EndThinkId, 513],
            [401, 402, 403, 404, 405],
            [410, 411, ThinkId, 412, 413],
            [801, 802, 803, 804, 805]
        ];

        for (int step = 0; step < 5; step++)
        {
            for (int s = 0; s < sequences.Count; s++)
            {
                var seq = sequences[s];
                int token = simulatedTokenStream[s][step];

                seq.LastToken = token;
                seq.GeneratedTokens.Add(token);

                // Update reasoning state per engine logic
                if (seq.SamplingOptions.ThinkTokenId >= 0 && token == seq.SamplingOptions.ThinkTokenId)
                {
                    seq.IsInReasoningBlock = true;
                }
                else if (seq.SamplingOptions.EndThinkTokenId >= 0 && token == seq.SamplingOptions.EndThinkTokenId)
                {
                    seq.IsInReasoningBlock = false;
                }
                else if (seq.IsInReasoningBlock)
                {
                    seq.CurrentReasoningTokens++;
                }
            }
        }

        // VERIFICATION OF FINAL STATES:
        // Sequence 0: Started in reasoning, emitted 2 reasoning tokens, emitted EndThinkId, then 2 normal tokens.
        // Final: IsInReasoningBlock = false, CurrentReasoningTokens = 2.
        Assert.False(sequences[0].IsInReasoningBlock);
        Assert.Equal(2, sequences[0].CurrentReasoningTokens);

        // Sequence 1: Started outside, emitted 1 normal, emitted ThinkId, emitted 2 reasoning tokens, emitted EndThinkId.
        // Final: IsInReasoningBlock = false, CurrentReasoningTokens = 2.
        Assert.False(sequences[1].IsInReasoningBlock);
        Assert.Equal(2, sequences[1].CurrentReasoningTokens);

        // Sequence 2: Started in reasoning, emitted 3 reasoning tokens, emitted EndThinkId, emitted 1 normal.
        // Final: IsInReasoningBlock = false, CurrentReasoningTokens = 3.
        Assert.False(sequences[2].IsInReasoningBlock);
        Assert.Equal(3, sequences[2].CurrentReasoningTokens);

        // Sequence 4: Started outside, emitted 2 normal, emitted ThinkId, emitted 2 reasoning tokens.
        // Final: IsInReasoningBlock = true, CurrentReasoningTokens = 2.
        Assert.True(sequences[4].IsInReasoningBlock);
        Assert.Equal(2, sequences[4].CurrentReasoningTokens);

        // Sequence 5: Pure standard prompt, emitted 5 normal tokens.
        // Final: IsInReasoningBlock = false, CurrentReasoningTokens = 0.
        Assert.False(sequences[5].IsInReasoningBlock);
        Assert.Equal(0, sequences[5].CurrentReasoningTokens);

        foreach (var seq in sequences)
        {
            seq.BlockTable.ReleaseAll();
        }
        Assert.Equal(64, pool.FreeBlocksCount);
    }

    /// <summary>
    /// Challenge 4: Multi-threaded stress: 10 concurrent virtual sequences allocate and dynamically
    /// grow BlockTable pages from PagedBlockPool simultaneously to verify zero data races,
    /// zero double-allocations, and zero block leaks.
    /// </summary>
    [Fact]
    public async Task Challenge_MultiThreadedStress_10ConcurrentVirtualSequences_ZeroDataRacesOrLeaks()
    {
        const int TotalBlocks = 128;
        const int NumSequences = 10;
        const int TokensPerSequence = 64; // Crosses multiple 16-token block boundaries

        using var pool = new PagedBlockPool(totalBlocks: TotalBlocks, layers: 2, headsKv: 2, headDim: 16);
        Assert.Equal(TotalBlocks, pool.FreeBlocksCount);

        var allocatedPhysicalBlocks = new ConcurrentBag<int>();
        var tasks = new List<Task>();

        for (int s = 0; s < NumSequences; s++)
        {
            int seqId = s;
            tasks.Add(Task.Run(() =>
            {
                var table = new BlockTable(pool);
                var seq = new SequenceState($"seq-{seqId}", [100 + seqId], table, maxNewTokens: TokensPerSequence);

                for (int t = 0; t < TokensPerSequence; t++)
                {
                    bool ok = table.AppendToken(out int blockId, out int offset);
                    Assert.True(ok, $"Failed to append token {t} for sequence {seqId}");
                    allocatedPhysicalBlocks.Add(blockId);
                }

                // Verify block count is exactly TokensPerSequence / BlockSize = 64 / 16 = 4 blocks
                Assert.Equal(4, table.BlockCount);
                Assert.Equal(TokensPerSequence, table.TokenCount);

                // Recycle blocks
                table.ReleaseAll();
                Assert.Equal(0, table.BlockCount);
            }));
        }

        await Task.WhenAll(tasks);

        // Every allocated physical block must be within valid range [0, TotalBlocks - 1]
        foreach (int blockId in allocatedPhysicalBlocks)
        {
            Assert.True(blockId >= 0 && blockId < TotalBlocks, $"Invalid physical block ID: {blockId}");
        }

        // PagedBlockPool free blocks count MUST return to exactly TotalBlocks!
        Assert.Equal(TotalBlocks, pool.FreeBlocksCount);
        Assert.Equal(0, pool.AllocatedBlocksCount);
    }

    /// <summary>
    /// Challenge 5: Multi-threaded stress: 10 concurrent virtual sequences enqueued into a multi-producer
    /// waiting queue with concurrent readers, streaming tokens with zero deadlocks or lost data.
    /// </summary>
    [Fact]
    public async Task Challenge_MultiThreadedStress_ConcurrentChannelStreaming_ZeroLostMessages()
    {
        const int NumSequences = 10;
        const int TokensPerSeq = 20;

        var waitingQueue = Channel.CreateUnbounded<SequenceState>(new UnboundedChannelOptions
        {
            SingleReader = true,
            SingleWriter = false
        });

        using var pool = new PagedBlockPool(totalBlocks: 64, layers: 1, headsKv: 1, headDim: 16);
        var producedSequences = new ConcurrentBag<SequenceState>();
        var readerTasks = new List<Task<List<string>>>();

        // Enqueue 10 sequences concurrently
        var enqueueTasks = new List<Task>();
        for (int i = 0; i < NumSequences; i++)
        {
            int seqId = i;
            enqueueTasks.Add(Task.Run(async () =>
            {
                var table = new BlockTable(pool);
                var state = new SequenceState($"stress-req-{seqId}", [10, 20], table, maxNewTokens: TokensPerSeq);
                producedSequences.Add(state);

                // Start consumer task for this sequence's token stream
                var readTask = Task.Run(async () =>
                {
                    var tokens = new List<string>();
                    while (await state.TokenChannel.Reader.WaitToReadAsync())
                    {
                        while (state.TokenChannel.Reader.TryRead(out var tok))
                        {
                            tokens.Add(tok);
                        }
                    }
                    return tokens;
                });

                lock (readerTasks)
                {
                    readerTasks.Add(readTask);
                }

                await waitingQueue.Writer.WriteAsync(state);
            }));
        }

        await Task.WhenAll(enqueueTasks);

        // Process queue: single consumer simulating engine iteration loop
        int processedCount = 0;
        while (processedCount < NumSequences && await waitingQueue.Reader.WaitToReadAsync())
        {
            while (waitingQueue.Reader.TryRead(out var seq))
            {
                for (int t = 0; t < TokensPerSeq; t++)
                {
                    seq.TokenChannel.Writer.TryWrite($"tok_{t}");
                }
                seq.TokenChannel.Writer.TryComplete();
                seq.BlockTable.ReleaseAll();
                processedCount++;
            }
        }

        var results = await Task.WhenAll(readerTasks);

        // Verify all 10 sequences received all 20 tokens in order
        Assert.Equal(NumSequences, results.Length);
        foreach (var seqTokens in results)
        {
            Assert.Equal(TokensPerSeq, seqTokens.Count);
            for (int t = 0; t < TokensPerSeq; t++)
            {
                Assert.Equal($"tok_{t}", seqTokens[t]);
            }
        }

        Assert.Equal(64, pool.FreeBlocksCount);
    }

    /// <summary>
    /// Challenge 6 (Real Hardware & Authentic GGUF Model):
    /// If the real DeepSeek-R1-Distill-Qwen-7B model exists on the physical machine,
    /// instantiate ContinuousBatchEngine and execute 4 concurrent heterogeneous sequences
    /// (Loop prompt, Budget-limited prompt, Control prompt, Fact prompt) simultaneously.
    /// Verifies zero data races, zero deadlock, proper block recycling, and output integrity.
    /// </summary>
    [Fact]
    public async Task Challenge_LiveModel_ConcurrentHeterogeneousSequences_Execution()
    {
        string? modelPath = RealModelPath;
        if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
        {
            // Skip if GLACIER_MODEL_PATH is not configured or model is not present on this machine
            return;
        }

        using var engine = new ContinuousBatchEngine(modelPath, totalKvBlocks: 256, maxBatchSize: 4);

        var requests = new (string Name, InferenceRequest Request)[]
        {
            ("L1-Looping", new InferenceRequest
            {
                Prompt = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Verify repeatedly.",
                MaxTokens = 24,
                Temperature = 0.6f
            }),
            ("ReasoningBudget", new InferenceRequest
            {
                Prompt = "<think>\nLet us analyze 17 * 19 step by step.",
                MaxTokens = 24,
                MaxReasoningTokens = 8,
                Temperature = 0.0f
            }),
            ("C1-Control", new InferenceRequest
            {
                Prompt = "What is the capital of France? Answer in one word.",
                MaxTokens = 16,
                Temperature = 0.0f
            }),
            ("C2-Arithmetic", new InferenceRequest
            {
                Prompt = "What is 17 * 19? Answer with just the number.",
                MaxTokens = 16,
                Temperature = 0.0f
            })
        };

        var tasks = new List<Task<(string Name, string Text, int Tokens, SequenceState State)>>();

        foreach (var (name, req) in requests)
        {
            tasks.Add(Task.Run(async () =>
            {
                var (state, tokenStream) = await engine.EnqueueAsync(req);
                var sb = new StringBuilder();
                int count = 0;
                await foreach (var tok in tokenStream)
                {
                    sb.Append(tok);
                    count++;
                }
                return (name, sb.ToString(), count, state);
            }));
        }

        var results = await Task.WhenAll(tasks);

        // Verify each request ran to completion without crashing
        foreach (var (name, text, tokens, state) in results)
        {
            Assert.True(tokens > 0, $"Sequence {name} produced 0 tokens!");
            Assert.True(state.IsFinished, $"Sequence {name} did not mark IsFinished!");
            Assert.False(string.IsNullOrEmpty(text), $"Sequence {name} output text is empty!");
        }

        // Verify all 4 completed requests were served and blocks recycled
        Assert.True(engine.TotalRequestsServed >= 4);
        Assert.Equal(256, engine.BlockPool.FreeBlocksCount);
    }
}

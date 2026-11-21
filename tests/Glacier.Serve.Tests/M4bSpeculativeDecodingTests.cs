using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Inference.PagedAttention;
using Xunit;

namespace Glacier.Serve.Tests;

public sealed class M4bSpeculativeDecodingTests
{
    [Fact]
    public void BlockTable_TruncateTo_ReleasesTrailingBlocksAndMaintainsPrefixConsistency()
    {
        using var pool = new PagedBlockPool(8, layers: 1, headsKv: 1, headDim: 16, blockSize: 4);
        var bt = new BlockTable(pool);

        // Append 10 tokens => 3 blocks (4 + 4 + 2)
        for (int i = 0; i < 10; i++)
        {
            Assert.True(bt.AppendToken(out _, out _));
        }
        Assert.Equal(10, bt.TokenCount);
        Assert.Equal(3, bt.BlockCount);
        Assert.Equal(5, pool.FreeBlocksCount);

        // Bind block 1 in prefix hash map and verify TruncateTo(4) evicts blocks 1 & 2
        ulong h1 = PagedBlockPool.ComputePrefixHash(1UL, new[] { 10, 11, 12, 13 });
        pool.BindPrefixBlock(h1, bt.GetBlockId(1));
        Assert.True(pool.TryFindPrefixBlock(h1, out int foundId));
        pool.ReleaseBlock(foundId); // refCount back to 1

        // Truncate within block 2 (9 tokens still needs 3 blocks)
        bt.TruncateTo(9);
        Assert.Equal(9, bt.TokenCount);
        Assert.Equal(3, bt.BlockCount);
        Assert.Equal(5, pool.FreeBlocksCount);

        // Truncate to 4 tokens => releases blocks 2 and 1 (and evicts h1 from prefix cache)
        bt.TruncateTo(4);
        Assert.Equal(4, bt.TokenCount);
        Assert.Equal(1, bt.BlockCount);
        Assert.Equal(7, pool.FreeBlocksCount);
        Assert.False(pool.TryFindPrefixBlock(h1, out _));

        // Bounds checks
        Assert.Throws<ArgumentOutOfRangeException>(() => bt.TruncateTo(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => bt.TruncateTo(5));

        bt.ReleaseAll();
        Assert.Equal(8, pool.FreeBlocksCount);
    }

    [Theory]
    [InlineData(KvCachePrecision.Fp32, false)]
    [InlineData(KvCachePrecision.Q8_0, false)]
    [InlineData(KvCachePrecision.Q4_0, false)]
    [InlineData(KvCachePrecision.Fp32, true)]
    public void PagedTransformerRunner_VerifyDraftChunk_MatchesForwardToken_AndRollsBackOnOom(
        KvCachePrecision precision,
        bool withLora)
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"m4b_verify_cpu_{Guid.NewGuid():N}");
        try
        {
            string ggufPath = M4bTestHelpers.CreateSyntheticDenseGguf(tempDir, dim: 64, headDim: 16);
            using var gguf = GgufFile.Open(ggufPath);
            using var weights = new ModelWeights(gguf);
            LoraAdapterWeights? lora = withLora
                ? LoraAdapterWeights.Load(M4bTestHelpers.CreateSyntheticLoraFile(tempDir))
                : null;

            using var runner = new PagedTransformerRunner(weights, maxSeqLen: 64, maxBatchSize: 8);
            using var poolRef = new PagedBlockPool(16, 2, 2, 16, blockSize: 4, kvPrecision: precision);
            using var poolVer = new PagedBlockPool(16, 2, 2, 16, blockSize: 4, kvPrecision: precision);

            int[] prefix = [1, 8, 9];
            int[] chunk = [10, 11, 12, 13, 14];
            var btRef = new BlockTable(poolRef);
            var btVer = new BlockTable(poolVer);

            var scratch = new float[weights.VocabSize];
            Assert.True(runner.ForwardPrefillChunk(prefix, 0, btRef, poolRef, scratch, computeLogits: true, lora));
            Assert.True(runner.ForwardPrefillChunk(prefix, 0, btVer, poolVer, scratch, computeLogits: true, lora));

            var refLogits = new float[chunk.Length * weights.VocabSize];
            for (int t = 0; t < chunk.Length; t++)
            {
                Assert.True(runner.ForwardToken(
                    chunk[t],
                    prefix.Length + t,
                    btRef,
                    poolRef,
                    refLogits.AsSpan(t * weights.VocabSize, weights.VocabSize),
                    computeLogits: true,
                    lora));
            }

            var verLogits = new float[chunk.Length * weights.VocabSize];
            Assert.True(runner.VerifyDraftChunk(chunk, prefix.Length, btVer, poolVer, verLogits, lora));

            Assert.Equal(btRef.TokenCount, btVer.TokenCount);
            Assert.True(M4bTestHelpers.MaxAbsDiff(refLogits, verLogits) < 1e-4f);

            // Rollback on OOM: only 1 block free, verifying 6 tokens across block boundary must roll back
            using var tinyPool = new PagedBlockPool(1, 2, 2, 16, blockSize: 4, kvPrecision: precision);
            var btTiny = new BlockTable(tinyPool);
            Assert.True(runner.ForwardPrefillChunk([1, 8, 9], 0, btTiny, tinyPool, scratch, computeLogits: true, lora));
            Assert.Equal(3, btTiny.TokenCount);
            Assert.Equal(0, tinyPool.FreeBlocksCount);

            var oomLogits = new float[3 * weights.VocabSize];
            Assert.False(runner.VerifyDraftChunk([10, 11, 12], 3, btTiny, tinyPool, oomLogits, lora));
            Assert.Equal(3, btTiny.TokenCount);
            Assert.Equal(1, btTiny.BlockCount);
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void CudaPagedTransformerRunner_VerifyDraftChunk_And_HiddenStateCaptureSemantics()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"m4b_verify_cuda_{Guid.NewGuid():N}");
        try
        {
            string ggufPath = M4bTestHelpers.CreateSyntheticDenseGguf(tempDir, dim: 64, headDim: 16);
            using var gguf = GgufFile.Open(ggufPath);
            using var weights = new ModelWeights(gguf);

            if (!CudaPagedTransformerRunner.TryCreate(weights, 16, 4, 64, 8, KvCachePrecision.Q4_0, out var cudaRunner))
            {
                return; // CUDA GPU unavailable on host
            }

            using (cudaRunner)
            {
                // Pure CUDA must report SupportsHiddenStateCapture == false and never return zeroed hidden states
                Assert.False(cudaRunner.SupportsHiddenStateCapture);
                Assert.True(cudaRunner.GetLastHiddenState().IsEmpty);
                var hidDst = new float[weights.EmbeddingLength];
                Assert.False(cudaRunner.TryGetHiddenStateAt(0, hidDst));

                using var pool = new PagedBlockPool(16, 2, 2, 16, blockSize: 4, allocateHostBuffers: false, kvPrecision: KvCachePrecision.Q4_0);
                var btRef = new BlockTable(pool);
                var btVer = new BlockTable(pool);
                int[] prefix = [1, 8, 9];
                int[] chunk = [10, 11, 12];
                var scratch = new float[weights.VocabSize];

                Assert.True(cudaRunner.ForwardPrefillChunk(prefix, 0, btRef, pool, scratch));
                Assert.True(cudaRunner.ForwardPrefillChunk(prefix, 0, btVer, pool, scratch));

                var refLogits = new float[chunk.Length * weights.VocabSize];
                for (int t = 0; t < chunk.Length; t++)
                {
                    Assert.True(cudaRunner.ForwardToken(chunk[t], prefix.Length + t, btRef, pool, refLogits.AsSpan(t * weights.VocabSize, weights.VocabSize)));
                }

                var verLogits = new float[chunk.Length * weights.VocabSize];
                Assert.True(cudaRunner.VerifyDraftChunk(chunk, prefix.Length, btVer, pool, verLogits));
                Assert.True(M4bTestHelpers.MaxAbsDiff(refLogits, verLogits) < 1e-3f);

                // LoRA fallback enables CPU hidden state capture and VerifyDraftChunk parity
                var lora = LoraAdapterWeights.Load(M4bTestHelpers.CreateSyntheticLoraFile(tempDir));
                cudaRunner.EnablePositionHistory();
                var btLora = new BlockTable(pool);
                Assert.True(cudaRunner.ForwardPrefillChunk(prefix, 0, btLora, pool, scratch, computeLogits: true, lora: lora));
                Assert.True(cudaRunner.SupportsHiddenStateCapture);
                Assert.True(cudaRunner.TryGetHiddenStateAt(0, hidDst));
                Assert.Contains(hidDst, v => v != 0f);
                Assert.True(cudaRunner.VerifyDraftChunk(chunk, prefix.Length, btLora, pool, verLogits, lora));
            }
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void PromptLookup_And_CostGatedDraftPolicy_SelectBestRouteAndRecordOutcome()
    {
        var lookup = new PromptLookupDraftProvider(minMatchLength: 2, maxMatchLength: 4);
        int[] tokens = [8, 9, 10, 11, 12, 8, 9];
        Span<int> drafts = stackalloc int[4];
        int count = lookup.DraftWithMatchLength(tokens, 4, drafts, out int matchLen);
        Assert.Equal(2, matchLen);
        Assert.Equal(3, count);
        Assert.Equal(new[] { 10, 11, 12 }, drafts[..count].ToArray());

        var policy = new CostGatedDraftPolicy();
        var serialDecision = policy.SelectRoute(maxDraftTokens: 4, hasMtp: false, lookupCandidateCount: 0, matchedNgramLength: 0);
        Assert.Equal(DraftRoute.Serial, serialDecision.Route);
        Assert.Equal(0, serialDecision.DraftCount);

        var lookupDecision = policy.SelectRoute(maxDraftTokens: 4, hasMtp: false, lookupCandidateCount: 3, matchedNgramLength: 2);
        Assert.Equal(DraftRoute.PromptLookup, lookupDecision.Route);
        Assert.True(lookupDecision.DraftCount >= 1);

        // Repeated rejections drive acceptance estimate down until policy falls back to Serial
        for (int r = 0; r < 25; r++)
        {
            policy.RecordOutcome(DraftRoute.PromptLookup, matchedNgramLength: 2, draftedCount: 3, acceptedCount: 0, roundElapsedMs: 16.0);
        }
        var degradedDecision = policy.SelectRoute(maxDraftTokens: 4, hasMtp: false, lookupCandidateCount: 3, matchedNgramLength: 2);
        Assert.Equal(DraftRoute.Serial, degradedDecision.Route);
    }

    [Theory]
    [InlineData(0.0f)]
    [InlineData(0.8f)]
    public async Task ContinuousBatchEngine_PromptLookupSpeculation_MatchesSerialTokensWithSameSeed(float temperature)
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"m4b_spec_lookup_{Guid.NewGuid():N}");
        try
        {
            string ggufPath = M4bTestHelpers.CreateSyntheticDenseGguf(
                tempDir,
                dim: 64,
                headDim: 16,
                maxSeqLen: 128,
                seed: 21,
                dominantTokenId: 10); // token 'a'

            // Repeated 'a' in prompt guarantees PromptLookup finds 2..4-gram matches and accepts drafts
            const string prompt = "aaaaaa";

            using var serialEngine = new ContinuousBatchEngine(
                ggufPath,
                totalKvBlocks: 32,
                maxBatchSize: 4,
                maxSeqLen: 128,
                seed: 555,
                useGpu: false,
                enableSpeculativeDecoding: false);

            using var specEngine = new ContinuousBatchEngine(
                ggufPath,
                totalKvBlocks: 32,
                maxBatchSize: 4,
                maxSeqLen: 128,
                seed: 555,
                useGpu: false,
                enableSpeculativeDecoding: true,
                maxDraftTokens: 4,
                speculativeBatchThreshold: 4);

            var (serialState, serialStream) = await serialEngine.EnqueueAsync(new InferenceRequest
            {
                Prompt = prompt,
                MaxTokens = 10,
                Temperature = temperature,
                TopP = 1.0f,
                RepetitionPenalty = 1.0f,
                Seed = 777
            });
            await foreach (var _ in serialStream) { }
            await serialState.Completion.Task;

            var (specState, specStream) = await specEngine.EnqueueAsync(new InferenceRequest
            {
                Prompt = prompt,
                MaxTokens = 10,
                Temperature = temperature,
                TopP = 1.0f,
                RepetitionPenalty = 1.0f,
                Seed = 777
            });
            await foreach (var _ in specStream) { }
            await specState.Completion.Task;

            Assert.Equal(0L, serialEngine.SpeculativeSteps);
            Assert.Equal(10, serialState.GeneratedTokens.Count);
            Assert.Equal(10, specState.GeneratedTokens.Count);

            if (temperature == 0.0f)
            {
                Assert.Equal(serialState.GeneratedTokens, specState.GeneratedTokens);
                Assert.True(specEngine.SpeculativeSteps > 0);
                Assert.True(specEngine.SpeculativePromptLookupRounds > 0);
                Assert.True(specEngine.SpeculativeDraftTokensAccepted > 0);
            }
            else
            {
                // Stochastic speculative verification draws valid samples (acceptance/correction/bonus)
                Assert.True(specEngine.SpeculativeSteps > 0);
                Assert.True(specEngine.SpeculativePromptLookupRounds > 0);
                Assert.All(specState.GeneratedTokens, tok => Assert.InRange(tok, 1, 95));
            }
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ContinuousBatchEngine_MtpEmbeddedAndSidecar_DraftsAndMatchesSerialGreedyOutput(bool useSidecar)
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"m4b_spec_mtp_{Guid.NewGuid():N}");
        try
        {
            string ggufPath = M4bTestHelpers.CreateSyntheticDenseGguf(
                tempDir,
                fileName: useSidecar ? "base.gguf" : "embedded_mtp.gguf",
                dim: 64,
                ffnDim: 128,
                nHeads: 4,
                nHeadsKv: 2,
                headDim: 16,
                layers: 2,
                vocabSize: 96,
                maxSeqLen: 128,
                includeEmbeddedMtp: !useSidecar,
                seed: 31);

            string? sidecarPath = useSidecar
                ? M4bTestHelpers.CreateSyntheticMtpSidecarGguf(tempDir, dim: 64, ffnDim: 128, nHeads: 4, nHeadsKv: 2, headDim: 16, vocabSize: 96, mtpBlockIdx: 2)
                : null;

            const string prompt = "hello alpha beta gamma";

            using var serialEngine = new ContinuousBatchEngine(
                ggufPath,
                totalKvBlocks: 32,
                maxBatchSize: 4,
                maxSeqLen: 128,
                seed: 101,
                useGpu: false,
                enableSpeculativeDecoding: false);

            using var mtpEngine = new ContinuousBatchEngine(
                ggufPath,
                totalKvBlocks: 32,
                maxBatchSize: 4,
                maxSeqLen: 128,
                seed: 101,
                useGpu: false,
                enableSpeculativeDecoding: true,
                maxDraftTokens: 3,
                speculativeBatchThreshold: 4,
                mtpSidecarPath: sidecarPath);

            var (serialState, serialStream) = await serialEngine.EnqueueAsync(new InferenceRequest
            {
                Prompt = prompt,
                MaxTokens = 8,
                Temperature = 0.0f,
                RepetitionPenalty = 1.0f
            });
            await foreach (var _ in serialStream) { }
            await serialState.Completion.Task;

            var (mtpState, mtpStream) = await mtpEngine.EnqueueAsync(new InferenceRequest
            {
                Prompt = prompt,
                MaxTokens = 8,
                Temperature = 0.0f,
                RepetitionPenalty = 1.0f
            });
            await foreach (var _ in mtpStream) { }
            await mtpState.Completion.Task;

            Assert.Equal(8, serialState.GeneratedTokens.Count);
            Assert.Equal(serialState.GeneratedTokens, mtpState.GeneratedTokens);
            Assert.True(mtpEngine.SpeculativeSteps > 0);
            Assert.True(mtpEngine.SpeculativeMtpRounds > 0);
            Assert.True(mtpEngine.SpeculativeDraftTokensProposed > 0);

            // Second sequential request reuses pooled MtpDraftProvider cleanly without double-free or stale state
            var (mtpState2, mtpStream2) = await mtpEngine.EnqueueAsync(new InferenceRequest
            {
                Prompt = "hello delta ok",
                MaxTokens = 6,
                Temperature = 0.0f,
                RepetitionPenalty = 1.0f
            });
            await foreach (var _ in mtpStream2) { }
            await mtpState2.Completion.Task;
            Assert.Equal(6, mtpState2.GeneratedTokens.Count);
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public async Task ContinuousBatchEngine_ActiveBatchSizeThreshold_GatesOffAbove4_AndAutoEnablesAtLe4()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"m4b_spec_gate_{Guid.NewGuid():N}");
        try
        {
            string ggufPath = M4bTestHelpers.CreateSyntheticDenseGguf(
                tempDir,
                dim: 64,
                ffnDim: 128,
                nHeads: 4,
                nHeadsKv: 2,
                headDim: 16,
                layers: 2,
                vocabSize: 96,
                maxSeqLen: 128,
                includeEmbeddedMtp: true,
                seed: 44);

            using var highBatchEngine = new ContinuousBatchEngine(
                ggufPath,
                totalKvBlocks: 128,
                maxBatchSize: 8,
                maxSeqLen: 128,
                seed: 44,
                useGpu: false,
                prefillChunkSize: 16,
                enableSpeculativeDecoding: true,
                maxDraftTokens: 3,
                speculativeBatchThreshold: 4);

            // 6 concurrent sequences (> 4 threshold): must gate off speculative decoding while > 4 active
            using var cancelTwo = new CancellationTokenSource();
            var streams = new IAsyncEnumerator<string>[6];
            var tasks = new Task<string>[6];
            for (int i = 0; i < 6; i++)
            {
                var ct = i < 2 ? cancelTwo.Token : CancellationToken.None;
                var (st, str) = await highBatchEngine.EnqueueAsync(new InferenceRequest
                {
                    Prompt = $"hello world alpha tok{16 + i}",
                    MaxTokens = 20,
                    Temperature = 0.0f,
                    RepetitionPenalty = 1.0f
                }, ct);
                streams[i] = str.GetAsyncEnumerator();
                tasks[i] = st.Completion.Task;
            }

            // Wait until all 6 sequences (including #5) have emitted their first token while none have finished (MaxTokens=20)
            for (int i = 0; i < 6; i++)
            {
                Assert.True(await streams[i].MoveNextAsync());
            }

            Assert.True(highBatchEngine.ActiveBatchSize > 4);
            Assert.False(highBatchEngine.IsSpeculativeDecodingActive);
            Assert.Equal(0L, highBatchEngine.SpeculativeSteps);
            Assert.Equal(0L, highBatchEngine.SpeculativeMtpRounds);

            // Cancel 2 sequences so active concurrency drops from 6 -> 4 (<= threshold), auto-enabling speculation
            cancelTwo.Cancel();
            for (int i = 0; i < 2; i++)
            {
                await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
                {
                    while (await streams[i].MoveNextAsync()) { }
                    await tasks[i];
                });
                await streams[i].DisposeAsync();
            }

            for (int i = 2; i < 6; i++)
            {
                while (await streams[i].MoveNextAsync()) { }
                await streams[i].DisposeAsync();
                await tasks[i];
            }

            Assert.True(highBatchEngine.SpeculativeSteps > 0);
            Assert.True(highBatchEngine.SpeculativeMtpRounds > 0);
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public async Task ContinuousBatchEngine_SpeculativeDecoding_RespectsReasoningBudgetForcedClose_StopSequence_AndMaxCompletionTokens()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"m4b_spec_bounds_{Guid.NewGuid():N}");
        try
        {
            string ggufPath = M4bTestHelpers.CreateSyntheticDenseGguf(
                tempDir,
                dim: 64,
                ffnDim: 128,
                nHeads: 4,
                nHeadsKv: 2,
                headDim: 16,
                layers: 2,
                vocabSize: 96,
                maxSeqLen: 128,
                includeEmbeddedMtp: true,
                seed: 55);

            using var engine = new ContinuousBatchEngine(
                ggufPath,
                totalKvBlocks: 64,
                maxBatchSize: 4,
                maxSeqLen: 128,
                seed: 55,
                useGpu: false,
                enableSpeculativeDecoding: true,
                maxDraftTokens: 3,
                speculativeBatchThreshold: 4);

            // 1. Reasoning budget forced close inside <think> under active MTP speculation
            var (thinkState, thinkStream) = await engine.EnqueueAsync(new InferenceRequest
            {
                Prompt = "<think>hello alpha",
                MaxCompletionTokens = 12,
                MaxThinkTokens = 3,
                Temperature = 0.0f,
                RepetitionPenalty = 1.0f
            });
            await foreach (var _ in thinkStream) { }
            string thinkOut = await thinkState.Completion.Task;
            Assert.True(thinkState.ForcedThinkClose);
            Assert.Contains("</think>", thinkOut, StringComparison.Ordinal);
            Assert.True(thinkState.GeneratedTokens.Count <= 12);

            // 2. MaxCompletionTokens exact cap
            var (capState, capStream) = await engine.EnqueueAsync(new InferenceRequest
            {
                Prompt = "hello alpha beta",
                MaxCompletionTokens = 5,
                Temperature = 0.0f,
                RepetitionPenalty = 1.0f
            });
            await foreach (var _ in capStream) { }
            await capState.Completion.Task;
            Assert.Equal(5, capState.GeneratedTokens.Count);
            Assert.Equal("length", capState.FinishReason);

            // 3. Stop sequence termination cleanly truncates and finishes
            var (serialRefEngine, serialStopOut, stopSeqText) = await RunSerialToDiscoverStopToken(ggufPath);
            using (serialRefEngine)
            {
                var (stopState, stopStream) = await engine.EnqueueAsync(new InferenceRequest
                {
                    Prompt = "hello alpha beta",
                    MaxCompletionTokens = 10,
                    Temperature = 0.0f,
                    RepetitionPenalty = 1.0f,
                    Stop = new[] { stopSeqText }
                });
                await foreach (var _ in stopStream) { }
                string actualStopOut = await stopState.Completion.Task;
                Assert.Equal("stop", stopState.FinishReason);
                Assert.DoesNotContain(stopSeqText, actualStopOut, StringComparison.Ordinal);
                Assert.True(actualStopOut.Length < serialStopOut.Length);
            }
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }

    private static async Task<(ContinuousBatchEngine Engine, string FullOutput, string StopCandidate)> RunSerialToDiscoverStopToken(string ggufPath)
    {
        var serialEngine = new ContinuousBatchEngine(
            ggufPath,
            totalKvBlocks: 32,
            maxBatchSize: 2,
            maxSeqLen: 128,
            seed: 55,
            useGpu: false,
            enableSpeculativeDecoding: false);

        var (st, stream) = await serialEngine.EnqueueAsync(new InferenceRequest
        {
            Prompt = "hello alpha beta",
            MaxCompletionTokens = 10,
            Temperature = 0.0f,
            RepetitionPenalty = 1.0f
        });
        await foreach (var _ in stream) { }
        string full = await st.Completion.Task;
        int targetIdx = Math.Min(3, st.GeneratedTokens.Count - 1);
        string stopCandidate = serialEngine.Tokenizer.Decode(new[] { st.GeneratedTokens[targetIdx] });
        return (serialEngine, full, stopCandidate);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ContinuousBatchEngine_EnqueueAsync_PreCanceledToken_DoesNotLeakPrefixBlocksOrHiddenState(bool useMtpSidecar)
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"m4b_precancel_{Guid.NewGuid():N}");
        try
        {
            string ggufPath = M4bTestHelpers.CreateSyntheticDenseGguf(
                tempDir,
                dim: 64,
                ffnDim: 128,
                nHeads: 4,
                nHeadsKv: 2,
                headDim: 16,
                layers: 2,
                vocabSize: 96,
                maxSeqLen: 128,
                seed: 61);

            string? sidecarPath = useMtpSidecar
                ? M4bTestHelpers.CreateSyntheticMtpSidecarGguf(tempDir, dim: 64, ffnDim: 128, nHeads: 4, nHeadsKv: 2, headDim: 16, vocabSize: 96, mtpBlockIdx: 2)
                : null;

            using var engine = new ContinuousBatchEngine(
                ggufPath,
                totalKvBlocks: 16,
                maxBatchSize: 4,
                maxSeqLen: 128,
                seed: 61,
                useGpu: false,
                enableSpeculativeDecoding: useMtpSidecar,
                maxDraftTokens: 3,
                speculativeBatchThreshold: 4,
                mtpSidecarPath: sidecarPath);

            // >= 16 tokens guarantees >= 1 full KV block (BlockSize=16) is bound into the prefix cache
            const string longPrefixPrompt = "abcdefghijklmnopqrstuvwxyz0123456789";
            Assert.True(engine.Tokenizer.Encode(longPrefixPrompt).Length >= 16);

            var (req1State, req1Stream) = await engine.EnqueueAsync(new InferenceRequest
            {
                Prompt = longPrefixPrompt,
                MaxTokens = 6,
                Temperature = 0.0f,
                RepetitionPenalty = 1.0f
            });

            await using var req1Enum = req1Stream.GetAsyncEnumerator();
            Assert.True(await req1Enum.MoveNextAsync());

            using var preCanceledCts = new CancellationTokenSource();
            preCanceledCts.Cancel();

            await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
                engine.EnqueueAsync(new InferenceRequest
                {
                    Prompt = longPrefixPrompt,
                    MaxTokens = 6,
                    Temperature = 0.0f,
                    RepetitionPenalty = 1.0f
                }, preCanceledCts.Token));

            while (await req1Enum.MoveNextAsync()) { }
            await req1State.Completion.Task;

            Assert.Equal(engine.BlockPool.TotalBlocks, engine.BlockPool.FreeBlocksCount);
            if (useMtpSidecar)
            {
                Assert.True(engine.HiddenHistoryPoolCount >= 1);
            }
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }

    [Theory]
    [InlineData(KvCachePrecision.Q4_0)]
    [InlineData(KvCachePrecision.Q8_0)]
    public async Task ContinuousBatchEngine_ThreeWay_MtpSidecar_QuantizedKv_Lora_And_ConcurrencyTransition(KvCachePrecision kvPrecision)
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"m4b_3way_{kvPrecision}_{Guid.NewGuid():N}");
        try
        {
            string ggufPath = M4bTestHelpers.CreateSyntheticDenseGguf(
                tempDir,
                fileName: "base.gguf",
                dim: 64,
                ffnDim: 128,
                nHeads: 4,
                nHeadsKv: 2,
                headDim: 16,
                layers: 2,
                vocabSize: 96,
                maxSeqLen: 128,
                includeEmbeddedMtp: false,
                seed: 73);

            string sidecarPath = M4bTestHelpers.CreateSyntheticMtpSidecarGguf(
                tempDir,
                dim: 64,
                ffnDim: 128,
                nHeads: 4,
                nHeadsKv: 2,
                headDim: 16,
                vocabSize: 96,
                mtpBlockIdx: 2,
                seed: 91);

            string loraPath = M4bTestHelpers.CreateSyntheticLoraFile(
                tempDir,
                layerCount: 2,
                rank: 4,
                alpha: 512.0f,
                dim: 64,
                qDim: 64,
                seed: 177);

            using var specEngine = new ContinuousBatchEngine(
                ggufPath,
                totalKvBlocks: 128,
                maxBatchSize: 8,
                maxSeqLen: 128,
                seed: 73,
                useGpu: false,
                prefillChunkSize: 16,
                kvCachePrecision: kvPrecision,
                enableSpeculativeDecoding: true,
                maxDraftTokens: 3,
                speculativeBatchThreshold: 4,
                mtpSidecarPath: sidecarPath);

            using var serialEngine = new ContinuousBatchEngine(
                ggufPath,
                totalKvBlocks: 64,
                maxBatchSize: 4,
                maxSeqLen: 128,
                seed: 73,
                useGpu: false,
                kvCachePrecision: kvPrecision,
                enableSpeculativeDecoding: false);

            specEngine.LoadLoraAdapter("coder", loraPath);
            serialEngine.LoadLoraAdapter("coder", loraPath);
            Assert.Equal(kvPrecision, specEngine.KvPrecision);

            // Wave 1: > 4 concurrent sequences (mix of base + LoRA) gates speculative decoding OFF,
            // then cancelling 2 drops concurrency from 6 -> 4 (<= 4) and re-enables speculation.
            using var cancelTwo = new CancellationTokenSource();
            var streams = new IAsyncEnumerator<string>[6];
            var tasks = new Task<string>[6];
            for (int i = 0; i < 6; i++)
            {
                var ct = i < 2 ? cancelTwo.Token : CancellationToken.None;
                string modelName = (i % 2 == 0) ? "default" : "default:coder";
                var (st, str) = await specEngine.EnqueueAsync(new InferenceRequest
                {
                    Model = modelName,
                    Prompt = $"hello world alpha tok{16 + i}",
                    MaxTokens = 16,
                    Temperature = 0.0f,
                    RepetitionPenalty = 1.0f
                }, "default", ct);
                streams[i] = str.GetAsyncEnumerator();
                tasks[i] = st.Completion.Task;
            }

            for (int i = 0; i < 6; i++)
            {
                Assert.True(await streams[i].MoveNextAsync());
            }

            Assert.True(specEngine.ActiveBatchSize > 4);
            Assert.False(specEngine.IsSpeculativeDecodingActive);
            Assert.Equal(0L, specEngine.SpeculativeSteps);

            cancelTwo.Cancel();
            for (int i = 0; i < 2; i++)
            {
                await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
                {
                    while (await streams[i].MoveNextAsync()) { }
                    await tasks[i];
                });
                await streams[i].DisposeAsync();
            }

            for (int i = 2; i < 6; i++)
            {
                while (await streams[i].MoveNextAsync()) { }
                await streams[i].DisposeAsync();
                await tasks[i];
            }

            Assert.True(specEngine.SpeculativeSteps > 0);
            Assert.True(specEngine.SpeculativeMtpRounds > 0);
            Assert.Equal(specEngine.BlockPool.TotalBlocks, specEngine.BlockPool.FreeBlocksCount);

            // Wave 2: sequential verification of base vs LoRA greedy token parity against serial baseline
            static async Task<string> RunOne(ContinuousBatchEngine eng, string model)
            {
                var (st, str) = await eng.EnqueueAsync(new InferenceRequest
                {
                    Model = model,
                    Prompt = "hello alpha beta",
                    MaxTokens = 8,
                    Temperature = 0.0f,
                    RepetitionPenalty = 1.0f
                }, "default");
                await foreach (var _ in str) { }
                return await st.Completion.Task;
            }

            string specBase = await RunOne(specEngine, "default");
            string serialBase = await RunOne(serialEngine, "default");
            string specLora = await RunOne(specEngine, "default:coder");
            string serialLora = await RunOne(serialEngine, "default:coder");

            Assert.Equal(serialBase, specBase);
            Assert.Equal(serialLora, specLora);
            Assert.NotEqual(specBase, specLora);
            Assert.Equal(specEngine.BlockPool.TotalBlocks, specEngine.BlockPool.FreeBlocksCount);
            Assert.Equal(serialEngine.BlockPool.TotalBlocks, serialEngine.BlockPool.FreeBlocksCount);
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }
}

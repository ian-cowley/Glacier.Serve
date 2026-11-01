namespace Glacier.Serve.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Pipelines;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Model;
using Glacier.Serve.Core;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Inference.Http;
using Glacier.Serve.Inference.PagedAttention;
using Glacier.Serve.Routing;
using Glacier.Serve.Serialization;
using Glacier.Serve.Server;
using Glacier.Tensor.Core;
using Xunit;
using ServeHttpMethod = Glacier.Serve.Routing.HttpMethod;

public sealed class PhaseCServeRemediationTests : IDisposable
{
    private readonly List<string> _tempFiles = [];

    public void Dispose()
    {
        foreach (string path in _tempFiles)
        {
            try
            {
                if (File.Exists(path))
                    File.Delete(path);
            }
            catch
            {
                // Best effort cleanup if Windows MMF handle release lags briefly
            }
        }
    }

    private string AllocateTempGgufPath(string tag)
    {
        string path = Path.Combine(Path.GetTempPath(), $"glacier_serve_phasec_{tag}_{Guid.NewGuid():N}.gguf");
        _tempFiles.Add(path);
        return path;
    }

    private static int GetAvailablePort()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    // =========================================================================
    // 1. PagedBlockPool Hardening (64-Byte Alignment & Double-Free Idempotency)
    // =========================================================================

    [Fact]
    public void PagedBlockPool_64ByteAlignedAndDoubleFreeIdempotent()
    {
        using var pool = new PagedBlockPool(totalBlocks: 8, layers: 2, headsKv: 2, headDim: 32, blockSize: 16);

        Assert.Equal(0L, pool.KeyBufferAddress % 64);
        Assert.Equal(0L, pool.ValueBufferAddress % 64);
        Assert.Equal(8, pool.FreeBlocksCount);

        Assert.True(pool.TryAllocateBlock(out int b0));
        Assert.True(pool.TryAllocateBlock(out int b1));
        Assert.True(b0 >= 0 && b1 >= 0 && b0 != b1);
        Assert.Equal(6, pool.FreeBlocksCount);

        // Double-free and out-of-state free must be strictly idempotent
        pool.FreeBlock(b0);
        Assert.Equal(7, pool.FreeBlocksCount);

        pool.FreeBlock(b0);
        Assert.Equal(7, pool.FreeBlocksCount);

        // Freeing unallocated block IDs must never corrupt the free stack
        for (int candidate = 0; candidate < pool.TotalBlocks; candidate++)
        {
            if (candidate != b1)
            {
                pool.FreeBlock(candidate);
            }
        }
        Assert.Equal(7, pool.FreeBlocksCount);

        // Exhaust all remaining blocks and assert every allocated block ID is unique
        var allocated = new HashSet<int> { b1 };
        for (int i = 0; i < 7; i++)
        {
            Assert.True(pool.TryAllocateBlock(out int next));
            Assert.True(next >= 0);
            Assert.True(allocated.Add(next));
        }
        Assert.Equal(0, pool.FreeBlocksCount);
        Assert.False(pool.TryAllocateBlock(out int exhausted));
        Assert.Equal(-1, exhausted);
    }

    // =========================================================================
    // 2. Sequence Length Bounds Enforcement (Kernel, Runner, Engine & HTTP 400)
    // =========================================================================

    [Fact]
    public unsafe void PagedAttentionKernel_RejectsOutOfBoundsPosition()
    {
        using var pool = new PagedBlockPool(totalBlocks: 4, layers: 1, headsKv: 1, headDim: 32, blockSize: 16);
        var bt = new BlockTable(pool);
        Assert.True(bt.AppendToken(out _, out _));

        float[] q = new float[32];
        float[] output = new float[32];
        float[] scores = new float[16];

        Assert.Throws<ArgumentOutOfRangeException>(() =>
        {
            fixed (float* qPtr = q)
            fixed (float* outPtr = output)
            fixed (float* scPtr = scores)
            {
                PagedAttentionKernel.ComputeAttention(
                    stageLayer: 0,
                    modelLayer: 0,
                    pos: 16,
                    qBase: qPtr,
                    outBase: outPtr,
                    blockTable: bt,
                    pool: pool,
                    nHeads: 1,
                    nHeadsKv: 1,
                    headDim: 32,
                    vHeadDim: 32,
                    attnScale: 1.0f / MathF.Sqrt(32f),
                    headScores: scPtr,
                    maxSeqLen: 16);
            }
        });

        Assert.Throws<ArgumentOutOfRangeException>(() =>
        {
            fixed (float* qPtr = q)
            fixed (float* outPtr = output)
            fixed (float* scPtr = scores)
            {
                PagedAttentionKernel.ComputeAttention(
                    stageLayer: 0,
                    modelLayer: 0,
                    pos: -1,
                    qBase: qPtr,
                    outBase: outPtr,
                    blockTable: bt,
                    pool: pool,
                    nHeads: 1,
                    nHeadsKv: 1,
                    headDim: 32,
                    vHeadDim: 32,
                    attnScale: 1.0f / MathF.Sqrt(32f),
                    headScores: scPtr,
                    maxSeqLen: 16);
            }
        });
    }

    [Fact]
    public async Task SequenceBounds_EngineRejectsOverflow_AutoClampsMaxTokens_AndHttpReturns400()
    {
        string ggufPath = AllocateTempGgufPath("bounds");
        int tokA = SyntheticGgufBuilder.GetByteTokenId((byte)'A');
        SyntheticGgufBuilder.WriteModel(
            ggufPath,
            includeThinkTokens: true,
            useQ8Weights: true,
            dim: 64,
            ffnDim: 128,
            nHeads: 2,
            nHeadsKv: 1,
            headDim: 32,
            vHeadDim: 32,
            forcedTargetTokenIds: [tokA]);

        using var engine = new ContinuousBatchEngine(
            ggufPath,
            totalKvBlocks: 16,
            maxBatchSize: 4,
            maxSeqLen: 24,
            useGpu: false);

        // 1. Empty prompt throws ArgumentOutOfRangeException
        var emptyReq = new InferenceRequest { Prompt = "" };
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
            engine.EnqueueAsync(emptyReq));

        // 2. Prompt length >= maxSeqLen throws ArgumentOutOfRangeException
        var overflowReq = new InferenceRequest { Prompt = new string('A', 28) };
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
            engine.EnqueueAsync(overflowReq));

        // 3. Valid prompt ("AAAA" = 4 tokens) with MaxTokens = 10000 auto-clamps cleanly at maxSeqLen (20 new tokens) without overrun
        var clampReq = new InferenceRequest
        {
            Prompt = "AAAA",
            MaxTokens = 10000,
            Temperature = 0f
        };
        var (clampState, _) = await engine.EnqueueAsync(clampReq);
        await clampState.Completion.Task.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Equal("length", clampState.FinishReason);
        Assert.Equal(20, clampState.GeneratedTokens.Count);
        Assert.Equal(engine.BlockPool.TotalBlocks, engine.BlockPool.FreeBlocksCount);

        // 4. HTTP endpoints return 400 Bad Request when prompt exceeds context window
        int port = GetAvailablePort();
        using var app = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .Build();
        app.MapOpenAiEndpoints(engine);
        app.MapOllamaEndpoints(engine);
        await app.StartAsync();

        using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };
        string hugePrompt = new string('A', 64); // 64 byte-tokens > maxSeqLen (24)

        var rChat = await client.PostAsync(
            "/v1/chat/completions",
            new StringContent($"{{\"messages\":[{{\"role\":\"user\",\"content\":\"{hugePrompt}\"}}],\"stream\":false}}", Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.BadRequest, rChat.StatusCode);

        var rComp = await client.PostAsync(
            "/v1/completions",
            new StringContent($"{{\"prompt\":\"{hugePrompt}\",\"stream\":false}}", Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.BadRequest, rComp.StatusCode);

        var rOllama = await client.PostAsync(
            "/api/generate",
            new StringContent($"{{\"prompt\":\"{hugePrompt}\",\"stream\":false}}", Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.BadRequest, rOllama.StatusCode);
    }

    // =========================================================================
    // 3. PagedTransformerRunner Stride / QK-Norm & CUDA Paged Parity
    // =========================================================================

    [Fact]
    public void PagedTransformerRunner_AsymmetricVHeadDimAndQkNorm_SinglePrefillAndBatchParity()
    {
        string ggufPath = AllocateTempGgufPath("asymmetric_qknorm");
        SyntheticGgufBuilder.WriteModel(
            ggufPath,
            includeThinkTokens: true,
            useQ8Weights: false, // F32 weights
            dim: 64,
            ffnDim: 128,
            nHeads: 2,
            nHeadsKv: 1,
            headDim: 32,
            vHeadDim: 64, // Asymmetric ValueDim != HeadDim
            includeQkNorm: true);

        using var gguf = GgufFile.Open(ggufPath);
        var weights = new ModelWeights(gguf);
        Assert.Equal(32, weights.HeadDim);
        Assert.Equal(64, weights.ValueDim);

        using var poolA = new PagedBlockPool(totalBlocks: 8, layers: weights.BlockCount, headsKv: weights.HeadCountKv, headDim: 32, blockSize: 4, vHeadDim: 64);
        using var poolB = new PagedBlockPool(totalBlocks: 8, layers: weights.BlockCount, headsKv: weights.HeadCountKv, headDim: 32, blockSize: 4, vHeadDim: 64);

        using var runnerA = new PagedTransformerRunner(weights, maxSeqLen: 32, maxBatchSize: 8);
        using var runnerB = new PagedTransformerRunner(weights, maxSeqLen: 32, maxBatchSize: 8);

        var btA = new BlockTable(poolA);
        var btB = new BlockTable(poolB);

        int[] prompt = [10, 20, 30, 40, 50, 60]; // 6 tokens crossing block boundary (blockSize=4)
        float[] logitsA = new float[weights.VocabSize];
        float[] logitsB = new float[weights.VocabSize];

        // Runner A: token-by-token ForwardToken
        for (int i = 0; i < prompt.Length; i++)
        {
            Assert.True(runnerA.ForwardToken(prompt[i], i, btA, poolA, logitsA, computeLogits: i == prompt.Length - 1));
        }

        // Runner B: batched ForwardPrefillChunk (chunk 1: 4 tokens, chunk 2: 2 tokens)
        Assert.True(runnerB.ForwardPrefillChunk(prompt.AsSpan(0, 4), startPos: 0, btB, poolB, Span<float>.Empty, computeLogits: false));
        Assert.True(runnerB.ForwardPrefillChunk(prompt.AsSpan(4, 2), startPos: 4, btB, poolB, logitsB, computeLogits: true));

        for (int v = 0; v < weights.VocabSize; v++)
        {
            Assert.Equal(logitsA[v], logitsB[v], precision: 4);
        }

        // Verify ForwardBatch matches ForwardToken at pos = 6
        bool[] okFlags = new bool[1];
        Assert.True(runnerA.ForwardToken(70, 6, btA, poolA, logitsA, computeLogits: true));
        Assert.True(runnerB.ForwardBatch([70], [6], [btB], poolB, logitsB, okFlags, computeLogits: true));
        Assert.True(okFlags[0]);

        for (int v = 0; v < weights.VocabSize; v++)
        {
            Assert.Equal(logitsA[v], logitsB[v], precision: 4);
        }

        // Verify pos >= maxSeqLen throws ArgumentOutOfRangeException
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            runnerA.ForwardToken(70, 32, btA, poolA, logitsA));
    }

    [Fact]
    public void CudaPagedTransformerRunner_GpuVsCpuParity_ChunkedPrefillAndMultiSeqBatchDecode()
    {
        if (!GpuContext.IsSupported)
        {
            return;
        }

        string ggufPath = AllocateTempGgufPath("cuda_parity");
        SyntheticGgufBuilder.WriteModel(
            ggufPath,
            includeThinkTokens: true,
            useQ8Weights: true,
            dim: 64,
            ffnDim: 128,
            nHeads: 2,
            nHeadsKv: 1,
            headDim: 32,
            vHeadDim: 32,
            includeQkNorm: true);

        using var gguf = GgufFile.Open(ggufPath);
        var weights = new ModelWeights(gguf);

        const int totalBlocks = 16;
        const int blockSize = 8;
        const int maxSeqLen = 64;

        bool created = CudaPagedTransformerRunner.TryCreate(
            weights,
            totalBlocks: totalBlocks,
            blockSize: blockSize,
            maxSeqLen: maxSeqLen,
            maxBatchSize: 16,
            out var cudaRunner);

        Assert.True(created, "Expected CudaPagedTransformerRunner.TryCreate to succeed on CUDA-capable machine.");
        Assert.NotNull(cudaRunner);

        using (cudaRunner!)
        {
            Assert.True(cudaRunner.IsGpuAccelerated);

            using var cpuPool = new PagedBlockPool(totalBlocks, weights.BlockCount, weights.HeadCountKv, weights.HeadDim, blockSize);
            using var gpuPool = new PagedBlockPool(totalBlocks, weights.BlockCount, weights.HeadCountKv, weights.HeadDim, blockSize);
            using var cpuRunner = new PagedTransformerRunner(weights, maxSeqLen, maxBatchSize: 16);

            var cpuBt0 = new BlockTable(cpuPool);
            var gpuBt0 = new BlockTable(gpuPool);
            var cpuBt1 = new BlockTable(cpuPool);
            var gpuBt1 = new BlockTable(gpuPool);

            // Seq 0: 20-token prompt crossing 3 blocks (blockSize = 8); Seq 1: 10-token prompt crossing 2 blocks
            int[] prompt0 = Enumerable.Range(10, 20).ToArray();
            int[] prompt1 = Enumerable.Range(40, 10).ToArray();

            float[] cpuLogits0 = new float[weights.VocabSize];
            float[] gpuLogits0 = new float[weights.VocabSize];
            float[] cpuLogits1 = new float[weights.VocabSize];
            float[] gpuLogits1 = new float[weights.VocabSize];

            // Multi-chunk prefill on Seq 0 (chunks of 7, 7, 6)
            int pos = 0;
            int[] chunks = [7, 7, 6];
            for (int c = 0; c < chunks.Length; c++)
            {
                int chunkLen = chunks[c];
                bool isLast = c == chunks.Length - 1;
                Assert.True(cpuRunner.ForwardPrefillChunk(prompt0.AsSpan(pos, chunkLen), pos, cpuBt0, cpuPool, cpuLogits0, computeLogits: isLast));
                Assert.True(cudaRunner.ForwardPrefillChunk(prompt0.AsSpan(pos, chunkLen), pos, gpuBt0, gpuPool, gpuLogits0, computeLogits: isLast));
                pos += chunkLen;
            }

            // Single-chunk prefill on Seq 1 (10 tokens)
            Assert.True(cpuRunner.ForwardPrefillChunk(prompt1, 0, cpuBt1, cpuPool, cpuLogits1, computeLogits: true));
            Assert.True(cudaRunner.ForwardPrefillChunk(prompt1, 0, gpuBt1, gpuPool, gpuLogits1, computeLogits: true));

            AssertLogitsClose(cpuLogits0, gpuLogits0, tolerance: 2.5e-2f);
            AssertLogitsClose(cpuLogits1, gpuLogits1, tolerance: 2.5e-2f);

            int nextTok0 = ArgMax(gpuLogits0);
            int nextTok1 = ArgMax(gpuLogits1);
            Assert.Equal(ArgMax(cpuLogits0), nextTok0);
            Assert.Equal(ArgMax(cpuLogits1), nextTok1);

            // Multi-sequence concurrent ForwardBatch decode step
            float[] cpuBatchLogits = new float[2 * weights.VocabSize];
            float[] gpuBatchLogits = new float[2 * weights.VocabSize];
            bool[] cpuOk = new bool[2];
            bool[] gpuOk = new bool[2];

            Assert.True(cpuRunner.ForwardBatch([nextTok0, nextTok1], [20, 10], [cpuBt0, cpuBt1], cpuPool, cpuBatchLogits, cpuOk, computeLogits: true));
            Assert.True(cudaRunner.ForwardBatch([nextTok0, nextTok1], [20, 10], [gpuBt0, gpuBt1], gpuPool, gpuBatchLogits, gpuOk, computeLogits: true));
            Assert.True(cpuOk[0] && cpuOk[1] && gpuOk[0] && gpuOk[1]);

            AssertLogitsClose(cpuBatchLogits.AsSpan(0, weights.VocabSize), gpuBatchLogits.AsSpan(0, weights.VocabSize), tolerance: 2.5e-2f);
            AssertLogitsClose(cpuBatchLogits.AsSpan(weights.VocabSize, weights.VocabSize), gpuBatchLogits.AsSpan(weights.VocabSize, weights.VocabSize), tolerance: 2.5e-2f);
            Assert.Equal(ArgMax(cpuBatchLogits.AsSpan(0, weights.VocabSize)), ArgMax(gpuBatchLogits.AsSpan(0, weights.VocabSize)));
            Assert.Equal(ArgMax(cpuBatchLogits.AsSpan(weights.VocabSize, weights.VocabSize)), ArgMax(gpuBatchLogits.AsSpan(weights.VocabSize, weights.VocabSize)));
        }
    }

    // =========================================================================
    // 4. ContinuousBatchEngine Correctness (REC-1 Guard, EOS, UTF-8 Emoji, Stop, Cancel)
    // =========================================================================

    [Fact]
    public async Task ContinuousBatchEngine_NonThinkingModel_NeverActivatesReasoningLoopGuard()
    {
        string ggufPath = AllocateTempGgufPath("non_thinking");
        // Omit <think> and </think> from vocabulary -> ThinkTokenId == -1
        // Bias output head strongly toward token 'A' so generation emits a pure periodic "AAAA..." stream
        int targetTokenId = SyntheticGgufBuilder.GetByteTokenId((byte)'A');
        SyntheticGgufBuilder.WriteModel(
            ggufPath,
            includeThinkTokens: false,
            useQ8Weights: false,
            forcedTargetTokenIds: [targetTokenId]);

        using var engine = new ContinuousBatchEngine(
            ggufPath,
            totalKvBlocks: 16,
            maxBatchSize: 4,
            maxSeqLen: 128,
            useGpu: false);

        var req = new InferenceRequest
        {
            Prompt = "Hello",
            MaxTokens = 40,
            Temperature = 0f,
            EnableReasoningLoopGuard = true
        };

        var (seq, _) = await engine.EnqueueAsync(req);
        Assert.Equal(-1, seq.SamplingOptions.ThinkTokenId);

        string text = await seq.Completion.Task.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Equal(new string('A', 40), text);
        Assert.Equal(0, seq.PeriodicLoopTriggers);
        Assert.Equal(0, seq.LoopRecoveryTriggers);
    }

    [Fact]
    public async Task ContinuousBatchEngine_EosNonLeakage_NeverAppendsOrEmitsEosToken()
    {
        string ggufPath = AllocateTempGgufPath("eos_noleak");
        int eosId = SyntheticGgufBuilder.EosTokenIdConst;
        SyntheticGgufBuilder.WriteModel(
            ggufPath,
            includeThinkTokens: true,
            useQ8Weights: false,
            forcedTargetTokenIds: [eosId]);

        using var engine = new ContinuousBatchEngine(
            ggufPath,
            totalKvBlocks: 16,
            maxBatchSize: 4,
            maxSeqLen: 64,
            useGpu: false);

        var req = new InferenceRequest
        {
            Prompt = "Hi",
            MaxTokens = 16,
            Temperature = 0f
        };

        var (seq, tokenStream) = await engine.EnqueueAsync(req);
        var streamedPieces = new List<string>();
        await foreach (string piece in tokenStream)
        {
            streamedPieces.Add(piece);
        }

        string finalOutput = await seq.Completion.Task.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Equal("stop", seq.FinishReason);
        Assert.DoesNotContain(eosId, seq.GeneratedTokens);
        Assert.Equal(string.Empty, finalOutput);
        Assert.DoesNotContain("<|im_end|>", string.Concat(streamedPieces));
    }

    [Fact]
    public async Task ContinuousBatchEngine_StatefulUtf8Decoder_Reassembles4ByteEmojiAcrossTokens()
    {
        string ggufPath = AllocateTempGgufPath("utf8_emoji");
        // "🚀" in UTF-8 is 4 separate bytes: 0xF0, 0x9F, 0x9A, 0x80.
        byte[] rocketBytes = Encoding.UTF8.GetBytes("🚀");
        int t0 = SyntheticGgufBuilder.GetByteTokenId(rocketBytes[0]);
        int t1 = SyntheticGgufBuilder.GetByteTokenId(rocketBytes[1]);
        int t2 = SyntheticGgufBuilder.GetByteTokenId(rocketBytes[2]);
        int t3 = SyntheticGgufBuilder.GetByteTokenId(rocketBytes[3]);
        int eos = SyntheticGgufBuilder.EosTokenIdConst;
        int promptTok = SyntheticGgufBuilder.GetByteTokenId((byte)'!');

        // 0-layer transition model: '!' -> 0xF0 -> 0x9F -> 0x9A -> 0x80 -> EOS
        SyntheticGgufBuilder.WriteTransitionChainModel(
            ggufPath,
            transitions: new Dictionary<int, int>
            {
                [promptTok] = t0,
                [t0] = t1,
                [t1] = t2,
                [t2] = t3,
                [t3] = eos
            });

        using var engine = new ContinuousBatchEngine(
            ggufPath,
            totalKvBlocks: 16,
            maxBatchSize: 4,
            maxSeqLen: 64,
            useGpu: false);

        var req = new InferenceRequest
        {
            Prompt = "!",
            MaxTokens = 10,
            Temperature = 0f
        };

        var (seq, tokenStream) = await engine.EnqueueAsync(req);
        var chunks = new List<string>();
        await foreach (string piece in tokenStream)
        {
            Assert.DoesNotContain('\uFFFD', piece);
            chunks.Add(piece);
        }

        string completed = await seq.Completion.Task.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Equal("stop", seq.FinishReason);
        Assert.Equal([t0, t1, t2, t3], seq.GeneratedTokens);
        Assert.Equal("🚀", completed);
        Assert.Equal("🚀", string.Concat(chunks));
    }

    [Fact]
    public async Task ContinuousBatchEngine_StopSequences_HoldbackAndTruncateCleanly()
    {
        string ggufPath = AllocateTempGgufPath("stop_seq");
        int p = SyntheticGgufBuilder.GetByteTokenId((byte)'!');
        int h = SyntheticGgufBuilder.GetByteTokenId((byte)'H');
        int i = SyntheticGgufBuilder.GetByteTokenId((byte)'i');
        int lt = SyntheticGgufBuilder.GetByteTokenId((byte)'<');
        int e = SyntheticGgufBuilder.GetByteTokenId((byte)'E');
        int n = SyntheticGgufBuilder.GetByteTokenId((byte)'N');
        int d = SyntheticGgufBuilder.GetByteTokenId((byte)'D');
        int gt = SyntheticGgufBuilder.GetByteTokenId((byte)'>');
        int x = SyntheticGgufBuilder.GetByteTokenId((byte)'X');

        SyntheticGgufBuilder.WriteTransitionChainModel(
            ggufPath,
            transitions: new Dictionary<int, int>
            {
                [p] = h,
                [h] = i,
                [i] = lt,
                [lt] = e,
                [e] = n,
                [n] = d,
                [d] = gt,
                [gt] = x,
                [x] = SyntheticGgufBuilder.EosTokenIdConst
            });

        using var engine = new ContinuousBatchEngine(
            ggufPath,
            totalKvBlocks: 16,
            maxBatchSize: 4,
            maxSeqLen: 64,
            useGpu: false);

        var req = new InferenceRequest
        {
            Prompt = "!",
            MaxTokens = 16,
            Temperature = 0f,
            Stop = ["<END>"]
        };

        var (seq, tokenStream) = await engine.EnqueueAsync(req);
        var streamed = new List<string>();
        await foreach (string piece in tokenStream)
        {
            streamed.Add(piece);
        }

        string fullText = await seq.Completion.Task.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Equal("stop", seq.FinishReason);
        Assert.Equal("Hi", fullText);
        Assert.Equal("Hi", string.Concat(streamed));
        Assert.DoesNotContain("<END>", fullText);
    }

    [Fact]
    public async Task ContinuousBatchEngine_CancellationToken_ImmediatelyFreesKvBlocks()
    {
        string ggufPath = AllocateTempGgufPath("cancel_blocks");
        int tokA = SyntheticGgufBuilder.GetByteTokenId((byte)'A');
        SyntheticGgufBuilder.WriteModel(
            ggufPath,
            includeThinkTokens: true,
            useQ8Weights: false,
            forcedTargetTokenIds: [tokA]);

        using var engine = new ContinuousBatchEngine(
            ggufPath,
            totalKvBlocks: 16,
            maxBatchSize: 4,
            maxSeqLen: 128,
            useGpu: false);

        using var cts = new CancellationTokenSource();
        var req = new InferenceRequest
        {
            Prompt = "Generate many tokens",
            MaxTokens = 100,
            Temperature = 0f
        };

        var (seq, tokenStream) = await engine.EnqueueAsync(req, cts.Token);

        await using (var enumerator = tokenStream.GetAsyncEnumerator())
        {
            Assert.True(await enumerator.MoveNextAsync());
            // Cancel mid-generation while sequence holds KV blocks!
            await cts.CancelAsync();
        }

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => seq.Completion.Task.WaitAsync(TimeSpan.FromSeconds(5)));

        for (int spin = 0; spin < 50 && engine.BlockPool.FreeBlocksCount < engine.BlockPool.TotalBlocks; spin++)
        {
            await Task.Delay(10);
        }

        Assert.Equal(engine.BlockPool.TotalBlocks, engine.BlockPool.FreeBlocksCount);
        Assert.Equal(0, engine.ActiveBatchSize);
    }

    // =========================================================================
    // 5. OpenAI & Ollama Wire Contracts (snake_case, /v1/completions, SSE, NDJSON vs JSON)
    // =========================================================================

    [Fact]
    public async Task OpenAiAndOllama_WireContracts_SnakeCaseCompletionsSseAndStreamToggle()
    {
        string ggufPath = AllocateTempGgufPath("wire_contracts");
        int p = SyntheticGgufBuilder.GetByteTokenId((byte)'Q');
        int o = SyntheticGgufBuilder.GetByteTokenId((byte)'O');
        int k = SyntheticGgufBuilder.GetByteTokenId((byte)'K');
        int s = SyntheticGgufBuilder.GetByteTokenId((byte)'#');

        SyntheticGgufBuilder.WriteTransitionChainModel(
            ggufPath,
            transitions: new Dictionary<int, int>
            {
                [p] = o,
                [o] = k,
                [k] = s,
                [s] = SyntheticGgufBuilder.EosTokenIdConst
            },
            defaultTargetTokenId: o);

        using var engine = new ContinuousBatchEngine(
            ggufPath,
            totalKvBlocks: 32,
            maxBatchSize: 8,
            maxSeqLen: 64,
            useGpu: false);

        int port = GetAvailablePort();
        using var app = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .Build();
        app.MapOpenAiEndpoints(engine);
        app.MapOllamaEndpoints(engine);
        await app.StartAsync();

        using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

        // 1. OpenAI Request Deserialization: snake_case, polymorphic stop string & array, seed, top_k, repetition_penalty
        string singleStopJson = """
            {"model":"test","prompt":"Q","max_tokens":5,"top_p":0.9,"top_k":20,"repetition_penalty":1.15,"seed":42,"stop":"#","stream":false}
            """;
        var parsedSingle = JsonSerializer.Deserialize(singleStopJson, ServeJsonContext.Default.InferenceRequest);
        Assert.NotNull(parsedSingle);
        Assert.Equal(5, parsedSingle!.EffectiveMaxTokens);
        Assert.Equal(0.9f, parsedSingle.EffectiveTopP);
        Assert.Equal(20, parsedSingle.EffectiveTopK);
        Assert.Equal(1.15f, parsedSingle.EffectiveRepetitionPenalty);
        Assert.Equal(42, parsedSingle.EffectiveSeed);
        Assert.NotNull(parsedSingle.EffectiveStop);
        Assert.Equal(["#"], parsedSingle.EffectiveStop!);

        string arrayStopJson = """
            {"model":"test","prompt":"Q","max_completion_tokens":7,"stop":["#","END"],"stream":false}
            """;
        var parsedArray = JsonSerializer.Deserialize(arrayStopJson, ServeJsonContext.Default.InferenceRequest);
        Assert.NotNull(parsedArray);
        Assert.Equal(7, parsedArray!.EffectiveMaxTokens);
        Assert.NotNull(parsedArray.EffectiveStop);
        Assert.Equal(["#", "END"], parsedArray.EffectiveStop!);

        // 2. POST /v1/completions (non-streaming)
        var compRes = await client.PostAsync(
            "/v1/completions",
            new StringContent("""{"model":"test","prompt":"Q","max_tokens":5,"temperature":0.0,"stop":"#","stream":false}""", Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.OK, compRes.StatusCode);
        string compBody = await compRes.Content.ReadAsStringAsync();
        using (var doc = JsonDocument.Parse(compBody))
        {
            Assert.Equal("text_completion", doc.RootElement.GetProperty("object").GetString());
            var choice0 = doc.RootElement.GetProperty("choices")[0];
            Assert.Equal("OK", choice0.GetProperty("text").GetString());
            Assert.Equal("stop", choice0.GetProperty("finish_reason").GetString());
        }

        // 3. POST /v1/completions (streaming SSE with terminal finish_reason chunk before [DONE])
        var compStreamRes = await client.PostAsync(
            "/v1/completions",
            new StringContent("""{"model":"test","prompt":"Q","max_tokens":5,"temperature":0.0,"stop":["#"],"stream":true}""", Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.OK, compStreamRes.StatusCode);
        string compSse = await compStreamRes.Content.ReadAsStringAsync();
        string[] sseLines = compSse.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        Assert.Equal("data: [DONE]", sseLines[^1]);
        Assert.StartsWith("data: ", sseLines[^2]);
        using (var finalChunkDoc = JsonDocument.Parse(sseLines[^2]["data: ".Length..]))
        {
            Assert.Equal("stop", finalChunkDoc.RootElement.GetProperty("choices")[0].GetProperty("finish_reason").GetString());
        }

        // 4. POST /v1/chat/completions (streaming SSE with terminal finish_reason chunk before [DONE])
        var chatStreamRes = await client.PostAsync(
            "/v1/chat/completions",
            new StringContent("""{"model":"test","messages":[{"role":"user","content":"Q"}],"max_tokens":3,"temperature":0.0,"stream":true}""", Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.OK, chatStreamRes.StatusCode);
        string chatSse = await chatStreamRes.Content.ReadAsStringAsync();
        string[] chatLines = chatSse.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        Assert.Equal("data: [DONE]", chatLines[^1]);
        using (var finalChatDoc = JsonDocument.Parse(chatLines[^2]["data: ".Length..]))
        {
            Assert.Equal("chat.completion.chunk", finalChatDoc.RootElement.GetProperty("object").GetString());
            Assert.NotEqual(JsonValueKind.Null, finalChatDoc.RootElement.GetProperty("choices")[0].GetProperty("finish_reason").ValueKind);
        }

        // 5. Ollama POST /api/generate with stream: false -> single JSON object with done: true & options respected
        var ollamaSyncRes = await client.PostAsync(
            "/api/generate",
            new StringContent("""{"model":"test","prompt":"Q","stream":false,"options":{"num_predict":5,"temperature":0.0,"stop":["#"]}}""", Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.OK, ollamaSyncRes.StatusCode);
        Assert.Contains("application/json", ollamaSyncRes.Content.Headers.ContentType?.ToString() ?? "");
        string ollamaSyncBody = await ollamaSyncRes.Content.ReadAsStringAsync();
        Assert.DoesNotContain("\n{", ollamaSyncBody.Trim());
        using (var ollamaDoc = JsonDocument.Parse(ollamaSyncBody))
        {
            Assert.True(ollamaDoc.RootElement.GetProperty("done").GetBoolean());
            Assert.Equal("OK", ollamaDoc.RootElement.GetProperty("response").GetString());
        }

        // 6. Ollama POST /api/generate with omitted stream (defaults to true) -> application/x-ndjson
        var ollamaStreamRes = await client.PostAsync(
            "/api/generate",
            new StringContent("""{"model":"test","prompt":"Q","options":{"num_predict":5,"temperature":0.0,"stop":["#"]}}""", Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.OK, ollamaStreamRes.StatusCode);
        Assert.Contains("application/x-ndjson", ollamaStreamRes.Content.Headers.ContentType?.ToString() ?? "");
        string ollamaNdjson = await ollamaStreamRes.Content.ReadAsStringAsync();
        string[] ndjsonLines = ollamaNdjson.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        Assert.True(ndjsonLines.Length >= 2);
        using (var lastNdjson = JsonDocument.Parse(ndjsonLines[^1]))
        {
            Assert.True(lastNdjson.RootElement.GetProperty("done").GetBoolean());
        }
    }

    // =========================================================================
    // 6. UTF-8 RadixTreeRouter, QueryString & 413 Payload Too Large Guard
    // =========================================================================

    [Fact]
    public void RadixTreeRouter_CompressedUtf8SpanPrefixParamsAndQueryStrip()
    {
        var router = new RadixTreeRouter();
        RequestDelegate hModels = _ => ValueTask.CompletedTask;
        RequestDelegate hModelById = _ => ValueTask.CompletedTask;
        RequestDelegate hChat = _ => ValueTask.CompletedTask;
        RequestDelegate hColonUser = _ => ValueTask.CompletedTask;

        router.AddRoute(ServeHttpMethod.Get, "/v1/models", hModels);
        router.AddRoute(ServeHttpMethod.Get, "/v1/models/{modelId}", hModelById);
        router.AddRoute(ServeHttpMethod.Post, "/v1/chat/completions", hChat);
        router.AddRoute(ServeHttpMethod.Get, "/org/:orgId/users/{userId}/profile", hColonUser);

        Assert.True(router.TryMatch(ServeHttpMethod.Get, "/v1/models?limit=10&q=qwen"u8, out var match1));
        Assert.Same(hModels, match1.Handler);
        Assert.True(match1.Parameters == null || match1.Parameters.Length == 0);

        Assert.True(router.TryMatch(ServeHttpMethod.Get, "/v1/models/qwen3-8b?verbose=true"u8, out var match2));
        Assert.Same(hModelById, match2.Handler);
        Assert.Equal("qwen3-8b", match2.GetParam("modelId"));

        Assert.True(router.TryMatch(ServeHttpMethod.Get, "/org/glacier/users/42/profile?tab=keys"u8, out var match3));
        Assert.Same(hColonUser, match3.Handler);
        Assert.Equal("glacier", match3.GetParam("orgId"));
        Assert.Equal("42", match3.GetParam("userId"));

        Assert.False(router.TryMatch(ServeHttpMethod.Post, "/v1/models"u8, out _));
        Assert.False(router.TryMatch(ServeHttpMethod.Get, "/v1/unknown"u8, out _));
    }

    [Fact]
    public async Task GlacierServeApp_QueryParsing_ChunkedUpload_And413PayloadTooLarge()
    {
        int port = GetAvailablePort();
        using var app = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .UseMaxRequestBodySize(1024)
            .MapGet("/search", async ctx =>
            {
                string q = ctx.Request.Query("q") ?? "";
                string limit = ctx.Request.Query("limit") ?? "";
                await ctx.Response.WriteUtf8Async($"path={ctx.Request.Path};qs={ctx.Request.QueryString};q={q};limit={limit}");
            })
            .MapPost("/echo", async ctx =>
            {
                await ctx.Response.WriteBytesAsync(ctx.Request.Body);
            })
            .Build();

        await app.StartAsync();

        using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

        // 1. Query string extraction + router path isolation
        var qRes = await client.GetAsync("/search?q=paged+attention&limit=25");
        Assert.Equal(HttpStatusCode.OK, qRes.StatusCode);
        string qText = await qRes.Content.ReadAsStringAsync();
        Assert.Equal("path=/search;qs=?q=paged+attention&limit=25;q=paged attention;limit=25", qText);

        // 2. Valid <= 1024-byte Content-Length body succeeds
        byte[] okPayload = Encoding.UTF8.GetBytes(new string('Z', 512));
        var okRes = await client.PostAsync("/echo", new ByteArrayContent(okPayload));
        Assert.Equal(HttpStatusCode.OK, okRes.StatusCode);
        Assert.Equal(okPayload, await okRes.Content.ReadAsByteArrayAsync());

        // 3. Oversized Content-Length (> 1024 bytes) -> 413 Payload Too Large
        byte[] bigPayload = Encoding.UTF8.GetBytes(new string('X', 2048));
        var tooLargeRes = await client.PostAsync("/echo", new ByteArrayContent(bigPayload));
        Assert.Equal(HttpStatusCode.RequestEntityTooLarge, tooLargeRes.StatusCode);

        // 4. Valid chunked transfer encoding succeeds
        var chunkedOkReq = new HttpRequestMessage(System.Net.Http.HttpMethod.Post, "/echo")
        {
            Content = new StreamContent(new MemoryStream(okPayload))
        };
        chunkedOkReq.Headers.TransferEncodingChunked = true;
        var chunkedOkRes = await client.SendAsync(chunkedOkReq);
        Assert.Equal(HttpStatusCode.OK, chunkedOkRes.StatusCode);
        Assert.Equal(okPayload, await chunkedOkRes.Content.ReadAsByteArrayAsync());

        // 5. Oversized chunked transfer encoding (> 1024 bytes) -> 413 Payload Too Large
        var chunkedBigReq = new HttpRequestMessage(System.Net.Http.HttpMethod.Post, "/echo")
        {
            Content = new StreamContent(new MemoryStream(bigPayload))
        };
        chunkedBigReq.Headers.TransferEncodingChunked = true;
        var chunkedBigRes = await client.SendAsync(chunkedBigReq);
        Assert.Equal(HttpStatusCode.RequestEntityTooLarge, chunkedBigRes.StatusCode);
    }

    // =========================================================================
    // 7. Adversarial Hardening Regressions (Pool Safety, SSE Escaping, Strided Tensor, 431, Zero Host KV on CUDA)
    // =========================================================================

    [Fact]
    public async Task HttpResponse_ConcurrentWriteJsonAsync_ZeroCrossRequestCorruptionAndCrlfGuard()
    {
        // 1. Reject CRLF header-splitting injection in both ContentType and Headers
        var crlfPipe1 = new Pipe();
        var badResp1 = new HttpResponse(crlfPipe1.Writer)
        {
            ContentType = "application/json\r\nX-Injected: pwned"
        };
        await Assert.ThrowsAsync<ArgumentException>(() => badResp1.WriteUtf8Async("hi").AsTask());

        var crlfPipe2 = new Pipe();
        var badResp2 = new HttpResponse(crlfPipe2.Writer);
        badResp2.Headers["X-Custom"] = "safe\r\nSet-Cookie: evil=1";
        await Assert.ThrowsAsync<ArgumentException>(() => badResp2.WriteUtf8Async("hi").AsTask());

        // 2. Hammer WriteJsonAsync concurrently across 64 async continuations to verify per-call pooled buffer isolation
        Task[] tasks = Enumerable.Range(0, 64).Select(async idx =>
        {
            await Task.Yield();
            var pipe = new Pipe();
            var resp = new HttpResponse(pipe.Writer);
            string expectedTag = $"model-{idx:D3}-" + new string((char)('a' + (idx % 26)), 128);
            var dto = new TensorDto
            {
                Rank = 1,
                Shape = [idx + 1],
                Length = idx + 1,
                Data = Enumerable.Range(0, idx + 1).Select(x => (float)(x + idx)).ToArray()
            };
            resp.Headers["X-Tag"] = expectedTag;

            await resp.WriteJsonAsync(dto, ServeJsonContext.Default.TensorDto);
            await pipe.Writer.CompleteAsync();

            using var ms = new MemoryStream();
            while (true)
            {
                var read = await pipe.Reader.ReadAsync();
                foreach (var seg in read.Buffer) ms.Write(seg.Span);
                pipe.Reader.AdvanceTo(read.Buffer.End);
                if (read.IsCompleted) break;
            }
            await pipe.Reader.CompleteAsync();

            byte[] raw = ms.ToArray();
            int sep = raw.AsSpan().IndexOf("\r\n\r\n"u8);
            Assert.True(sep > 0);
            string headers = Encoding.ASCII.GetString(raw, 0, sep + 2);
            Assert.Contains($"X-Tag: {expectedTag}\r\n", headers);

            var roundTrip = JsonSerializer.Deserialize(raw.AsSpan(sep + 4), ServeJsonContext.Default.TensorDto);
            Assert.NotNull(roundTrip);
            Assert.Equal(idx + 1, roundTrip!.Length);
            Assert.Equal((float)idx, roundTrip.Data[0]);
        }).ToArray();

        await Task.WhenAll(tasks);
    }

    [Fact]
    public async Task OpenAiSse_ControlCharModelNameAndToken_SufficientBufferNoOverflow()
    {
        string ggufPath = AllocateTempGgufPath("sse_ctrl_escape");
        int promptTok = SyntheticGgufBuilder.GetByteTokenId((byte)'Q');
        int ctrlTok = SyntheticGgufBuilder.GetByteTokenId(0x01); // SOH -> expands 6x to "\u0001" in JSON!

        SyntheticGgufBuilder.WriteTransitionChainModel(
            ggufPath,
            transitions: new Dictionary<int, int>
            {
                [promptTok] = ctrlTok,
                [ctrlTok] = SyntheticGgufBuilder.EosTokenIdConst
            });

        using var engine = new ContinuousBatchEngine(
            ggufPath,
            totalKvBlocks: 16,
            maxBatchSize: 4,
            maxSeqLen: 64,
            useGpu: false);

        int port = GetAvailablePort();
        using var app = GlacierServeApp.CreateBuilder().UsePort(port).Build();
        app.MapOpenAiEndpoints(engine);
        await app.StartAsync();

        using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

        // Model name packed with 96 ASCII control chars (each expands 1 byte -> 6 bytes \u0001 in Utf8JsonWriter)
        string controlHeavyModel = new string('\u0001', 96);
        string payload = JsonSerializer.Serialize(new InferenceRequest
        {
            Model = controlHeavyModel,
            Prompt = "Q",
            MaxTokens = 4,
            Temperature = 0f,
            Stream = true
        }, ServeJsonContext.Default.InferenceRequest);

        var res = await client.PostAsync("/v1/completions", new StringContent(payload, Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);

        string sse = await res.Content.ReadAsStringAsync();
        string[] lines = sse.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        Assert.Equal("data: [DONE]", lines[^1]);

        using var firstChunk = JsonDocument.Parse(lines[0]["data: ".Length..]);
        Assert.Equal(controlHeavyModel, firstChunk.RootElement.GetProperty("model").GetString());
        Assert.Equal("\u0001", firstChunk.RootElement.GetProperty("choices")[0].GetProperty("text").GetString());
    }

    [Fact]
    public async Task TensorSerialization_NonContiguousTransposedView_AndNonFiniteJsonLiterals()
    {
        // Original [2, 3] matrix:
        // [ 1,  2,  3 ]
        // [ 4, NaN, Inf ]
        // Transposed [3, 2] view (non-contiguous!) logical row-major order:
        // [ 1, 4, 2, NaN, 3, Inf ]
        float[] raw = [1f, 2f, 3f, 4f, float.NaN, float.PositiveInfinity];
        using var baseTensor = Tensor<float>.FromSpan(raw, [2, 3]);
        using var transposed = baseTensor.Transpose(0, 1);
        Assert.False(transposed.IsContiguous);

        // 1. Binary round-trip on non-contiguous view
        var binPipe = new Pipe();
        var binResp = new HttpResponse(binPipe.Writer);
        await binResp.WriteBinaryTensorAsync(transposed);
        await binPipe.Writer.CompleteAsync();

        using var binMs = new MemoryStream();
        while (true)
        {
            var r = await binPipe.Reader.ReadAsync();
            foreach (var seg in r.Buffer) binMs.Write(seg.Span);
            binPipe.Reader.AdvanceTo(r.Buffer.End);
            if (r.IsCompleted) break;
        }
        await binPipe.Reader.CompleteAsync();

        byte[] binBytes = binMs.ToArray();
        int binOffset = binBytes.AsSpan().IndexOf("\r\n\r\n"u8) + 4;
        Assert.Equal(2, BitConverter.ToInt32(binBytes, binOffset));
        Assert.Equal(3, BitConverter.ToInt32(binBytes, binOffset + 4));
        Assert.Equal(2, BitConverter.ToInt32(binBytes, binOffset + 8));

        float[] expectedLogical = [1f, 4f, 2f, float.NaN, 3f, float.PositiveInfinity];
        for (int i = 0; i < expectedLogical.Length; i++)
        {
            float v = BitConverter.ToSingle(binBytes, binOffset + 12 + i * sizeof(float));
            if (float.IsNaN(expectedLogical[i]))
                Assert.True(float.IsNaN(v));
            else
                Assert.Equal(expectedLogical[i], v);
        }

        // 2. JSON round-trip on non-contiguous view containing NaN & PositiveInfinity
        var jsonPipe = new Pipe();
        var jsonResp = new HttpResponse(jsonPipe.Writer);
        await jsonResp.WriteJsonTensorAsync(transposed);
        await jsonPipe.Writer.CompleteAsync();

        using var jsonMs = new MemoryStream();
        while (true)
        {
            var r = await jsonPipe.Reader.ReadAsync();
            foreach (var seg in r.Buffer) jsonMs.Write(seg.Span);
            jsonPipe.Reader.AdvanceTo(r.Buffer.End);
            if (r.IsCompleted) break;
        }
        await jsonPipe.Reader.CompleteAsync();

        byte[] jsonBytes = jsonMs.ToArray();
        int jsonOffset = jsonBytes.AsSpan().IndexOf("\r\n\r\n"u8) + 4;
        var dto = JsonSerializer.Deserialize(jsonBytes.AsSpan(jsonOffset), ServeJsonContext.Default.TensorDto);
        Assert.NotNull(dto);
        Assert.Equal([3, 2], dto!.Shape);
        Assert.Equal(1f, dto.Data[0]);
        Assert.Equal(4f, dto.Data[1]);
        Assert.True(float.IsNaN(dto.Data[3]));
        Assert.True(float.IsPositiveInfinity(dto.Data[5]));
    }

    [Fact]
    public void ContinuousBatchEngine_CudaSkipsDuplicateHostKvBuffers()
    {
        using (var metadataOnlyPool = new PagedBlockPool(totalBlocks: 32, layers: 4, headsKv: 2, headDim: 64, allocateHostBuffers: false))
        {
            Assert.False(metadataOnlyPool.HasHostBuffers);
            Assert.Equal(0L, metadataOnlyPool.TotalMemoryBytes);
            Assert.Equal(IntPtr.Zero, metadataOnlyPool.KeyBufferAddress);
            Assert.Equal(IntPtr.Zero, metadataOnlyPool.ValueBufferAddress);

            // Virtual page allocation/recycling still works identically for CUDA BlockTables
            Assert.True(metadataOnlyPool.TryAllocateBlock(out int b0));
            Assert.True(b0 >= 0);
            Assert.Equal(31, metadataOnlyPool.FreeBlocksCount);
            metadataOnlyPool.FreeBlock(b0);
            Assert.Equal(32, metadataOnlyPool.FreeBlocksCount);
        }

        if (!GpuContext.IsSupported)
        {
            return;
        }

        string ggufPath = AllocateTempGgufPath("cuda_zero_host_kv");
        SyntheticGgufBuilder.WriteModel(
            ggufPath,
            includeThinkTokens: false,
            useQ8Weights: true,
            dim: 64,
            ffnDim: 128,
            nHeads: 2,
            nHeadsKv: 1,
            headDim: 32,
            vHeadDim: 32);

        using var engine = new ContinuousBatchEngine(
            ggufPath,
            totalKvBlocks: 16,
            maxBatchSize: 4,
            maxSeqLen: 64,
            useGpu: true);

        Assert.True(engine.IsGpuAccelerated);
        Assert.False(engine.BlockPool.HasHostBuffers);
        Assert.Equal(0L, engine.BlockPool.TotalMemoryBytes);
    }

    [Fact]
    public async Task HttpHardening_431OversizedHeaders_AndPercentDecodedRouteParams()
    {
        int port = GetAvailablePort();
        using var app = GlacierServeApp.CreateBuilder()
            .UsePort(port)
            .MapGet("/models/{modelName}", async ctx =>
            {
                await ctx.Response.WriteUtf8Async(ctx.Request.Param("modelName") ?? "");
            })
            .Build();

        await app.StartAsync();

        // 1. Percent-decoded route parameters (%20 and %2F)
        using (var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") })
        {
            var res = await client.GetAsync("/models/Qwen%203.5-8B");
            Assert.Equal(HttpStatusCode.OK, res.StatusCode);
            Assert.Equal("Qwen 3.5-8B", await res.Content.ReadAsStringAsync());
        }

        // 2. Oversized HTTP request headers (> 16 KB) -> 431 Request Header Fields Too Large
        using var tcp = new TcpClient();
        await tcp.ConnectAsync(IPAddress.Loopback, port);
        await using var stream = tcp.GetStream();
        string giantHeader = "GET /models/test HTTP/1.1\r\nHost: localhost\r\nX-Giant: " + new string('H', 18000) + "\r\n\r\n";
        byte[] reqBytes = Encoding.ASCII.GetBytes(giantHeader);
        await stream.WriteAsync(reqBytes);
        await stream.FlushAsync();

        byte[] respBuf = new byte[512];
        int n = await stream.ReadAsync(respBuf);
        string statusText = Encoding.ASCII.GetString(respBuf, 0, n);
        Assert.StartsWith("HTTP/1.1 431 Request Header Fields Too Large", statusText);
    }

    // =========================================================================
    // Helpers & Synthetic GGUF v3 Model Builder
    // =========================================================================

    private static int ArgMax(ReadOnlySpan<float> values)
    {
        int bestIdx = 0;
        float bestVal = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > bestVal)
            {
                bestVal = values[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    private static void AssertLogitsClose(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            Assert.True(diff <= tolerance, $"Logit mismatch at index {i}: expected={expected[i]}, actual={actual[i]}, diff={diff}");
        }
    }

    private static class SyntheticGgufBuilder
    {
        public const int BosTokenIdConst = 256;
        public const int EosTokenIdConst = 257;

        private static readonly string[] ByteVocab = BuildGpt2ByteVocab();

        public static int GetByteTokenId(byte b) => b;

        private static string[] BuildGpt2ByteVocab()
        {
            var bs = new List<int>();
            for (int i = '!'; i <= '~'; i++) bs.Add(i);
            for (int i = '¡'; i <= '¬'; i++) bs.Add(i);
            for (int i = '®'; i <= 'ÿ'; i++) bs.Add(i);

            var cs = new List<int>(bs);
            int n = 0;
            for (int b = 0; b < 256; b++)
            {
                if (!bs.Contains(b))
                {
                    bs.Add(b);
                    cs.Add(256 + n);
                    n++;
                }
            }

            string[] lut = new string[256];
            for (int i = 0; i < 256; i++)
            {
                lut[bs[i]] = ((char)cs[i]).ToString();
            }
            return lut;
        }

        private static List<string> BuildTokenList(bool includeThinkTokens)
        {
            var tokens = new List<string>(288);
            // 0..255: complete 256-byte GPT-2 BPE alphabet so any UTF-8 string encodes losslessly
            tokens.AddRange(ByteVocab);
            // 256..258: standard chat/control tokens
            tokens.Add("<|endoftext|>"); // 256 (BOS)
            tokens.Add("<|im_end|>");    // 257 (EOS)
            tokens.Add("<|im_start|>");  // 258
            if (includeThinkTokens)
            {
                tokens.Add("<think>");   // 259
                tokens.Add("</think>");  // 260
            }
            // Pad vocab size to a clean multiple of 32 (288)
            while (tokens.Count < 288)
            {
                tokens.Add($"<|extra_{tokens.Count}|>");
            }
            return tokens;
        }

        /// <summary>
        /// Builds a deterministic 1-layer zero-residual GGUF model whose greedy argmax implements
        /// an exact token-to-token state machine <paramref name="transitions"/>.
        /// </summary>
        public static void WriteTransitionChainModel(
            string path,
            IReadOnlyDictionary<int, int> transitions,
            int defaultTargetTokenId = EosTokenIdConst)
        {
            var tokens = BuildTokenList(includeThinkTokens: true);
            int vocabSize = tokens.Count;
            const int dim = 64;
            const int ffnDim = 128;
            const int nHeads = 2;
            const int nHeadsKv = 1;
            const int headDim = 32;

            float[] embd = new float[vocabSize * dim];
            float[] outWeight = new float[vocabSize * dim];
            float[] onesDim = Enumerable.Repeat(1.0f, dim).ToArray();

            var assignedBasis = new Dictionary<int, int>();
            int nextBasis = 1;
            foreach (int src in transitions.Keys)
            {
                if (!assignedBasis.ContainsKey(src) && nextBasis < dim)
                {
                    assignedBasis[src] = nextBasis++;
                }
            }

            for (int v = 0; v < vocabSize; v++)
            {
                int basis = assignedBasis.TryGetValue(v, out int b) ? b : 0;
                embd[v * dim + basis] = 1.0f;
            }

            outWeight[defaultTargetTokenId * dim + 0] = 10.0f;
            foreach (var (src, dst) in transitions)
            {
                int basis = assignedBasis[src];
                outWeight[dst * dim + basis] = 10.0f;
            }

            int qDim = nHeads * headDim;
            int kDim = nHeadsKv * headDim;

            var tensors = new List<TensorSpec>
            {
                TensorSpec.CreateF32("token_embd.weight", [(ulong)dim, (ulong)vocabSize], embd),
                TensorSpec.CreateF32("output_norm.weight", [(ulong)dim], onesDim),
                TensorSpec.CreateF32("output.weight", [(ulong)dim, (ulong)vocabSize], outWeight),
                TensorSpec.CreateF32("blk.0.attn_norm.weight", [(ulong)dim], onesDim),
                TensorSpec.CreateF32("blk.0.attn_q.weight", [(ulong)dim, (ulong)qDim], new float[qDim * dim]),
                TensorSpec.CreateF32("blk.0.attn_k.weight", [(ulong)dim, (ulong)kDim], new float[kDim * dim]),
                TensorSpec.CreateF32("blk.0.attn_v.weight", [(ulong)dim, (ulong)kDim], new float[kDim * dim]),
                TensorSpec.CreateF32("blk.0.attn_output.weight", [(ulong)qDim, (ulong)dim], new float[dim * qDim]),
                TensorSpec.CreateF32("blk.0.ffn_norm.weight", [(ulong)dim], onesDim),
                TensorSpec.CreateF32("blk.0.ffn_gate.weight", [(ulong)dim, (ulong)ffnDim], new float[ffnDim * dim]),
                TensorSpec.CreateF32("blk.0.ffn_up.weight", [(ulong)dim, (ulong)ffnDim], new float[ffnDim * dim]),
                TensorSpec.CreateF32("blk.0.ffn_down.weight", [(ulong)ffnDim, (ulong)dim], new float[dim * ffnDim])
            };

            WriteGgufFile(
                path,
                tokens,
                blockCount: 1,
                dim: dim,
                ffnDim: ffnDim,
                nHeads: nHeads,
                nHeadsKv: nHeadsKv,
                headDim: headDim,
                vHeadDim: headDim,
                tensors);
        }

        public static void WriteModel(
            string path,
            bool includeThinkTokens,
            bool useQ8Weights,
            int dim = 64,
            int ffnDim = 128,
            int nHeads = 2,
            int nHeadsKv = 1,
            int headDim = 32,
            int vHeadDim = 32,
            bool includeQkNorm = false,
            int[]? forcedTargetTokenIds = null)
        {
            var tokens = BuildTokenList(includeThinkTokens);
            int vocabSize = tokens.Count;
            var rng = new Random(1337);

            int qDim = nHeads * headDim;
            int kDim = nHeadsKv * headDim;
            int vDim = nHeadsKv * vHeadDim;
            int attnInDim = nHeads * vHeadDim;

            float[] embd = RandomFloats(rng, vocabSize * dim, 0.08f);
            float[] outNorm = PositiveFloats(rng, dim, 0.8f, 1.2f);
            float[] outW = RandomFloats(rng, vocabSize * dim, 0.08f);

            if (forcedTargetTokenIds is { Length: > 0 })
            {
                Array.Clear(embd);
                for (int v = 0; v < vocabSize; v++) embd[v * dim + 0] = 1.0f;
                Array.Fill(outNorm, 1.0f);
                Array.Clear(outW);
                outW[forcedTargetTokenIds[0] * dim + 0] = 25.0f;
            }

            bool zeroLayerResiduals = forcedTargetTokenIds is { Length: > 0 };
            float scale = zeroLayerResiduals ? 0.0f : 0.06f;

            float[] attnNorm = PositiveFloats(rng, dim, 0.9f, 1.1f);
            float[] wQ = RandomFloats(rng, qDim * dim, scale);
            float[] wK = RandomFloats(rng, kDim * dim, scale);
            float[] wV = RandomFloats(rng, vDim * dim, scale);
            float[] wO = RandomFloats(rng, dim * attnInDim, scale);
            float[] ffnNorm = PositiveFloats(rng, dim, 0.9f, 1.1f);
            float[] wGate = RandomFloats(rng, ffnDim * dim, scale);
            float[] wUp = RandomFloats(rng, ffnDim * dim, scale);
            float[] wDown = RandomFloats(rng, dim * ffnDim, scale);

            var tensors = new List<TensorSpec>
            {
                TensorSpec.CreateF32("token_embd.weight", [(ulong)dim, (ulong)vocabSize], embd),
                TensorSpec.CreateF32("output_norm.weight", [(ulong)dim], outNorm),
                CreateMatrix("output.weight", dim, vocabSize, outW, useQ8Weights),
                TensorSpec.CreateF32("blk.0.attn_norm.weight", [(ulong)dim], attnNorm),
                CreateMatrix("blk.0.attn_q.weight", dim, qDim, wQ, useQ8Weights),
                CreateMatrix("blk.0.attn_k.weight", dim, kDim, wK, useQ8Weights),
                CreateMatrix("blk.0.attn_v.weight", dim, vDim, wV, useQ8Weights),
                CreateMatrix("blk.0.attn_output.weight", attnInDim, dim, wO, useQ8Weights),
                TensorSpec.CreateF32("blk.0.ffn_norm.weight", [(ulong)dim], ffnNorm),
                CreateMatrix("blk.0.ffn_gate.weight", dim, ffnDim, wGate, useQ8Weights),
                CreateMatrix("blk.0.ffn_up.weight", dim, ffnDim, wUp, useQ8Weights),
                CreateMatrix("blk.0.ffn_down.weight", ffnDim, dim, wDown, useQ8Weights)
            };

            if (includeQkNorm)
            {
                tensors.Add(TensorSpec.CreateF32("blk.0.attn_q_norm.weight", [(ulong)headDim], PositiveFloats(rng, headDim, 0.7f, 1.3f)));
                tensors.Add(TensorSpec.CreateF32("blk.0.attn_k_norm.weight", [(ulong)headDim], PositiveFloats(rng, headDim, 0.7f, 1.3f)));
            }

            WriteGgufFile(
                path,
                tokens,
                blockCount: 1,
                dim: dim,
                ffnDim: ffnDim,
                nHeads: nHeads,
                nHeadsKv: nHeadsKv,
                headDim: headDim,
                vHeadDim: vHeadDim,
                tensors);
        }

        private static TensorSpec CreateMatrix(string name, int cols, int rows, float[] data, bool useQ8)
        {
            if (!useQ8)
            {
                return TensorSpec.CreateF32(name, [(ulong)cols, (ulong)rows], data);
            }
            return TensorSpec.CreateQ8_0(name, [(ulong)cols, (ulong)rows], data);
        }

        private static float[] RandomFloats(Random rng, int count, float amplitude)
        {
            float[] arr = new float[count];
            for (int i = 0; i < count; i++)
            {
                arr[i] = ((float)rng.NextDouble() * 2.0f - 1.0f) * amplitude;
            }
            return arr;
        }

        private static float[] PositiveFloats(Random rng, int count, float min, float max)
        {
            float[] arr = new float[count];
            for (int i = 0; i < count; i++)
            {
                arr[i] = min + (float)rng.NextDouble() * (max - min);
            }
            return arr;
        }

        private static void WriteGgufFile(
            string path,
            List<string> tokens,
            int blockCount,
            int dim,
            int ffnDim,
            int nHeads,
            int nHeadsKv,
            int headDim,
            int vHeadDim,
            List<TensorSpec> tensors)
        {
            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
            using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: true);

            var kvEntries = new List<Action<BinaryWriter>>
            {
                w => WriteKvString(w, "general.architecture", "qwen2"),
                w => WriteKvUInt32(w, "general.alignment", 32u),
                w => WriteKvUInt32(w, "qwen2.block_count", (uint)blockCount),
                w => WriteKvUInt32(w, "qwen2.context_length", 256u),
                w => WriteKvUInt32(w, "qwen2.embedding_length", (uint)dim),
                w => WriteKvUInt32(w, "qwen2.feed_forward_length", (uint)ffnDim),
                w => WriteKvUInt32(w, "qwen2.attention.head_count", (uint)nHeads),
                w => WriteKvUInt32(w, "qwen2.attention.head_count_kv", (uint)nHeadsKv),
                w => WriteKvUInt32(w, "qwen2.attention.key_length", (uint)headDim),
                w => WriteKvUInt32(w, "qwen2.attention.value_length", (uint)vHeadDim),
                w => WriteKvFloat32(w, "qwen2.attention.layer_norm_rms_epsilon", 1e-6f),
                w => WriteKvFloat32(w, "qwen2.rope.freq_base", 10000.0f),
                w => WriteKvUInt32(w, "tokenizer.ggml.bos_token_id", BosTokenIdConst),
                w => WriteKvUInt32(w, "tokenizer.ggml.eos_token_id", EosTokenIdConst),
                w => WriteKvBool(w, "tokenizer.ggml.add_bos_token", false),
                w => WriteKvBool(w, "tokenizer.ggml.add_eos_token", false),
                w => WriteKvStringArray(w, "tokenizer.ggml.tokens", tokens)
            };

            // 1. Header
            bw.Write(0x46554747u); // 'GGUF'
            bw.Write(3u);          // Version 3
            bw.Write((ulong)tensors.Count);
            bw.Write((ulong)kvEntries.Count);

            // 2. Metadata KVs
            foreach (var writeKv in kvEntries)
            {
                writeKv(bw);
            }

            // 3. Compute 32-byte aligned tensor offsets
            ulong currentDataOffset = 0;
            ulong[] offsets = new ulong[tensors.Count];
            for (int i = 0; i < tensors.Count; i++)
            {
                ulong rem = currentDataOffset % 32ul;
                if (rem != 0) currentDataOffset += 32ul - rem;
                offsets[i] = currentDataOffset;
                currentDataOffset += (ulong)tensors[i].Bytes.Length;
            }

            // 4. Tensor directory
            for (int i = 0; i < tensors.Count; i++)
            {
                var t = tensors[i];
                WriteGgufString(bw, t.Name);
                bw.Write((uint)t.Dims.Length);
                foreach (ulong d in t.Dims)
                {
                    bw.Write(d);
                }
                bw.Write((uint)t.Type);
                bw.Write(offsets[i]);
            }

            // 5. Align to 32-byte boundary for TensorDataOffset
            bw.Flush();
            long headerPos = fs.Position;
            long padHeader = (32 - (headerPos % 32)) % 32;
            for (int p = 0; p < padHeader; p++) bw.Write((byte)0);

            long dataBase = fs.Position;
            for (int i = 0; i < tensors.Count; i++)
            {
                long targetPos = dataBase + (long)offsets[i];
                while (fs.Position < targetPos)
                {
                    bw.Write((byte)0);
                }
                bw.Write(tensors[i].Bytes);
            }
            bw.Flush();
        }

        private static void WriteGgufString(BinaryWriter bw, string s)
        {
            byte[] utf8 = Encoding.UTF8.GetBytes(s);
            bw.Write((ulong)utf8.Length);
            bw.Write(utf8);
        }

        private static void WriteKvString(BinaryWriter bw, string key, string value)
        {
            WriteGgufString(bw, key);
            bw.Write((uint)GgufValueType.String);
            WriteGgufString(bw, value);
        }

        private static void WriteKvUInt32(BinaryWriter bw, string key, uint value)
        {
            WriteGgufString(bw, key);
            bw.Write((uint)GgufValueType.Uint32);
            bw.Write(value);
        }

        private static void WriteKvFloat32(BinaryWriter bw, string key, float value)
        {
            WriteGgufString(bw, key);
            bw.Write((uint)GgufValueType.Float32);
            bw.Write(value);
        }

        private static void WriteKvBool(BinaryWriter bw, string key, bool value)
        {
            WriteGgufString(bw, key);
            bw.Write((uint)GgufValueType.Bool);
            bw.Write(value ? (byte)1 : (byte)0);
        }

        private static void WriteKvStringArray(BinaryWriter bw, string key, List<string> values)
        {
            WriteGgufString(bw, key);
            bw.Write((uint)GgufValueType.Array);
            bw.Write((uint)GgufValueType.String);
            bw.Write((ulong)values.Count);
            foreach (string v in values)
            {
                WriteGgufString(bw, v);
            }
        }

        private sealed record TensorSpec(string Name, ulong[] Dims, GgufType Type, byte[] Bytes)
        {
            public static TensorSpec CreateF32(string name, ulong[] dims, float[] values)
            {
                byte[] bytes = new byte[values.Length * sizeof(float)];
                Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
                return new TensorSpec(name, dims, GgufType.F32, bytes);
            }

            public static TensorSpec CreateQ8_0(string name, ulong[] dims, float[] values)
            {
                if (values.Length % 32 != 0)
                    throw new ArgumentException($"Q8_0 tensor '{name}' element count ({values.Length}) must be divisible by 32.");

                int numBlocks = values.Length / 32;
                byte[] bytes = new byte[numBlocks * 34];

                for (int b = 0; b < numBlocks; b++)
                {
                    int srcOffset = b * 32;
                    float maxAbs = 0f;
                    for (int i = 0; i < 32; i++)
                    {
                        float a = MathF.Abs(values[srcOffset + i]);
                        if (a > maxAbs) maxAbs = a;
                    }

                    Half dHalf = (Half)(maxAbs / 127.0f);
                    float d = (float)dHalf;
                    float invD = d > 0f ? 1.0f / d : 0f;

                    int dstOffset = b * 34;
                    ushort bits = BitConverter.HalfToUInt16Bits(dHalf);
                    bytes[dstOffset] = (byte)(bits & 0xFF);
                    bytes[dstOffset + 1] = (byte)(bits >> 8);

                    for (int i = 0; i < 32; i++)
                    {
                        int q = (int)MathF.Round(values[srcOffset + i] * invD);
                        q = Math.Clamp(q, -127, 127);
                        bytes[dstOffset + 2 + i] = unchecked((byte)(sbyte)q);
                    }
                }

                return new TensorSpec(name, dims, GgufType.Q8_0, bytes);
            }
        }
    }
}

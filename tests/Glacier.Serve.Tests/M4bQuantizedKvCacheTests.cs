using System;
using System.IO;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Inference.PagedAttention;
using Xunit;

namespace Glacier.Serve.Tests;

public sealed class M4bQuantizedKvCacheTests
{
    [Fact]
    public void PagedBlockPool_AutoResolution_HostResolvesToFp32_HostlessStaysAuto()
    {
        using var hostAuto = new PagedBlockPool(4, layers: 2, headsKv: 2, headDim: 64, allocateHostBuffers: true);
        Assert.Equal(KvCachePrecision.Fp32, hostAuto.Precision);
        Assert.True(hostAuto.HasHostBuffers);
        Assert.Equal(64 * sizeof(float), hostAuto.KeySlotBytes);

        using var deviceAuto = new PagedBlockPool(4, layers: 2, headsKv: 2, headDim: 64, allocateHostBuffers: false);
        Assert.Equal(KvCachePrecision.Auto, deviceAuto.Precision);
        Assert.False(deviceAuto.HasHostBuffers);
        Assert.Equal(0L, deviceAuto.TotalMemoryBytes);
        Assert.Equal(64 * sizeof(ushort), deviceAuto.KeySlotBytes);

        using var int4Pool = new PagedBlockPool(4, layers: 2, headsKv: 2, headDim: 64, allocateHostBuffers: true, kvPrecision: KvCachePrecision.Int4);
        Assert.Equal(KvCachePrecision.Q4_0, int4Pool.Precision);
    }

    [Fact]
    public void PagedBlockPool_MemoryFootprint_Delivers3_5xTo8xReduction()
    {
        const int blocks = 16;
        const int layers = 4;
        const int headsKv = 4;
        const int headDim = 128;
        const int blockSize = 16;

        using var fp32 = new PagedBlockPool(blocks, layers, headsKv, headDim, blockSize, kvPrecision: KvCachePrecision.Fp32);
        using var fp16 = new PagedBlockPool(blocks, layers, headsKv, headDim, blockSize, kvPrecision: KvCachePrecision.Fp16);
        using var fp8 = new PagedBlockPool(blocks, layers, headsKv, headDim, blockSize, kvPrecision: KvCachePrecision.Fp8);
        using var q8 = new PagedBlockPool(blocks, layers, headsKv, headDim, blockSize, kvPrecision: KvCachePrecision.Q8_0);
        using var q4 = new PagedBlockPool(blocks, layers, headsKv, headDim, blockSize, kvPrecision: KvCachePrecision.Q4_0);

        Assert.Equal(512L, PagedBlockPool.GetSlotByteSize(headDim, KvCachePrecision.Fp32));
        Assert.Equal(256L, PagedBlockPool.GetSlotByteSize(headDim, KvCachePrecision.Fp16));
        Assert.Equal(128L, PagedBlockPool.GetSlotByteSize(headDim, KvCachePrecision.Fp8));
        Assert.Equal(136L, PagedBlockPool.GetSlotByteSize(headDim, KvCachePrecision.Q8_0));
        Assert.Equal(72L, PagedBlockPool.GetSlotByteSize(headDim, KvCachePrecision.Q4_0));

        double q8VsFp32 = (double)fp32.TotalMemoryBytes / q8.TotalMemoryBytes;
        double q8VsFp16 = (double)fp16.TotalMemoryBytes / q8.TotalMemoryBytes;
        double q4VsFp32 = (double)fp32.TotalMemoryBytes / q4.TotalMemoryBytes;
        double q4VsFp16 = (double)fp16.TotalMemoryBytes / q4.TotalMemoryBytes;

        Assert.InRange(q8VsFp32, 3.76, 3.77); // 128 / 34 = 3.7647x
        Assert.InRange(q8VsFp16, 1.88, 1.89); // 64 / 34 = 1.8824x
        Assert.InRange(q4VsFp32, 7.11, 7.12); // 128 / 18 = 7.1111x
        Assert.InRange(q4VsFp16, 3.55, 3.56); // 64 / 18 = 3.5556x
        Assert.Equal(fp32.TotalMemoryBytes / 4L, fp8.TotalMemoryBytes);

        // Non-32-divisible asymmetric K/V dims
        Assert.Equal(68L, PagedBlockPool.GetSlotByteSize(48, KvCachePrecision.Q8_0));
        Assert.Equal(36L, PagedBlockPool.GetSlotByteSize(36, KvCachePrecision.Q4_0));
    }

    [Theory]
    [InlineData(KvCachePrecision.Fp32, 0.99999f)]
    [InlineData(KvCachePrecision.Fp16, 0.9999f)]
    [InlineData(KvCachePrecision.Fp8, 0.990f)]
    [InlineData(KvCachePrecision.Q8_0, 0.999f)]
    [InlineData(KvCachePrecision.Q4_0, 0.985f)]
    public unsafe void PagedBlockPool_StoreAndRead_RoundTripAccuracy_IncludingNon32Dims(KvCachePrecision precision, float minCosine)
    {
        const int headDim = 48;
        const int vHeadDim = 36;
        const int headsKv = 2;

        using var pool = new PagedBlockPool(4, layers: 2, headsKv, headDim, blockSize: 8, vHeadDim: vHeadDim, kvPrecision: precision);
        Assert.True(pool.TryAllocateBlock(out int blockId));

        var rng = new Random(17);
        var kSrc = new float[headsKv * headDim];
        var vSrc = new float[headsKv * vHeadDim];
        for (int i = 0; i < kSrc.Length; i++) kSrc[i] = ((float)rng.NextDouble() * 2f - 1f) * 2.0f;
        for (int i = 0; i < vSrc.Length; i++) vSrc[i] = ((float)rng.NextDouble() * 2f - 1f) * 2.0f;

        fixed (float* pK = kSrc)
        fixed (float* pV = vSrc)
        {
            pool.Store(blockId, layer: 1, offsetInBlock: 3, pK, pV);
        }

        if (precision != KvCachePrecision.Fp32)
        {
            Assert.Throws<InvalidOperationException>(() => pool.GetKeyPtr(blockId, 1, 0, 3));
            Assert.Throws<InvalidOperationException>(() => pool.GetValuePtr(blockId, 1, 0, 3));
        }
        else
        {
            Assert.True(pool.GetKeyPtr(blockId, 1, 0, 3) != null);
        }

        var kOut = new float[headDim];
        var vOut = new float[vHeadDim];
        for (int h = 0; h < headsKv; h++)
        {
            pool.ReadKey(blockId, 1, h, 3, kOut);
            pool.ReadValue(blockId, 1, h, 3, vOut);
            Assert.True(M4bTestHelpers.CosineSimilarity(kSrc.AsSpan(h * headDim, headDim), kOut) >= minCosine);
            Assert.True(M4bTestHelpers.CosineSimilarity(vSrc.AsSpan(h * vHeadDim, vHeadDim), vOut) >= minCosine);
        }
    }

    [Theory]
    [InlineData(KvCachePrecision.Fp32)]
    [InlineData(KvCachePrecision.Fp16)]
    [InlineData(KvCachePrecision.Fp8)]
    [InlineData(KvCachePrecision.Q8_0)]
    [InlineData(KvCachePrecision.Q4_0)]
    public unsafe void PagedBlockPool_CopyBlockData_And_TryForkBlock_AreBitExact(KvCachePrecision precision)
    {
        const int headDim = 48;
        const int vHeadDim = 36;
        const int headsKv = 2;
        const int blockSize = 8;

        using var pool = new PagedBlockPool(4, layers: 2, headsKv, headDim, blockSize: blockSize, vHeadDim: vHeadDim, kvPrecision: precision);
        Assert.True(pool.TryAllocateBlock(out int srcId));

        var rng = new Random(33);
        var kSrc = new float[headsKv * headDim];
        var vSrc = new float[headsKv * vHeadDim];

        for (int t = 0; t < 5; t++)
        {
            for (int i = 0; i < kSrc.Length; i++) kSrc[i] = ((float)rng.NextDouble() * 2f - 1f);
            for (int i = 0; i < vSrc.Length; i++) vSrc[i] = ((float)rng.NextDouble() * 2f - 1f);
            fixed (float* pK = kSrc)
            fixed (float* pV = vSrc)
            {
                pool.Store(srcId, layer: 0, offsetInBlock: t, pK, pV);
                pool.Store(srcId, layer: 1, offsetInBlock: t, pK, pV);
            }
        }

        Assert.True(pool.TryForkBlock(srcId, validTokens: 5, out int dstId));
        Assert.NotEqual(srcId, dstId);

        for (int l = 0; l < 2; l++)
        {
            for (int h = 0; h < headsKv; h++)
            {
                for (int t = 0; t < 5; t++)
                {
                    var sK = new ReadOnlySpan<byte>(pool.GetKeySlotPtr(srcId, l, h, t), (int)pool.KeySlotBytes);
                    var dK = new ReadOnlySpan<byte>(pool.GetKeySlotPtr(dstId, l, h, t), (int)pool.KeySlotBytes);
                    Assert.True(sK.SequenceEqual(dK));

                    var sV = new ReadOnlySpan<byte>(pool.GetValueSlotPtr(srcId, l, h, t), (int)pool.ValueSlotBytes);
                    var dV = new ReadOnlySpan<byte>(pool.GetValueSlotPtr(dstId, l, h, t), (int)pool.ValueSlotBytes);
                    Assert.True(sV.SequenceEqual(dV));
                }
            }
        }
    }

    [Theory]
    [InlineData(64, 64, KvCachePrecision.Fp16, 0.9999f)]
    [InlineData(64, 64, KvCachePrecision.Fp8, 0.990f)]
    [InlineData(64, 64, KvCachePrecision.Q8_0, 0.999f)]
    [InlineData(64, 64, KvCachePrecision.Q4_0, 0.975f)]
    [InlineData(48, 36, KvCachePrecision.Q8_0, 0.999f)]
    [InlineData(48, 36, KvCachePrecision.Q4_0, 0.975f)]
    public unsafe void PagedAttentionKernel_QuantizedAttention_MatchesFp32Baseline(
        int headDim,
        int vHeadDim,
        KvCachePrecision precision,
        float minCosine)
    {
        const int layers = 1;
        const int nHeads = 4;
        const int nHeadsKv = 2;
        const int blockSize = 8;
        const int totalTokens = 19; // spans 3 blocks (8 + 8 + 3)
        const int maxSeqLen = 32;

        using var fp32Pool = new PagedBlockPool(8, layers, nHeadsKv, headDim, blockSize, vHeadDim, kvPrecision: KvCachePrecision.Fp32);
        using var qPool = new PagedBlockPool(8, layers, nHeadsKv, headDim, blockSize, vHeadDim, kvPrecision: precision);
        var btFp32 = new BlockTable(fp32Pool);
        var btQ = new BlockTable(qPool);

        var rng = new Random(77);
        var kVec = new float[nHeadsKv * headDim];
        var vVec = new float[nHeadsKv * vHeadDim];

        for (int t = 0; t < totalTokens; t++)
        {
            for (int i = 0; i < kVec.Length; i++) kVec[i] = ((float)rng.NextDouble() * 2f - 1f);
            for (int i = 0; i < vVec.Length; i++) vVec[i] = ((float)rng.NextDouble() * 2f - 1f);

            Assert.True(btFp32.AppendToken(out int pb32, out int off32));
            Assert.True(btQ.AppendToken(out int pbQ, out int offQ));

            fixed (float* pK = kVec)
            fixed (float* pV = vVec)
            {
                fp32Pool.Store(pb32, 0, off32, pK, pV);
                qPool.Store(pbQ, 0, offQ, pK, pV);
            }
        }

        var qVec = new float[nHeads * headDim];
        for (int i = 0; i < qVec.Length; i++) qVec[i] = ((float)rng.NextDouble() * 2f - 1f);

        var outFp32 = new float[nHeads * vHeadDim];
        var outQ = new float[nHeads * vHeadDim];
        Array.Fill(outQ, 12345.0f); // Verify quantized kernel explicitly clears outHead before += accumulation

        var scores = new float[nHeads * maxSeqLen];
        float scale = 1.0f / MathF.Sqrt(headDim);

        fixed (float* pQuery = qVec)
        fixed (float* pOut32 = outFp32)
        fixed (float* pOutQ = outQ)
        fixed (float* pScores = scores)
        {
            PagedAttentionKernel.ComputeAttention(0, 0, totalTokens - 1, pQuery, pOut32, btFp32, fp32Pool, nHeads, nHeadsKv, headDim, vHeadDim, scale, pScores, maxSeqLen);
            PagedAttentionKernel.ComputeAttention(0, 0, totalTokens - 1, pQuery, pOutQ, btQ, qPool, nHeads, nHeadsKv, headDim, vHeadDim, scale, pScores, maxSeqLen);
        }

        float cos = M4bTestHelpers.CosineSimilarity(outFp32, outQ);
        Assert.True(cos >= minCosine, $"Precision {precision} ({headDim}/{vHeadDim}) attention cosine {cos:F6} < {minCosine:F6}");
    }

    [Fact]
    public void CudaPagedTransformerRunner_SupportsQuantizedKv_AndFallsBackCleanlyForFp32Fp8()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"m4b_cuda_kv_{Guid.NewGuid():N}");
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
                useQ8Weights: false);

            using var gguf = GgufFile.Open(ggufPath);
            using var weights = new ModelWeights(gguf);

            // Fp32 and Fp8 must return false cleanly (never throw NotSupportedException)
            Assert.False(CudaPagedTransformerRunner.TryCreate(weights, 32, 16, 128, 4, KvCachePrecision.Fp32, out var fp32Runner));
            Assert.Null(fp32Runner);
            Assert.False(CudaPagedTransformerRunner.TryCreate(weights, 32, 16, 128, 4, KvCachePrecision.Fp8, out var fp8Runner));
            Assert.Null(fp8Runner);

            if (!CudaPagedTransformerRunner.TryCreate(weights, 32, 16, 128, 4, KvCachePrecision.Fp16, out var cudaFp16))
            {
                return; // CUDA GPU unavailable on host; fallback test above already verified
            }

            using (cudaFp16)
            using (var cpuRunner = new PagedTransformerRunner(weights, maxSeqLen: 128, maxBatchSize: 8))
            using (var cpuFp32Pool = new PagedBlockPool(32, 2, 2, 16, blockSize: 16, allocateHostBuffers: true, kvPrecision: KvCachePrecision.Fp32))
            {
                Assert.Equal(KvCachePrecision.Fp16, cudaFp16.KvPrecision);

                Assert.True(CudaPagedTransformerRunner.TryCreate(weights, 32, 16, 128, 4, KvCachePrecision.Q8_0, out var cudaQ8));
                Assert.True(CudaPagedTransformerRunner.TryCreate(weights, 32, 16, 128, 4, KvCachePrecision.Q4_0, out var cudaQ4));
                using (cudaQ8!)
                using (cudaQ4!)
                {
                    Assert.Equal(KvCachePrecision.Q8_0, cudaQ8.KvPrecision);
                    Assert.Equal(KvCachePrecision.Q4_0, cudaQ4.KvPrecision);

                    // For headDim=16 (<32), Q8_0 pads to 1 block (34B vs 32B Fp16), Q4_0 is 18B vs 32B Fp16
                    Assert.True(cudaQ4.DeviceKvPoolBytes < cudaFp16.DeviceKvPoolBytes);

                    // Explicit host pool mismatch must throw ArgumentException
                    var btMismatch = new BlockTable(cpuFp32Pool);
                    var mismatchLogits = new float[weights.VocabSize];
                    Assert.Throws<ArgumentException>(() => cudaQ4.ForwardToken(8, 0, btMismatch, cpuFp32Pool, mismatchLogits));

                    // End-to-end logit parity vs CPU Fp32 baseline
                    int[] prompt = [1, 8, 9, 10, 11];
                    var cpuLogits = new float[weights.VocabSize];
                    var fp16Logits = new float[weights.VocabSize];
                    var q8Logits = new float[weights.VocabSize];
                    var q4Logits = new float[weights.VocabSize];

                    var btCpu = new BlockTable(cpuFp32Pool);
                    Assert.True(cpuRunner.ForwardPrefillChunk(prompt, 0, btCpu, cpuFp32Pool, cpuLogits));

                    using var poolFp16 = new PagedBlockPool(32, 2, 2, 16, 16, allocateHostBuffers: false, kvPrecision: KvCachePrecision.Fp16);
                    using var poolQ8 = new PagedBlockPool(32, 2, 2, 16, 16, allocateHostBuffers: false, kvPrecision: KvCachePrecision.Q8_0);
                    using var poolQ4 = new PagedBlockPool(32, 2, 2, 16, 16, allocateHostBuffers: false, kvPrecision: KvCachePrecision.Q4_0);

                    var bt16 = new BlockTable(poolFp16);
                    var bt8 = new BlockTable(poolQ8);
                    var bt4 = new BlockTable(poolQ4);

                    Assert.True(cudaFp16.ForwardPrefillChunk(prompt, 0, bt16, poolFp16, fp16Logits));
                    Assert.True(cudaQ8.ForwardPrefillChunk(prompt, 0, bt8, poolQ8, q8Logits));
                    Assert.True(cudaQ4.ForwardPrefillChunk(prompt, 0, bt4, poolQ4, q4Logits));

                    Assert.True(M4bTestHelpers.CosineSimilarity(cpuLogits, fp16Logits) >= 0.999f);
                    Assert.True(M4bTestHelpers.CosineSimilarity(cpuLogits, q8Logits) >= 0.998f);
                    Assert.True(M4bTestHelpers.CosineSimilarity(cpuLogits, q4Logits) >= 0.985f);

                    // Verify GPU + _hostLoraPool partial-block COW hook
                    string loraPath = M4bTestHelpers.CreateSyntheticLoraFile(tempDir);
                    var lora = LoraAdapterWeights.Load(loraPath);
                    var btShared = new BlockTable(poolQ4);
                    Assert.True(cudaQ4.ForwardPrefillChunk(prompt, 0, btShared, poolQ4, q4Logits, computeLogits: true, lora: lora));
                    int sharedBlock = btShared.GetBlockId(0);
                    poolQ4.RetainBlock(sharedBlock);
                    Assert.Equal(2, poolQ4.GetRefCount(sharedBlock));
                    Assert.True(cudaQ4.ForwardToken(12, prompt.Length, btShared, poolQ4, q4Logits, computeLogits: true, lora: lora));
                    Assert.NotEqual(sharedBlock, btShared.GetBlockId(0));
                    poolQ4.ReleaseBlock(sharedBlock);
                }
            }
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }

    [Theory]
    [InlineData(KvCachePrecision.Auto, false, KvCachePrecision.Fp32)]
    [InlineData(KvCachePrecision.Fp16, false, KvCachePrecision.Fp16)]
    [InlineData(KvCachePrecision.Fp8, false, KvCachePrecision.Fp8)]
    [InlineData(KvCachePrecision.Q8_0, false, KvCachePrecision.Q8_0)]
    [InlineData(KvCachePrecision.Q4_0, false, KvCachePrecision.Q4_0)]
    [InlineData(KvCachePrecision.Fp32, true, KvCachePrecision.Fp32)]
    [InlineData(KvCachePrecision.Fp8, true, KvCachePrecision.Fp8)]
    public async Task ContinuousBatchEngine_KvCachePrecision_ConfiguresCpuAndGpuFallbackCorrectly(
        KvCachePrecision requested,
        bool useGpu,
        KvCachePrecision expectedResolved)
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"m4b_engine_kv_{Guid.NewGuid():N}");
        try
        {
            string ggufPath = M4bTestHelpers.CreateSyntheticDenseGguf(tempDir, dim: 64, headDim: 16);
            using var engine = new ContinuousBatchEngine(
                ggufPath,
                totalKvBlocks: 32,
                maxBatchSize: 4,
                maxSeqLen: 64,
                seed: 7,
                useGpu: useGpu,
                kvCachePrecision: requested);

            Assert.Equal(expectedResolved, engine.KvPrecision);
            Assert.Equal(expectedResolved, engine.BlockPool.Precision);

            if (useGpu && (requested == KvCachePrecision.Fp32 || requested == KvCachePrecision.Fp8))
            {
                Assert.False(engine.IsGpuAccelerated);
            }

            var (state, stream) = await engine.EnqueueAsync(new InferenceRequest
            {
                Prompt = "hello world",
                MaxTokens = 6,
                Temperature = 0.0f
            });

            await foreach (var _ in stream) { }
            string output = await state.Completion.Task;
            Assert.False(string.IsNullOrEmpty(output));
            Assert.Equal(6, state.GeneratedTokens.Count);
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public async Task ContinuousBatchEngine_GpuQuantizedKv_Q8_0_And_Q4_0_GenerateValidOutput()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"m4b_engine_gpukv_{Guid.NewGuid():N}");
        try
        {
            string ggufPath = M4bTestHelpers.CreateSyntheticDenseGguf(tempDir, dim: 128, ffnDim: 256, nHeads: 4, nHeadsKv: 2, headDim: 32, layers: 2);
            foreach (var prec in new[] { KvCachePrecision.Fp16, KvCachePrecision.Q8_0, KvCachePrecision.Q4_0 })
            {
                using var engine = new ContinuousBatchEngine(
                    ggufPath,
                    totalKvBlocks: 32,
                    maxBatchSize: 4,
                    maxSeqLen: 64,
                    seed: 11,
                    useGpu: true,
                    kvCachePrecision: prec);

                Assert.Equal(prec, engine.KvPrecision);
                var (state, stream) = await engine.EnqueueAsync(new InferenceRequest
                {
                    Prompt = "hello alpha beta",
                    MaxTokens = 6,
                    Temperature = 0.0f
                });
                await foreach (var _ in stream) { }
                await state.Completion.Task;
                Assert.Equal(6, state.GeneratedTokens.Count);
            }
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }
}

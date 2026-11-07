using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Glacier.Inference.Diagnostics;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;

namespace Glacier.Serve.Inference.PagedAttention;

/// <summary>
/// VRAM-resident NVIDIA CUDA PagedAttention runner backed by <see cref="Qwen2GpuModel"/>.
/// Allocates FP16 physical Key/Value pools (<c>dKPool</c>, <c>dVPool</c>) and a reusable
/// device block-table buffer (<c>dBlockTable</c>) directly in GPU memory, eliminating
/// host-side contiguous KV duplication and executing prefill/decode via custom PTX kernels.
/// </summary>
public sealed unsafe class CudaPagedTransformerRunner : IPagedModelRunner
{
    private const ulong SafetyVramHeadroomBytes = 128UL * 1024 * 1024;

    private readonly GpuContext _gpu;
    private readonly Qwen2GpuModel _gpuModel;
    private readonly ModelWeights _weights;
    private readonly int _vocabSize;
    private readonly int _maxSeqLen;
    private readonly int _maxBatchSize;
    private readonly int _blockSize;
    private readonly int _maxBlocksPerSeq;
    private readonly bool _supportsNativeBatchGemm;

    private IntPtr _dKPool;
    private IntPtr _dVPool;
    private IntPtr _dBlockTable;
    private bool _disposed;

    public bool IsGpuAccelerated => true;
    public string BackendName { get; }
    public int VocabSize => _vocabSize;
    public int MaxSeqLen => _maxSeqLen;
    public int MaxBatchSize => _maxBatchSize;
    public bool SupportsNativeBatchGemm => _supportsNativeBatchGemm;

    private CudaPagedTransformerRunner(
        GpuContext gpu,
        Qwen2GpuModel gpuModel,
        ModelWeights weights,
        IntPtr dKPool,
        IntPtr dVPool,
        IntPtr dBlockTable,
        int blockSize,
        int maxSeqLen,
        int maxBatchSize,
        int maxBlocksPerSeq,
        bool supportsNativeBatchGemm)
    {
        _gpu = gpu;
        _gpuModel = gpuModel;
        _weights = weights;
        _dKPool = dKPool;
        _dVPool = dVPool;
        _dBlockTable = dBlockTable;
        _blockSize = blockSize;
        _maxSeqLen = maxSeqLen;
        _maxBatchSize = maxBatchSize;
        _maxBlocksPerSeq = maxBlocksPerSeq;
        _vocabSize = weights.VocabSize;
        _supportsNativeBatchGemm = supportsNativeBatchGemm;
        BackendName = $"CUDA ({gpu.DeviceName})";
    }

    /// <summary>
    /// Attempts to initialize a VRAM-resident CUDA PagedAttention runner when CUDA hardware is available,
    /// the loaded model architecture/quantization is supported by the PTX kernels, and sufficient VRAM exists.
    /// Falls back cleanly (returns false and disposes any partial allocations) on any unsupported configuration or OOM.
    /// </summary>
    public static bool TryCreate(
        ModelWeights weights,
        int totalBlocks,
        int blockSize,
        int maxSeqLen,
        int maxBatchSize,
        [NotNullWhen(true)] out CudaPagedTransformerRunner? runner)
    {
        runner = null;
        if (weights == null || totalBlocks <= 0 || blockSize <= 0 || maxSeqLen <= 0)
            return false;

        if (!GpuContext.IsSupported)
            return false;

        if (!IsModelSupportedOnGpu(weights, out bool supportsNativeBatchGemm, out string? unsupportedReason))
        {
            GlacierDiagnostics.LogInformation(
                $"[CudaPagedTransformerRunner] Falling back to CPU PagedTransformerRunner: {unsupportedReason}");
            return false;
        }

        GpuContext? gpu = null;
        Qwen2GpuModel? gpuModel = null;
        IntPtr dKPool = IntPtr.Zero;
        IntPtr dVPool = IntPtr.Zero;
        IntPtr dBlockTable = IntPtr.Zero;

        try
        {
            gpu = new GpuContext(0);
            CuDriver.CtxSetCurrent(gpu.Handle);

            int layers = weights.BlockCount;
            int headsKv = weights.HeadCountKv;
            int headDim = weights.HeadDim;
            int vHeadDim = weights.ValueDim > 0 ? weights.ValueDim : headDim;
            int maxBlocksPerSeq = (maxSeqLen + blockSize - 1) / blockSize;

            int effectiveMaxBatch = Math.Max(maxBatchSize, Qwen2GpuModel.MaxBatchSize);
            nuint poolBytesK = checked((nuint)((long)totalBlocks * layers * headsKv * blockSize * headDim * sizeof(ushort)));
            nuint poolBytesV = checked((nuint)((long)totalBlocks * layers * headsKv * blockSize * vHeadDim * sizeof(ushort)));
            nuint blockTableBytes = checked((nuint)((long)effectiveMaxBatch * Math.Max(1, maxBlocksPerSeq) * sizeof(int)));

            ulong estimatedModelBytes = EstimateModelVramBytes(weights);
            ulong requiredBytes = estimatedModelBytes + poolBytesK + poolBytesV + blockTableBytes + SafetyVramHeadroomBytes;

            var (freeBytes, _) = gpu.GetMemoryInfo();
            if (freeBytes < requiredBytes)
            {
                GlacierDiagnostics.LogWarning(
                    $"[CudaPagedTransformerRunner] Insufficient free VRAM ({freeBytes / (1024 * 1024)} MB free < {requiredBytes / (1024 * 1024)} MB required); falling back to CPU.");
                gpu.Dispose();
                return false;
            }

            gpuModel = new Qwen2GpuModel(
                gpu,
                weights,
                maxSeqLen: maxSeqLen,
                kvPrecision: KvCachePrecision.Fp16,
                allocateContiguousKvCache: false);

            dKPool = gpu.AllocateDevice(poolBytesK);
            dVPool = gpu.AllocateDevice(poolBytesV);
            dBlockTable = gpu.AllocateDevice(blockTableBytes);

            CuDriver.Check(CuDriver.MemsetD8(dKPool, 0, poolBytesK), "MemsetD8(dKPool)");
            CuDriver.Check(CuDriver.MemsetD8(dVPool, 0, poolBytesV), "MemsetD8(dVPool)");
            CuDriver.Check(CuDriver.MemsetD8(dBlockTable, 0, blockTableBytes), "MemsetD8(dBlockTable)");
            gpu.Synchronize();

            runner = new CudaPagedTransformerRunner(
                gpu,
                gpuModel,
                weights,
                dKPool,
                dVPool,
                dBlockTable,
                blockSize,
                maxSeqLen,
                Math.Max(maxBatchSize, Qwen2GpuModel.MaxBatchSize),
                maxBlocksPerSeq,
                supportsNativeBatchGemm);

            return true;
        }
        catch (Exception ex)
        {
            GlacierDiagnostics.LogWarning(
                $"[CudaPagedTransformerRunner] CUDA initialization failed ({ex.Message}); falling back to CPU PagedTransformerRunner.",
                ex);

            try
            {
                if (gpu != null)
                {
                    if (dBlockTable != IntPtr.Zero) gpu.FreeDevice(dBlockTable);
                    if (dVPool != IntPtr.Zero) gpu.FreeDevice(dVPool);
                    if (dKPool != IntPtr.Zero) gpu.FreeDevice(dKPool);
                }
                gpuModel?.Dispose();
                gpu?.Dispose();
            }
            catch
            {
                // Ignore secondary cleanup errors during fallback
            }

            runner = null;
            return false;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public bool ForwardToken(
        int token,
        int pos,
        BlockTable blockTable,
        PagedBlockPool pool,
        Span<float> logits,
        bool computeLogits = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (pos < 0 || pos >= _maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(pos), pos, $"Token position {pos} exceeds valid sequence range [0, {_maxSeqLen - 1}].");

        CuDriver.CtxSetCurrent(_gpu.Handle);

        if (pos == blockTable.TokenCount)
        {
            if (!blockTable.AppendToken(out _, out _))
                return false;
        }
        else if (pos > blockTable.TokenCount)
        {
            throw new ArgumentOutOfRangeException(nameof(pos), pos, $"Cannot evaluate non-contiguous position {pos} when BlockTable has {blockTable.TokenCount} tokens.");
        }

        UploadBlockTable(blockTable);
        _gpuModel.ForwardPagedToken(
            token,
            pos,
            _dKPool,
            _dVPool,
            _dBlockTable,
            _blockSize,
            computeLogits ? logits : Span<float>.Empty,
            computeLogits && !logits.IsEmpty);

        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public bool ForwardPrefillChunk(
        ReadOnlySpan<int> tokens,
        int startPos,
        BlockTable blockTable,
        PagedBlockPool pool,
        Span<float> logits,
        bool computeLogits = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (tokens.IsEmpty) return true;
        if (startPos < 0 || startPos + tokens.Length > _maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(startPos), startPos, $"Prefill chunk [{startPos}, {startPos + tokens.Length}) exceeds maxSeqLen {_maxSeqLen}.");

        CuDriver.CtxSetCurrent(_gpu.Handle);

        // Reserve physical block slots in BlockTable for the entire chunk so free-block accounting stays exact
        for (int t = 0; t < tokens.Length; t++)
        {
            int pos = startPos + t;
            if (pos == blockTable.TokenCount)
            {
                if (!blockTable.AppendToken(out _, out _))
                    return false;
            }
        }

        UploadBlockTable(blockTable);

        if (tokens.Length == 1)
        {
            _gpuModel.ForwardPagedToken(
                tokens[0],
                startPos,
                _dKPool,
                _dVPool,
                _dBlockTable,
                _blockSize,
                computeLogits ? logits : Span<float>.Empty,
                computeLogits && !logits.IsEmpty);
            return true;
        }

        if (_supportsNativeBatchGemm)
        {
            _gpuModel.ForwardPagedBatch(
                tokens,
                startPos,
                _dKPool,
                _dVPool,
                _dBlockTable,
                _blockSize,
                computeLogits ? logits : Span<float>.Empty,
                computeLogits && !logits.IsEmpty);
        }
        else
        {
            for (int t = 0; t < tokens.Length; t++)
            {
                bool isLast = (t == tokens.Length - 1) && computeLogits && !logits.IsEmpty;
                _gpuModel.ForwardPagedToken(
                    tokens[t],
                    startPos + t,
                    _dKPool,
                    _dVPool,
                    _dBlockTable,
                    _blockSize,
                    isLast ? logits : Span<float>.Empty,
                    isLast);
            }
        }

        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public bool ForwardBatch(
        ReadOnlySpan<int> tokens,
        ReadOnlySpan<int> positions,
        IReadOnlyList<BlockTable> blockTables,
        PagedBlockPool pool,
        Span<float> logitsBatch,
        Span<bool> successFlags,
        bool computeLogits = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int batchSize = tokens.Length;
        if (batchSize == 0) return true;
        if (batchSize > _maxBatchSize)
            throw new ArgumentOutOfRangeException(nameof(tokens), $"Batch size {batchSize} exceeds maximum runner capacity {_maxBatchSize}.");

        for (int b = 0; b < batchSize; b++)
        {
            if (positions[b] < 0 || positions[b] >= _maxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions), positions[b], $"Sequence {b} position {positions[b]} exceeds valid range [0, {_maxSeqLen - 1}].");
        }

        CuDriver.CtxSetCurrent(_gpu.Handle);

        for (int b = 0; b < batchSize; b++)
        {
            int pos = positions[b];
            var blockTable = blockTables[b];
            if (pos == blockTable.TokenCount)
            {
                if (!blockTable.AppendToken(out _, out _))
                {
                    successFlags[b] = false;
                    continue;
                }
            }

            successFlags[b] = true;
            UploadBlockTable(blockTable, b);

            Span<float> seqLogits = (computeLogits && !logitsBatch.IsEmpty)
                ? logitsBatch.Slice(b * _vocabSize, _vocabSize)
                : Span<float>.Empty;

            IntPtr seqBlockTable = _dBlockTable + (b * _maxBlocksPerSeq * sizeof(int));

            _gpuModel.ForwardPagedToken(
                tokens[b],
                pos,
                _dKPool,
                _dVPool,
                seqBlockTable,
                _blockSize,
                seqLogits,
                computeLogits && !seqLogits.IsEmpty);
        }

        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void UploadBlockTable(BlockTable blockTable, int batchIndex = 0)
    {
        ReadOnlySpan<int> hostBlocks = blockTable.AsSpan();
        if (hostBlocks.IsEmpty) return;
        if (hostBlocks.Length > _maxBlocksPerSeq)
            throw new ArgumentOutOfRangeException(nameof(blockTable), hostBlocks.Length, $"Sequence block count {hostBlocks.Length} exceeds maxBlocksPerSeq {_maxBlocksPerSeq}.");

        IntPtr targetDevicePtr = _dBlockTable + (batchIndex * _maxBlocksPerSeq * sizeof(int));
        fixed (int* pBlocks = hostBlocks)
        {
            _gpu.CopyToDevice(targetDevicePtr, (IntPtr)pBlocks, (nuint)(hostBlocks.Length * sizeof(int)));
        }
    }

    private static bool IsModelSupportedOnGpu(ModelWeights weights, out bool supportsNativeBatchGemm, out string? reason)
    {
        supportsNativeBatchGemm = true;
        reason = null;

        if (weights.IsMoe || weights.IsMla || weights.IsHybridSsm)
        {
            reason = "MoE, MLA, and Hybrid SSM architectures are not supported by the dense CUDA PagedAttention kernel.";
            return false;
        }

        if (weights.ArchitectureFamily is not (
            UniversalArchitecture.Qwen or
            UniversalArchitecture.Llama or
            UniversalArchitecture.Mistral or
            UniversalArchitecture.Generic))
        {
            reason = $"Architecture '{weights.ArchitectureFamily}' uses specialized layer features requiring CPU runner.";
            return false;
        }

        if (weights.ValueDim > 0 && weights.ValueDim != weights.HeadDim)
        {
            reason = $"Asymmetric vHeadDim ({weights.ValueDim}) != headDim ({weights.HeadDim}) is not supported by Qwen2GpuModel.";
            return false;
        }

        if (weights.HeadDim <= 0 || (weights.HeadDim & 1) != 0 || weights.HeadDim > 256)
        {
            reason = $"Unsupported headDim {weights.HeadDim} for CUDA PagedAttention kernel.";
            return false;
        }

        if (weights.RopeDimensionCount != weights.HeadDim ||
            weights.RopeFreqsWeight != null ||
            weights.Gguf.RopeScalingType.Equals("yarn", StringComparison.OrdinalIgnoreCase))
        {
            reason = "Partial RoPE or YaRN frequency scaling requires CPU PagedTransformerRunner.";
            return false;
        }

        if (!IsSupportedGemvType(weights.OutType))
        {
            reason = $"LM head quantization type '{weights.OutType}' is not supported by CUDA GEMV.";
            return false;
        }

        for (int l = 0; l < weights.BlockCount; l++)
        {
            var lw = weights.Layers[l];
            if (lw.HasFusedQkv || lw.HasFusedGateUp || lw.AttnSinksWeight != null || lw.AttnOutBias != null)
            {
                reason = $"Layer {l} uses fused projections, attention sinks, or output bias unsupported on CUDA.";
                return false;
            }

            if (!IsSupportedGemvType(lw.QType) ||
                !IsSupportedGemvType(lw.KType) ||
                !IsSupportedGemvType(lw.VType) ||
                !IsSupportedGemvType(lw.AttnOutType) ||
                !IsSupportedGemvType(lw.FfnGateType) ||
                !IsSupportedGemvType(lw.FfnUpType) ||
                !IsSupportedGemvType(lw.FfnDownType))
            {
                reason = $"Layer {l} uses quantization types unsupported by CUDA GEMV.";
                return false;
            }

            if (!IsSupportedBatchGemmType(lw.QType) ||
                !IsSupportedBatchGemmType(lw.KType) ||
                !IsSupportedBatchGemmType(lw.VType) ||
                !IsSupportedBatchGemmType(lw.AttnOutType) ||
                !IsSupportedBatchGemmType(lw.FfnDownType))
            {
                supportsNativeBatchGemm = false;
            }
        }

        return true;
    }

    private static bool IsSupportedGemvType(GgufType type) => type is
        GgufType.Q4_K or
        GgufType.Q5_K or
        GgufType.Q3_K or
        GgufType.Q6_K or
        GgufType.Q8_0 or
        GgufType.F32;

    private static bool IsSupportedBatchGemmType(GgufType type) => type is
        GgufType.Q4_K or
        GgufType.Q6_K or
        GgufType.Q8_0;

    private static ulong EstimateModelVramBytes(ModelWeights weights)
    {
        // Conservative upper-bound estimate of weights + batch scratch in VRAM
        long dim = weights.EmbeddingLength;
        long ffn = weights.FeedForwardLength;
        long qDim = (long)weights.HeadCount * weights.HeadDim;
        long kvDim = (long)weights.HeadCountKv * weights.HeadDim;
        long perLayerParams = dim * (qDim + 2 * kvDim + dim + 3 * ffn);
        long totalParams = perLayerParams * weights.BlockCount + dim * weights.VocabSize;
        bool isF32 = weights.OutType == GgufType.F32 || (weights.BlockCount > 0 && weights.Layers[0].QType == GgufType.F32);
        ulong weightBytes = isF32 ? (ulong)totalParams * 4UL : (ulong)totalParams; // ~1 byte/param upper bound for quantized GGUF
        ulong scratchBytes = (ulong)((dim + qDim + 2 * kvDim + ffn * 2 + weights.VocabSize) * 4L * 40L);
        return weightBytes + scratchBytes;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            try
            {
                CuDriver.CtxSetCurrent(_gpu.Handle);
                if (_dBlockTable != IntPtr.Zero)
                {
                    _gpu.FreeDevice(_dBlockTable);
                    _dBlockTable = IntPtr.Zero;
                }
                if (_dVPool != IntPtr.Zero)
                {
                    _gpu.FreeDevice(_dVPool);
                    _dVPool = IntPtr.Zero;
                }
                if (_dKPool != IntPtr.Zero)
                {
                    _gpu.FreeDevice(_dKPool);
                    _dKPool = IntPtr.Zero;
                }
            }
            catch
            {
                // Ignore context teardown errors
            }

            _gpuModel.Dispose();
            _gpu.Dispose();
        }
        GC.SuppressFinalize(this);
    }
}

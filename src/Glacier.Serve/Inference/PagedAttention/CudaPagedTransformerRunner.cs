using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Glacier.Inference.Diagnostics;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;

namespace Glacier.Serve.Inference.PagedAttention;

/// <summary>
/// VRAM-resident NVIDIA CUDA PagedAttention runner backed by <see cref="Qwen2GpuModel"/>.
/// Allocates quantized (<see cref="KvCachePrecision.Fp16"/>, <see cref="KvCachePrecision.Q8_0"/>, or
/// <see cref="KvCachePrecision.Q4_0"/>) physical Key/Value pools (<c>dKPool</c>, <c>dVPool</c>) and a reusable
/// device block-table buffer (<c>dBlockTable</c>) directly in GPU memory, eliminating
/// host-side contiguous KV duplication and executing prefill/decode via custom PTX kernels.
/// </summary>
public sealed unsafe class CudaPagedTransformerRunner : IPagedModelRunner, IHiddenStateSource, IPagedHiddenStateContext
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
    private readonly KvCachePrecision _requestedKvPrecision;
    private readonly KvCachePrecision _kvPrecision;
    private readonly long _deviceKvPoolBytes;
    private readonly long _blockByteStrideK;
    private readonly long _blockByteStrideV;

    private IntPtr _dKPool;
    private IntPtr _dVPool;
    private IntPtr _dBlockTable;
    private readonly int _totalBlocks;
    private readonly object _loraFallbackLock = new();
    private PagedTransformerRunner? _cpuLoraRunner;
    private PagedBlockPool? _hostLoraPool;
    private PagedBlockPool? _attachedEnginePool;
    private bool _historyEnabled;
    private float[]? _pendingHiddenHistory;
    private IReadOnlyList<float[]?>? _pendingBatchHistories;
    private bool _disposed;

    public bool IsGpuAccelerated => true;
    public string BackendName { get; }
    public int HiddenDim => _weights.EmbeddingLength;
    public int VocabSize => _vocabSize;
    public int MaxSeqLen => _maxSeqLen;
    public int MaxBatchSize => _maxBatchSize;
    public bool SupportsNativeBatchGemm => _supportsNativeBatchGemm;
    public KvCachePrecision KvPrecision => _kvPrecision;
    public long DeviceKvPoolBytes => _deviceKvPoolBytes;

    public bool SupportsHiddenStateCapture
    {
        get
        {
            lock (_loraFallbackLock)
            {
                return _cpuLoraRunner != null;
            }
        }
    }

    bool IPagedHiddenStateContext.SupportsHiddenStateCapture(LoraAdapterWeights? lora) => lora != null;

    void IPagedHiddenStateContext.BindHiddenStateHistory(float[]? history)
    {
        lock (_loraFallbackLock)
        {
            _pendingHiddenHistory = history;
            _cpuLoraRunner?.BindHiddenStateHistory(history);
        }
    }

    void IPagedHiddenStateContext.BindBatchHiddenStateHistories(IReadOnlyList<float[]?>? histories)
    {
        lock (_loraFallbackLock)
        {
            _pendingBatchHistories = histories;
            _cpuLoraRunner?.BindBatchHiddenStateHistories(histories);
        }
    }

    public ReadOnlySpan<float> GetLastHiddenState()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        lock (_loraFallbackLock)
        {
            return _cpuLoraRunner != null ? _cpuLoraRunner.GetLastHiddenState() : ReadOnlySpan<float>.Empty;
        }
    }

    public void CopyTokenEmbedding(int tokenId, Span<float> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int dim = _weights.EmbeddingLength;
        if (destination.Length < dim)
            throw new ArgumentException($"Destination span too small ({destination.Length} < {dim}).", nameof(destination));
        if (_weights.EmbdWeight == null)
            throw new InvalidOperationException("Model has no token_embd.weight pointer.");
        fixed (float* pDst = destination)
        {
            QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, tokenId, pDst, dim);
        }
    }

    public void EnablePositionHistory()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        lock (_loraFallbackLock)
        {
            _historyEnabled = true;
            _cpuLoraRunner?.EnablePositionHistory();
        }
    }

    public bool TryGetHiddenStateAt(int position, Span<float> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        lock (_loraFallbackLock)
        {
            return _cpuLoraRunner != null && _cpuLoraRunner.TryGetHiddenStateAt(position, destination);
        }
    }

    private CudaPagedTransformerRunner(
        GpuContext gpu,
        Qwen2GpuModel gpuModel,
        ModelWeights weights,
        IntPtr dKPool,
        IntPtr dVPool,
        IntPtr dBlockTable,
        int totalBlocks,
        int blockSize,
        int maxSeqLen,
        int maxBatchSize,
        int maxBlocksPerSeq,
        bool supportsNativeBatchGemm,
        KvCachePrecision requestedKvPrecision,
        KvCachePrecision kvPrecision,
        long deviceKvPoolBytes,
        long blockByteStrideK,
        long blockByteStrideV)
    {
        _gpu = gpu;
        _gpuModel = gpuModel;
        _weights = weights;
        _dKPool = dKPool;
        _dVPool = dVPool;
        _dBlockTable = dBlockTable;
        _totalBlocks = totalBlocks;
        _blockSize = blockSize;
        _maxSeqLen = maxSeqLen;
        _maxBatchSize = maxBatchSize;
        _maxBlocksPerSeq = maxBlocksPerSeq;
        _vocabSize = weights.VocabSize;
        _supportsNativeBatchGemm = supportsNativeBatchGemm;
        _requestedKvPrecision = requestedKvPrecision;
        _kvPrecision = kvPrecision;
        _deviceKvPoolBytes = deviceKvPoolBytes;
        _blockByteStrideK = blockByteStrideK;
        _blockByteStrideV = blockByteStrideV;
        BackendName = $"CUDA ({gpu.DeviceName})";
    }

    /// <summary>
    /// Attempts to initialize a VRAM-resident CUDA PagedAttention runner using <see cref="KvCachePrecision.Auto"/> (<see cref="KvCachePrecision.Fp16"/>).
    /// </summary>
    public static bool TryCreate(
        ModelWeights weights,
        int totalBlocks,
        int blockSize,
        int maxSeqLen,
        int maxBatchSize,
        [NotNullWhen(true)] out CudaPagedTransformerRunner? runner) =>
        TryCreate(weights, totalBlocks, blockSize, maxSeqLen, maxBatchSize, KvCachePrecision.Auto, out runner);

    /// <summary>
    /// Attempts to initialize a VRAM-resident CUDA PagedAttention runner for the requested <paramref name="kvPrecision"/>
    /// (<see cref="KvCachePrecision.Auto"/> -> <see cref="KvCachePrecision.Fp16"/>, <see cref="KvCachePrecision.Q8_0"/>, or <see cref="KvCachePrecision.Q4_0"/>).
    /// Returns false cleanly for unsupported precisions (<see cref="KvCachePrecision.Fp32"/>, <see cref="KvCachePrecision.Fp8"/>) or OOM so callers fall back to CPU.
    /// </summary>
    public static bool TryCreate(
        ModelWeights weights,
        int totalBlocks,
        int blockSize,
        int maxSeqLen,
        int maxBatchSize,
        KvCachePrecision kvPrecision,
        [NotNullWhen(true)] out CudaPagedTransformerRunner? runner)
    {
        runner = null;
        if (weights == null || totalBlocks <= 0 || blockSize <= 0 || maxSeqLen <= 0)
            return false;

        KvCachePrecision resolvedPrecision = kvPrecision == KvCachePrecision.Auto ? KvCachePrecision.Fp16 : kvPrecision;
        if (resolvedPrecision is not (KvCachePrecision.Fp16 or KvCachePrecision.Q8_0 or KvCachePrecision.Q4_0))
        {
            GlacierDiagnostics.LogInformation(
                $"[CudaPagedTransformerRunner] KV cache precision '{resolvedPrecision}' requires CPU PagedTransformerRunner; falling back to CPU.");
            return false;
        }

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
            var (poolBytesK, poolBytesV) = Qwen2GpuModel.GetPagedPoolByteSizes(
                totalBlocks, layers, headsKv, headDim, vHeadDim, blockSize, resolvedPrecision);
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
                kvPrecision: resolvedPrecision,
                allocateContiguousKvCache: false);

            dKPool = gpu.AllocateDevice(poolBytesK);
            dVPool = gpu.AllocateDevice(poolBytesV);
            dBlockTable = gpu.AllocateDevice(blockTableBytes);

            CuDriver.Check(CuDriver.MemsetD8(dKPool, 0, poolBytesK), "MemsetD8(dKPool)");
            CuDriver.Check(CuDriver.MemsetD8(dVPool, 0, poolBytesV), "MemsetD8(dVPool)");
            CuDriver.Check(CuDriver.MemsetD8(dBlockTable, 0, blockTableBytes), "MemsetD8(dBlockTable)");
            gpu.Synchronize();

            long totalPoolBytes = checked((long)poolBytesK + (long)poolBytesV);
            long blockStrideK = checked((long)poolBytesK / totalBlocks);
            long blockStrideV = checked((long)poolBytesV / totalBlocks);

            runner = new CudaPagedTransformerRunner(
                gpu,
                gpuModel,
                weights,
                dKPool,
                dVPool,
                dBlockTable,
                totalBlocks,
                blockSize,
                maxSeqLen,
                effectiveMaxBatch,
                maxBlocksPerSeq,
                supportsNativeBatchGemm,
                kvPrecision,
                resolvedPrecision,
                totalPoolBytes,
                blockStrideK,
                blockStrideV);

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
            catch (Exception cleanupEx)
            {
                GlacierDiagnostics.LogDebug($"[CudaPagedTransformerRunner] Secondary cleanup during fallback failed: {cleanupEx.Message}");
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
        bool computeLogits = true) =>
        ForwardToken(token, pos, blockTable, pool, logits, computeLogits, lora: null);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public bool ForwardToken(
        int token,
        int pos,
        BlockTable blockTable,
        PagedBlockPool pool,
        Span<float> logits,
        bool computeLogits,
        LoraAdapterWeights? lora)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (pos < 0 || pos >= _maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(pos), pos, $"Token position {pos} exceeds valid sequence range [0, {_maxSeqLen - 1}].");

        ValidatePoolPrecision(pool);

        if (lora != null)
        {
            PagedBlockPool targetPool = EnsureLoraFallbackPool(pool);
            return _cpuLoraRunner!.ForwardToken(token, pos, blockTable, targetPool, logits, computeLogits, lora);
        }

        EnsureEnginePoolForkMirror(pool);
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
            _kvPrecision,
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
        bool computeLogits = true) =>
        ForwardPrefillChunk(tokens, startPos, blockTable, pool, logits, computeLogits, lora: null);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public bool ForwardPrefillChunk(
        ReadOnlySpan<int> tokens,
        int startPos,
        BlockTable blockTable,
        PagedBlockPool pool,
        Span<float> logits,
        bool computeLogits,
        LoraAdapterWeights? lora)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (tokens.IsEmpty) return true;
        if (startPos < 0 || startPos + tokens.Length > _maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(startPos), startPos, $"Prefill chunk [{startPos}, {startPos + tokens.Length}) exceeds maxSeqLen {_maxSeqLen}.");

        ValidatePoolPrecision(pool);

        if (lora != null)
        {
            PagedBlockPool targetPool = EnsureLoraFallbackPool(pool);
            return _cpuLoraRunner!.ForwardPrefillChunk(tokens, startPos, blockTable, targetPool, logits, computeLogits, lora);
        }

        EnsureEnginePoolForkMirror(pool);
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
            else if (pos > blockTable.TokenCount)
            {
                throw new ArgumentOutOfRangeException(nameof(startPos), pos, $"Cannot evaluate non-contiguous position {pos} when BlockTable has {blockTable.TokenCount} tokens.");
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
                _kvPrecision,
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
                _kvPrecision,
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
                    _kvPrecision,
                    isLast ? logits : Span<float>.Empty,
                    isLast);
            }
        }

        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public bool VerifyDraftChunk(
        ReadOnlySpan<int> tokens,
        int startPos,
        BlockTable blockTable,
        PagedBlockPool pool,
        Span<float> logitsOut) =>
        VerifyDraftChunk(tokens, startPos, blockTable, pool, logitsOut, lora: null);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public bool VerifyDraftChunk(
        ReadOnlySpan<int> tokens,
        int startPos,
        BlockTable blockTable,
        PagedBlockPool pool,
        Span<float> logitsOut,
        LoraAdapterWeights? lora)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (tokens.IsEmpty) return true;
        if (startPos < 0 || startPos + tokens.Length > _maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(startPos), startPos, $"Draft verify chunk [{startPos}, {startPos + tokens.Length}) exceeds maxSeqLen {_maxSeqLen}.");
        if (logitsOut.Length < tokens.Length * _vocabSize)
            throw new ArgumentException($"logitsOut too small ({logitsOut.Length} < {tokens.Length * _vocabSize}).", nameof(logitsOut));

        ValidatePoolPrecision(pool);

        if (lora != null)
        {
            PagedBlockPool targetPool = EnsureLoraFallbackPool(pool);
            return _cpuLoraRunner!.VerifyDraftChunk(tokens, startPos, blockTable, targetPool, logitsOut, lora);
        }

        EnsureEnginePoolForkMirror(pool);
        CuDriver.CtxSetCurrent(_gpu.Handle);

        int initialTokens = blockTable.TokenCount;
        for (int t = 0; t < tokens.Length; t++)
        {
            int pos = startPos + t;
            if (pos == blockTable.TokenCount)
            {
                if (!blockTable.AppendToken(out _, out _))
                {
                    blockTable.TruncateTo(initialTokens);
                    return false;
                }
            }
            else if (pos > blockTable.TokenCount)
            {
                blockTable.TruncateTo(initialTokens);
                throw new ArgumentOutOfRangeException(nameof(startPos), pos, $"Cannot verify non-contiguous position {pos} when BlockTable has {blockTable.TokenCount} tokens.");
            }
        }

        UploadBlockTable(blockTable);
        for (int t = 0; t < tokens.Length; t++)
        {
            _gpuModel.ForwardPagedToken(
                tokens[t],
                startPos + t,
                _dKPool,
                _dVPool,
                _dBlockTable,
                _blockSize,
                _kvPrecision,
                logitsOut.Slice(t * _vocabSize, _vocabSize),
                computeLogits: true);
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
        bool computeLogits = true) =>
        ForwardBatch(tokens, positions, blockTables, pool, logitsBatch, successFlags, computeLogits, loras: null);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public bool ForwardBatch(
        ReadOnlySpan<int> tokens,
        ReadOnlySpan<int> positions,
        IReadOnlyList<BlockTable> blockTables,
        PagedBlockPool pool,
        Span<float> logitsBatch,
        Span<bool> successFlags,
        bool computeLogits,
        IReadOnlyList<LoraAdapterWeights?>? loras)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int batchSize = tokens.Length;
        if (batchSize == 0) return true;
        if (batchSize > _maxBatchSize)
            throw new ArgumentOutOfRangeException(nameof(tokens), $"Batch size {batchSize} exceeds maximum runner capacity {_maxBatchSize}.");
        if (loras != null && loras.Count < batchSize)
            throw new ArgumentException($"LoRA adapter list length ({loras.Count}) must be at least batch size ({batchSize}).", nameof(loras));

        ValidatePoolPrecision(pool);

        for (int b = 0; b < batchSize; b++)
        {
            if (positions[b] < 0 || positions[b] >= _maxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions), positions[b], $"Sequence {b} position {positions[b]} exceeds valid range [0, {_maxSeqLen - 1}].");
        }

        int loraCount = 0;
        if (loras != null)
        {
            for (int b = 0; b < batchSize; b++)
            {
                if (loras[b] != null) loraCount++;
            }
        }

        if (loraCount < batchSize)
        {
            EnsureEnginePoolForkMirror(pool);
            CuDriver.CtxSetCurrent(_gpu.Handle);
            for (int b = 0; b < batchSize; b++)
            {
                if (loras != null && loras[b] != null)
                    continue;

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
                    _kvPrecision,
                    seqLogits,
                    computeLogits && !seqLogits.IsEmpty);
            }
        }

        if (loraCount > 0)
        {
            PagedBlockPool targetPool = EnsureLoraFallbackPool(pool);
            if (loraCount == batchSize)
            {
                return _cpuLoraRunner!.ForwardBatch(tokens, positions, blockTables, targetPool, logitsBatch, successFlags, computeLogits, loras);
            }

            int[] subTokens = new int[loraCount];
            int[] subPositions = new int[loraCount];
            var subBlockTables = new BlockTable[loraCount];
            var subLoras = new LoraAdapterWeights?[loraCount];
            float[]?[]? subHistories = _pendingBatchHistories != null ? new float[]?[loraCount] : null;
            bool[] subSuccess = new bool[loraCount];
            int[] origIndices = new int[loraCount];
            int k = 0;
            for (int b = 0; b < batchSize; b++)
            {
                if (loras![b] != null)
                {
                    subTokens[k] = tokens[b];
                    subPositions[k] = positions[b];
                    subBlockTables[k] = blockTables[b];
                    subLoras[k] = loras[b];
                    if (subHistories != null && b < _pendingBatchHistories!.Count)
                        subHistories[k] = _pendingBatchHistories[b];
                    origIndices[k] = b;
                    k++;
                }
            }

            bool emitLogits = computeLogits && !logitsBatch.IsEmpty;
            float[]? rentedLogits = emitLogits ? ArrayPool<float>.Shared.Rent(loraCount * _vocabSize) : null;
            try
            {
                Span<float> subLogits = rentedLogits != null
                    ? rentedLogits.AsSpan(0, loraCount * _vocabSize)
                    : Span<float>.Empty;

                if (subHistories != null)
                    _cpuLoraRunner!.BindBatchHiddenStateHistories(subHistories);

                _cpuLoraRunner!.ForwardBatch(subTokens, subPositions, subBlockTables, targetPool, subLogits, subSuccess, computeLogits, subLoras);

                for (int i = 0; i < loraCount; i++)
                {
                    int origIdx = origIndices[i];
                    successFlags[origIdx] = subSuccess[i];
                    if (emitLogits && subSuccess[i])
                    {
                        subLogits.Slice(i * _vocabSize, _vocabSize).CopyTo(logitsBatch.Slice(origIdx * _vocabSize, _vocabSize));
                    }
                }
            }
            finally
            {
                if (subHistories != null)
                    _cpuLoraRunner!.BindBatchHiddenStateHistories(_pendingBatchHistories);
                if (rentedLogits != null)
                    ArrayPool<float>.Shared.Return(rentedLogits);
            }
        }

        return true;
    }

    private void ValidatePoolPrecision(PagedBlockPool pool)
    {
        ArgumentNullException.ThrowIfNull(pool);
        if (pool.HasHostBuffers)
        {
            if (pool.HasExplicitPrecision && pool.Precision != _kvPrecision)
            {
                throw new ArgumentException(
                    $"Host PagedBlockPool precision '{pool.Precision}' does not match CudaPagedTransformerRunner precision '{_kvPrecision}'.",
                    nameof(pool));
            }
        }
        else if (pool.Precision != KvCachePrecision.Auto && pool.Precision != _kvPrecision)
        {
            throw new ArgumentException(
                $"PagedBlockPool precision '{pool.Precision}' does not match CudaPagedTransformerRunner precision '{_kvPrecision}'.",
                nameof(pool));
        }
    }

    private void EnsureEnginePoolForkMirror(PagedBlockPool enginePool)
    {
        if (enginePool.HasHostBuffers || ReferenceEquals(_attachedEnginePool, enginePool))
            return;

        lock (_loraFallbackLock)
        {
            if (_disposed || ReferenceEquals(_attachedEnginePool, enginePool))
                return;

            _attachedEnginePool = enginePool;
            enginePool.OnBlockForked = HandleEngineBlockForked;
        }
    }

    private void HandleEngineBlockForked(int srcBlockId, int dstBlockId, int validTokens)
    {
        if (validTokens <= 0)
            return;

        lock (_loraFallbackLock)
        {
            if (_disposed)
                return;

            CuDriver.CtxSetCurrent(_gpu.Handle);
            CuDriver.Check(
                CuDriver.MemcpyDtoD(
                    _dKPool + (nint)((long)dstBlockId * _blockByteStrideK),
                    _dKPool + (nint)((long)srcBlockId * _blockByteStrideK),
                    (nuint)_blockByteStrideK),
                "MemcpyDtoD(dKPool)");
            CuDriver.Check(
                CuDriver.MemcpyDtoD(
                    _dVPool + (nint)((long)dstBlockId * _blockByteStrideV),
                    _dVPool + (nint)((long)srcBlockId * _blockByteStrideV),
                    (nuint)_blockByteStrideV),
                "MemcpyDtoD(dVPool)");

            _hostLoraPool?.CopyBlockData(srcBlockId, dstBlockId, validTokens);
        }
    }

    private PagedBlockPool EnsureLoraFallbackPool(PagedBlockPool enginePool)
    {
        lock (_loraFallbackLock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_cpuLoraRunner == null)
            {
                _cpuLoraRunner = new PagedTransformerRunner(_weights, _maxSeqLen, _maxBatchSize);
                if (_historyEnabled)
                    _cpuLoraRunner.EnablePositionHistory();
                if (_pendingHiddenHistory != null)
                    _cpuLoraRunner.BindHiddenStateHistory(_pendingHiddenHistory);
                if (_pendingBatchHistories != null)
                    _cpuLoraRunner.BindBatchHiddenStateHistories(_pendingBatchHistories);
            }

            if (enginePool.HasHostBuffers)
                return enginePool;

            if (!ReferenceEquals(_attachedEnginePool, enginePool))
            {
                _attachedEnginePool = enginePool;
                enginePool.OnBlockForked = HandleEngineBlockForked;
            }

            int desiredBlocks = Math.Max(_totalBlocks, enginePool.TotalBlocks);
            if (_hostLoraPool == null || _hostLoraPool.TotalBlocks < desiredBlocks)
            {
                _hostLoraPool?.Dispose();
                int vHeadDim = _weights.ValueDim > 0 ? _weights.ValueDim : _weights.HeadDim;
                KvCachePrecision hostLoraPrecision = _requestedKvPrecision == KvCachePrecision.Auto
                    ? KvCachePrecision.Auto
                    : _kvPrecision;
                _hostLoraPool = new PagedBlockPool(
                    desiredBlocks,
                    _weights.BlockCount,
                    _weights.HeadCountKv,
                    _weights.HeadDim,
                    blockSize: _blockSize,
                    vHeadDim: vHeadDim,
                    allocateHostBuffers: true,
                    kvPrecision: hostLoraPrecision);
            }
            return _hostLoraPool;
        }
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
        lock (_loraFallbackLock)
        {
            if (_disposed)
            {
                GC.SuppressFinalize(this);
                return;
            }

            _disposed = true;
            if (_attachedEnginePool != null)
            {
                _attachedEnginePool.OnBlockForked = null;
                _attachedEnginePool = null;
            }
            _cpuLoraRunner?.Dispose();
            _cpuLoraRunner = null;
            _hostLoraPool?.Dispose();
            _hostLoraPool = null;
        }

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
        catch (Exception ex)
        {
            GlacierDiagnostics.LogDebug($"[CudaPagedTransformerRunner] CUDA context teardown skipped: {ex.Message}");
        }

        _gpuModel.Dispose();
        _gpu.Dispose();
        GC.SuppressFinalize(this);
    }
}

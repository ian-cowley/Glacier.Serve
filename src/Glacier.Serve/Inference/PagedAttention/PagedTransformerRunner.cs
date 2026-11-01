using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;

namespace Glacier.Serve.Inference.PagedAttention;

/// <summary>
/// CPU SIMD execution runner for Transformer inference using PagedAttention.
/// Holds per-runner activation scratch buffers and evaluates single-token,
/// causal batched prefill chunks, and multi-sequence decode steps using virtual block tables.
/// </summary>
public sealed unsafe class PagedTransformerRunner : IPagedModelRunner
{
    private readonly ModelWeights _weights;
    private readonly int _dim;
    private readonly int _ffnDim;
    private readonly int _nHeads;
    private readonly int _nHeadsKv;
    private readonly int _headDim;
    private readonly int _vHeadDim;
    private readonly float _attnScale;
    private readonly int _layerCount;
    private readonly int _vocabSize;
    private readonly int _maxSeqLen;
    private readonly int _maxBatchSize;

    private readonly float* _x;
    private readonly float* _normX;
    private readonly float* _normXSums;
    private readonly float* _q;
    private readonly float* _k;
    private readonly float* _v;
    private readonly float* _attnOut;
    private readonly float* _attnOutSums;
    private readonly float* _attnProj;
    private readonly float* _gate;
    private readonly float* _up;
    private readonly float* _gateSums;
    private readonly float* _ffnProj;
    private readonly float* _headScores;
    private readonly float* _yarnInvFreq;
    private readonly float _yarnMscale;

    // Batched activation scratch buffers for unified GEMM execution
    private readonly float* _xBatch;
    private readonly float* _normXBatch;
    private readonly float* _normXSumsBatch;
    private readonly float* _qBatch;
    private readonly float* _kBatch;
    private readonly float* _vBatch;
    private readonly float* _attnOutBatch;
    private readonly float* _attnOutSumsBatch;
    private readonly float* _attnProjBatch;
    private readonly float* _gateBatch;
    private readonly float* _upBatch;
    private readonly float* _gateSumsBatch;
    private readonly float* _ffnProjBatch;

    private bool _disposed;

    public bool IsGpuAccelerated => false;
    public string BackendName => "CPU-SIMD";
    public ModelWeights Weights => _weights;
    public int VocabSize => _vocabSize;
    public int MaxSeqLen => _maxSeqLen;
    public int MaxBatchSize => _maxBatchSize;

    public PagedTransformerRunner(ModelWeights weights, int maxSeqLen = 4096, int maxBatchSize = 64)
    {
        _weights = weights ?? throw new ArgumentNullException(nameof(weights));
        ArgumentOutOfRangeException.ThrowIfLessThan(maxSeqLen, 1);
        ValidateSupportedDenseWeights(_weights);

        try
        {
            _dim = _weights.EmbeddingLength;
            _ffnDim = _weights.FeedForwardLength;
            _nHeads = _weights.HeadCount;
            _nHeadsKv = _weights.HeadCountKv;
            _headDim = _weights.HeadDim;
            _vHeadDim = _weights.ValueDim > 0 ? _weights.ValueDim : _headDim;
            int qkRopeDim = _weights.RopeDimensionCount;
            int halfRope = qkRopeDim / 2;
            if (_weights.Gguf.RopeScalingType.Equals("yarn", StringComparison.OrdinalIgnoreCase))
            {
                _yarnInvFreq = (float*)NativeMemory.AllocZeroed((nuint)(halfRope * sizeof(float)));
                QuantKernels.PrecomputeYarnFrequencies(
                    new Span<float>(_yarnInvFreq, halfRope),
                    qkRopeDim,
                    _weights.RopeFreqBase,
                    _weights.Gguf.RopeScalingFactor > 0 ? _weights.Gguf.RopeScalingFactor : 1.0f,
                    _weights.Gguf.RopeScalingOriginalContextLength > 0 ? _weights.Gguf.RopeScalingOriginalContextLength : 4096);

                float logMul = _weights.Gguf.RopeScalingYarnLogMultiplier > 0 ? _weights.Gguf.RopeScalingYarnLogMultiplier : 0.0707f;
                float factor = _weights.Gguf.RopeScalingFactor > 0 ? _weights.Gguf.RopeScalingFactor : 1.0f;
                _yarnMscale = QuantKernels.ComputeYarnMscale(factor, mscale: 1.0f, mscaleAllDim: logMul / 0.1f);
                // QuantKernels.RoPE scales both Q and K by _yarnMscale, resulting in (mscale)^2 in Q*K;
                // scaling _attnScale by _yarnMscale again would scale attention logits by (mscale)^3, over-concentrating softmax weights.
                _attnScale = 1.0f / MathF.Sqrt(_headDim);
            }
            else
            {
                _yarnInvFreq = null;
                _yarnMscale = 1.0f;
                _attnScale = 1.0f / MathF.Sqrt(_headDim);
            }

            _layerCount = _weights.BlockCount;
            _vocabSize = _weights.VocabSize;
            _maxSeqLen = maxSeqLen;
            _maxBatchSize = maxBatchSize > 0 ? maxBatchSize : 64;

            int qDim = _nHeads * _headDim;
            int kvDim = _nHeadsKv * _headDim;
            int vDim = _nHeadsKv * _vHeadDim;
            int outDim = _nHeads * _vHeadDim;
            int sumsDim = (_dim + 31) / 32;
            int outSumsDim = (outDim + 31) / 32;
            int gateSumsDim = (_ffnDim + 31) / 32;

            _x = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
            _normX = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
            _normXSums = (float*)NativeMemory.AllocZeroed((nuint)(sumsDim * sizeof(float)));

            _q = (float*)NativeMemory.AllocZeroed((nuint)(qDim * sizeof(float)));
            _k = (float*)NativeMemory.AllocZeroed((nuint)(kvDim * sizeof(float)));
            _v = (float*)NativeMemory.AllocZeroed((nuint)(vDim * sizeof(float)));

            _attnOut = (float*)NativeMemory.AllocZeroed((nuint)(outDim * sizeof(float)));
            _attnOutSums = (float*)NativeMemory.AllocZeroed((nuint)(outSumsDim * sizeof(float)));
            _attnProj = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));

            _gate = (float*)NativeMemory.AllocZeroed((nuint)(_ffnDim * sizeof(float)));
            _up = (float*)NativeMemory.AllocZeroed((nuint)(_ffnDim * sizeof(float)));
            _gateSums = (float*)NativeMemory.AllocZeroed((nuint)(gateSumsDim * sizeof(float)));
            _ffnProj = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));

            _headScores = (float*)NativeMemory.AllocZeroed((nuint)((long)_nHeads * _maxSeqLen * sizeof(float)));

            // Allocate unified batch buffers
            _xBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)_maxBatchSize * _dim * sizeof(float)));
            _normXBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)_maxBatchSize * _dim * sizeof(float)));
            _normXSumsBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)_maxBatchSize * sumsDim * sizeof(float)));

            _qBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)_maxBatchSize * qDim * sizeof(float)));
            _kBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)_maxBatchSize * kvDim * sizeof(float)));
            _vBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)_maxBatchSize * vDim * sizeof(float)));

            _attnOutBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)_maxBatchSize * outDim * sizeof(float)));
            _attnOutSumsBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)_maxBatchSize * outSumsDim * sizeof(float)));
            _attnProjBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)_maxBatchSize * _dim * sizeof(float)));

            _gateBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)_maxBatchSize * _ffnDim * sizeof(float)));
            _upBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)_maxBatchSize * _ffnDim * sizeof(float)));
            _gateSumsBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)_maxBatchSize * gateSumsDim * sizeof(float)));
            _ffnProjBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)_maxBatchSize * _dim * sizeof(float)));
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    private static void ValidateSupportedDenseWeights(ModelWeights weights)
    {
        if (weights.EmbdWeight == null || weights.OutNormWeight == null || weights.OutWeight == null)
        {
            throw new NotSupportedException("ModelWeights is missing required embedding or output projection tensors for PagedAttention.");
        }

        if (weights.IsMoe || weights.IsMla || weights.ArchitectureFamily == UniversalArchitecture.HybridSsm)
        {
            throw new NotSupportedException(
                $"Architecture '{weights.Gguf.Architecture}' (MoE/MLA/HybridSSM) is not supported by dense PagedTransformerRunner; use InferenceSession directly.");
        }

        for (int l = 0; l < weights.BlockCount; l++)
        {
            var lw = weights.Layers[l];
            if (lw.HasFusedQkv || lw.HasFusedGateUp ||
                lw.AttnNormWeight == null || lw.QWeight == null || lw.KWeight == null || lw.VWeight == null || lw.AttnOutWeight == null ||
                lw.FfnNormWeight == null || lw.FfnGateWeight == null || lw.FfnUpWeight == null || lw.FfnDownWeight == null)
            {
                throw new NotSupportedException(
                    $"Layer {l} of '{weights.Gguf.Architecture}' uses fused or specialized weights unsupported by dense PagedTransformerRunner.");
            }
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

        // 1. Embedding lookup
        QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, token, _x, _dim);

        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;
        int vDim = _nHeadsKv * _vHeadDim;
        int outDim = _nHeads * _vHeadDim;

        // Ensure token slot is allocated in BlockTable
        int pBlockId;
        int offsetInBlock;
        if (pos == blockTable.TokenCount)
        {
            if (!blockTable.AppendToken(out pBlockId, out offsetInBlock))
            {
                return false; // Pool out of memory
            }
        }
        else
        {
            blockTable.GetTokenLocation(pos, out pBlockId, out offsetInBlock);
        }

        // 2. Transformer layers
        for (int l = 0; l < _layerCount; l++)
        {
            var layer = _weights.Layers[l];

            // Attention pre-norm
            QuantKernels.RMSNorm(_x, layer.AttnNormWeight, _normX, _dim, _weights.RmsNormEps);
            QuantKernels.ComputeBlockSums32(_normX, _normXSums, _dim);

            // Q, K, V projections
            QuantKernels.MatVecMul(layer.QType, layer.QWeight, _normX, _q, _dim, qDim, _normXSums);
            QuantKernels.MatVecMul(layer.KType, layer.KWeight, _normX, _k, _dim, kvDim, _normXSums);
            QuantKernels.MatVecMul(layer.VType, layer.VWeight, _normX, _v, _dim, vDim, _normXSums);

            if (layer.QBias != null) AddVector(_q, layer.QBias, qDim);
            if (layer.KBias != null) AddVector(_k, layer.KBias, kvDim);
            if (layer.VBias != null) AddVector(_v, layer.VBias, vDim);

            // Per-head QK RMSNorm (Qwen3 / Gemma3 / DeepSeek-R1-Qwen3)
            ApplyQkNorm(layer, _q, _k);

            // Rotary Position Embedding
            QuantKernels.RoPE(
                _q,
                _k,
                _nHeads,
                _nHeadsKv,
                _headDim,
                pos,
                _weights.RopeFreqBase,
                ropeFreqs: _weights.RopeFreqsWeight,
                ropeDim: _weights.RopeDimensionCount,
                invFreqTable: _yarnInvFreq,
                mscale: _yarnMscale);

            // Store Key and Value into PagedBlockPool
            pool.Store(pBlockId, l, offsetInBlock, _k, _v);

            // Paged Attention
            float? sinkLogit = layer.AttnSinksWeight != null ? (float?)layer.AttnSinksWeight[0] : null;
            PagedAttentionKernel.ComputeAttention(
                stageLayer: l,
                modelLayer: l,
                pos: pos,
                qBase: _q,
                outBase: _attnOut,
                blockTable: blockTable,
                pool: pool,
                nHeads: _nHeads,
                nHeadsKv: _nHeadsKv,
                headDim: _headDim,
                vHeadDim: _vHeadDim,
                attnScale: _attnScale,
                headScores: _headScores,
                maxSeqLen: _maxSeqLen,
                sinkLogit: sinkLogit);

            // Attention out projection (uses outDim = _nHeads * _vHeadDim)
            QuantKernels.ComputeBlockSums32(_attnOut, _attnOutSums, outDim);
            QuantKernels.MatVecMul(layer.AttnOutType, layer.AttnOutWeight, _attnOut, _attnProj, outDim, _dim, _attnOutSums);
            if (layer.AttnOutBias != null) AddVector(_attnProj, layer.AttnOutBias, _dim);
            AddVector(_x, _attnProj, _dim);

            // FFN pre-norm
            QuantKernels.RMSNorm(_x, layer.FfnNormWeight, _normX, _dim, _weights.RmsNormEps);
            QuantKernels.ComputeBlockSums32(_normX, _normXSums, _dim);

            // SwiGLU FFN
            QuantKernels.MatVecMul(layer.FfnGateType, layer.FfnGateWeight, _normX, _gate, _dim, _ffnDim, _normXSums);
            QuantKernels.MatVecMul(layer.FfnUpType, layer.FfnUpWeight, _normX, _up, _dim, _ffnDim, _normXSums);
            QuantKernels.SwiGLU(_gate, _up, _gate, _ffnDim);
            QuantKernels.ComputeBlockSums32(_gate, _gateSums, _ffnDim);
            QuantKernels.MatVecMul(layer.FfnDownType, layer.FfnDownWeight, _gate, _ffnProj, _ffnDim, _dim, _gateSums);
            AddVector(_x, _ffnProj, _dim);
        }

        // 3. Compute final logits if requested
        if (computeLogits && !logits.IsEmpty)
        {
            QuantKernels.RMSNorm(_x, _weights.OutNormWeight, _normX, _dim, _weights.RmsNormEps);
            QuantKernels.ComputeBlockSums32(_normX, _normXSums, _dim);

            fixed (float* pLogits = logits)
            {
                QuantKernels.MatVecMul(_weights.OutType, _weights.OutWeight, _normX, pLogits, _dim, _vocabSize, _normXSums);
            }
        }

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

        if (tokens.Length == 1)
            return ForwardToken(tokens[0], startPos, blockTable, pool, logits, computeLogits);

        int total = tokens.Length;
        int offset = 0;
        while (offset < total)
        {
            int chunkLen = Math.Min(total - offset, _maxBatchSize);
            bool isLastSubChunk = (offset + chunkLen == total);
            if (!ForwardPrefillSubBatch(
                tokens.Slice(offset, chunkLen),
                startPos + offset,
                blockTable,
                pool,
                isLastSubChunk && computeLogits ? logits : Span<float>.Empty,
                isLastSubChunk && computeLogits))
            {
                return false;
            }
            offset += chunkLen;
        }

        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private bool ForwardPrefillSubBatch(
        ReadOnlySpan<int> chunkTokens,
        int chunkStartPos,
        BlockTable blockTable,
        PagedBlockPool pool,
        Span<float> logits,
        bool computeLogits)
    {
        int batchSize = chunkTokens.Length;
        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;
        int vDim = _nHeadsKv * _vHeadDim;
        int outDim = _nHeads * _vHeadDim;
        int sumsDim = (_dim + 31) / 32;
        int outSumsDim = (outDim + 31) / 32;
        int gateSumsDim = (_ffnDim + 31) / 32;

        Span<int> pBlockIds = stackalloc int[batchSize];
        Span<int> offsetsInBlock = stackalloc int[batchSize];

        // 1. Allocate block table slots for all tokens in this causal prefill sub-batch
        for (int t = 0; t < batchSize; t++)
        {
            int pos = chunkStartPos + t;
            if (pos == blockTable.TokenCount)
            {
                if (!blockTable.AppendToken(out int blockId, out int offset))
                {
                    return false;
                }
                pBlockIds[t] = blockId;
                offsetsInBlock[t] = offset;
            }
            else
            {
                blockTable.GetTokenLocation(pos, out int blockId, out int offset);
                pBlockIds[t] = blockId;
                offsetsInBlock[t] = offset;
            }
        }

        // 2. Extract embeddings into _xBatch
        for (int t = 0; t < batchSize; t++)
        {
            QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, chunkTokens[t], _xBatch + t * _dim, _dim);
        }

        // 3. Layer-by-layer causal batched transformer execution
        for (int l = 0; l < _layerCount; l++)
        {
            var layer = _weights.Layers[l];

            for (int t = 0; t < batchSize; t++)
            {
                QuantKernels.RMSNorm(_xBatch + t * _dim, layer.AttnNormWeight, _normXBatch + t * _dim, _dim, _weights.RmsNormEps);
                QuantKernels.ComputeBlockSums32(_normXBatch + t * _dim, _normXSumsBatch + t * sumsDim, _dim);
            }

            QuantKernels.MatMulBatch(layer.QType, layer.QWeight, _normXBatch, _qBatch, _dim, qDim, batchSize, _normXSumsBatch);
            QuantKernels.MatMulBatch(layer.KType, layer.KWeight, _normXBatch, _kBatch, _dim, kvDim, batchSize, _normXSumsBatch);
            QuantKernels.MatMulBatch(layer.VType, layer.VWeight, _normXBatch, _vBatch, _dim, vDim, batchSize, _normXSumsBatch);

            // Store all K/V for this layer first, then compute causal attention (bounded by pos + 1)
            for (int t = 0; t < batchSize; t++)
            {
                float* qPtr = _qBatch + t * qDim;
                float* kPtr = _kBatch + t * kvDim;
                float* vPtr = _vBatch + t * vDim;

                if (layer.QBias != null) AddVector(qPtr, layer.QBias, qDim);
                if (layer.KBias != null) AddVector(kPtr, layer.KBias, kvDim);
                if (layer.VBias != null) AddVector(vPtr, layer.VBias, vDim);

                ApplyQkNorm(layer, qPtr, kPtr);

                QuantKernels.RoPE(
                    qPtr,
                    kPtr,
                    _nHeads,
                    _nHeadsKv,
                    _headDim,
                    chunkStartPos + t,
                    _weights.RopeFreqBase,
                    ropeFreqs: _weights.RopeFreqsWeight,
                    ropeDim: _weights.RopeDimensionCount,
                    invFreqTable: _yarnInvFreq,
                    mscale: _yarnMscale);

                pool.Store(pBlockIds[t], l, offsetsInBlock[t], kPtr, vPtr);
            }

            float? sinkLogit = layer.AttnSinksWeight != null ? (float?)layer.AttnSinksWeight[0] : null;
            for (int t = 0; t < batchSize; t++)
            {
                PagedAttentionKernel.ComputeAttention(
                    stageLayer: l,
                    modelLayer: l,
                    pos: chunkStartPos + t,
                    qBase: _qBatch + t * qDim,
                    outBase: _attnOutBatch + t * outDim,
                    blockTable: blockTable,
                    pool: pool,
                    nHeads: _nHeads,
                    nHeadsKv: _nHeadsKv,
                    headDim: _headDim,
                    vHeadDim: _vHeadDim,
                    attnScale: _attnScale,
                    headScores: _headScores,
                    maxSeqLen: _maxSeqLen,
                    sinkLogit: sinkLogit);

                QuantKernels.ComputeBlockSums32(_attnOutBatch + t * outDim, _attnOutSumsBatch + t * outSumsDim, outDim);
            }

            QuantKernels.MatMulBatch(layer.AttnOutType, layer.AttnOutWeight, _attnOutBatch, _attnProjBatch, outDim, _dim, batchSize, _attnOutSumsBatch);

            for (int t = 0; t < batchSize; t++)
            {
                if (layer.AttnOutBias != null) AddVector(_attnProjBatch + t * _dim, layer.AttnOutBias, _dim);
                AddVector(_xBatch + t * _dim, _attnProjBatch + t * _dim, _dim);

                QuantKernels.RMSNorm(_xBatch + t * _dim, layer.FfnNormWeight, _normXBatch + t * _dim, _dim, _weights.RmsNormEps);
                QuantKernels.ComputeBlockSums32(_normXBatch + t * _dim, _normXSumsBatch + t * sumsDim, _dim);
            }

            QuantKernels.MatMulBatch(layer.FfnGateType, layer.FfnGateWeight, _normXBatch, _gateBatch, _dim, _ffnDim, batchSize, _normXSumsBatch);
            QuantKernels.MatMulBatch(layer.FfnUpType, layer.FfnUpWeight, _normXBatch, _upBatch, _dim, _ffnDim, batchSize, _normXSumsBatch);

            for (int t = 0; t < batchSize; t++)
            {
                QuantKernels.SwiGLU(_gateBatch + t * _ffnDim, _upBatch + t * _ffnDim, _gateBatch + t * _ffnDim, _ffnDim);
                QuantKernels.ComputeBlockSums32(_gateBatch + t * _ffnDim, _gateSumsBatch + t * gateSumsDim, _ffnDim);
            }

            QuantKernels.MatMulBatch(layer.FfnDownType, layer.FfnDownWeight, _gateBatch, _ffnProjBatch, _ffnDim, _dim, batchSize, _gateSumsBatch);

            for (int t = 0; t < batchSize; t++)
            {
                AddVector(_xBatch + t * _dim, _ffnProjBatch + t * _dim, _dim);
            }
        }

        // 4. Compute logits only for the final token of the prefill chunk
        if (computeLogits && !logits.IsEmpty)
        {
            float* lastX = _xBatch + (batchSize - 1) * _dim;
            QuantKernels.RMSNorm(lastX, _weights.OutNormWeight, _normX, _dim, _weights.RmsNormEps);
            QuantKernels.ComputeBlockSums32(_normX, _normXSums, _dim);

            fixed (float* pLogits = logits)
            {
                QuantKernels.MatVecMul(_weights.OutType, _weights.OutWeight, _normX, pLogits, _dim, _vocabSize, _normXSums);
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
        int batchSize = tokens.Length;
        if (batchSize == 0) return true;
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (batchSize > _maxBatchSize)
            throw new ArgumentOutOfRangeException(nameof(tokens), $"Batch size {batchSize} exceeds maximum runner capacity {_maxBatchSize}.");

        for (int b = 0; b < batchSize; b++)
        {
            if (positions[b] < 0 || positions[b] >= _maxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions), positions[b], $"Sequence {b} position {positions[b]} exceeds valid range [0, {_maxSeqLen - 1}].");
        }

        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;
        int vDim = _nHeadsKv * _vHeadDim;
        int outDim = _nHeads * _vHeadDim;
        int sumsDim = (_dim + 31) / 32;
        int outSumsDim = (outDim + 31) / 32;
        int gateSumsDim = (_ffnDim + 31) / 32;

        Span<int> pBlockIds = stackalloc int[batchSize];
        Span<int> offsetsInBlock = stackalloc int[batchSize];

        // 1. Allocate block table slots for each sequence
        for (int b = 0; b < batchSize; b++)
        {
            int pos = positions[b];
            var blockTable = blockTables[b];
            if (pos == blockTable.TokenCount)
            {
                if (!blockTable.AppendToken(out int blockId, out int offset))
                {
                    successFlags[b] = false;
                    pBlockIds[b] = -1;
                    offsetsInBlock[b] = -1;
                    continue;
                }
                pBlockIds[b] = blockId;
                offsetsInBlock[b] = offset;
                successFlags[b] = true;
            }
            else
            {
                blockTable.GetTokenLocation(pos, out int blockId, out int offset);
                pBlockIds[b] = blockId;
                offsetsInBlock[b] = offset;
                successFlags[b] = true;
            }
        }

        // 2. Extract embeddings into _xBatch
        for (int b = 0; b < batchSize; b++)
        {
            if (!successFlags[b]) continue;
            QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, tokens[b], _xBatch + b * _dim, _dim);
        }

        // 3. Layer by layer batched transformer execution
        for (int l = 0; l < _layerCount; l++)
        {
            var layer = _weights.Layers[l];

            // Attention pre-norm across all batch rows
            for (int b = 0; b < batchSize; b++)
            {
                if (!successFlags[b]) continue;
                QuantKernels.RMSNorm(_xBatch + b * _dim, layer.AttnNormWeight, _normXBatch + b * _dim, _dim, _weights.RmsNormEps);
                QuantKernels.ComputeBlockSums32(_normXBatch + b * _dim, _normXSumsBatch + b * sumsDim, _dim);
            }

            // Unified Batch GEMMs: load weights ONCE, multiply across all sequences simultaneously
            QuantKernels.MatMulBatch(layer.QType, layer.QWeight, _normXBatch, _qBatch, _dim, qDim, batchSize, _normXSumsBatch);
            QuantKernels.MatMulBatch(layer.KType, layer.KWeight, _normXBatch, _kBatch, _dim, kvDim, batchSize, _normXSumsBatch);
            QuantKernels.MatMulBatch(layer.VType, layer.VWeight, _normXBatch, _vBatch, _dim, vDim, batchSize, _normXSumsBatch);

            for (int b = 0; b < batchSize; b++)
            {
                if (!successFlags[b]) continue;
                float* qPtr = _qBatch + b * qDim;
                float* kPtr = _kBatch + b * kvDim;
                float* vPtr = _vBatch + b * vDim;

                if (layer.QBias != null) AddVector(qPtr, layer.QBias, qDim);
                if (layer.KBias != null) AddVector(kPtr, layer.KBias, kvDim);
                if (layer.VBias != null) AddVector(vPtr, layer.VBias, vDim);

                // Per-head QK RMSNorm
                ApplyQkNorm(layer, qPtr, kPtr);

                // RoPE per sequence position
                QuantKernels.RoPE(
                    qPtr,
                    kPtr,
                    _nHeads,
                    _nHeadsKv,
                    _headDim,
                    positions[b],
                    _weights.RopeFreqBase,
                    ropeFreqs: _weights.RopeFreqsWeight,
                    ropeDim: _weights.RopeDimensionCount,
                    invFreqTable: _yarnInvFreq,
                    mscale: _yarnMscale);

                // Store Key & Value into PagedBlockPool (vPtr indexed by vDim)
                pool.Store(pBlockIds[b], l, offsetsInBlock[b], kPtr, vPtr);

                // Compute Attention for sequence b
                float? sinkLogit = layer.AttnSinksWeight != null ? (float?)layer.AttnSinksWeight[0] : null;
                PagedAttentionKernel.ComputeAttention(
                    stageLayer: l,
                    modelLayer: l,
                    pos: positions[b],
                    qBase: qPtr,
                    outBase: _attnOutBatch + b * outDim,
                    blockTable: blockTables[b],
                    pool: pool,
                    nHeads: _nHeads,
                    nHeadsKv: _nHeadsKv,
                    headDim: _headDim,
                    vHeadDim: _vHeadDim,
                    attnScale: _attnScale,
                    headScores: _headScores,
                    maxSeqLen: _maxSeqLen,
                    sinkLogit: sinkLogit);

                QuantKernels.ComputeBlockSums32(_attnOutBatch + b * outDim, _attnOutSumsBatch + b * outSumsDim, outDim);
            }

            // Unified Batch GEMM: Attention Out Projection (uses outDim)
            QuantKernels.MatMulBatch(layer.AttnOutType, layer.AttnOutWeight, _attnOutBatch, _attnProjBatch, outDim, _dim, batchSize, _attnOutSumsBatch);

            for (int b = 0; b < batchSize; b++)
            {
                if (!successFlags[b]) continue;
                if (layer.AttnOutBias != null) AddVector(_attnProjBatch + b * _dim, layer.AttnOutBias, _dim);
                AddVector(_xBatch + b * _dim, _attnProjBatch + b * _dim, _dim);

                // FFN pre-norm
                QuantKernels.RMSNorm(_xBatch + b * _dim, layer.FfnNormWeight, _normXBatch + b * _dim, _dim, _weights.RmsNormEps);
                QuantKernels.ComputeBlockSums32(_normXBatch + b * _dim, _normXSumsBatch + b * sumsDim, _dim);
            }

            // Unified Batch GEMMs: FFN Gate & Up
            QuantKernels.MatMulBatch(layer.FfnGateType, layer.FfnGateWeight, _normXBatch, _gateBatch, _dim, _ffnDim, batchSize, _normXSumsBatch);
            QuantKernels.MatMulBatch(layer.FfnUpType, layer.FfnUpWeight, _normXBatch, _upBatch, _dim, _ffnDim, batchSize, _normXSumsBatch);

            for (int b = 0; b < batchSize; b++)
            {
                if (!successFlags[b]) continue;
                QuantKernels.SwiGLU(_gateBatch + b * _ffnDim, _upBatch + b * _ffnDim, _gateBatch + b * _ffnDim, _ffnDim);
                QuantKernels.ComputeBlockSums32(_gateBatch + b * _ffnDim, _gateSumsBatch + b * gateSumsDim, _ffnDim);
            }

            // Unified Batch GEMM: FFN Down Projection
            QuantKernels.MatMulBatch(layer.FfnDownType, layer.FfnDownWeight, _gateBatch, _ffnProjBatch, _ffnDim, _dim, batchSize, _gateSumsBatch);

            for (int b = 0; b < batchSize; b++)
            {
                if (!successFlags[b]) continue;
                AddVector(_xBatch + b * _dim, _ffnProjBatch + b * _dim, _dim);
            }
        }

        // 4. Compute final logits if requested
        if (computeLogits && !logitsBatch.IsEmpty)
        {
            for (int b = 0; b < batchSize; b++)
            {
                if (!successFlags[b]) continue;
                QuantKernels.RMSNorm(_xBatch + b * _dim, _weights.OutNormWeight, _normXBatch + b * _dim, _dim, _weights.RmsNormEps);
                QuantKernels.ComputeBlockSums32(_normXBatch + b * _dim, _normXSumsBatch + b * sumsDim, _dim);
            }

            fixed (float* pLogitsBatch = logitsBatch)
            {
                QuantKernels.MatMulBatch(_weights.OutType, _weights.OutWeight, _normXBatch, pLogitsBatch, _dim, _vocabSize, batchSize, _normXSumsBatch);
            }
        }

        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ApplyQkNorm(LayerWeights layer, float* qPtr, float* kPtr)
    {
        if (layer.AttnQNormWeight != null)
        {
            for (int h = 0; h < _nHeads; h++)
            {
                float* qHead = qPtr + h * _headDim;
                QuantKernels.RMSNorm(qHead, layer.AttnQNormWeight, qHead, _headDim, _weights.RmsNormEps);
            }
        }

        if (layer.AttnKNormWeight != null)
        {
            for (int h = 0; h < _nHeadsKv; h++)
            {
                float* kHead = kPtr + h * _headDim;
                QuantKernels.RMSNorm(kHead, layer.AttnKNormWeight, kHead, _headDim, _weights.RmsNormEps);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddVector(float* a, float* b, int count)
    {
        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            int limit = count - 8;
            for (; i <= limit; i += 8)
            {
                var va = Vector256.Load(a + i);
                var vb = Vector256.Load(b + i);
                (va + vb).Store(a + i);
            }
        }
        for (; i < count; i++)
        {
            a[i] += b[i];
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_x != null) NativeMemory.Free(_x);
            if (_normX != null) NativeMemory.Free(_normX);
            if (_normXSums != null) NativeMemory.Free(_normXSums);
            if (_q != null) NativeMemory.Free(_q);
            if (_k != null) NativeMemory.Free(_k);
            if (_v != null) NativeMemory.Free(_v);
            if (_attnOut != null) NativeMemory.Free(_attnOut);
            if (_attnOutSums != null) NativeMemory.Free(_attnOutSums);
            if (_attnProj != null) NativeMemory.Free(_attnProj);
            if (_gate != null) NativeMemory.Free(_gate);
            if (_up != null) NativeMemory.Free(_up);
            if (_gateSums != null) NativeMemory.Free(_gateSums);
            if (_ffnProj != null) NativeMemory.Free(_ffnProj);
            if (_headScores != null) NativeMemory.Free(_headScores);
            if (_yarnInvFreq != null) NativeMemory.Free(_yarnInvFreq);

            if (_xBatch != null) NativeMemory.Free(_xBatch);
            if (_normXBatch != null) NativeMemory.Free(_normXBatch);
            if (_normXSumsBatch != null) NativeMemory.Free(_normXSumsBatch);
            if (_qBatch != null) NativeMemory.Free(_qBatch);
            if (_kBatch != null) NativeMemory.Free(_kBatch);
            if (_vBatch != null) NativeMemory.Free(_vBatch);
            if (_attnOutBatch != null) NativeMemory.Free(_attnOutBatch);
            if (_attnOutSumsBatch != null) NativeMemory.Free(_attnOutSumsBatch);
            if (_attnProjBatch != null) NativeMemory.Free(_attnProjBatch);
            if (_gateBatch != null) NativeMemory.Free(_gateBatch);
            if (_upBatch != null) NativeMemory.Free(_upBatch);
            if (_gateSumsBatch != null) NativeMemory.Free(_gateSumsBatch);
            if (_ffnProjBatch != null) NativeMemory.Free(_ffnProjBatch);
        }
        GC.SuppressFinalize(this);
    }

    ~PagedTransformerRunner()
    {
        Dispose();
    }
}

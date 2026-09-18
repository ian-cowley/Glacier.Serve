using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;

namespace Glacier.Serve.Inference.PagedAttention;

/// <summary>
/// Execution runner for Transformer inference using PagedAttention.
/// Holds per-thread/per-runner activation scratch buffers and evaluates
/// token forward steps using virtual block tables.
/// </summary>
public sealed unsafe class PagedTransformerRunner : IDisposable
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

    public ModelWeights Weights => _weights;
    public int VocabSize => _vocabSize;
    public int MaxSeqLen => _maxSeqLen;
    public int MaxBatchSize => _maxBatchSize;

    public PagedTransformerRunner(ModelWeights weights, int maxSeqLen = 4096, int maxBatchSize = 64)
    {
        _weights = weights ?? throw new ArgumentNullException(nameof(weights));
        _dim = _weights.EmbeddingLength;
        _ffnDim = _weights.FeedForwardLength;
        _nHeads = _weights.HeadCount;
        _nHeadsKv = _weights.HeadCountKv;
        _headDim = _weights.HeadDim;
        _vHeadDim = _weights.ValueDim > 0 ? _weights.ValueDim : _headDim;
        _attnScale = 1.0f / MathF.Sqrt(_headDim);
        _layerCount = _weights.BlockCount;
        _vocabSize = _weights.VocabSize;
        _maxSeqLen = maxSeqLen;
        _maxBatchSize = maxBatchSize > 0 ? maxBatchSize : 64;

        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;
        int vDim = _nHeadsKv * _vHeadDim;
        int outDim = _nHeads * _vHeadDim;
        int sumsDim = (_dim + 31) / 32;
        int gateSumsDim = (_ffnDim + 31) / 32;

        _x = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _normX = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _normXSums = (float*)NativeMemory.AllocZeroed((nuint)(sumsDim * sizeof(float)));

        _q = (float*)NativeMemory.AllocZeroed((nuint)(qDim * sizeof(float)));
        _k = (float*)NativeMemory.AllocZeroed((nuint)(kvDim * sizeof(float)));
        _v = (float*)NativeMemory.AllocZeroed((nuint)(vDim * sizeof(float)));

        _attnOut = (float*)NativeMemory.AllocZeroed((nuint)(outDim * sizeof(float)));
        _attnOutSums = (float*)NativeMemory.AllocZeroed((nuint)(sumsDim * sizeof(float)));
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
        _attnOutSumsBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)_maxBatchSize * sumsDim * sizeof(float)));
        _attnProjBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)_maxBatchSize * _dim * sizeof(float)));

        _gateBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)_maxBatchSize * _ffnDim * sizeof(float)));
        _upBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)_maxBatchSize * _ffnDim * sizeof(float)));
        _gateSumsBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)_maxBatchSize * gateSumsDim * sizeof(float)));
        _ffnProjBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)_maxBatchSize * _dim * sizeof(float)));
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
        // 1. Embedding lookup
        QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, token, _x, _dim);

        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;

        // Ensure token slot is allocated in BlockTable
        int pBlockId = -1;
        int offsetInBlock = -1;
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
            QuantKernels.MatVecMul(layer.VType, layer.VWeight, _normX, _v, _dim, kvDim, _normXSums);

            if (layer.QBias != null) AddVector(_q, layer.QBias, qDim);
            if (layer.KBias != null) AddVector(_k, layer.KBias, kvDim);
            if (layer.VBias != null) AddVector(_v, layer.VBias, kvDim);

            // Rotary Position Embedding
            QuantKernels.RoPE(_q, _k, _nHeads, _nHeadsKv, _headDim, pos, _weights.RopeFreqBase);

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

            // Attention out projection
            QuantKernels.ComputeBlockSums32(_attnOut, _attnOutSums, qDim);
            QuantKernels.MatVecMul(layer.AttnOutType, layer.AttnOutWeight, _attnOut, _attnProj, qDim, _dim, _attnOutSums);
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
        if (batchSize > _maxBatchSize)
            throw new ArgumentOutOfRangeException(nameof(tokens), $"Batch size {batchSize} exceeds maximum runner capacity {_maxBatchSize}.");

        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;
        int outDim = _nHeads * _vHeadDim;
        int sumsDim = (_dim + 31) / 32;
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
            QuantKernels.MatMulBatch(layer.VType, layer.VWeight, _normXBatch, _vBatch, _dim, kvDim, batchSize, _normXSumsBatch);

            for (int b = 0; b < batchSize; b++)
            {
                if (!successFlags[b]) continue;
                if (layer.QBias != null) AddVector(_qBatch + b * qDim, layer.QBias, qDim);
                if (layer.KBias != null) AddVector(_kBatch + b * kvDim, layer.KBias, kvDim);
                if (layer.VBias != null) AddVector(_vBatch + b * kvDim, layer.VBias, kvDim);

                // RoPE per sequence position
                QuantKernels.RoPE(_qBatch + b * qDim, _kBatch + b * kvDim, _nHeads, _nHeadsKv, _headDim, positions[b], _weights.RopeFreqBase);

                // Store Key & Value into PagedBlockPool
                pool.Store(pBlockIds[b], l, offsetsInBlock[b], _kBatch + b * kvDim, _vBatch + b * kvDim);

                // Compute Attention for sequence b
                float? sinkLogit = layer.AttnSinksWeight != null ? (float?)layer.AttnSinksWeight[0] : null;
                PagedAttentionKernel.ComputeAttention(
                    stageLayer: l,
                    modelLayer: l,
                    pos: positions[b],
                    qBase: _qBatch + b * qDim,
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

                QuantKernels.ComputeBlockSums32(_attnOutBatch + b * outDim, _attnOutSumsBatch + b * sumsDim, qDim);
            }

            // Unified Batch GEMM: Attention Out Projection
            QuantKernels.MatMulBatch(layer.AttnOutType, layer.AttnOutWeight, _attnOutBatch, _attnProjBatch, qDim, _dim, batchSize, _attnOutSumsBatch);

            for (int b = 0; b < batchSize; b++)
            {
                if (!successFlags[b]) continue;
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

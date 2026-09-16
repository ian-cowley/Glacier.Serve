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

    private bool _disposed;

    public ModelWeights Weights => _weights;
    public int VocabSize => _vocabSize;
    public int MaxSeqLen => _maxSeqLen;

    public PagedTransformerRunner(ModelWeights weights, int maxSeqLen = 4096)
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

        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;
        int vDim = _nHeadsKv * _vHeadDim;
        int outDim = _nHeads * _vHeadDim;

        _x = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _normX = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _normXSums = (float*)NativeMemory.AllocZeroed((nuint)(((_dim + 31) / 32) * sizeof(float)));

        _q = (float*)NativeMemory.AllocZeroed((nuint)(qDim * sizeof(float)));
        _k = (float*)NativeMemory.AllocZeroed((nuint)(kvDim * sizeof(float)));
        _v = (float*)NativeMemory.AllocZeroed((nuint)(vDim * sizeof(float)));

        _attnOut = (float*)NativeMemory.AllocZeroed((nuint)(outDim * sizeof(float)));
        _attnOutSums = (float*)NativeMemory.AllocZeroed((nuint)(((_dim + 31) / 32) * sizeof(float)));
        _attnProj = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));

        _gate = (float*)NativeMemory.AllocZeroed((nuint)(_ffnDim * sizeof(float)));
        _up = (float*)NativeMemory.AllocZeroed((nuint)(_ffnDim * sizeof(float)));
        _gateSums = (float*)NativeMemory.AllocZeroed((nuint)(((_ffnDim + 31) / 32) * sizeof(float)));
        _ffnProj = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));

        _headScores = (float*)NativeMemory.AllocZeroed((nuint)((long)_nHeads * _maxSeqLen * sizeof(float)));
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
        }
        GC.SuppressFinalize(this);
    }

    ~PagedTransformerRunner()
    {
        Dispose();
    }
}

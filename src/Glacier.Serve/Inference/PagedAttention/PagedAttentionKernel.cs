using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;
using Glacier.Inference.Quant;

namespace Glacier.Serve.Inference.PagedAttention;

/// <summary>
/// Hardware-accelerated PagedAttention kernel.
/// Computes Multi-Head and Grouped Query Attention (GQA) across non-contiguous physical
/// memory blocks stored in PagedBlockPool, achieving zero memory copy overhead and vector efficiency.
/// </summary>
public static unsafe class PagedAttentionKernel
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void ComputeAttention(
        int stageLayer,
        int modelLayer,
        int pos,
        float* qBase,
        float* outBase,
        BlockTable blockTable,
        PagedBlockPool pool,
        int nHeads,
        int nHeadsKv,
        int headDim,
        int vHeadDim,
        float attnScale,
        float* headScores,
        int maxSeqLen,
        float? sinkLogit = null)
    {
        int totalTokens = blockTable.TokenCount;
        if (totalTokens == 0) return;

        int groupSize = nHeads / nHeadsKv;
        int blockCount = blockTable.BlockCount;
        int blockSize = pool.BlockSize;

        Parallel.For(0, nHeads, h =>
        {
            int hKv = h / groupSize;
            float* qHead = qBase + h * headDim;
            float* scores = headScores + (long)h * maxSeqLen;

            // 1. Block-by-block Q * K^T dot products
            for (int b = 0; b < blockCount; b++)
            {
                int pBlockId = blockTable.GetBlockId(b);
                int baseTokenIdx = b * blockSize;
                int tokensInBlock = Math.Min(blockSize, totalTokens - baseTokenIdx);

                for (int t = 0; t < tokensInBlock; t++)
                {
                    float* kPast = pool.GetKeyPtr(pBlockId, stageLayer, hKv, t);
                    float dot = QuantKernels.VecDotF32(qHead, kPast, headDim);
                    scores[baseTokenIdx + t] = dot * attnScale;
                }
            }

            // 2. Softmax over all valid tokens in the sequence
            QuantKernels.Softmax(scores, totalTokens, sinkLogit);

            // 3. Block-by-block Value aggregation
            float* outHead = outBase + h * vHeadDim;
            for (int d = 0; d < vHeadDim; d++) outHead[d] = 0f;

            for (int b = 0; b < blockCount; b++)
            {
                int pBlockId = blockTable.GetBlockId(b);
                int baseTokenIdx = b * blockSize;
                int tokensInBlock = Math.Min(blockSize, totalTokens - baseTokenIdx);

                for (int t = 0; t < tokensInBlock; t++)
                {
                    float w = scores[baseTokenIdx + t];
                    float* vPast = pool.GetValuePtr(pBlockId, stageLayer, hKv, t);

                    if (Vector256.IsHardwareAccelerated)
                    {
                        var vw = Vector256.Create(w);
                        int d = 0;
                        int limit = vHeadDim - 8;
                        for (; d <= limit; d += 8)
                        {
                            var vo = Vector256.Load(outHead + d);
                            var vv = Vector256.Load(vPast + d);
                            vo += vw * vv;
                            vo.Store(outHead + d);
                        }
                        for (; d < vHeadDim; d++)
                        {
                            outHead[d] += w * vPast[d];
                        }
                    }
                    else
                    {
                        for (int d = 0; d < vHeadDim; d++)
                        {
                            outHead[d] += w * vPast[d];
                        }
                    }
                }
            }
        });
    }
}

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

            // 3. Block-by-block Value aggregation: Hoard accumulators in CPU registers
            float* outHead = outBase + h * vHeadDim;

            if (Vector512.IsHardwareAccelerated && vHeadDim == 128)
            {
                var acc0 = Vector512<float>.Zero;
                var acc1 = Vector512<float>.Zero;
                var acc2 = Vector512<float>.Zero;
                var acc3 = Vector512<float>.Zero;
                var acc4 = Vector512<float>.Zero;
                var acc5 = Vector512<float>.Zero;
                var acc6 = Vector512<float>.Zero;
                var acc7 = Vector512<float>.Zero;

                for (int b = 0; b < blockCount; b++)
                {
                    int pBlockId = blockTable.GetBlockId(b);
                    int baseTokenIdx = b * blockSize;
                    int tokensInBlock = Math.Min(blockSize, totalTokens - baseTokenIdx);

                    for (int t = 0; t < tokensInBlock; t++)
                    {
                        float w = scores[baseTokenIdx + t];
                        float* vPast = pool.GetValuePtr(pBlockId, stageLayer, hKv, t);
                        var vw = Vector512.Create(w);

                        acc0 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast), acc0);
                        acc1 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 16), acc1);
                        acc2 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 32), acc2);
                        acc3 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 48), acc3);
                        acc4 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 64), acc4);
                        acc5 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 80), acc5);
                        acc6 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 96), acc6);
                        acc7 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 112), acc7);
                    }
                }

                acc0.Store(outHead);
                acc1.Store(outHead + 16);
                acc2.Store(outHead + 32);
                acc3.Store(outHead + 48);
                acc4.Store(outHead + 64);
                acc5.Store(outHead + 80);
                acc6.Store(outHead + 96);
                acc7.Store(outHead + 112);
            }
            else if (Vector512.IsHardwareAccelerated && vHeadDim == 64)
            {
                var acc0 = Vector512<float>.Zero;
                var acc1 = Vector512<float>.Zero;
                var acc2 = Vector512<float>.Zero;
                var acc3 = Vector512<float>.Zero;

                for (int b = 0; b < blockCount; b++)
                {
                    int pBlockId = blockTable.GetBlockId(b);
                    int baseTokenIdx = b * blockSize;
                    int tokensInBlock = Math.Min(blockSize, totalTokens - baseTokenIdx);

                    for (int t = 0; t < tokensInBlock; t++)
                    {
                        float w = scores[baseTokenIdx + t];
                        float* vPast = pool.GetValuePtr(pBlockId, stageLayer, hKv, t);
                        var vw = Vector512.Create(w);

                        acc0 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast), acc0);
                        acc1 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 16), acc1);
                        acc2 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 32), acc2);
                        acc3 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 48), acc3);
                    }
                }

                acc0.Store(outHead);
                acc1.Store(outHead + 16);
                acc2.Store(outHead + 32);
                acc3.Store(outHead + 48);
            }
            else if (Vector512.IsHardwareAccelerated && vHeadDim == 256)
            {
                var acc0 = Vector512<float>.Zero;
                var acc1 = Vector512<float>.Zero;
                var acc2 = Vector512<float>.Zero;
                var acc3 = Vector512<float>.Zero;
                var acc4 = Vector512<float>.Zero;
                var acc5 = Vector512<float>.Zero;
                var acc6 = Vector512<float>.Zero;
                var acc7 = Vector512<float>.Zero;
                var acc8 = Vector512<float>.Zero;
                var acc9 = Vector512<float>.Zero;
                var acc10 = Vector512<float>.Zero;
                var acc11 = Vector512<float>.Zero;
                var acc12 = Vector512<float>.Zero;
                var acc13 = Vector512<float>.Zero;
                var acc14 = Vector512<float>.Zero;
                var acc15 = Vector512<float>.Zero;

                for (int b = 0; b < blockCount; b++)
                {
                    int pBlockId = blockTable.GetBlockId(b);
                    int baseTokenIdx = b * blockSize;
                    int tokensInBlock = Math.Min(blockSize, totalTokens - baseTokenIdx);

                    for (int t = 0; t < tokensInBlock; t++)
                    {
                        float w = scores[baseTokenIdx + t];
                        float* vPast = pool.GetValuePtr(pBlockId, stageLayer, hKv, t);
                        var vw = Vector512.Create(w);

                        acc0 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast), acc0);
                        acc1 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 16), acc1);
                        acc2 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 32), acc2);
                        acc3 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 48), acc3);
                        acc4 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 64), acc4);
                        acc5 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 80), acc5);
                        acc6 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 96), acc6);
                        acc7 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 112), acc7);
                        acc8 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 128), acc8);
                        acc9 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 144), acc9);
                        acc10 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 160), acc10);
                        acc11 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 176), acc11);
                        acc12 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 192), acc12);
                        acc13 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 208), acc13);
                        acc14 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 224), acc14);
                        acc15 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(vPast + 240), acc15);
                    }
                }

                acc0.Store(outHead);
                acc1.Store(outHead + 16);
                acc2.Store(outHead + 32);
                acc3.Store(outHead + 48);
                acc4.Store(outHead + 64);
                acc5.Store(outHead + 80);
                acc6.Store(outHead + 96);
                acc7.Store(outHead + 112);
                acc8.Store(outHead + 128);
                acc9.Store(outHead + 144);
                acc10.Store(outHead + 160);
                acc11.Store(outHead + 176);
                acc12.Store(outHead + 192);
                acc13.Store(outHead + 208);
                acc14.Store(outHead + 224);
                acc15.Store(outHead + 240);
            }
            else if (Vector256.IsHardwareAccelerated && vHeadDim == 64)
            {
                var acc0 = Vector256<float>.Zero;
                var acc1 = Vector256<float>.Zero;
                var acc2 = Vector256<float>.Zero;
                var acc3 = Vector256<float>.Zero;
                var acc4 = Vector256<float>.Zero;
                var acc5 = Vector256<float>.Zero;
                var acc6 = Vector256<float>.Zero;
                var acc7 = Vector256<float>.Zero;

                for (int b = 0; b < blockCount; b++)
                {
                    int pBlockId = blockTable.GetBlockId(b);
                    int baseTokenIdx = b * blockSize;
                    int tokensInBlock = Math.Min(blockSize, totalTokens - baseTokenIdx);

                    for (int t = 0; t < tokensInBlock; t++)
                    {
                        float w = scores[baseTokenIdx + t];
                        float* vPast = pool.GetValuePtr(pBlockId, stageLayer, hKv, t);
                        var vw = Vector256.Create(w);

                        acc0 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast), acc0);
                        acc1 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast + 8), acc1);
                        acc2 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast + 16), acc2);
                        acc3 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast + 24), acc3);
                        acc4 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast + 32), acc4);
                        acc5 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast + 40), acc5);
                        acc6 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast + 48), acc6);
                        acc7 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast + 56), acc7);
                    }
                }

                acc0.Store(outHead);
                acc1.Store(outHead + 8);
                acc2.Store(outHead + 16);
                acc3.Store(outHead + 24);
                acc4.Store(outHead + 32);
                acc5.Store(outHead + 40);
                acc6.Store(outHead + 48);
                acc7.Store(outHead + 56);
            }
            else if (Vector256.IsHardwareAccelerated && vHeadDim == 128)
            {
                var acc0 = Vector256<float>.Zero;
                var acc1 = Vector256<float>.Zero;
                var acc2 = Vector256<float>.Zero;
                var acc3 = Vector256<float>.Zero;
                var acc4 = Vector256<float>.Zero;
                var acc5 = Vector256<float>.Zero;
                var acc6 = Vector256<float>.Zero;
                var acc7 = Vector256<float>.Zero;
                var acc8 = Vector256<float>.Zero;
                var acc9 = Vector256<float>.Zero;
                var acc10 = Vector256<float>.Zero;
                var acc11 = Vector256<float>.Zero;
                var acc12 = Vector256<float>.Zero;
                var acc13 = Vector256<float>.Zero;
                var acc14 = Vector256<float>.Zero;
                var acc15 = Vector256<float>.Zero;

                for (int b = 0; b < blockCount; b++)
                {
                    int pBlockId = blockTable.GetBlockId(b);
                    int baseTokenIdx = b * blockSize;
                    int tokensInBlock = Math.Min(blockSize, totalTokens - baseTokenIdx);

                    for (int t = 0; t < tokensInBlock; t++)
                    {
                        float w = scores[baseTokenIdx + t];
                        float* vPast = pool.GetValuePtr(pBlockId, stageLayer, hKv, t);
                        var vw = Vector256.Create(w);

                        acc0 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast), acc0);
                        acc1 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast + 8), acc1);
                        acc2 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast + 16), acc2);
                        acc3 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast + 24), acc3);
                        acc4 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast + 32), acc4);
                        acc5 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast + 40), acc5);
                        acc6 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast + 48), acc6);
                        acc7 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast + 56), acc7);
                        acc8 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast + 64), acc8);
                        acc9 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast + 72), acc9);
                        acc10 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast + 80), acc10);
                        acc11 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast + 88), acc11);
                        acc12 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast + 96), acc12);
                        acc13 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast + 104), acc13);
                        acc14 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast + 112), acc14);
                        acc15 = Vector256.FusedMultiplyAdd(vw, Vector256.Load(vPast + 120), acc15);
                    }
                }

                acc0.Store(outHead);
                acc1.Store(outHead + 8);
                acc2.Store(outHead + 16);
                acc3.Store(outHead + 24);
                acc4.Store(outHead + 32);
                acc5.Store(outHead + 40);
                acc6.Store(outHead + 48);
                acc7.Store(outHead + 56);
                acc8.Store(outHead + 64);
                acc9.Store(outHead + 72);
                acc10.Store(outHead + 80);
                acc11.Store(outHead + 88);
                acc12.Store(outHead + 96);
                acc13.Store(outHead + 104);
                acc14.Store(outHead + 112);
                acc15.Store(outHead + 120);
            }
            else
            {
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

                        int d = 0;
                        if (Vector512.IsHardwareAccelerated && vHeadDim >= 16)
                        {
                            var vw512 = Vector512.Create(w);
                            int limit512 = vHeadDim - 16;
                            for (; d <= limit512; d += 16)
                            {
                                var vo = Vector512.Load(outHead + d);
                                var vv = Vector512.Load(vPast + d);
                                Vector512.FusedMultiplyAdd(vw512, vv, vo).Store(outHead + d);
                            }
                        }
                        if (Vector256.IsHardwareAccelerated && (vHeadDim - d) >= 8)
                        {
                            var vw256 = Vector256.Create(w);
                            int limit256 = vHeadDim - 8;
                            for (; d <= limit256; d += 8)
                            {
                                var vo = Vector256.Load(outHead + d);
                                var vv = Vector256.Load(vPast + d);
                                Vector256.FusedMultiplyAdd(vw256, vv, vo).Store(outHead + d);
                            }
                        }
                        for (; d < vHeadDim; d++)
                        {
                            outHead[d] += w * vPast[d];
                        }
                    }
                }
            }
        });
    }
}

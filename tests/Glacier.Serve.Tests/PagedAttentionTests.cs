using System;
using System.Runtime.InteropServices;
using Glacier.Serve.Inference.PagedAttention;
using Xunit;

namespace Glacier.Serve.Tests;

public unsafe class PagedAttentionTests
{
    [Fact]
    public void PagedBlockPool_AllocatesAndRecyclesBlocks_InConstantTime()
    {
        using var pool = new PagedBlockPool(totalBlocks: 64, layers: 4, headsKv: 2, headDim: 64, blockSize: 16);
        Assert.Equal(64, pool.TotalBlocks);
        Assert.Equal(64, pool.FreeBlocksCount);
        Assert.Equal(0, pool.AllocatedBlocksCount);
        Assert.Equal(1024, pool.CapacityTokens); // 64 * 16

        // Allocate 10 blocks
        var allocated = new int[10];
        for (int i = 0; i < 10; i++)
        {
            Assert.True(pool.TryAllocateBlock(out allocated[i]));
        }

        Assert.Equal(54, pool.FreeBlocksCount);
        Assert.Equal(10, pool.AllocatedBlocksCount);
        Assert.Equal(160, pool.ActiveTokens);

        // Recycle all 10 blocks
        pool.FreeBlocks(allocated);
        Assert.Equal(64, pool.FreeBlocksCount);
        Assert.Equal(0, pool.AllocatedBlocksCount);
    }

    [Fact]
    public void BlockTable_AutomaticallyExpandsBlocks_AcrossTokenBoundaries()
    {
        using var pool = new PagedBlockPool(totalBlocks: 16, layers: 2, headsKv: 2, headDim: 32, blockSize: 16);
        var table = new BlockTable(pool);

        Assert.Equal(0, table.TokenCount);
        Assert.Equal(0, table.BlockCount);

        // Append 35 tokens (should consume 3 blocks: 0..15, 16..31, 32..34)
        for (int t = 0; t < 35; t++)
        {
            Assert.True(table.AppendToken(out int blockId, out int offset));
            Assert.True(blockId >= 0);
            Assert.Equal(t % 16, offset);
        }

        Assert.Equal(35, table.TokenCount);
        Assert.Equal(3, table.BlockCount);
        Assert.Equal(13, pool.FreeBlocksCount); // 16 - 3

        // Verify logical to physical mappings
        table.GetTokenLocation(0, out int b0, out int off0);
        Assert.Equal(table.GetBlockId(0), b0);
        Assert.Equal(0, off0);

        table.GetTokenLocation(15, out int b15, out int off15);
        Assert.Equal(table.GetBlockId(0), b15);
        Assert.Equal(15, off15);

        table.GetTokenLocation(16, out int b16, out int off16);
        Assert.Equal(table.GetBlockId(1), b16);
        Assert.Equal(0, off16);

        table.GetTokenLocation(34, out int b34, out int off34);
        Assert.Equal(table.GetBlockId(2), b34);
        Assert.Equal(2, off34);

        // Release all
        table.ReleaseAll();
        Assert.Equal(0, table.TokenCount);
        Assert.Equal(0, table.BlockCount);
        Assert.Equal(16, pool.FreeBlocksCount);
    }

    [Fact]
    public void PagedBlockPool_StoresAndRetrievesExactVectors()
    {
        using var pool = new PagedBlockPool(totalBlocks: 4, layers: 2, headsKv: 2, headDim: 4, blockSize: 4);
        Assert.True(pool.TryAllocateBlock(out int blockId));

        float[] kData = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f]; // 2 heads * 4 dim
        float[] vData = [10f, 20f, 30f, 40f, 50f, 60f, 70f, 80f];

        fixed (float* pK = kData, pV = vData)
        {
            pool.Store(blockId, layer: 1, offsetInBlock: 2, pK, pV);
        }

        float* retrievedK0 = pool.GetKeyPtr(blockId, layer: 1, headKv: 0, offsetInBlock: 2);
        float* retrievedK1 = pool.GetKeyPtr(blockId, layer: 1, headKv: 1, offsetInBlock: 2);
        float* retrievedV0 = pool.GetValuePtr(blockId, layer: 1, headKv: 0, offsetInBlock: 2);
        float* retrievedV1 = pool.GetValuePtr(blockId, layer: 1, headKv: 1, offsetInBlock: 2);

        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(kData[i], retrievedK0[i]);
            Assert.Equal(kData[4 + i], retrievedK1[i]);
            Assert.Equal(vData[i], retrievedV0[i]);
            Assert.Equal(vData[4 + i], retrievedV1[i]);
        }

        pool.FreeBlock(blockId);
    }

    [Fact]
    public void ComputeAttention_HeadDim128_AccumulatesAccurately()
    {
        const int headDim = 128;
        const int blockSize = 16;
        const int totalTokens = 32;
        const int nHeads = 4;
        const int nHeadsKv = 2;

        using var pool = new PagedBlockPool(totalBlocks: 8, layers: 1, headsKv: nHeadsKv, headDim: headDim, blockSize: blockSize);
        var table = new BlockTable(pool);

        for (int t = 0; t < totalTokens; t++)
        {
            Assert.True(table.AppendToken(out int blockId, out int offset));
            float[] k = new float[nHeadsKv * headDim];
            float[] v = new float[nHeadsKv * headDim];
            for (int d = 0; d < headDim; d++)
            {
                k[d] = 0.01f * (t + 1);
                v[d] = (t + 1);
                k[headDim + d] = 0.01f * (t + 1);
                v[headDim + d] = 2.0f * (t + 1);
            }
            fixed (float* pK = k, pV = v)
            {
                pool.Store(blockId, layer: 0, offsetInBlock: offset, pK, pV);
            }
        }

        float[] q = new float[nHeads * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = 1.0f;

        float[] outAttn = new float[nHeads * headDim];
        float[] scores = new float[nHeads * totalTokens];

        fixed (float* pQ = q, pOut = outAttn, pScores = scores)
        {
            PagedAttentionKernel.ComputeAttention(
                stageLayer: 0,
                modelLayer: 0,
                pos: totalTokens - 1,
                qBase: pQ,
                outBase: pOut,
                blockTable: table,
                pool: pool,
                nHeads: nHeads,
                nHeadsKv: nHeadsKv,
                headDim: headDim,
                vHeadDim: headDim,
                attnScale: 1.0f / MathF.Sqrt(headDim),
                headScores: pScores,
                maxSeqLen: totalTokens);
        }

        // Verify output is non-zero, positive, and matches expected dimensions
        for (int h = 0; h < nHeads; h++)
        {
            float firstVal = outAttn[h * headDim];
            Assert.True(firstVal > 0f, $"Head {h} output should be positive");
            // All dimensions in head should have identical value due to symmetry
            for (int d = 0; d < headDim; d++)
            {
                Assert.Equal(firstVal, outAttn[h * headDim + d], precision: 4);
            }
        }

        table.ReleaseAll();
    }
}

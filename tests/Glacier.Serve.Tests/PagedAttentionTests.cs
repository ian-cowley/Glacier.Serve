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
}

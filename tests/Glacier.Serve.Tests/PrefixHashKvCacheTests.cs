using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Glacier.Serve.Inference.PagedAttention;
using Xunit;

namespace Glacier.Serve.Tests;

public unsafe class PrefixHashKvCacheTests
{
    [Fact]
    public void ComputePrefixHash_Deterministic()
    {
        int[] tokensA = [101, 2054, 2003, 1037, 3231, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        int[] tokensB = [101, 2054, 2003, 1037, 3231, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        ulong hashA = PagedBlockPool.ComputePrefixHash(0UL, tokensA);
        ulong hashB = PagedBlockPool.ComputePrefixHash(0UL, tokensB);

        Assert.NotEqual(0UL, hashA);
        Assert.Equal(hashA, hashB);
    }

    [Fact]
    public void ComputePrefixHash_Avalanche_ChangesDownstreamBlocks()
    {
        int[] block0A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        int[] block0B = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 99]; // 1 token changed

        int[] block1 = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115];

        ulong hash0A = PagedBlockPool.ComputePrefixHash(0UL, block0A);
        ulong hash0B = PagedBlockPool.ComputePrefixHash(0UL, block0B);
        Assert.NotEqual(hash0A, hash0B);

        ulong hash1A = PagedBlockPool.ComputePrefixHash(hash0A, block1);
        ulong hash1B = PagedBlockPool.ComputePrefixHash(hash0B, block1);
        Assert.NotEqual(hash1A, hash1B);
    }

    [Fact]
    public void ComputePrefixHash_OrderDependent()
    {
        int[] tokens1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        int[] tokens2 = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];

        ulong hash1 = PagedBlockPool.ComputePrefixHash(0UL, tokens1);
        ulong hash2 = PagedBlockPool.ComputePrefixHash(0UL, tokens2);

        Assert.NotEqual(hash1, hash2);
    }

    [Fact]
    public void AtomicRefCount_InitializesToOne()
    {
        using var pool = new PagedBlockPool(totalBlocks: 8, layers: 2, headsKv: 2, headDim: 32, blockSize: 16);
        Assert.True(pool.TryAllocateBlock(out int blockId));
        Assert.Equal(1, pool.GetRefCount(blockId));

        pool.ReleaseBlock(blockId);
        Assert.Equal(0, pool.GetRefCount(blockId));
        Assert.Equal(8, pool.FreeBlocksCount);
    }

    [Fact]
    public void AtomicRefCount_RetainAndRelease_RefCountsAccurately()
    {
        using var pool = new PagedBlockPool(totalBlocks: 8, layers: 2, headsKv: 2, headDim: 32, blockSize: 16);
        Assert.True(pool.TryAllocateBlock(out int blockId));
        Assert.Equal(1, pool.GetRefCount(blockId));

        Assert.Equal(2, pool.RetainBlock(blockId));
        Assert.Equal(3, pool.RetainBlock(blockId));
        Assert.Equal(3, pool.GetRefCount(blockId));

        // Release 2 references
        Assert.Equal(2, pool.ReleaseBlock(blockId));
        Assert.Equal(1, pool.ReleaseBlock(blockId));
        Assert.Equal(7, pool.FreeBlocksCount); // Still allocated

        // Final release
        Assert.Equal(0, pool.ReleaseBlock(blockId));
        Assert.Equal(8, pool.FreeBlocksCount); // Now free
    }

    [Fact]
    public void AtomicRefCount_ConcurrentRetainRelease_NoLeaksOrDoubleFree()
    {
        using var pool = new PagedBlockPool(totalBlocks: 32, layers: 2, headsKv: 2, headDim: 32, blockSize: 16);
        int iterations = 10_000;

        Parallel.For(0, iterations, _ =>
        {
            if (pool.TryAllocateBlock(out int bId))
            {
                pool.RetainBlock(bId);
                pool.ReleaseBlock(bId);
                pool.ReleaseBlock(bId);
            }
        });

        Assert.Equal(32, pool.FreeBlocksCount);
        Assert.Equal(0, pool.AllocatedBlocksCount);
    }

    [Fact]
    public void PrefixBlock_BindAndFind_RegistersAndRecycles()
    {
        using var pool = new PagedBlockPool(totalBlocks: 16, layers: 2, headsKv: 2, headDim: 32, blockSize: 16);
        int[] tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        ulong hash = PagedBlockPool.ComputePrefixHash(0UL, tokens);

        Assert.True(pool.TryAllocateBlock(out int blockId));
        pool.BindPrefixBlock(hash, blockId);

        // Sequence 2 finds it in prefix cache
        Assert.True(pool.TryFindPrefixBlock(hash, out int foundBlockId));
        Assert.Equal(blockId, foundBlockId);
        Assert.Equal(2, pool.GetRefCount(blockId)); // Incremented by TryFindPrefixBlock

        // First sequence releases its reference
        pool.ReleaseBlock(blockId);
        Assert.Equal(1, pool.GetRefCount(blockId));

        // Second sequence releases its reference
        pool.ReleaseBlock(foundBlockId);
        Assert.Equal(0, pool.GetRefCount(blockId));

        // Block has been returned to pool and purged from prefix map
        Assert.False(pool.TryFindPrefixBlock(hash, out _));
        Assert.Equal(16, pool.FreeBlocksCount);
    }

    [Fact]
    public void CopyOnWrite_AppendToken_SharedBlock_TriggersFork()
    {
        using var pool = new PagedBlockPool(totalBlocks: 16, layers: 2, headsKv: 2, headDim: 32, blockSize: 16);
        var tableA = new BlockTable(pool);

        // Append 10 tokens to Sequence A
        for (int i = 0; i < 10; i++)
        {
            Assert.True(tableA.AppendToken(out _, out _));
        }

        int sharedBlockId = tableA.GetBlockId(0);
        Assert.Equal(1, pool.GetRefCount(sharedBlockId));

        // Write sentinel float to KV buffer
        float* kPtrA = pool.GetKeyPtr(sharedBlockId, layer: 0, headKv: 0, offsetInBlock: 0);
        *kPtrA = 42.0f;

        // Sequence B shares this block
        var tableB = new BlockTable(pool);
        pool.RetainBlock(sharedBlockId);
        tableB.AttachPrefixBlocks([sharedBlockId], tokenCount: 10);
        Assert.Equal(2, pool.GetRefCount(sharedBlockId));

        // Sequence A appends 11th token -> should trigger Copy-On-Write fork!
        Assert.True(tableA.AppendToken(out int blockAfterFork, out int offsetAfterFork));
        Assert.Equal(10, offsetAfterFork);
        Assert.NotEqual(sharedBlockId, blockAfterFork);

        // Verify refcounts: original block should now have refcount 1 (owned by B), forked block has refcount 1 (owned by A)
        Assert.Equal(1, pool.GetRefCount(sharedBlockId));
        Assert.Equal(1, pool.GetRefCount(blockAfterFork));

        // Verify data in forked block matches original block
        float* kPtrForked = pool.GetKeyPtr(blockAfterFork, layer: 0, headKv: 0, offsetInBlock: 0);
        Assert.Equal(42.0f, *kPtrForked);

        // Mutate forked block in Sequence A
        *kPtrForked = 999.0f;

        // Sequence B's original block is unaltered
        Assert.Equal(42.0f, *kPtrA);

        tableA.ReleaseAll();
        tableB.ReleaseAll();
        Assert.Equal(16, pool.FreeBlocksCount);
    }

    [Fact]
    public void BlockTable_AttachPrefixBlocks_AttachesCorrectly()
    {
        using var pool = new PagedBlockPool(totalBlocks: 16, layers: 2, headsKv: 2, headDim: 32, blockSize: 16);
        Assert.True(pool.TryAllocateBlock(out int block0));
        Assert.True(pool.TryAllocateBlock(out int block1));

        var table = new BlockTable(pool);
        table.AttachPrefixBlocks([block0, block1], tokenCount: 32);

        Assert.Equal(32, table.TokenCount);
        Assert.Equal(2, table.BlockCount);
        Assert.Equal(block0, table.GetBlockId(0));
        Assert.Equal(block1, table.GetBlockId(1));

        table.ReleaseAll();
        Assert.Equal(16, pool.FreeBlocksCount);
    }
}

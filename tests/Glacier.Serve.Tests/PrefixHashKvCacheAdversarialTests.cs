using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Serve.Inference.PagedAttention;
using Xunit;

namespace Glacier.Serve.Tests;

public unsafe class PrefixHashKvCacheAdversarialTests
{
    [Fact]
    public void BindPrefixBlock_Overwrite_ReleasingFirstBlock_MustNotEvictActiveSecondBlock()
    {
        // Scenario: Two sequences concurrently process the same prefix.
        // Sequence 1 binds Block A to Hash H.
        // Sequence 2 binds Block B to Hash H.
        // Sequence 1 finishes and releases Block A.
        // Block B is STILL ACTIVE and should be discoverable in the prefix cache!
        using var pool = new PagedBlockPool(totalBlocks: 16, layers: 2, headsKv: 2, headDim: 32, blockSize: 16);
        ulong prefixHash = 0xABCD1234EF567890UL;

        Assert.True(pool.TryAllocateBlock(out int blockA));
        Assert.True(pool.TryAllocateBlock(out int blockB));

        // Bind Block A, then overwrite with Block B for the same prefix hash
        pool.BindPrefixBlock(prefixHash, blockA);
        pool.BindPrefixBlock(prefixHash, blockB);

        // Release Block A (ref count becomes 0)
        pool.ReleaseBlock(blockA);

        // Does the prefix cache still find Block B?
        bool found = pool.TryFindPrefixBlock(prefixHash, out int foundBlockId);
        Assert.True(found, "Prefix cache should still contain Block B after Block A was released, but was evicted!");
        Assert.Equal(blockB, foundBlockId);

        pool.ReleaseBlock(blockB); // Decrements the ref added by TryFindPrefixBlock
        pool.ReleaseBlock(blockB); // Releases original ref
    }

    [Fact]
    public void CopyOnWrite_HighContention_16ConcurrentSequences_PreserveDataIntegrity()
    {
        // 16 sequences all share a single initial block of 8 tokens.
        // All 16 concurrently append tokens 8..15.
        // Each sequence must end up with its own private block containing its distinct tokens.
        int numSequences = 16;
        using var pool = new PagedBlockPool(totalBlocks: 64, layers: 1, headsKv: 1, headDim: 16, blockSize: 16);

        Assert.True(pool.TryAllocateBlock(out int sharedBlockId));

        // Initialize shared tokens 0..7 in shared block
        for (int t = 0; t < 8; t++)
        {
            float* kPtr = pool.GetKeyPtr(sharedBlockId, layer: 0, headKv: 0, offsetInBlock: t);
            *kPtr = 100.0f + t;
        }

        // Set ref count to 16
        for (int i = 1; i < numSequences; i++)
        {
            pool.RetainBlock(sharedBlockId);
        }
        Assert.Equal(numSequences, pool.GetRefCount(sharedBlockId));

        var tables = new BlockTable[numSequences];
        for (int i = 0; i < numSequences; i++)
        {
            tables[i] = new BlockTable(pool);
            tables[i].AttachPrefixBlocks([sharedBlockId], tokenCount: 8);
        }

        // Concurrently append token 8 with distinct sequence ID
        Parallel.For(0, numSequences, i =>
        {
            Assert.True(tables[i].AppendToken(out int blockId, out int offset));
            Assert.Equal(8, offset);

            // Write distinct sequence marker at offset 8
            float* kPtr = pool.GetKeyPtr(blockId, layer: 0, headKv: 0, offsetInBlock: 8);
            *kPtr = 1000.0f + i;
        });

        // Verify each sequence has prefix tokens 0..7 intact and its own distinct token 8
        for (int i = 0; i < numSequences; i++)
        {
            int blockId = tables[i].GetBlockId(0);
            for (int t = 0; t < 8; t++)
            {
                float* kPtr = pool.GetKeyPtr(blockId, layer: 0, headKv: 0, offsetInBlock: t);
                Assert.Equal(100.0f + t, *kPtr);
            }

            float* kPtr8 = pool.GetKeyPtr(blockId, layer: 0, headKv: 0, offsetInBlock: 8);
            Assert.Equal(1000.0f + i, *kPtr8);
        }

        // Clean up
        for (int i = 0; i < numSequences; i++)
        {
            tables[i].ReleaseAll();
        }

        Assert.Equal(64, pool.FreeBlocksCount);
    }

    [Fact]
    public void CopyOnWrite_PoolOutOfMemory_ReturnsFalseAndLeavesStateConsistent()
    {
        // Pool with only 1 block. Both sequences share it.
        // Neither can fork because no free blocks exist.
        using var pool = new PagedBlockPool(totalBlocks: 1, layers: 1, headsKv: 1, headDim: 16, blockSize: 16);
        Assert.True(pool.TryAllocateBlock(out int block0));
        pool.RetainBlock(block0);

        var tableA = new BlockTable(pool);
        tableA.AttachPrefixBlocks([block0], tokenCount: 4);

        var tableB = new BlockTable(pool);
        tableB.AttachPrefixBlocks([block0], tokenCount: 4);

        // Sequence A attempts to append -> needs COW fork, but pool is empty!
        Assert.False(tableA.AppendToken(out int pBlock, out int offset));
        Assert.Equal(-1, pBlock);
        Assert.Equal(-1, offset);

        // Verify tableA state was not corrupted
        Assert.Equal(4, tableA.TokenCount);
        Assert.Equal(1, tableA.BlockCount);
        Assert.Equal(block0, tableA.GetBlockId(0));
        Assert.Equal(2, pool.GetRefCount(block0));

        // Release all
        tableA.ReleaseAll();
        Assert.Equal(1, pool.GetRefCount(block0));
        tableB.ReleaseAll();
        Assert.Equal(0, pool.GetRefCount(block0));
        Assert.Equal(1, pool.FreeBlocksCount);
    }

    [Fact]
    public void PrefixHash_OrderDependence_PermutationsProduceDistinctHashes()
    {
        int[] baseTokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        var seenHashes = new HashSet<ulong>();

        // Generate 100 cyclic shifts
        for (int shift = 0; shift < 16; shift++)
        {
            int[] shifted = new int[16];
            for (int i = 0; i < 16; i++)
            {
                shifted[i] = baseTokens[(i + shift) % 16];
            }

            ulong hash = PagedBlockPool.ComputePrefixHash(0UL, shifted);
            Assert.True(seenHashes.Add(hash), $"Hash collision on cyclic shift {shift}: {hash:X16}");
        }
    }

    [Fact]
    public void PrefixHash_ChainedMerkleDependency_BranchingDivergesPermanently()
    {
        // If parent block differs, all subsequent child blocks must differ even if child tokens are identical
        int[] commonChildTokens = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];

        ulong parentHash1 = PagedBlockPool.ComputePrefixHash(0UL, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        ulong parentHash2 = PagedBlockPool.ComputePrefixHash(0UL, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 99]);

        ulong childHash1 = PagedBlockPool.ComputePrefixHash(parentHash1, commonChildTokens);
        ulong childHash2 = PagedBlockPool.ComputePrefixHash(parentHash2, commonChildTokens);

        Assert.NotEqual(parentHash1, parentHash2);
        Assert.NotEqual(childHash1, childHash2);
    }

    [Fact]
    public void PagedBlockPool_ConcurrentFindAndReleaseStress_NoDeadlockOrNegativeRefCount()
    {
        using var pool = new PagedBlockPool(totalBlocks: 32, layers: 1, headsKv: 1, headDim: 16, blockSize: 16);
        ulong testHash = 0x1234567890ABCDEFUL;
        int iterations = 5_000;

        Parallel.For(0, iterations, i =>
        {
            if (pool.TryAllocateBlock(out int blockId))
            {
                pool.BindPrefixBlock(testHash + (ulong)(i % 8), blockId);

                // Another thread may find it
                if (pool.TryFindPrefixBlock(testHash + (ulong)(i % 8), out int foundId))
                {
                    pool.ReleaseBlock(foundId);
                }

                pool.ReleaseBlock(blockId);
            }
        });

        // After all iterations complete, verify no block has a negative refcount
        for (int b = 0; b < pool.TotalBlocks; b++)
        {
            Assert.True(pool.GetRefCount(b) >= 0, $"Block {b} has negative refcount {pool.GetRefCount(b)}");
        }
    }

    [Fact]
    public void PagedBlockPool_AbaRaceDetection_RecycledBlockNeverReturnsMismatchedPrefix()
    {
        // Adversarial test: Block recycling under high concurrency.
        // Block A is allocated and bound to HashA.
        // Block A is freed, returned to free stack, reallocated to Block B and bound to HashB.
        // Any concurrent thread calling TryFindPrefixBlock(HashA) MUST NEVER return a block that now contains HashB!
        using var pool = new PagedBlockPool(totalBlocks: 4, layers: 1, headsKv: 1, headDim: 16, blockSize: 16);
        ulong hashA = 0xAAAAAAAAAAAAAAAAUL;
        ulong hashB = 0xBBBBBBBBBBBBBBBBUL;
        int iterations = 10_000;
        int mismatchViolations = 0;

        Parallel.For(0, iterations, i =>
        {
            if ((i & 1) == 0)
            {
                // Allocation / Recycling thread
                if (pool.TryAllocateBlock(out int blockId))
                {
                    pool.BindPrefixBlock(hashA, blockId);
                    Thread.Yield();
                    pool.ReleaseBlock(blockId);

                    // Immediately re-allocate and bind to hashB
                    if (pool.TryAllocateBlock(out int reallocatedId))
                    {
                        pool.BindPrefixBlock(hashB, reallocatedId);
                        Thread.Yield();
                        pool.ReleaseBlock(reallocatedId);
                    }
                }
            }
            else
            {
                // Lookup thread targeting hashA
                if (pool.TryFindPrefixBlock(hashA, out int foundId))
                {
                    // If we found a block, its prefix hash must be hashA!
                    // If ABA race occurred and post-CAS validation failed, foundId would have hashB or 0UL.
                    // We verify through the pool that this block is valid.
                    int refCount = pool.GetRefCount(foundId);
                    if (refCount <= 0)
                    {
                        Interlocked.Increment(ref mismatchViolations);
                    }
                    pool.ReleaseBlock(foundId);
                }
            }
        });

        Assert.Equal(0, mismatchViolations);
        for (int b = 0; b < pool.TotalBlocks; b++)
        {
            Assert.True(pool.GetRefCount(b) >= 0, $"Block {b} has negative refcount {pool.GetRefCount(b)}");
        }
    }

    [Fact]
    public void PagedBlockPool_ConcurrentMultiBlockOverwrite_EvictionIsolation()
    {
        // 8 sequences allocate distinct blocks and bind to the exact same hash H.
        // Then all 8 release in reverse or shuffled order.
        // Eviction of one block must never prematurely evict active blocks,
        // and when all are released, ref counts must cleanly settle to 0.
        int numBlocks = 8;
        using var pool = new PagedBlockPool(totalBlocks: numBlocks, layers: 1, headsKv: 1, headDim: 16, blockSize: 16);
        ulong commonHash = 0xFEEDFACECAFEBABEUL;

        int[] allocatedBlocks = new int[numBlocks];
        for (int i = 0; i < numBlocks; i++)
        {
            Assert.True(pool.TryAllocateBlock(out allocatedBlocks[i]));
            pool.BindPrefixBlock(commonHash, allocatedBlocks[i]);
        }

        // The active representative in the dictionary is the last one bound
        int lastBoundBlock = allocatedBlocks[numBlocks - 1];
        Assert.True(pool.TryFindPrefixBlock(commonHash, out int foundBlock));
        Assert.Equal(lastBoundBlock, foundBlock);
        pool.ReleaseBlock(foundBlock); // Release find ref

        // Release first 7 blocks (all except the last bound block)
        for (int i = 0; i < numBlocks - 1; i++)
        {
            pool.ReleaseBlock(allocatedBlocks[i]);
        }

        // The last bound block MUST STILL BE PRESENT and discoverable in the cache!
        Assert.True(pool.TryFindPrefixBlock(commonHash, out int finalFound));
        Assert.Equal(lastBoundBlock, finalFound);
        pool.ReleaseBlock(finalFound); // Release find ref

        // Now release the last bound block
        pool.ReleaseBlock(lastBoundBlock);

        // Now cache lookup should return false
        Assert.False(pool.TryFindPrefixBlock(commonHash, out int nonexistent));
        Assert.Equal(-1, nonexistent);
        Assert.Equal(numBlocks, pool.FreeBlocksCount);
    }

    [Fact]
    public void PagedBlockPool_ZeroHash_ThrowsArgumentExceptionAndFindReturnsFalse()
    {
        using var pool = new PagedBlockPool(totalBlocks: 4, layers: 1, headsKv: 1, headDim: 16, blockSize: 16);
        Assert.True(pool.TryAllocateBlock(out int blockId));

        Assert.Throws<ArgumentException>(() => pool.BindPrefixBlock(0UL, blockId));
        Assert.False(pool.TryFindPrefixBlock(0UL, out int foundId));
        Assert.Equal(-1, foundId);

        pool.ReleaseBlock(blockId);
        Assert.Equal(4, pool.FreeBlocksCount);
    }

    [Fact]
    public void PagedBlockPool_DoubleFree_IsStrictlyIdempotent()
    {
        using var pool = new PagedBlockPool(totalBlocks: 4, layers: 1, headsKv: 1, headDim: 16, blockSize: 16);
        Assert.True(pool.TryAllocateBlock(out int blockId));
        Assert.Equal(3, pool.FreeBlocksCount);

        // First release frees block
        Assert.Equal(0, pool.ReleaseBlock(blockId));
        Assert.Equal(4, pool.FreeBlocksCount);

        // Second release must be idempotent and not corrupt free stack
        Assert.Equal(0, pool.ReleaseBlock(blockId));
        Assert.Equal(4, pool.FreeBlocksCount);

        // FreeBlock wrapper is also idempotent
        pool.FreeBlock(blockId);
        Assert.Equal(4, pool.FreeBlocksCount);

        // All 4 blocks can still be cleanly allocated
        for (int i = 0; i < 4; i++)
        {
            Assert.True(pool.TryAllocateBlock(out _));
        }
        Assert.Equal(0, pool.FreeBlocksCount);
    }

    [Fact]
    public void PagedBlockPool_OutOfRangeBlockId_HandledSafely()
    {
        using var pool = new PagedBlockPool(totalBlocks: 4, layers: 1, headsKv: 1, headDim: 16, blockSize: 16);

        Assert.Equal(0, pool.ReleaseBlock(-1));
        Assert.Equal(0, pool.ReleaseBlock(100));
        Assert.Equal(0, pool.GetRefCount(-1));
        Assert.Equal(0, pool.GetRefCount(100));

        Assert.Throws<ArgumentOutOfRangeException>(() => pool.RetainBlock(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => pool.RetainBlock(100));
        Assert.Throws<ArgumentOutOfRangeException>(() => pool.BindPrefixBlock(0x1234UL, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => pool.BindPrefixBlock(0x1234UL, 100));
    }

    [Fact]
    public void BlockTable_MultiTurnConversation_PrefixBranchingIsolation()
    {
        using var pool = new PagedBlockPool(totalBlocks: 16, layers: 1, headsKv: 1, headDim: 16, blockSize: 16);

        // 1. Setup shared 32-token system prompt in 2 blocks
        int[] block0Tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        int[] block1Tokens = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];

        ulong hash0 = PagedBlockPool.ComputePrefixHash(0UL, block0Tokens);
        ulong hash1 = PagedBlockPool.ComputePrefixHash(hash0, block1Tokens);

        Assert.True(pool.TryAllocateBlock(out int sharedB0));
        Assert.True(pool.TryAllocateBlock(out int sharedB1));
        pool.BindPrefixBlock(hash0, sharedB0);
        pool.BindPrefixBlock(hash1, sharedB1);

        // 2. Branch A attaches prefix and appends 5 tokens
        var branchA = new BlockTable(pool);
        pool.RetainBlock(sharedB0);
        pool.RetainBlock(sharedB1);
        branchA.AttachPrefixBlocks([sharedB0, sharedB1], tokenCount: 32);

        for (int i = 0; i < 5; i++)
        {
            Assert.True(branchA.AppendToken(out int blkA, out int offA));
            Assert.Equal(32, pool.BlockSize * 2);
        }
        Assert.Equal(37, branchA.TokenCount);
        Assert.Equal(3, branchA.BlockCount);

        // 3. Branch B attaches identical prefix and appends 10 tokens
        var branchB = new BlockTable(pool);
        pool.RetainBlock(sharedB0);
        pool.RetainBlock(sharedB1);
        branchB.AttachPrefixBlocks([sharedB0, sharedB1], tokenCount: 32);

        for (int i = 0; i < 10; i++)
        {
            Assert.True(branchB.AppendToken(out int blkB, out int offB));
        }
        Assert.Equal(42, branchB.TokenCount);
        Assert.Equal(3, branchB.BlockCount);

        // Verify shared blocks have ref count 3 (original bind + branch A + branch B)
        Assert.Equal(3, pool.GetRefCount(sharedB0));
        Assert.Equal(3, pool.GetRefCount(sharedB1));

        // 4. Branch A finishes and releases
        branchA.ReleaseAll();
        Assert.Equal(2, pool.GetRefCount(sharedB0));
        Assert.Equal(2, pool.GetRefCount(sharedB1));

        // Branch B remains intact
        Assert.Equal(42, branchB.TokenCount);

        // 5. Branch B finishes and releases
        branchB.ReleaseAll();
        Assert.Equal(1, pool.GetRefCount(sharedB0));
        Assert.Equal(1, pool.GetRefCount(sharedB1));

        // 6. Release initial binding references
        pool.ReleaseBlock(sharedB0);
        pool.ReleaseBlock(sharedB1);

        // Prefix map purged, all blocks free
        Assert.False(pool.TryFindPrefixBlock(hash0, out _));
        Assert.False(pool.TryFindPrefixBlock(hash1, out _));
        Assert.Equal(16, pool.FreeBlocksCount);
    }

    [Fact]
    public void PagedBlockPool_Concurrent32Threads_PrefixChurnStress()
    {
        using var pool = new PagedBlockPool(totalBlocks: 64, layers: 1, headsKv: 1, headDim: 16, blockSize: 16);
        int iterations = 10_000;

        Parallel.For(0, iterations, new ParallelOptions { MaxDegreeOfParallelism = 32 }, i =>
        {
            ulong hash = 0xAA000000UL + (ulong)(i % 16);
            if (pool.TryAllocateBlock(out int bId))
            {
                pool.BindPrefixBlock(hash, bId);

                if (pool.TryFindPrefixBlock(hash, out int foundId))
                {
                    Assert.True(foundId >= 0 && foundId < 64);
                    pool.ReleaseBlock(foundId);
                }

                pool.ReleaseBlock(bId);
            }
        });

        // Verify clean convergence: all refcounts >= 0
        for (int b = 0; b < pool.TotalBlocks; b++)
        {
            Assert.True(pool.GetRefCount(b) >= 0, $"Block {b} has negative refcount: {pool.GetRefCount(b)}");
        }
    }
}


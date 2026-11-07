using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Glacier.Serve.Inference.PagedAttention;

/// <summary>
/// Logical-to-physical virtual page table for an individual sequence.
/// Tracks allocated physical blocks from the PagedBlockPool, allowing sequences
/// to dynamically grow without requiring contiguous physical memory allocation.
/// </summary>
public sealed class BlockTable
{
    private readonly PagedBlockPool _pool;
    private readonly List<int> _physicalBlockIds = new(16);

    public int TokenCount { get; private set; }
    public int BlockSize => _pool.BlockSize;
    public int BlockCount => _physicalBlockIds.Count;
    public IReadOnlyList<int> PhysicalBlockIds => _physicalBlockIds;

    public BlockTable(PagedBlockPool pool)
    {
        _pool = pool ?? throw new ArgumentNullException(nameof(pool));
    }

    /// <summary>
    /// Attaches pre-allocated shared prefix blocks to this sequence's virtual page table.
    /// </summary>
    public void AttachPrefixBlocks(ReadOnlySpan<int> prefixBlockIds, int tokenCount)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(tokenCount, 0);
        for (int i = 0; i < prefixBlockIds.Length; i++)
        {
            _physicalBlockIds.Add(prefixBlockIds[i]);
        }
        TokenCount = tokenCount;
    }

    /// <summary>
    /// Allocates physical slot for next token. If block boundary is crossed, allocates new physical block.
    /// Enforces Copy-On-Write if the current block is shared.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool AppendToken(out int physicalBlockId, out int offsetInBlock)
    {
        if (TokenCount % BlockSize == 0)
        {
            if (!_pool.TryAllocateBlock(out int newBlockId))
            {
                physicalBlockId = -1;
                offsetInBlock = -1;
                return false;
            }
            _physicalBlockIds.Add(newBlockId);
        }
        else
        {
            int lastBlockId = _physicalBlockIds[^1];
            if (_pool.GetRefCount(lastBlockId) > 1)
            {
                // COW: Fork shared partial block
                int tokensInBlock = TokenCount % BlockSize;
                if (!_pool.TryForkBlock(lastBlockId, tokensInBlock, out int forkedBlockId))
                {
                    physicalBlockId = -1;
                    offsetInBlock = -1;
                    return false;
                }
                _pool.ReleaseBlock(lastBlockId);
                _physicalBlockIds[^1] = forkedBlockId;
            }
        }

        physicalBlockId = _physicalBlockIds[^1];
        offsetInBlock = TokenCount % BlockSize;
        TokenCount++;
        return true;
    }

    /// <summary>
    /// Maps a token index t to its physical block ID and offset within the block.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void GetTokenLocation(int tokenIndex, out int physicalBlockId, out int offsetInBlock)
    {
        int blockIndex = tokenIndex / BlockSize;
        physicalBlockId = _physicalBlockIds[blockIndex];
        offsetInBlock = tokenIndex % BlockSize;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int GetBlockId(int blockIndex)
    {
        return _physicalBlockIds[blockIndex];
    }

    /// <summary>
    /// Zero-allocation contiguous span view of physical block IDs for direct HtoD transfer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<int> AsSpan() => CollectionsMarshal.AsSpan(_physicalBlockIds);

    /// <summary>
    /// Recycles all physical blocks back to the pool and resets sequence state.
    /// </summary>
    public void ReleaseAll()
    {
        for (int i = 0; i < _physicalBlockIds.Count; i++)
        {
            _pool.ReleaseBlock(_physicalBlockIds[i]);
        }
        _physicalBlockIds.Clear();
        TokenCount = 0;
    }
}

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace Glacier.Serve.Inference.PagedAttention;

/// <summary>
/// Unmanaged physical Key-Value memory pool for PagedAttention.
/// Partitions KV memory into uniform blocks of fixed size (default: 16 tokens),
/// eliminating internal and external VRAM/RAM fragmentation.
/// Supports thread-safe O(1) lock-free block allocation and idempotent recycling.
/// </summary>
public sealed unsafe class PagedBlockPool : IDisposable
{
    public const int DefaultBlockSize = 16;
    public const nuint CacheLineAlignment = 64;

    private readonly int _totalBlocks;
    private readonly int _blockSize;
    private readonly int _layers;
    private readonly int _headsKv;
    private readonly int _headDim;
    private readonly int _vHeadDim;

    private readonly long _headStrideK;
    private readonly long _layerStrideK;
    private readonly long _blockStrideK;

    private readonly long _headStrideV;
    private readonly long _layerStrideV;
    private readonly long _blockStrideV;

    private readonly float* _kBuffer;
    private readonly float* _vBuffer;
    private readonly bool _allocateHostBuffers;
    private readonly ConcurrentStack<int> _freeBlocks;
    private readonly int[] _blockAllocatedState;
    private readonly int[] _refCounts;
    private readonly ulong[] _blockPrefixHashes;
    private readonly ConcurrentDictionary<ulong, int> _prefixHashMap;
    private bool _disposed;

    public int TotalBlocks => _totalBlocks;
    public int BlockSize => _blockSize;
    public int Layers => _layers;
    public int HeadsKv => _headsKv;
    public int HeadDim => _headDim;
    public int ValueHeadDim => _vHeadDim;
    public int FreeBlocksCount => _freeBlocks.Count;
    public int AllocatedBlocksCount => _totalBlocks - _freeBlocks.Count;
    public int CapacityTokens => _totalBlocks * _blockSize;
    public int ActiveTokens => AllocatedBlocksCount * _blockSize;
    public bool HasHostBuffers => _allocateHostBuffers;
    public long TotalMemoryBytes => _allocateHostBuffers ? (_totalBlocks * _blockStrideK + _totalBlocks * _blockStrideV) * sizeof(float) : 0L;
    public IntPtr KeyBufferAddress => (IntPtr)_kBuffer;
    public IntPtr ValueBufferAddress => (IntPtr)_vBuffer;

    public PagedBlockPool(int totalBlocks, int layers, int headsKv, int headDim, int blockSize = DefaultBlockSize, int vHeadDim = -1, bool allocateHostBuffers = true)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(totalBlocks, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(layers, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(headsKv, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(headDim, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(blockSize, 1);

        _totalBlocks = totalBlocks;
        _layers = layers;
        _headsKv = headsKv;
        _headDim = headDim;
        _vHeadDim = vHeadDim > 0 ? vHeadDim : headDim;
        _blockSize = blockSize;
        _allocateHostBuffers = allocateHostBuffers;

        _headStrideK = (long)_blockSize * _headDim;
        _layerStrideK = (long)_headsKv * _headStrideK;
        _blockStrideK = (long)_layers * _layerStrideK;

        _headStrideV = (long)_blockSize * _vHeadDim;
        _layerStrideV = (long)_headsKv * _headStrideV;
        _blockStrideV = (long)_layers * _layerStrideV;

        if (_allocateHostBuffers)
        {
            nuint totalBytesK = checked((nuint)((long)_totalBlocks * _blockStrideK * sizeof(float)));
            nuint totalBytesV = checked((nuint)((long)_totalBlocks * _blockStrideV * sizeof(float)));

            _kBuffer = (float*)NativeMemory.AlignedAlloc(totalBytesK, CacheLineAlignment);
            try
            {
                NativeMemory.Clear(_kBuffer, totalBytesK);
                _vBuffer = (float*)NativeMemory.AlignedAlloc(totalBytesV, CacheLineAlignment);
                NativeMemory.Clear(_vBuffer, totalBytesV);
            }
            catch
            {
                if (_kBuffer != null) NativeMemory.AlignedFree(_kBuffer);
                if (_vBuffer != null) NativeMemory.AlignedFree(_vBuffer);
                _kBuffer = null;
                _vBuffer = null;
                throw;
            }
        }
        else
        {
            _kBuffer = null;
            _vBuffer = null;
        }

        _blockAllocatedState = new int[_totalBlocks];
        _refCounts = new int[_totalBlocks];
        _blockPrefixHashes = new ulong[_totalBlocks];
        _prefixHashMap = new ConcurrentDictionary<ulong, int>();
        _freeBlocks = new ConcurrentStack<int>();
        for (int i = _totalBlocks - 1; i >= 0; i--)
        {
            _freeBlocks.Push(i);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetKeyPtr(int physicalBlockId, int layer, int headKv, int offsetInBlock)
    {
        if (!_allocateHostBuffers)
            throw new InvalidOperationException("Host KV buffers were omitted because KV pages reside exclusively in GPU VRAM.");
        long offset = (long)physicalBlockId * _blockStrideK +
                      (long)layer * _layerStrideK +
                      (long)headKv * _headStrideK +
                      (long)offsetInBlock * _headDim;
        return _kBuffer + offset;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetValuePtr(int physicalBlockId, int layer, int headKv, int offsetInBlock)
    {
        if (!_allocateHostBuffers)
            throw new InvalidOperationException("Host KV buffers were omitted because KV pages reside exclusively in GPU VRAM.");
        long offset = (long)physicalBlockId * _blockStrideV +
                      (long)layer * _layerStrideV +
                      (long)headKv * _headStrideV +
                      (long)offsetInBlock * _vHeadDim;
        return _vBuffer + offset;
    }

    /// <summary>
    /// Stores the projected Key and Value vectors for a specific layer and block offset.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Store(int physicalBlockId, int layer, int offsetInBlock, float* kSrc, float* vSrc)
    {
        for (int h = 0; h < _headsKv; h++)
        {
            float* kDst = GetKeyPtr(physicalBlockId, layer, h, offsetInBlock);
            float* vDst = GetValuePtr(physicalBlockId, layer, h, offsetInBlock);

            Buffer.MemoryCopy(kSrc + h * _headDim, kDst, _headDim * sizeof(float), _headDim * sizeof(float));
            Buffer.MemoryCopy(vSrc + h * _vHeadDim, vDst, _vHeadDim * sizeof(float), _vHeadDim * sizeof(float));
        }
    }

    /// <summary>
    /// Computes the 64-bit xxHash3 cumulative Merkle prefix hash over a block of token IDs.
    /// Uses the predecessor block's cumulative hash as seed.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong ComputePrefixHash(ulong parentHash, ReadOnlySpan<int> tokens)
    {
        ReadOnlySpan<byte> bytes = MemoryMarshal.AsBytes(tokens);
        return System.IO.Hashing.XxHash3.HashToUInt64(bytes, (long)parentHash);
    }

    /// <summary>
    /// Attempts to allocate a physical block from the free list in O(1).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryAllocateBlock(out int physicalBlockId)
    {
        if (_freeBlocks.TryPop(out physicalBlockId))
        {
            Interlocked.Exchange(ref _blockAllocatedState[physicalBlockId], 1);
            Interlocked.Exchange(ref _refCounts[physicalBlockId], 1);
            Volatile.Write(ref _blockPrefixHashes[physicalBlockId], 0UL);
            return true;
        }
        physicalBlockId = -1;
        return false;
    }

    /// <summary>
    /// Atomically searches for a cached block matching prefixHash and increments its ref-count if found.
    /// </summary>
    public bool TryFindPrefixBlock(ulong prefixHash, out int physicalBlockId)
    {
        if (prefixHash != 0UL && _prefixHashMap.TryGetValue(prefixHash, out physicalBlockId))
        {
            while (true)
            {
                int current = Volatile.Read(ref _refCounts[physicalBlockId]);
                if (current <= 0)
                {
                    physicalBlockId = -1;
                    return false;
                }
                if (Interlocked.CompareExchange(ref _refCounts[physicalBlockId], current + 1, current) == current)
                {
                    if (Volatile.Read(ref _blockPrefixHashes[physicalBlockId]) != prefixHash)
                    {
                        ReleaseBlock(physicalBlockId);
                        physicalBlockId = -1;
                        return false;
                    }
                    return true;
                }
            }
        }
        physicalBlockId = -1;
        return false;
    }

    /// <summary>
    /// Registers a full block's cumulative prefix hash in the cache.
    /// </summary>
    public void BindPrefixBlock(ulong prefixHash, int physicalBlockId)
    {
        if ((uint)physicalBlockId >= (uint)_totalBlocks)
            throw new ArgumentOutOfRangeException(nameof(physicalBlockId));
        if (prefixHash == 0UL)
            throw new ArgumentException("Prefix hash cannot be 0.", nameof(prefixHash));

        Volatile.Write(ref _blockPrefixHashes[physicalBlockId], prefixHash);
        _prefixHashMap[prefixHash] = physicalBlockId;
    }

    /// <summary>
    /// Increments atomic reference count for physical block.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int RetainBlock(int physicalBlockId)
    {
        if ((uint)physicalBlockId >= (uint)_totalBlocks)
            throw new ArgumentOutOfRangeException(nameof(physicalBlockId));
        return Interlocked.Increment(ref _refCounts[physicalBlockId]);
    }

    /// <summary>
    /// Decrements atomic reference count. Recycles block to free pool when ref-count reaches 0.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int ReleaseBlock(int physicalBlockId)
    {
        if ((uint)physicalBlockId >= (uint)_totalBlocks)
            return 0;

        int remaining = Interlocked.Decrement(ref _refCounts[physicalBlockId]);
        if (remaining < 0)
        {
            Interlocked.Exchange(ref _refCounts[physicalBlockId], 0);
            return 0;
        }

        if (remaining == 0)
        {
            ulong hash = Volatile.Read(ref _blockPrefixHashes[physicalBlockId]);
            if (hash != 0UL)
            {
                _prefixHashMap.TryRemove(new KeyValuePair<ulong, int>(hash, physicalBlockId));
                Volatile.Write(ref _blockPrefixHashes[physicalBlockId], 0UL);
            }

            if (Interlocked.CompareExchange(ref _blockAllocatedState[physicalBlockId], 0, 1) == 1)
            {
                _freeBlocks.Push(physicalBlockId);
            }
        }

        return remaining;
    }

    /// <summary>
    /// Returns current atomic reference count for inspection.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int GetRefCount(int physicalBlockId)
    {
        if ((uint)physicalBlockId >= (uint)_totalBlocks) return 0;
        return Volatile.Read(ref _refCounts[physicalBlockId]);
    }

    /// <summary>
    /// Recycles an allocated physical block back to the free list in O(1).
    /// Strictly idempotent: double-freeing or freeing an unallocated block is safely ignored.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void FreeBlock(int physicalBlockId)
    {
        ReleaseBlock(physicalBlockId);
    }

    /// <summary>
    /// Duplicates KV data up to validTokens from srcBlockId to dstBlockId for Copy-on-Write.
    /// </summary>
    public void CopyBlockData(int srcBlockId, int dstBlockId, int validTokens)
    {
        if (!_allocateHostBuffers)
            throw new InvalidOperationException("Host KV buffers were omitted.");
        if (validTokens <= 0) return;
        int count = Math.Min(validTokens, _blockSize);

        for (int l = 0; l < _layers; l++)
        {
            for (int h = 0; h < _headsKv; h++)
            {
                for (int t = 0; t < count; t++)
                {
                    float* srcK = GetKeyPtr(srcBlockId, l, h, t);
                    float* dstK = GetKeyPtr(dstBlockId, l, h, t);
                    Buffer.MemoryCopy(srcK, dstK, _headDim * sizeof(float), _headDim * sizeof(float));

                    float* srcV = GetValuePtr(srcBlockId, l, h, t);
                    float* dstV = GetValuePtr(dstBlockId, l, h, t);
                    Buffer.MemoryCopy(srcV, dstV, _vHeadDim * sizeof(float), _vHeadDim * sizeof(float));
                }
            }
        }
    }

    /// <summary>
    /// Allocates a new block and copies valid token KV data for Copy-On-Write isolation.
    /// </summary>
    public bool TryForkBlock(int srcBlockId, int validTokens, out int forkedBlockId)
    {
        if (!TryAllocateBlock(out forkedBlockId))
            return false;

        if (_allocateHostBuffers)
        {
            CopyBlockData(srcBlockId, forkedBlockId, validTokens);
        }
        return true;
    }

    /// <summary>
    /// Recycles multiple physical blocks back to the free list.
    /// </summary>
    public void FreeBlocks(ReadOnlySpan<int> physicalBlockIds)
    {
        for (int i = 0; i < physicalBlockIds.Length; i++)
        {
            FreeBlock(physicalBlockIds[i]);
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_kBuffer != null) NativeMemory.AlignedFree(_kBuffer);
            if (_vBuffer != null) NativeMemory.AlignedFree(_vBuffer);
        }
        GC.SuppressFinalize(this);
    }

    ~PagedBlockPool()
    {
        Dispose();
    }
}

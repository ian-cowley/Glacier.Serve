using System;
using System.Collections.Concurrent;
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
    /// Attempts to allocate a physical block from the free list in O(1).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryAllocateBlock(out int physicalBlockId)
    {
        if (_freeBlocks.TryPop(out physicalBlockId))
        {
            Interlocked.Exchange(ref _blockAllocatedState[physicalBlockId], 1);
            return true;
        }
        physicalBlockId = -1;
        return false;
    }

    /// <summary>
    /// Recycles an allocated physical block back to the free list in O(1).
    /// Strictly idempotent: double-freeing or freeing an unallocated block is safely ignored.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void FreeBlock(int physicalBlockId)
    {
        if ((uint)physicalBlockId < (uint)_totalBlocks &&
            Interlocked.CompareExchange(ref _blockAllocatedState[physicalBlockId], 0, 1) == 1)
        {
            _freeBlocks.Push(physicalBlockId);
        }
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

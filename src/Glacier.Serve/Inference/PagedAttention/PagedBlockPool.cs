using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Glacier.Serve.Inference.PagedAttention;

/// <summary>
/// Unmanaged physical Key-Value memory pool for PagedAttention.
/// Partitions KV memory into uniform blocks of fixed size (default: 16 tokens),
/// eliminating internal and external VRAM/RAM fragmentation.
/// Supports thread-safe O(1) lock-free block allocation and recycling.
/// </summary>
public sealed unsafe class PagedBlockPool : IDisposable
{
    public const int DefaultBlockSize = 16;

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
    private readonly ConcurrentStack<int> _freeBlocks;
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
    public long TotalMemoryBytes => (_totalBlocks * _blockStrideK + _totalBlocks * _blockStrideV) * sizeof(float);

    public PagedBlockPool(int totalBlocks, int layers, int headsKv, int headDim, int blockSize = DefaultBlockSize, int vHeadDim = -1)
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

        _headStrideK = (long)_blockSize * _headDim;
        _layerStrideK = (long)_headsKv * _headStrideK;
        _blockStrideK = (long)_layers * _layerStrideK;

        _headStrideV = (long)_blockSize * _vHeadDim;
        _layerStrideV = (long)_headsKv * _headStrideV;
        _blockStrideV = (long)_layers * _layerStrideV;

        long totalBytesK = (long)_totalBlocks * _blockStrideK * sizeof(float);
        long totalBytesV = (long)_totalBlocks * _blockStrideV * sizeof(float);

        _kBuffer = (float*)NativeMemory.AllocZeroed((nuint)totalBytesK);
        _vBuffer = (float*)NativeMemory.AllocZeroed((nuint)totalBytesV);

        _freeBlocks = new ConcurrentStack<int>();
        for (int i = _totalBlocks - 1; i >= 0; i--)
        {
            _freeBlocks.Push(i);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetKeyPtr(int physicalBlockId, int layer, int headKv, int offsetInBlock)
    {
        long offset = (long)physicalBlockId * _blockStrideK +
                      (long)layer * _layerStrideK +
                      (long)headKv * _headStrideK +
                      (long)offsetInBlock * _headDim;
        return _kBuffer + offset;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetValuePtr(int physicalBlockId, int layer, int headKv, int offsetInBlock)
    {
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
        return _freeBlocks.TryPop(out physicalBlockId);
    }

    /// <summary>
    /// Recycles a physical block back to the free list in O(1).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void FreeBlock(int physicalBlockId)
    {
        if (physicalBlockId >= 0 && physicalBlockId < _totalBlocks)
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
            if (_kBuffer != null) NativeMemory.Free(_kBuffer);
            if (_vBuffer != null) NativeMemory.Free(_vBuffer);
        }
        GC.SuppressFinalize(this);
    }

    ~PagedBlockPool()
    {
        Dispose();
    }
}

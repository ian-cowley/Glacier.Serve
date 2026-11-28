using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using Glacier.Inference.Memory;
using Glacier.Inference.Quant;

namespace Glacier.Serve.Inference.PagedAttention;

/// <summary>
/// Unmanaged physical Key-Value memory pool for PagedAttention.
/// Partitions KV memory into uniform blocks of fixed size (default: 16 tokens),
/// eliminating internal and external VRAM/RAM fragmentation.
/// Supports <see cref="KvCachePrecision.Fp32"/>, <see cref="KvCachePrecision.Fp16"/>,
/// <see cref="KvCachePrecision.Fp8"/>, <see cref="KvCachePrecision.Q8_0"/>, and
/// <see cref="KvCachePrecision.Q4_0"/> (<see cref="KvCachePrecision.Int4"/>) with
/// thread-safe O(1) block allocation, token-verified prefix caching, and LRU eviction.
/// </summary>
public sealed unsafe class PagedBlockPool : IDisposable
{
    public const int DefaultBlockSize = 16;
    public const nuint CacheLineAlignment = 64;

    private const int StateFree = 0;
    private const int StateActive = 1;
    private const int StateCached = 2;

    private readonly int _totalBlocks;
    private readonly int _blockSize;
    private readonly int _layers;
    private readonly int _headsKv;
    private readonly int _headDim;
    private readonly int _vHeadDim;
    private readonly KvCachePrecision _precision;
    private readonly bool _hasExplicitPrecision;

    private readonly long _slotBytesK;
    private readonly long _headByteStrideK;
    private readonly long _layerByteStrideK;
    private readonly long _blockByteStrideK;

    private readonly long _slotBytesV;
    private readonly long _headByteStrideV;
    private readonly long _layerByteStrideV;
    private readonly long _blockByteStrideV;

    private byte* _kBuffer;
    private byte* _vBuffer;
    private readonly bool _allocateHostBuffers;
    private readonly Lock _allocLock = new();
    private readonly int[] _freeStack;
    private int _freeCount;
    private readonly int[] _blockState;
    private readonly int[] _refCounts;
    private readonly ulong[] _blockPrefixHashes;
    private readonly int[] _blockTokenStorage;
    private readonly int[] _blockTokenCounts;
    private readonly Dictionary<ulong, int> _prefixHashMap;
    private readonly int[] _lruPrev;
    private readonly int[] _lruNext;
    private int _lruHead = -1;
    private int _lruTail = -1;
    private int _lruCount;
    private int _disposed;

    internal Action<int, int, int>? OnBlockForked { get; set; }
    internal bool HasExplicitPrecision => _hasExplicitPrecision;

    public int TotalBlocks => _totalBlocks;
    public int BlockSize => _blockSize;
    public int Layers => _layers;
    public int HeadsKv => _headsKv;
    public int HeadDim => _headDim;
    public int ValueHeadDim => _vHeadDim;
    public KvCachePrecision Precision => _precision;
    public long KeySlotBytes => _slotBytesK;
    public long ValueSlotBytes => _slotBytesV;
    public int FreeBlocksCount
    {
        get
        {
            lock (_allocLock)
            {
                return _freeCount + _lruCount;
            }
        }
    }
    public int CachedPrefixBlocksCount
    {
        get
        {
            lock (_allocLock)
            {
                return _lruCount;
            }
        }
    }
    public int AllocatedBlocksCount => _totalBlocks - FreeBlocksCount;
    public int CapacityTokens => _totalBlocks * _blockSize;
    public int ActiveTokens => AllocatedBlocksCount * _blockSize;
    public bool HasHostBuffers => _allocateHostBuffers;
    public long TotalMemoryBytes => _allocateHostBuffers ? checked((long)_totalBlocks * (_blockByteStrideK + _blockByteStrideV)) : 0L;
    public IntPtr KeyBufferAddress => (IntPtr)_kBuffer;
    public IntPtr ValueBufferAddress => (IntPtr)_vBuffer;

    public static long GetSlotByteSize(int dim, KvCachePrecision precision)
    {
        KvCachePrecision effective = precision == KvCachePrecision.Auto ? KvCachePrecision.Fp32 : precision;
        return KVCache.GetSlotByteSize(dim, effective);
    }

    public PagedBlockPool(
        int totalBlocks,
        int layers,
        int headsKv,
        int headDim,
        int blockSize = DefaultBlockSize,
        int vHeadDim = -1,
        bool allocateHostBuffers = true,
        KvCachePrecision kvPrecision = KvCachePrecision.Auto)
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
        _hasExplicitPrecision = kvPrecision != KvCachePrecision.Auto;
        _precision = kvPrecision == KvCachePrecision.Auto
            ? (allocateHostBuffers ? KvCachePrecision.Fp32 : KvCachePrecision.Auto)
            : kvPrecision;

        KvCachePrecision stridePrecision = _precision == KvCachePrecision.Auto ? KvCachePrecision.Fp16 : _precision;

        checked
        {
            _slotBytesK = KVCache.GetSlotByteSize(_headDim, stridePrecision);
            _headByteStrideK = (long)_blockSize * _slotBytesK;
            _layerByteStrideK = (long)_headsKv * _headByteStrideK;
            _blockByteStrideK = (long)_layers * _layerByteStrideK;

            _slotBytesV = KVCache.GetSlotByteSize(_vHeadDim, stridePrecision);
            _headByteStrideV = (long)_blockSize * _slotBytesV;
            _layerByteStrideV = (long)_headsKv * _headByteStrideV;
            _blockByteStrideV = (long)_layers * _layerByteStrideV;
        }

        if (_allocateHostBuffers)
        {
            long totalBytesKLong = checked((long)_totalBlocks * _blockByteStrideK);
            long totalBytesVLong = checked((long)_totalBlocks * _blockByteStrideV);
            if (sizeof(nuint) == 4 && (totalBytesKLong > uint.MaxValue || totalBytesVLong > uint.MaxValue))
            {
                throw new OutOfMemoryException("Requested paged KV cache size exceeds addressable memory limit.");
            }

            nuint totalBytesK = checked((nuint)totalBytesKLong);
            nuint totalBytesV = checked((nuint)totalBytesVLong);

            _kBuffer = (byte*)NativeMemory.AlignedAlloc(totalBytesK, CacheLineAlignment);
            try
            {
                NativeMemory.Clear(_kBuffer, totalBytesK);
                _vBuffer = (byte*)NativeMemory.AlignedAlloc(totalBytesV, CacheLineAlignment);
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

        _blockState = new int[_totalBlocks];
        _refCounts = new int[_totalBlocks];
        _blockPrefixHashes = new ulong[_totalBlocks];
        _blockTokenStorage = new int[checked(_totalBlocks * _blockSize)];
        _blockTokenCounts = new int[_totalBlocks];
        _prefixHashMap = new Dictionary<ulong, int>();
        _freeStack = new int[_totalBlocks];
        _lruPrev = new int[_totalBlocks];
        _lruNext = new int[_totalBlocks];
        Array.Fill(_lruPrev, -1);
        Array.Fill(_lruNext, -1);
        for (int i = 0; i < _totalBlocks; i++)
        {
            // Push in reverse so block 0 is popped first, matching historical LIFO order
            _freeStack[i] = _totalBlocks - 1 - i;
        }
        _freeCount = _totalBlocks;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public byte* GetKeySlotPtr(int physicalBlockId, int layer, int headKv, int offsetInBlock)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!_allocateHostBuffers)
            throw new InvalidOperationException("Host KV buffers were omitted because KV pages reside exclusively in GPU VRAM.");
        long offset = (long)physicalBlockId * _blockByteStrideK +
                      (long)layer * _layerByteStrideK +
                      (long)headKv * _headByteStrideK +
                      (long)offsetInBlock * _slotBytesK;
        return _kBuffer + offset;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public byte* GetValueSlotPtr(int physicalBlockId, int layer, int headKv, int offsetInBlock)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!_allocateHostBuffers)
            throw new InvalidOperationException("Host KV buffers were omitted because KV pages reside exclusively in GPU VRAM.");
        long offset = (long)physicalBlockId * _blockByteStrideV +
                      (long)layer * _layerByteStrideV +
                      (long)headKv * _headByteStrideV +
                      (long)offsetInBlock * _slotBytesV;
        return _vBuffer + offset;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetKeyPtr(int physicalBlockId, int layer, int headKv, int offsetInBlock)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!_allocateHostBuffers)
            throw new InvalidOperationException("Host KV buffers were omitted because KV pages reside exclusively in GPU VRAM.");
        if (_precision != KvCachePrecision.Fp32)
            throw new InvalidOperationException("Direct float* pointer access requires KvCachePrecision.Fp32. Use GetKeySlotPtr or ReadKey for quantized KV pools.");
        return (float*)GetKeySlotPtr(physicalBlockId, layer, headKv, offsetInBlock);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetValuePtr(int physicalBlockId, int layer, int headKv, int offsetInBlock)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!_allocateHostBuffers)
            throw new InvalidOperationException("Host KV buffers were omitted because KV pages reside exclusively in GPU VRAM.");
        if (_precision != KvCachePrecision.Fp32)
            throw new InvalidOperationException("Direct float* pointer access requires KvCachePrecision.Fp32. Use GetValueSlotPtr or ReadValue for quantized KV pools.");
        return (float*)GetValueSlotPtr(physicalBlockId, layer, headKv, offsetInBlock);
    }

    /// <summary>
    /// Stores the projected Key and Value vectors for a specific layer and block offset,
    /// quantizing into the configured <see cref="Precision"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Store(int physicalBlockId, int layer, int offsetInBlock, float* kSrc, float* vSrc)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!_allocateHostBuffers)
            throw new InvalidOperationException("Host KV buffers were omitted because KV pages reside exclusively in GPU VRAM.");

        switch (_precision)
        {
            case KvCachePrecision.Fp32:
                for (int h = 0; h < _headsKv; h++)
                {
                    byte* kDst = GetKeySlotPtr(physicalBlockId, layer, h, offsetInBlock);
                    byte* vDst = GetValueSlotPtr(physicalBlockId, layer, h, offsetInBlock);
                    Buffer.MemoryCopy(kSrc + h * _headDim, kDst, _slotBytesK, _slotBytesK);
                    Buffer.MemoryCopy(vSrc + h * _vHeadDim, vDst, _slotBytesV, _slotBytesV);
                }
                break;

            case KvCachePrecision.Fp16:
                for (int h = 0; h < _headsKv; h++)
                {
                    Half* kDst = (Half*)GetKeySlotPtr(physicalBlockId, layer, h, offsetInBlock);
                    Half* vDst = (Half*)GetValueSlotPtr(physicalBlockId, layer, h, offsetInBlock);
                    QuantKernels.QuantizeF16(kSrc + h * _headDim, kDst, _headDim);
                    QuantKernels.QuantizeF16(vSrc + h * _vHeadDim, vDst, _vHeadDim);
                }
                break;

            case KvCachePrecision.Fp8:
                for (int h = 0; h < _headsKv; h++)
                {
                    byte* kDst = GetKeySlotPtr(physicalBlockId, layer, h, offsetInBlock);
                    byte* vDst = GetValueSlotPtr(physicalBlockId, layer, h, offsetInBlock);
                    QuantKernels.QuantizeFp8E4M3(kSrc + h * _headDim, kDst, _headDim);
                    QuantKernels.QuantizeFp8E4M3(vSrc + h * _vHeadDim, vDst, _vHeadDim);
                }
                break;

            case KvCachePrecision.Q8_0:
                for (int h = 0; h < _headsKv; h++)
                {
                    BlockQ8_0* kDst = (BlockQ8_0*)GetKeySlotPtr(physicalBlockId, layer, h, offsetInBlock);
                    BlockQ8_0* vDst = (BlockQ8_0*)GetValueSlotPtr(physicalBlockId, layer, h, offsetInBlock);
                    QuantKernels.QuantizeQ8_0(kSrc + h * _headDim, kDst, _headDim);
                    QuantKernels.QuantizeQ8_0(vSrc + h * _vHeadDim, vDst, _vHeadDim);
                }
                break;

            case KvCachePrecision.Q4_0:
                for (int h = 0; h < _headsKv; h++)
                {
                    BlockQ4_0* kDst = (BlockQ4_0*)GetKeySlotPtr(physicalBlockId, layer, h, offsetInBlock);
                    BlockQ4_0* vDst = (BlockQ4_0*)GetValueSlotPtr(physicalBlockId, layer, h, offsetInBlock);
                    QuantKernels.QuantizeQ4_0(kSrc + h * _headDim, kDst, _headDim);
                    QuantKernels.QuantizeQ4_0(vSrc + h * _vHeadDim, vDst, _vHeadDim);
                }
                break;

            default:
                throw new NotSupportedException($"Unsupported KV cache precision: {_precision}.");
        }
    }

    /// <summary>
    /// Dequantizes a cached Key head vector at (<paramref name="physicalBlockId"/>, <paramref name="layer"/>,
    /// <paramref name="headKv"/>, <paramref name="offsetInBlock"/>) into <paramref name="dst"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void ReadKey(int physicalBlockId, int layer, int headKv, int offsetInBlock, float* dst)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        byte* src = GetKeySlotPtr(physicalBlockId, layer, headKv, offsetInBlock);
        switch (_precision)
        {
            case KvCachePrecision.Fp32:
                Buffer.MemoryCopy(src, dst, _slotBytesK, _slotBytesK);
                break;
            case KvCachePrecision.Fp16:
                QuantKernels.DequantizeF16((Half*)src, dst, _headDim);
                break;
            case KvCachePrecision.Fp8:
                QuantKernels.DequantizeFp8E4M3(src, dst, _headDim);
                break;
            case KvCachePrecision.Q8_0:
                QuantKernels.DequantizeQ8_0((BlockQ8_0*)src, dst, _headDim);
                break;
            case KvCachePrecision.Q4_0:
                QuantKernels.DequantizeQ4_0((BlockQ4_0*)src, dst, _headDim);
                break;
            default:
                throw new NotSupportedException($"Unsupported KV cache precision: {_precision}.");
        }
    }

    public void ReadKey(int physicalBlockId, int layer, int headKv, int offsetInBlock, Span<float> dst)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (dst.Length < _headDim)
            throw new ArgumentException($"Destination length {dst.Length} is smaller than HeadDim {_headDim}.", nameof(dst));

        fixed (float* pDst = dst)
        {
            ReadKey(physicalBlockId, layer, headKv, offsetInBlock, pDst);
        }
    }

    /// <summary>
    /// Dequantizes a cached Value head vector at (<paramref name="physicalBlockId"/>, <paramref name="layer"/>,
    /// <paramref name="headKv"/>, <paramref name="offsetInBlock"/>) into <paramref name="dst"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void ReadValue(int physicalBlockId, int layer, int headKv, int offsetInBlock, float* dst)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        byte* src = GetValueSlotPtr(physicalBlockId, layer, headKv, offsetInBlock);
        switch (_precision)
        {
            case KvCachePrecision.Fp32:
                Buffer.MemoryCopy(src, dst, _slotBytesV, _slotBytesV);
                break;
            case KvCachePrecision.Fp16:
                QuantKernels.DequantizeF16((Half*)src, dst, _vHeadDim);
                break;
            case KvCachePrecision.Fp8:
                QuantKernels.DequantizeFp8E4M3(src, dst, _vHeadDim);
                break;
            case KvCachePrecision.Q8_0:
                QuantKernels.DequantizeQ8_0((BlockQ8_0*)src, dst, _vHeadDim);
                break;
            case KvCachePrecision.Q4_0:
                QuantKernels.DequantizeQ4_0((BlockQ4_0*)src, dst, _vHeadDim);
                break;
            default:
                throw new NotSupportedException($"Unsupported KV cache precision: {_precision}.");
        }
    }

    public void ReadValue(int physicalBlockId, int layer, int headKv, int offsetInBlock, Span<float> dst)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (dst.Length < _vHeadDim)
            throw new ArgumentException($"Destination length {dst.Length} is smaller than ValueHeadDim {_vHeadDim}.", nameof(dst));

        fixed (float* pDst = dst)
        {
            ReadValue(physicalBlockId, layer, headKv, offsetInBlock, pDst);
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void LruAppendTail(int blockId)
    {
        int prevTail = _lruTail;
        _lruPrev[blockId] = prevTail;
        _lruNext[blockId] = -1;
        if (prevTail >= 0)
            _lruNext[prevTail] = blockId;
        else
            _lruHead = blockId;
        _lruTail = blockId;
        _lruCount++;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void LruRemove(int blockId)
    {
        int prev = _lruPrev[blockId];
        int next = _lruNext[blockId];
        if (prev >= 0)
            _lruNext[prev] = next;
        else
            _lruHead = next;

        if (next >= 0)
            _lruPrev[next] = prev;
        else
            _lruTail = prev;

        _lruPrev[blockId] = -1;
        _lruNext[blockId] = -1;
        _lruCount--;
    }

    /// <summary>
    /// Attempts to allocate a physical block in O(1), evicting the least-recently-used
    /// unreferenced cached prefix block when unallocated free blocks are exhausted.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryAllocateBlock(out int physicalBlockId)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        lock (_allocLock)
        {
            if (_freeCount > 0)
            {
                physicalBlockId = _freeStack[--_freeCount];
                _blockState[physicalBlockId] = StateActive;
                Volatile.Write(ref _refCounts[physicalBlockId], 1);
                _blockPrefixHashes[physicalBlockId] = 0UL;
                _blockTokenCounts[physicalBlockId] = 0;
                return true;
            }

            if (_lruHead >= 0)
            {
                physicalBlockId = _lruHead;
                LruRemove(physicalBlockId);
                ulong oldHash = _blockPrefixHashes[physicalBlockId];
                if (oldHash != 0UL && _prefixHashMap.TryGetValue(oldHash, out int mapped) && mapped == physicalBlockId)
                {
                    _prefixHashMap.Remove(oldHash);
                }
                _blockPrefixHashes[physicalBlockId] = 0UL;
                _blockTokenCounts[physicalBlockId] = 0;
                _blockState[physicalBlockId] = StateActive;
                Volatile.Write(ref _refCounts[physicalBlockId], 1);
                return true;
            }
        }

        physicalBlockId = -1;
        return false;
    }

    /// <summary>
    /// Atomically searches for a cached block matching <paramref name="prefixHash"/>,
    /// resurrecting it from the LRU cache if unreferenced and incrementing its ref-count.
    /// </summary>
    public bool TryFindPrefixBlock(ulong prefixHash, out int physicalBlockId)
        => TryFindPrefixBlockCore(prefixHash, ReadOnlySpan<int>.Empty, verifyTokens: false, out physicalBlockId);

    /// <summary>
    /// Atomically searches for a cached block matching both <paramref name="prefixHash"/> and
    /// exact <paramref name="tokens"/> sequence (rejecting 64-bit hash collisions),
    /// resurrecting it from the LRU cache if unreferenced and incrementing its ref-count.
    /// </summary>
    public bool TryFindPrefixBlock(ulong prefixHash, ReadOnlySpan<int> tokens, out int physicalBlockId)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (tokens.IsEmpty || tokens.Length > _blockSize)
        {
            physicalBlockId = -1;
            return false;
        }
        return TryFindPrefixBlockCore(prefixHash, tokens, verifyTokens: true, out physicalBlockId);
    }

    private bool TryFindPrefixBlockCore(
        ulong prefixHash,
        ReadOnlySpan<int> tokens,
        bool verifyTokens,
        out int physicalBlockId)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (prefixHash == 0UL)
        {
            physicalBlockId = -1;
            return false;
        }

        lock (_allocLock)
        {
            if (!_prefixHashMap.TryGetValue(prefixHash, out int id))
            {
                physicalBlockId = -1;
                return false;
            }

            if (_blockPrefixHashes[id] != prefixHash)
            {
                physicalBlockId = -1;
                return false;
            }

            if (verifyTokens)
            {
                int storedCount = _blockTokenCounts[id];
                if (storedCount > 0 &&
                    (storedCount != tokens.Length ||
                     !_blockTokenStorage.AsSpan(id * _blockSize, storedCount).SequenceEqual(tokens)))
                {
                    physicalBlockId = -1;
                    return false;
                }
            }

            int state = _blockState[id];
            if (state == StateActive && _refCounts[id] > 0)
            {
                Volatile.Write(ref _refCounts[id], _refCounts[id] + 1);
                physicalBlockId = id;
                return true;
            }

            if (state == StateCached)
            {
                LruRemove(id);
                _blockState[id] = StateActive;
                Volatile.Write(ref _refCounts[id], 1);
                physicalBlockId = id;
                return true;
            }
        }

        physicalBlockId = -1;
        return false;
    }

    /// <summary>
    /// Registers a full block's cumulative prefix hash in the cache.
    /// </summary>
    public void BindPrefixBlock(ulong prefixHash, int physicalBlockId)
        => BindPrefixBlockCore(prefixHash, physicalBlockId, ReadOnlySpan<int>.Empty, hasTokens: false);

    /// <summary>
    /// Registers a full block's cumulative prefix hash and exact token sequence in the cache.
    /// </summary>
    public void BindPrefixBlock(ulong prefixHash, int physicalBlockId, ReadOnlySpan<int> tokens)
    {
        if (tokens.IsEmpty || tokens.Length > _blockSize)
            throw new ArgumentOutOfRangeException(nameof(tokens), $"Token count must be between 1 and BlockSize ({_blockSize}).");

        BindPrefixBlockCore(prefixHash, physicalBlockId, tokens, hasTokens: true);
    }

    private void BindPrefixBlockCore(ulong prefixHash, int physicalBlockId, ReadOnlySpan<int> tokens, bool hasTokens)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if ((uint)physicalBlockId >= (uint)_totalBlocks)
            throw new ArgumentOutOfRangeException(nameof(physicalBlockId));
        if (prefixHash == 0UL)
            throw new ArgumentException("Prefix hash cannot be 0.", nameof(prefixHash));

        lock (_allocLock)
        {
            ulong prevHash = _blockPrefixHashes[physicalBlockId];
            if (prevHash != 0UL && prevHash != prefixHash &&
                _prefixHashMap.TryGetValue(prevHash, out int prevMapped) && prevMapped == physicalBlockId)
            {
                _prefixHashMap.Remove(prevHash);
            }

            _blockPrefixHashes[physicalBlockId] = prefixHash;
            if (hasTokens)
            {
                tokens.CopyTo(_blockTokenStorage.AsSpan(physicalBlockId * _blockSize, tokens.Length));
                _blockTokenCounts[physicalBlockId] = tokens.Length;
            }
            else
            {
                _blockTokenCounts[physicalBlockId] = 0;
            }
            _prefixHashMap[prefixHash] = physicalBlockId;
        }
    }

    /// <summary>
    /// Evicts all unreferenced (<c>refCount == 0</c>) prefix blocks from the LRU cache back to the free stack.
    /// </summary>
    public int EvictPrefixCache()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        lock (_allocLock)
        {
            int evicted = 0;
            while (_lruHead >= 0)
            {
                int id = _lruHead;
                LruRemove(id);
                ulong hash = _blockPrefixHashes[id];
                if (hash != 0UL && _prefixHashMap.TryGetValue(hash, out int mapped) && mapped == id)
                {
                    _prefixHashMap.Remove(hash);
                }
                _blockPrefixHashes[id] = 0UL;
                _blockTokenCounts[id] = 0;
                _blockState[id] = StateFree;
                _freeStack[_freeCount++] = id;
                evicted++;
            }
            return evicted;
        }
    }

    /// <summary>
    /// Increments atomic reference count for physical block.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int RetainBlock(int physicalBlockId)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if ((uint)physicalBlockId >= (uint)_totalBlocks)
            throw new ArgumentOutOfRangeException(nameof(physicalBlockId));

        lock (_allocLock)
        {
            if (_blockState[physicalBlockId] == StateCached)
            {
                LruRemove(physicalBlockId);
                _blockState[physicalBlockId] = StateActive;
            }
            int next = _refCounts[physicalBlockId] + 1;
            Volatile.Write(ref _refCounts[physicalBlockId], next);
            return next;
        }
    }

    /// <summary>
    /// Decrements reference count. When ref-count reaches 0, retains bound prefix blocks in the
    /// LRU cache (evictable under allocation pressure) or returns unpinned blocks directly to the free stack.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int ReleaseBlock(int physicalBlockId)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if ((uint)physicalBlockId >= (uint)_totalBlocks)
            return 0;

        lock (_allocLock)
        {
            if (_blockState[physicalBlockId] != StateActive || _refCounts[physicalBlockId] <= 0)
                return 0;

            int remaining = _refCounts[physicalBlockId] - 1;
            Volatile.Write(ref _refCounts[physicalBlockId], remaining);

            if (remaining == 0)
            {
                ulong hash = _blockPrefixHashes[physicalBlockId];
                if (hash != 0UL && _prefixHashMap.TryGetValue(hash, out int mappedId) && mappedId == physicalBlockId)
                {
                    _blockState[physicalBlockId] = StateCached;
                    LruAppendTail(physicalBlockId);
                }
                else
                {
                    _blockPrefixHashes[physicalBlockId] = 0UL;
                    _blockTokenCounts[physicalBlockId] = 0;
                    _blockState[physicalBlockId] = StateFree;
                    _freeStack[_freeCount++] = physicalBlockId;
                }
            }

            return remaining;
        }
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
    /// Duplicates raw quantized/float KV slot bytes up to validTokens from srcBlockId to dstBlockId for Copy-on-Write.
    /// </summary>
    public void CopyBlockData(int srcBlockId, int dstBlockId, int validTokens)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!_allocateHostBuffers)
            throw new InvalidOperationException("Host KV buffers were omitted.");
        if (validTokens <= 0) return;
        int count = Math.Min(validTokens, _blockSize);
        long copyBytesK = (long)count * _slotBytesK;
        long copyBytesV = (long)count * _slotBytesV;

        for (int l = 0; l < _layers; l++)
        {
            for (int h = 0; h < _headsKv; h++)
            {
                byte* srcK = GetKeySlotPtr(srcBlockId, l, h, 0);
                byte* dstK = GetKeySlotPtr(dstBlockId, l, h, 0);
                Buffer.MemoryCopy(srcK, dstK, copyBytesK, copyBytesK);

                byte* srcV = GetValueSlotPtr(srcBlockId, l, h, 0);
                byte* dstV = GetValueSlotPtr(dstBlockId, l, h, 0);
                Buffer.MemoryCopy(srcV, dstV, copyBytesV, copyBytesV);
            }
        }
    }

    /// <summary>
    /// Allocates a new block and copies valid token KV data for Copy-On-Write isolation.
    /// </summary>
    public bool TryForkBlock(int srcBlockId, int validTokens, out int forkedBlockId)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!TryAllocateBlock(out forkedBlockId))
            return false;

        if (_allocateHostBuffers)
        {
            CopyBlockData(srcBlockId, forkedBlockId, validTokens);
        }
        else if (validTokens > 0)
        {
            OnBlockForked?.Invoke(srcBlockId, forkedBlockId, validTokens);
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
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            if (_kBuffer != null)
            {
                NativeMemory.AlignedFree(_kBuffer);
                _kBuffer = null;
            }
            if (_vBuffer != null)
            {
                NativeMemory.AlignedFree(_vBuffer);
                _vBuffer = null;
            }
        }
        GC.SuppressFinalize(this);
    }

    ~PagedBlockPool()
    {
        Dispose();
    }
}

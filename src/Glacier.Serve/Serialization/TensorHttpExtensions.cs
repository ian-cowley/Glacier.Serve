namespace Glacier.Serve.Serialization;

using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Glacier.Serve.Core;
using Glacier.Tensor.Core;

public static class TensorHttpExtensions
{
    private const int StreamingChunkFloats = 16384; // 64 KB chunks

    /// <summary>
    /// Streams a <see cref="Tensor{T}"/> of <see cref="float"/> directly into <see cref="HttpResponse.BodyWriter"/>
    /// in bounded 64 KB chunks using 64-bit <c>Content-Length</c> arithmetic (eliminating <c>int</c> overflow
    /// and large contiguous <c>byte[]</c> heap allocations).
    /// </summary>
    public static async ValueTask WriteBinaryTensorAsync(this HttpResponse response, Tensor<float> tensor)
    {
        ArgumentNullException.ThrowIfNull(response);
        ArgumentNullException.ThrowIfNull(tensor);

        response.ContentType = "application/octet-stream";

        int rank = tensor.Rank;
        int headerSize = checked((1 + rank) * sizeof(int));
        long elementCount = tensor.ElementCount;
        long dataBytes = checked(elementCount * sizeof(float));
        long totalContentLength = checked((long)headerSize + dataBytes);

        // Materialize non-contiguous views BEFORE sending HTTP 200 OK headers so no response is truncated mid-stream
        Tensor<float>? contiguousOwner = tensor.IsContiguous ? null : tensor.Contiguous();
        try
        {
            Tensor<float> src = contiguousOwner ?? tensor;

            await response.EnsureHeadersSentAsync(totalContentLength);

            // 1. Write rank + shape header directly to PipeWriter
            Span<byte> headerBytesSpan = response.BodyWriter.GetSpan(headerSize);
            Span<int> headerInts = MemoryMarshal.Cast<byte, int>(headerBytesSpan[..headerSize]);
            headerInts[0] = rank;
            for (int i = 0; i < rank; i++)
            {
                headerInts[i + 1] = tensor.Shape[i];
            }
            response.BodyWriter.Advance(headerSize);

            // 2. Stream raw float payload in bounded chunks using 64-bit pointer offsets (no int overflow > 2B elements)
            long offset = 0;
            while (offset < elementCount)
            {
                int count = (int)Math.Min(StreamingChunkFloats, elementCount - offset);
                int chunkByteLen = count * sizeof(float);
                CopyContiguousChunk(src, offset, count, response.BodyWriter.GetSpan(chunkByteLen));
                response.BodyWriter.Advance(chunkByteLen);
                offset += count;

                if (offset < elementCount)
                {
                    await response.BodyWriter.FlushAsync();
                }
            }

            await response.BodyWriter.FlushAsync();
        }
        finally
        {
            contiguousOwner?.Dispose();
        }
    }

    private static unsafe void CopyContiguousChunk(Tensor<float> contiguousTensor, long offset, int count, Span<byte> dstBytes)
    {
        ReadOnlySpan<float> chunkSpan = new ReadOnlySpan<float>(contiguousTensor.DataPointer + offset, count);
        MemoryMarshal.AsBytes(chunkSpan).CopyTo(dstBytes);
    }

    public static ValueTask WriteJsonTensorAsync(this HttpResponse response, Tensor<float> tensor)
    {
        ArgumentNullException.ThrowIfNull(response);
        ArgumentNullException.ThrowIfNull(tensor);

        Tensor<float>? contiguousOwner = tensor.IsContiguous ? null : tensor.Contiguous();
        float[] data;
        try
        {
            Tensor<float> src = contiguousOwner ?? tensor;
            data = src.AsSpan().ToArray();
        }
        finally
        {
            contiguousOwner?.Dispose();
        }

        int[] shape = new int[tensor.Rank];
        for (int i = 0; i < tensor.Rank; i++) shape[i] = tensor.Shape[i];

        var dto = new TensorDto
        {
            Rank = tensor.Rank,
            Shape = shape,
            Length = checked((int)tensor.ElementCount),
            Data = data
        };

        return response.WriteJsonAsync(dto, ServeJsonContext.Default.TensorDto);
    }
}

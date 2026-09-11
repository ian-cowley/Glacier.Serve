namespace Glacier.Serve.Serialization;

using System;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Threading.Tasks;
using Glacier.Serve.Core;
using Glacier.Tensor.Core;

public static class TensorHttpExtensions
{
    public static async ValueTask WriteBinaryTensorAsync(this HttpResponse response, Tensor<float> tensor)
    {
        response.ContentType = "application/octet-stream";

        int rank = tensor.Rank;
        int headerSize = (1 + rank) * sizeof(int);
        int dataSize = (int)tensor.ElementCount * sizeof(float);
        byte[] payload = new byte[headerSize + dataSize];

        var intSpan = MemoryMarshal.Cast<byte, int>(payload.AsSpan(0, headerSize));
        intSpan[0] = rank;
        for (int i = 0; i < rank; i++) intSpan[i + 1] = tensor.Shape[i];

        var floatSpan = MemoryMarshal.Cast<byte, float>(payload.AsSpan(headerSize));
        tensor.AsSpan().CopyTo(floatSpan);

        await response.EnsureHeadersSentAsync(payload.Length);
        await response.BodyWriter.WriteAsync(payload);
        await response.BodyWriter.FlushAsync();
    }

    public static async ValueTask WriteJsonTensorAsync(this HttpResponse response, Tensor<float> tensor)
    {
        response.ContentType = "application/json; charset=utf-8";

        int[] shape = new int[tensor.Rank];
        for (int i = 0; i < tensor.Rank; i++) shape[i] = tensor.Shape[i];

        var dto = new
        {
            rank = tensor.Rank,
            shape = shape,
            length = (int)tensor.ElementCount,
            data = tensor.AsSpan().ToArray()
        };

        byte[] json = JsonSerializer.SerializeToUtf8Bytes(dto);
        await response.EnsureHeadersSentAsync(json.Length);
        await response.BodyWriter.WriteAsync(json);
        await response.BodyWriter.FlushAsync();
    }
}

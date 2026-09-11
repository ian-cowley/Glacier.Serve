namespace Glacier.Serve.Tests;

using System;
using System.Buffers;
using System.IO.Pipelines;
using System.Text;
using System.Threading.Tasks;
using Glacier.Polaris;
using Glacier.Polaris.Data;
using Glacier.Serve.Core;
using Glacier.Serve.Serialization;
using Glacier.Tensor.Core;
using Xunit;

public class SerializationTests
{
    private static int FindBodyOffset(ReadOnlySpan<byte> responseBytes)
    {
        int idx = responseBytes.IndexOf("\r\n\r\n"u8);
        return idx >= 0 ? idx + 4 : 0;
    }

    [Fact]
    public async Task DataFrame_SerializesToJsonAsync()
    {
        var s1 = new Float32Series("Temperature", 3);
        new float[] { 21.5f, 22.0f, 21.8f }.CopyTo(s1.Memory.Span);

        var s2 = new Int32Series("SensorId", 3);
        new int[] { 101, 102, 103 }.CopyTo(s2.Memory.Span);

        var df = new DataFrame([s1, s2]);

        var pipe = new Pipe();
        var response = new HttpResponse(pipe.Writer);

        await response.WriteDataFrameJsonAsync(df);

        var readResult = await pipe.Reader.ReadAsync();
        byte[] bytes = BuffersExtensions.ToArray(readResult.Buffer);

        int bodyOffset = FindBodyOffset(bytes);
        string json = Encoding.UTF8.GetString(bytes.AsSpan(bodyOffset));

        Assert.Contains("Temperature", json);
        Assert.Contains("SensorId", json);
        Assert.Contains("rowCount", json);
    }

    [Fact]
    public async Task DataFrame_SerializesToArrowIpcAsync()
    {
        var s1 = new Int32Series("Value", 2);
        new int[] { 100, 200 }.CopyTo(s1.Memory.Span);
        var df = new DataFrame([s1]);

        var pipe = new Pipe();
        var response = new HttpResponse(pipe.Writer);

        await response.WriteArrowIpcAsync(df);

        var readResult = await pipe.Reader.ReadAsync();
        byte[] bytes = BuffersExtensions.ToArray(readResult.Buffer);

        Assert.NotEmpty(bytes);
        Assert.Equal("application/vnd.apache.arrow.stream", response.ContentType);
    }

    [Fact]
    public async Task Tensor_SerializesToBinaryAsync()
    {
        using var tensor = Tensor<float>.FromSpan([10f, 20f, 30f, 40f, 50f, 60f], [2, 3]);

        var pipe = new Pipe();
        var response = new HttpResponse(pipe.Writer);

        await response.WriteBinaryTensorAsync(tensor);

        var readResult = await pipe.Reader.ReadAsync();
        byte[] bytes = BuffersExtensions.ToArray(readResult.Buffer);

        Assert.NotEmpty(bytes);
        Assert.Equal("application/octet-stream", response.ContentType);

        int bodyOffset = FindBodyOffset(bytes);
        byte[] body = bytes[bodyOffset..];

        // Verify Rank header = 2
        int rank = BitConverter.ToInt32(body, 0);
        int dim0 = BitConverter.ToInt32(body, 4);
        int dim1 = BitConverter.ToInt32(body, 8);

        Assert.Equal(2, rank);
        Assert.Equal(2, dim0);
        Assert.Equal(3, dim1);
    }
}

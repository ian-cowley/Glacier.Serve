namespace Glacier.Serve.Tests;

using System;
using System.Buffers;
using System.IO;
using System.IO.Pipelines;
using System.Text;
using System.Threading.Tasks;
using Glacier.Polaris;
using Glacier.Polaris.Data;
using Glacier.Serve.Core;
using Glacier.Serve.Serialization;
using Glacier.Storage.Arrow;
using Glacier.Tensor.Core;
using Xunit;

public class SerializationTests
{
    private static int FindBodyOffset(ReadOnlySpan<byte> responseBytes)
    {
        int idx = responseBytes.IndexOf("\r\n\r\n"u8);
        return idx >= 0 ? idx + 4 : 0;
    }

    private static async Task<byte[]> ReadPipeToEndAsync(PipeReader reader)
    {
        using var ms = new MemoryStream();
        while (true)
        {
            ReadResult res = await reader.ReadAsync();
            foreach (ReadOnlyMemory<byte> seg in res.Buffer)
            {
                ms.Write(seg.Span);
            }
            reader.AdvanceTo(res.Buffer.End);
            if (res.IsCompleted) break;
        }
        await reader.CompleteAsync();
        return ms.ToArray();
    }

    private static async Task<byte[]> DrainPipeAsync(Pipe pipe)
    {
        await pipe.Writer.CompleteAsync();
        return await ReadPipeToEndAsync(pipe.Reader);
    }

    [Fact]
    public async Task DataFrame_SerializesToJsonAsync()
    {
        using var s1 = Float32Series.FromValues("Temperature", [21.5f, null, 21.8f]);
        using var s2 = Int32Series.FromValues("SensorId", [101, 102, 103]);
        using var df = new DataFrame([s1, s2]);

        var pipe = new Pipe();
        var response = new HttpResponse(pipe.Writer);

        await response.WriteDataFrameJsonAsync(df);

        byte[] bytes = await DrainPipeAsync(pipe);
        int bodyOffset = FindBodyOffset(bytes);
        string json = Encoding.UTF8.GetString(bytes.AsSpan(bodyOffset));

        Assert.Contains("Temperature", json);
        Assert.Contains("SensorId", json);
        Assert.Contains("rowCount", json);
        Assert.Contains("null", json);
    }

    [Fact]
    public async Task DataFrame_ArrowIpc_FullRoundTrip_AllTypesNullBitmapsAndEos()
    {
        using var colInt32 = Int32Series.FromValues("Ids", [10, null, 30, 40]);
        using var colInt64 = Int64Series.FromValues("Timestamps", [10000000000L, 20000000000L, null, 40000000000L]);
        using var colFloat32 = Float32Series.FromValues("Scores", [1.25f, 2.5f, 3.75f, null]);
        using var colFloat64 = Float64Series.FromValues("Latencies", [null, 0.123456789, 1.987654321, 42.5]);
        using var colUtf8 = new Utf8StringSeries("Labels", ["alpha", "", "glacier🚀", "omega"]);
        colUtf8.ValidityMask.SetNull(1);
        using var colBool = BooleanSeries.FromValues("Flags", [true, false, null, true]);

        using var df = new DataFrame([colInt32, colInt64, colFloat32, colFloat64, colUtf8, colBool]);

        var pipe = new Pipe();
        var response = new HttpResponse(pipe.Writer);

        await response.WriteArrowIpcAsync(df);

        byte[] fullResponse = await DrainPipeAsync(pipe);
        Assert.Equal("application/vnd.apache.arrow.stream", response.ContentType);

        int bodyOffset = FindBodyOffset(fullResponse);
        byte[] ipcPayload = fullResponse[bodyOffset..];

        using var stream = new MemoryStream(ipcPayload);
        using var reader = new ArrowStreamReader(stream);

        ArrowSchema schema = reader.ReadSchema();
        Assert.Equal(6, schema.FieldCount);
        Assert.Equal("Ids", schema.GetField(0).Name);
        Assert.Equal(ArrowTypeId.Int32, schema.GetField(0).DataType.Id);
        Assert.Equal(ArrowTypeId.Int64, schema.GetField(1).DataType.Id);
        Assert.Equal(ArrowTypeId.Float, schema.GetField(2).DataType.Id);
        Assert.Equal(ArrowTypeId.Double, schema.GetField(3).DataType.Id);
        Assert.Equal(ArrowTypeId.Utf8, schema.GetField(4).DataType.Id);
        Assert.Equal(ArrowTypeId.Boolean, schema.GetField(5).DataType.Id);

        ArrowRecordBatch? batch = reader.ReadNextRecordBatch();
        Assert.NotNull(batch);
        Assert.Equal(4, batch!.RowCount);

        // Column 0: Int32 + Null
        var c0 = batch.Column(0);
        Assert.Equal(1, c0.NullCount);
        Assert.False(c0.IsNull(0));
        Assert.Equal(10, c0.GetInt32(0));
        Assert.True(c0.IsNull(1));
        Assert.Equal(30, c0.GetInt32(2));
        Assert.Equal(40, c0.GetInt32(3));

        // Column 1: Int64 + Null
        var c1 = batch.Column(1);
        Assert.Equal(1, c1.NullCount);
        Assert.Equal(10000000000L, c1.GetInt64(0));
        Assert.Equal(20000000000L, c1.GetInt64(1));
        Assert.True(c1.IsNull(2));
        Assert.Equal(40000000000L, c1.GetInt64(3));

        // Column 2: Float32 + Null
        var c2 = batch.Column(2);
        Assert.Equal(1, c2.NullCount);
        Assert.Equal(1.25f, c2.GetFloat(0));
        Assert.Equal(2.5f, c2.GetFloat(1));
        Assert.Equal(3.75f, c2.GetFloat(2));
        Assert.True(c2.IsNull(3));

        // Column 3: Float64 + Null
        var c3 = batch.Column(3);
        Assert.Equal(1, c3.NullCount);
        Assert.True(c3.IsNull(0));
        Assert.Equal(0.123456789, c3.GetDouble(1), precision: 9);
        Assert.Equal(1.987654321, c3.GetDouble(2), precision: 9);
        Assert.Equal(42.5, c3.GetDouble(3), precision: 9);

        // Column 4: Utf8 variable-width offsets + Null
        var c4 = batch.Column(4);
        Assert.Equal(1, c4.NullCount);
        Assert.Equal("alpha", c4.GetString(0));
        Assert.True(c4.IsNull(1));
        Assert.Equal("glacier🚀", c4.GetString(2));
        Assert.Equal("omega", c4.GetString(3));

        // Column 5: Bit-packed Boolean + Null
        var c5 = batch.Column(5);
        Assert.Equal(1, c5.NullCount);
        Assert.True(c5.GetBoolean(0));
        Assert.False(c5.GetBoolean(1));
        Assert.True(c5.IsNull(2));
        Assert.True(c5.GetBoolean(3));

        // Verify clean IPC End-Of-Stream marker (next batch returns null without throwing)
        Assert.Null(reader.ReadNextRecordBatch());
    }

    [Fact]
    public async Task DataFrame_SerializesToGlacierStorageArrowIpcAsync()
    {
        using var s1 = new Int32Series("Value", [100, 200]);
        using var df = new DataFrame([s1]);

        var pipe = new Pipe();
        var response = new HttpResponse(pipe.Writer);

        await response.WriteGlacierStorageArrowIpcAsync(df);

        byte[] bytes = await DrainPipeAsync(pipe);
        Assert.NotEmpty(bytes);
        Assert.Equal("application/vnd.apache.arrow.stream", response.ContentType);
    }

    [Fact]
    public async Task Tensor_SerializesToBinaryAsync_MultiChunkRoundTrip()
    {
        // 40,000 floats = 160,000 bytes (> 2.4x the 64 KB chunk size)
        const int rows = 200;
        const int cols = 200;
        int totalElements = rows * cols;
        float[] expected = new float[totalElements];
        for (int i = 0; i < totalElements; i++)
        {
            expected[i] = (i * 0.125f) - 500.0f;
        }

        using var tensor = Tensor<float>.FromSpan(expected, [rows, cols]);

        var pipe = new Pipe();
        var response = new HttpResponse(pipe.Writer);

        // Drain reader concurrently so >64 KB multi-chunk flushes exercise real Pipe backpressure without deadlocking
        Task<byte[]> readTask = ReadPipeToEndAsync(pipe.Reader);
        await response.WriteBinaryTensorAsync(tensor);
        await pipe.Writer.CompleteAsync();

        byte[] fullBytes = await readTask;
        Assert.Equal("application/octet-stream", response.ContentType);

        int bodyOffset = FindBodyOffset(fullBytes);
        string headersText = Encoding.ASCII.GetString(fullBytes, 0, bodyOffset);
        long expectedBodyBytes = (1 + 2) * sizeof(int) + (long)totalElements * sizeof(float);
        Assert.Contains($"Content-Length: {expectedBodyBytes}\r\n", headersText);

        byte[] body = fullBytes[bodyOffset..];
        Assert.Equal(expectedBodyBytes, body.Length);

        int rank = BitConverter.ToInt32(body, 0);
        int dim0 = BitConverter.ToInt32(body, 4);
        int dim1 = BitConverter.ToInt32(body, 8);

        Assert.Equal(2, rank);
        Assert.Equal(rows, dim0);
        Assert.Equal(cols, dim1);

        for (int i = 0; i < totalElements; i++)
        {
            float actual = BitConverter.ToSingle(body, 12 + i * sizeof(float));
            Assert.Equal(expected[i], actual);
        }
    }
}

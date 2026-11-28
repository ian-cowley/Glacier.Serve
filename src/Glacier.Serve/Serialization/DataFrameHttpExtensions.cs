namespace Glacier.Serve.Serialization;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.IO.Pipelines;
using System.Text.Json;
using System.Threading.Tasks;
using Glacier.Formats.Arrow;
using Glacier.Polaris;
using Glacier.Polaris.Data;
using Glacier.Serve.Core;

public static class DataFrameHttpExtensions
{
    private sealed class CountingStream : Stream
    {
        private long _length;

        public override bool CanRead => false;
        public override bool CanSeek => false;
        public override bool CanWrite => true;
        public override long Length => _length;
        public override long Position { get => _length; set => throw new NotSupportedException(); }

        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => _length += count;
        public override void Write(ReadOnlySpan<byte> buffer) => _length += buffer.Length;
        public override void WriteByte(byte value) => _length++;
    }

    private sealed class PipeWriterSyncStream : Stream
    {
        private const int FlushThresholdBytes = 65536;
        private readonly PipeWriter _writer;
        private long _written;
        private int _unflushed;

        public PipeWriterSyncStream(PipeWriter writer)
        {
            _writer = writer;
        }

        public override bool CanRead => false;
        public override bool CanSeek => false;
        public override bool CanWrite => true;
        public override long Length => _written;
        public override long Position { get => _written; set => throw new NotSupportedException(); }

        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();

        public override void Write(byte[] buffer, int offset, int count)
            => Write(buffer.AsSpan(offset, count));

        public override void Write(ReadOnlySpan<byte> buffer)
        {
            while (!buffer.IsEmpty)
            {
                int chunk = Math.Min(buffer.Length, FlushThresholdBytes);
                Span<byte> dest = _writer.GetSpan(chunk);
                int copyLen = Math.Min(chunk, dest.Length);
                buffer[..copyLen].CopyTo(dest);
                _writer.Advance(copyLen);
                _written += copyLen;
                _unflushed += copyLen;
                buffer = buffer[copyLen..];

                if (_unflushed >= FlushThresholdBytes)
                {
                    ValueTask<FlushResult> flushTask = _writer.FlushAsync();
                    if (!flushTask.IsCompletedSuccessfully)
                    {
                        flushTask.AsTask().GetAwaiter().GetResult();
                    }
                    else
                    {
                        _ = flushTask.Result;
                    }
                    _unflushed = 0;
                }
            }
        }

        public override void WriteByte(byte value)
        {
            Span<byte> dest = _writer.GetSpan(1);
            dest[0] = value;
            _writer.Advance(1);
            _written++;
            _unflushed++;
            if (_unflushed >= FlushThresholdBytes)
            {
                ValueTask<FlushResult> flushTask = _writer.FlushAsync();
                if (!flushTask.IsCompletedSuccessfully)
                {
                    flushTask.AsTask().GetAwaiter().GetResult();
                }
                else
                {
                    _ = flushTask.Result;
                }
                _unflushed = 0;
            }
        }
    }

    /// <summary>
    /// Streams a Glacier.Polaris DataFrame as Apache Arrow IPC stream format directly to the HTTP response
    /// using Glacier.Formats managed streaming.
    /// </summary>
    /// <remarks>
    /// Streams in 64 KB increments to <see cref="HttpResponse.BodyWriter"/>; callers using an in-memory
    /// <see cref="Pipe"/> for payloads exceeding <c>PauseWriterThreshold</c> (64 KB) must drain
    /// <see cref="PipeReader"/> concurrently, matching <see cref="HttpResponse.WriteBytesAsync"/>.
    /// </remarks>
    public static ValueTask WriteArrowIpcAsync(this HttpResponse response, DataFrame df)
        => WriteGlacierStorageArrowIpcAsync(response, df);

    /// <summary>
    /// Streams a Glacier.Polaris DataFrame using first-party Glacier.Formats pure managed ArrowStreamWriter,
    /// preserving full validity bitmaps, variable-width UTF-8/Binary offsets, bit-packed booleans, and IPC EOS markers
    /// without materializing an intermediate MemoryStream payload.
    /// </summary>
    /// <remarks>
    /// Flushes incrementally every 64 KB to <see cref="HttpResponse.BodyWriter"/>; callers using an in-memory
    /// <see cref="Pipe"/> for payloads exceeding <c>PauseWriterThreshold</c> (64 KB) must drain
    /// <see cref="PipeReader"/> concurrently, matching <see cref="TensorHttpExtensions.WriteBinaryTensorAsync"/>.
    /// </remarks>
    public static async ValueTask WriteGlacierStorageArrowIpcAsync(this HttpResponse response, DataFrame df)
    {
        ArgumentNullException.ThrowIfNull(response);
        ArgumentNullException.ThrowIfNull(df);

        response.ContentType = "application/vnd.apache.arrow.stream";

        int colCount = df.Columns.Count;
        var columns = new List<ArrowColumn>(colCount);
        var fields = new List<ArrowField>(colCount);

        for (int i = 0; i < colCount; i++)
        {
            var arrowCol = df.Columns[i].ToArrowColumn();
            columns.Add(arrowCol);
            fields.Add(arrowCol.Field);
        }

        var schema = new ArrowSchema(fields);
        var batch = new ArrowRecordBatch(schema, df.RowCount, columns);

        long contentLength;
        using (var counter = new CountingStream())
        using (var countWriter = new ArrowStreamWriter(counter, leaveOpen: true))
        {
            countWriter.WriteSchema(schema);
            countWriter.WriteRecordBatch(batch);
            countWriter.WriteEndOfStream();
            contentLength = counter.Length;
        }

        await response.EnsureHeadersSentAsync(contentLength);

        using (var pipeStream = new PipeWriterSyncStream(response.BodyWriter))
        using (var ipcWriter = new ArrowStreamWriter(pipeStream, leaveOpen: true))
        {
            ipcWriter.WriteSchema(schema);
            ipcWriter.WriteRecordBatch(batch);
            ipcWriter.WriteEndOfStream();
        }

        await response.BodyWriter.FlushAsync();
    }

    /// <summary>
    /// Serializes a DataFrame into high-performance JSON format using typed span accessors and pooled Utf8JsonWriter buffers.
    /// </summary>
    public static async ValueTask WriteDataFrameJsonAsync(this HttpResponse response, DataFrame df)
    {
        ArgumentNullException.ThrowIfNull(response);
        ArgumentNullException.ThrowIfNull(df);

        response.ContentType = "application/json; charset=utf-8";

        var bufferWriter = HttpResponse.RentJsonBuffer();
        try
        {
            using (var jsonWriter = new Utf8JsonWriter(bufferWriter))
            {
                jsonWriter.WriteStartObject();
                jsonWriter.WriteNumber("rowCount", df.RowCount);
                jsonWriter.WriteStartArray("columns");

                foreach (var col in df.Columns)
                {
                    jsonWriter.WriteStartObject();
                    jsonWriter.WriteString("name", col.Name);
                    jsonWriter.WriteString("type", col.DataType.Name);
                    jsonWriter.WriteStartArray("data");

                    WriteSeriesJsonValues(jsonWriter, col);

                    jsonWriter.WriteEndArray();
                    jsonWriter.WriteEndObject();
                }

                jsonWriter.WriteEndArray();
                jsonWriter.WriteEndObject();
            }

            await response.WriteBytesAsync(bufferWriter.WrittenMemory);
        }
        finally
        {
            HttpResponse.ReturnJsonBuffer(bufferWriter);
        }
    }

    private static void WriteSeriesJsonValues(Utf8JsonWriter jsonWriter, ISeries col)
    {
        int len = col.Length;
        switch (col)
        {
            case NullSeries:
                for (int i = 0; i < len; i++)
                    jsonWriter.WriteNullValue();
                break;

            case CategoricalSeries cat:
                for (int i = 0; i < len; i++)
                {
                    if (cat.ValidityMask.IsNull(i)) jsonWriter.WriteNullValue();
                    else jsonWriter.WriteStringValue(cat.Get(i)?.ToString());
                }
                break;

            case EnumSeries en:
                for (int i = 0; i < len; i++)
                {
                    if (en.ValidityMask.IsNull(i)) jsonWriter.WriteNullValue();
                    else jsonWriter.WriteStringValue(en.Get(i)?.ToString());
                }
                break;

            case Utf8StringSeries utf8:
                for (int i = 0; i < len; i++)
                {
                    if (utf8.ValidityMask.IsNull(i)) jsonWriter.WriteNullValue();
                    else jsonWriter.WriteStringValue(utf8.GetStringSpan(i));
                }
                break;

            case BinarySeries bin:
                for (int i = 0; i < len; i++)
                {
                    if (bin.ValidityMask.IsNull(i)) jsonWriter.WriteNullValue();
                    else jsonWriter.WriteBase64StringValue(bin.GetSpan(i));
                }
                break;

            case Series<float> sFloat:
            {
                var span = sFloat.Memory.Span;
                for (int i = 0; i < len; i++)
                {
                    if (sFloat.ValidityMask.IsNull(i) || !float.IsFinite(span[i])) jsonWriter.WriteNullValue();
                    else jsonWriter.WriteNumberValue(span[i]);
                }
                break;
            }

            case Series<double> sDouble:
            {
                var span = sDouble.Memory.Span;
                for (int i = 0; i < len; i++)
                {
                    if (sDouble.ValidityMask.IsNull(i) || !double.IsFinite(span[i])) jsonWriter.WriteNullValue();
                    else jsonWriter.WriteNumberValue(span[i]);
                }
                break;
            }

            case Series<int> sInt:
            {
                var span = sInt.Memory.Span;
                for (int i = 0; i < len; i++)
                {
                    if (sInt.ValidityMask.IsNull(i)) jsonWriter.WriteNullValue();
                    else jsonWriter.WriteNumberValue(span[i]);
                }
                break;
            }

            case Series<long> sLong:
            {
                var span = sLong.Memory.Span;
                for (int i = 0; i < len; i++)
                {
                    if (sLong.ValidityMask.IsNull(i)) jsonWriter.WriteNullValue();
                    else jsonWriter.WriteNumberValue(span[i]);
                }
                break;
            }

            case Series<uint> sUInt:
            {
                var span = sUInt.Memory.Span;
                for (int i = 0; i < len; i++)
                {
                    if (sUInt.ValidityMask.IsNull(i)) jsonWriter.WriteNullValue();
                    else jsonWriter.WriteNumberValue(span[i]);
                }
                break;
            }

            case Series<ulong> sULong:
            {
                var span = sULong.Memory.Span;
                for (int i = 0; i < len; i++)
                {
                    if (sULong.ValidityMask.IsNull(i)) jsonWriter.WriteNullValue();
                    else jsonWriter.WriteNumberValue(span[i]);
                }
                break;
            }

            case Series<short> sShort:
            {
                var span = sShort.Memory.Span;
                for (int i = 0; i < len; i++)
                {
                    if (sShort.ValidityMask.IsNull(i)) jsonWriter.WriteNullValue();
                    else jsonWriter.WriteNumberValue(span[i]);
                }
                break;
            }

            case Series<ushort> sUShort:
            {
                var span = sUShort.Memory.Span;
                for (int i = 0; i < len; i++)
                {
                    if (sUShort.ValidityMask.IsNull(i)) jsonWriter.WriteNullValue();
                    else jsonWriter.WriteNumberValue(span[i]);
                }
                break;
            }

            case Series<byte> sByte:
            {
                var span = sByte.Memory.Span;
                for (int i = 0; i < len; i++)
                {
                    if (sByte.ValidityMask.IsNull(i)) jsonWriter.WriteNullValue();
                    else jsonWriter.WriteNumberValue(span[i]);
                }
                break;
            }

            case Series<sbyte> sSByte:
            {
                var span = sSByte.Memory.Span;
                for (int i = 0; i < len; i++)
                {
                    if (sSByte.ValidityMask.IsNull(i)) jsonWriter.WriteNullValue();
                    else jsonWriter.WriteNumberValue(span[i]);
                }
                break;
            }

            case Series<decimal> sDec:
            {
                var span = sDec.Memory.Span;
                for (int i = 0; i < len; i++)
                {
                    if (sDec.ValidityMask.IsNull(i)) jsonWriter.WriteNullValue();
                    else jsonWriter.WriteNumberValue(span[i]);
                }
                break;
            }

            case Series<bool> sBool:
            {
                var span = sBool.Memory.Span;
                for (int i = 0; i < len; i++)
                {
                    if (sBool.ValidityMask.IsNull(i)) jsonWriter.WriteNullValue();
                    else jsonWriter.WriteBooleanValue(span[i]);
                }
                break;
            }

            default:
                for (int i = 0; i < len; i++)
                {
                    object? val = col.Get(i);
                    if (val == null) jsonWriter.WriteNullValue();
                    else jsonWriter.WriteStringValue(val.ToString());
                }
                break;
        }
    }
}

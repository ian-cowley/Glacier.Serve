namespace Glacier.Serve.Serialization;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Glacier.Polaris;
using Glacier.Serve.Core;
using Glacier.Storage.Arrow;

public static class DataFrameHttpExtensions
{
    /// <summary>
    /// Streams a Glacier.Polaris DataFrame as Apache Arrow IPC stream format directly to the HTTP response
    /// using Glacier.Storage managed streaming.
    /// </summary>
    public static ValueTask WriteArrowIpcAsync(this HttpResponse response, DataFrame df)
        => WriteGlacierStorageArrowIpcAsync(response, df);

    /// <summary>
    /// Streams a Glacier.Polaris DataFrame using first-party Glacier.Storage pure managed ArrowStreamWriter,
    /// preserving full validity bitmaps, variable-width UTF-8/Binary offsets, bit-packed booleans, and IPC EOS markers.
    /// </summary>
    public static async ValueTask WriteGlacierStorageArrowIpcAsync(this HttpResponse response, DataFrame df)
    {
        ArgumentNullException.ThrowIfNull(response);
        ArgumentNullException.ThrowIfNull(df);

        response.ContentType = "application/vnd.apache.arrow.stream";

        using var ms = new MemoryStream();
        using (var writer = new ArrowStreamWriter(ms, leaveOpen: true))
        {
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

            writer.WriteSchema(schema);
            writer.WriteRecordBatch(batch);
            writer.WriteEndOfStream();
        }

        if (ms.TryGetBuffer(out ArraySegment<byte> bufferSegment))
        {
            await response.WriteBytesAsync(bufferSegment.AsMemory());
        }
        else
        {
            await response.WriteBytesAsync(ms.ToArray());
        }
    }

    /// <summary>
    /// Serializes a DataFrame into high-performance JSON format using Utf8JsonWriter.
    /// </summary>
    public static async ValueTask WriteDataFrameJsonAsync(this HttpResponse response, DataFrame df)
    {
        ArgumentNullException.ThrowIfNull(response);
        ArgumentNullException.ThrowIfNull(df);

        response.ContentType = "application/json; charset=utf-8";

        var bufferWriter = new ArrayBufferWriter<byte>(1024);
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

                for (int i = 0; i < col.Length; i++)
                {
                    object? val = col.Get(i);
                    if (val == null) jsonWriter.WriteNullValue();
                    else if (val is float f) { if (float.IsFinite(f)) jsonWriter.WriteNumberValue(f); else jsonWriter.WriteNullValue(); }
                    else if (val is double d) { if (double.IsFinite(d)) jsonWriter.WriteNumberValue(d); else jsonWriter.WriteNullValue(); }
                    else if (val is int n) jsonWriter.WriteNumberValue(n);
                    else if (val is long l) jsonWriter.WriteNumberValue(l);
                    else if (val is uint un) jsonWriter.WriteNumberValue(un);
                    else if (val is ulong ul) jsonWriter.WriteNumberValue(ul);
                    else if (val is short s) jsonWriter.WriteNumberValue(s);
                    else if (val is ushort us) jsonWriter.WriteNumberValue(us);
                    else if (val is byte by) jsonWriter.WriteNumberValue(by);
                    else if (val is sbyte sb) jsonWriter.WriteNumberValue(sb);
                    else if (val is decimal dec) jsonWriter.WriteNumberValue(dec);
                    else if (val is bool b) jsonWriter.WriteBooleanValue(b);
                    else jsonWriter.WriteStringValue(val.ToString());
                }

                jsonWriter.WriteEndArray();
                jsonWriter.WriteEndObject();
            }

            jsonWriter.WriteEndArray();
            jsonWriter.WriteEndObject();
        }

        await response.WriteBytesAsync(bufferWriter.WrittenMemory);
    }
}

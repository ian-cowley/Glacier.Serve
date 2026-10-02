namespace Glacier.Serve.Serialization;

using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Glacier.Polaris;
using Glacier.Serve.Core;

public static class DataFrameHttpExtensions
{
    /// <summary>
    /// Streams a Glacier.Polaris DataFrame as Apache Arrow IPC stream format directly to the HTTP response
    /// using Glacier.Storage managed streaming.
    /// </summary>
    public static ValueTask WriteArrowIpcAsync(this HttpResponse response, DataFrame df)
        => WriteGlacierStorageArrowIpcAsync(response, df);

    /// <summary>
    /// Streams a Glacier.Polaris DataFrame using first-party Glacier.Storage pure managed ArrowStreamWriter.
    /// </summary>
    public static async ValueTask WriteGlacierStorageArrowIpcAsync(this HttpResponse response, DataFrame df)
    {
        response.ContentType = "application/vnd.apache.arrow.stream";
        using var ms = new MemoryStream();
        using (var writer = new Glacier.Storage.Arrow.ArrowStreamWriter(ms, leaveOpen: true))
        {
            var fields = new System.Collections.Generic.List<Glacier.Storage.Arrow.ArrowField>(df.Columns.Count);
            for (int i = 0; i < df.Columns.Count; i++)
            {
                var col = df.Columns[i];
                var dt = col switch
                {
                    Glacier.Polaris.Data.Int32Series => Glacier.Storage.Arrow.ArrowType.Int32,
                    Glacier.Polaris.Data.Int64Series => Glacier.Storage.Arrow.ArrowType.Int64,
                    Glacier.Polaris.Data.Float32Series => Glacier.Storage.Arrow.ArrowType.Float,
                    Glacier.Polaris.Data.Float64Series => Glacier.Storage.Arrow.ArrowType.Double,
                    Glacier.Polaris.Data.Utf8StringSeries => Glacier.Storage.Arrow.ArrowType.Utf8,
                    Glacier.Polaris.Data.BooleanSeries => Glacier.Storage.Arrow.ArrowType.Boolean,
                    _ => Glacier.Storage.Arrow.ArrowType.Binary
                };
                fields.Add(new Glacier.Storage.Arrow.ArrowField(col.Name, dt, isNullable: true));
            }
            var schema = new Glacier.Storage.Arrow.ArrowSchema(fields);
            writer.WriteSchema(schema);

            var columns = new System.Collections.Generic.List<Glacier.Storage.Arrow.ArrowColumn>(df.Columns.Count);
            for (int i = 0; i < df.Columns.Count; i++)
            {
                var col = df.Columns[i];
                var field = fields[i];
                ReadOnlyMemory<byte> dataMem = col switch
                {
                    Glacier.Polaris.Data.Int32Series s32 => System.Runtime.InteropServices.MemoryMarshal.AsBytes(s32.Memory.Span).ToArray(),
                    Glacier.Polaris.Data.Float32Series sf32 => System.Runtime.InteropServices.MemoryMarshal.AsBytes(sf32.Memory.Span).ToArray(),
                    Glacier.Polaris.Data.Float64Series sf64 => System.Runtime.InteropServices.MemoryMarshal.AsBytes(sf64.Memory.Span).ToArray(),
                    Glacier.Polaris.Data.Int64Series s64 => System.Runtime.InteropServices.MemoryMarshal.AsBytes(s64.Memory.Span).ToArray(),
                    _ => ReadOnlyMemory<byte>.Empty
                };
                columns.Add(new Glacier.Storage.Arrow.ArrowColumn(field, df.RowCount, 0, ReadOnlyMemory<byte>.Empty, ReadOnlyMemory<byte>.Empty, dataMem));
            }
            var batch = new Glacier.Storage.Arrow.ArrowRecordBatch(schema, df.RowCount, columns);
            writer.WriteRecordBatch(batch);
        }

        byte[] bytes = ms.ToArray();
        await response.EnsureHeadersSentAsync(bytes.Length);
        await response.BodyWriter.WriteAsync(bytes);
        await response.BodyWriter.FlushAsync();
    }

    /// <summary>
    /// Serializes a DataFrame into high-performance JSON format using Utf8JsonWriter.
    /// </summary>
    public static async ValueTask WriteDataFrameJsonAsync(this HttpResponse response, DataFrame df)
    {
        response.ContentType = "application/json; charset=utf-8";

        using var ms = new MemoryStream();
        using (var jsonWriter = new Utf8JsonWriter(ms))
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
                    else if (val is float f) jsonWriter.WriteNumberValue(f);
                    else if (val is double d) jsonWriter.WriteNumberValue(d);
                    else if (val is int n) jsonWriter.WriteNumberValue(n);
                    else if (val is long l) jsonWriter.WriteNumberValue(l);
                    else if (val is bool b) jsonWriter.WriteBooleanValue(b);
                    else jsonWriter.WriteStringValue(val.ToString());
                }

                jsonWriter.WriteEndArray();
                jsonWriter.WriteEndObject();
            }

            jsonWriter.WriteEndArray();
            jsonWriter.WriteEndObject();
        }

        byte[] bytes = ms.ToArray();
        await response.EnsureHeadersSentAsync(bytes.Length);
        await response.BodyWriter.WriteAsync(bytes);
        await response.BodyWriter.FlushAsync();
    }
}

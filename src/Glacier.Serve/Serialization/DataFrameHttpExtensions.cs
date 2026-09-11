namespace Glacier.Serve.Serialization;

using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Apache.Arrow.Ipc;
using Glacier.Polaris;
using Glacier.Serve.Core;

public static class DataFrameHttpExtensions
{
    /// <summary>
    /// Streams a Glacier.Polaris DataFrame as Apache Arrow IPC stream format directly to the HTTP response.
    /// </summary>
    public static async ValueTask WriteArrowIpcAsync(this HttpResponse response, DataFrame df)
    {
        response.ContentType = "application/vnd.apache.arrow.stream";
        var recordBatch = df.ToArrowRecordBatch();

        using var ms = new MemoryStream();
        using (var writer = new ArrowStreamWriter(ms, recordBatch.Schema))
        {
            await writer.WriteRecordBatchAsync(recordBatch);
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

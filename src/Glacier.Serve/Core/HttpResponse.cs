namespace Glacier.Serve.Core;

using System;
using System.Buffers;
using System.Buffers.Text;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO.Pipelines;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization.Metadata;
using System.Threading;
using System.Threading.Tasks;

public sealed class HttpResponse
{
    private const int MaxPooledBuffers = 64;
    private const int MaxPooledBufferCapacity = 65536;
    private static readonly ConcurrentQueue<ArrayBufferWriter<byte>> s_jsonBufferPool = new();
    private static int s_pooledBufferCount;

    private readonly PipeWriter _writer;
    private bool _headersSent;

    public int StatusCode { get; set; } = 200;
    public string ContentType { get; set; } = "text/plain; charset=utf-8";
    public Dictionary<string, string> Headers { get; } = new(StringComparer.OrdinalIgnoreCase);
    public PipeWriter BodyWriter => _writer;
    public bool HeadersSent => _headersSent;

    public HttpResponse(PipeWriter writer)
    {
        _writer = writer;
    }

    private static ArrayBufferWriter<byte> RentJsonBuffer()
    {
        if (s_jsonBufferPool.TryDequeue(out var writer))
        {
            Interlocked.Decrement(ref s_pooledBufferCount);
            writer.Clear();
            return writer;
        }
        return new ArrayBufferWriter<byte>(1024);
    }

    private static void ReturnJsonBuffer(ArrayBufferWriter<byte> writer)
    {
        if (writer.Capacity > MaxPooledBufferCapacity)
            return;

        writer.Clear();
        if (Interlocked.Increment(ref s_pooledBufferCount) <= MaxPooledBuffers)
        {
            s_jsonBufferPool.Enqueue(writer);
        }
        else
        {
            Interlocked.Decrement(ref s_pooledBufferCount);
        }
    }

    private static void ValidateHeaderNoCrlf(string value, string paramName)
    {
        if (value.AsSpan().IndexOfAny('\r', '\n') >= 0)
        {
            throw new ArgumentException("HTTP header name/value must not contain CR or LF characters.", paramName);
        }
    }

    public async ValueTask EnsureHeadersSentAsync(long? contentLength = null, CancellationToken ct = default)
    {
        if (_headersSent) return;
        ValidateHeaderNoCrlf(ContentType, nameof(ContentType));
        foreach (var (hk, hv) in Headers)
        {
            ValidateHeaderNoCrlf(hk, nameof(Headers));
            ValidateHeaderNoCrlf(hv, nameof(Headers));
        }

        _headersSent = true;

        string reason = StatusCode switch
        {
            200 => "OK",
            201 => "Created",
            202 => "Accepted",
            204 => "No Content",
            400 => "Bad Request",
            401 => "Unauthorized",
            403 => "Forbidden",
            404 => "Not Found",
            405 => "Method Not Allowed",
            408 => "Request Timeout",
            413 => "Payload Too Large",
            422 => "Unprocessable Entity",
            429 => "Too Many Requests",
            431 => "Request Header Fields Too Large",
            500 => "Internal Server Error",
            503 => "Service Unavailable",
            _ => "Status"
        };

        ReadOnlySpan<byte> httpPrefix = "HTTP/1.1 "u8;
        ReadOnlySpan<byte> ctPrefix = "\r\nContent-Type: "u8;
        ReadOnlySpan<byte> serverHeader = "\r\nServer: Glacier.Serve/1.2.0 (.NET 10)\r\n"u8;
        ReadOnlySpan<byte> clPrefix = "Content-Length: "u8;
        ReadOnlySpan<byte> crlf = "\r\n"u8;
        ReadOnlySpan<byte> colonSpace = ": "u8;

        int initialSize = 512 + ContentType.Length;
        var span = _writer.GetSpan(initialSize);
        int written = 0;

        httpPrefix.CopyTo(span[written..]); written += httpPrefix.Length;
        Utf8Formatter.TryFormat(StatusCode, span[written..], out int scBytes); written += scBytes;
        span[written++] = (byte)' ';
        written += Encoding.ASCII.GetBytes(reason, span[written..]);
        ctPrefix.CopyTo(span[written..]); written += ctPrefix.Length;
        written += Encoding.ASCII.GetBytes(ContentType, span[written..]);
        serverHeader.CopyTo(span[written..]); written += serverHeader.Length;

        if (contentLength.HasValue)
        {
            clPrefix.CopyTo(span[written..]); written += clPrefix.Length;
            Utf8Formatter.TryFormat(contentLength.Value, span[written..], out int clBytes); written += clBytes;
            crlf.CopyTo(span[written..]); written += crlf.Length;
        }

        foreach (var (k, v) in Headers)
        {
            int required = k.Length + v.Length + 4;
            if (span.Length - written < required)
            {
                _writer.Advance(written);
                written = 0;
                span = _writer.GetSpan(Math.Max(512, required + 64));
            }

            written += Encoding.ASCII.GetBytes(k, span[written..]);
            colonSpace.CopyTo(span[written..]); written += colonSpace.Length;
            written += Encoding.ASCII.GetBytes(v, span[written..]);
            crlf.CopyTo(span[written..]); written += crlf.Length;
        }

        if (span.Length - written < crlf.Length)
        {
            _writer.Advance(written);
            written = 0;
            span = _writer.GetSpan(crlf.Length);
        }

        crlf.CopyTo(span[written..]); written += crlf.Length;
        _writer.Advance(written);
        await _writer.FlushAsync(ct);
    }

    public async ValueTask WriteUtf8Async(string text, CancellationToken ct = default)
    {
        int byteCount = Encoding.UTF8.GetByteCount(text);
        await EnsureHeadersSentAsync(byteCount, ct);
        if (byteCount > 0)
        {
            Span<byte> dest = _writer.GetSpan(byteCount);
            int written = Encoding.UTF8.GetBytes(text, dest);
            _writer.Advance(written);
        }
        await _writer.FlushAsync(ct);
    }

    public async ValueTask WriteBytesAsync(ReadOnlyMemory<byte> bytes, CancellationToken ct = default)
    {
        await EnsureHeadersSentAsync(bytes.Length, ct);
        if (!bytes.IsEmpty)
        {
            await _writer.WriteAsync(bytes, ct);
        }
        await _writer.FlushAsync(ct);
    }

    /// <summary>
    /// Serializes <paramref name="value"/> using Native AOT source-generated <see cref="JsonTypeInfo{T}"/>
    /// via <see cref="Utf8JsonWriter"/> directly into pooled buffer memory with exact Content-Length.
    /// </summary>
    public async ValueTask WriteJsonAsync<T>(T value, JsonTypeInfo<T> jsonTypeInfo, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(jsonTypeInfo);
        ContentType = "application/json; charset=utf-8";

        var buffer = RentJsonBuffer();
        try
        {
            using (var jsonWriter = new Utf8JsonWriter(buffer))
            {
                JsonSerializer.Serialize(jsonWriter, value, jsonTypeInfo);
            }

            ReadOnlyMemory<byte> payload = buffer.WrittenMemory;
            await EnsureHeadersSentAsync(payload.Length, ct);
            if (!payload.IsEmpty)
            {
                await _writer.WriteAsync(payload, ct);
            }
        }
        finally
        {
            ReturnJsonBuffer(buffer);
        }

        await _writer.FlushAsync(ct);
    }

    /// <summary>
    /// Writes a single NDJSON line directly into <see cref="BodyWriter"/> via <see cref="Utf8JsonWriter"/>
    /// with zero intermediate <c>byte[]</c> allocations.
    /// </summary>
    public async ValueTask WriteNdjsonLineAsync<T>(T value, JsonTypeInfo<T> jsonTypeInfo, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(jsonTypeInfo);
        await EnsureHeadersSentAsync(null, ct);
        using (var jsonWriter = new Utf8JsonWriter(_writer))
        {
            JsonSerializer.Serialize(jsonWriter, value, jsonTypeInfo);
        }

        Span<byte> nl = _writer.GetSpan(1);
        nl[0] = (byte)'\n';
        _writer.Advance(1);
        await _writer.FlushAsync(ct);
    }
}

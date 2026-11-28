namespace Glacier.Serve.Core;

using System;
using System.Buffers;
using System.Buffers.Text;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.IO.Pipelines;
using System.Net.Sockets;
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
    private bool _chunkedCompleted;

    internal Action? OnWriteAborted { get; set; }

    public int StatusCode { get; set; } = 200;
    public string ContentType { get; set; } = "text/plain; charset=utf-8";
    public Dictionary<string, string> Headers { get; } = new(StringComparer.OrdinalIgnoreCase);
    public PipeWriter BodyWriter => _writer;
    public bool HeadersSent => _headersSent;
    public bool UseChunkedTransfer { get; set; }

    public HttpResponse(PipeWriter writer)
    {
        _writer = writer;
    }

    internal static ArrayBufferWriter<byte> RentJsonBuffer()
    {
        if (s_jsonBufferPool.TryDequeue(out var writer))
        {
            Interlocked.Decrement(ref s_pooledBufferCount);
            writer.Clear();
            return writer;
        }
        return new ArrayBufferWriter<byte>(1024);
    }

    internal static void ReturnJsonBuffer(ArrayBufferWriter<byte> writer)
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
            501 => "Not Implemented",
            503 => "Service Unavailable",
            _ => "Status"
        };

        ReadOnlySpan<byte> httpPrefix = "HTTP/1.1 "u8;
        ReadOnlySpan<byte> ctPrefix = "\r\nContent-Type: "u8;
        ReadOnlySpan<byte> serverHeader = "\r\nServer: Glacier.Serve/1.9.0 (.NET 10)\r\n"u8;
        ReadOnlySpan<byte> clPrefix = "Content-Length: "u8;
        ReadOnlySpan<byte> teChunked = "Transfer-Encoding: chunked\r\n"u8;
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
        else if (!Headers.ContainsKey("Content-Length") && !Headers.ContainsKey("Transfer-Encoding"))
        {
            if (UseChunkedTransfer)
            {
                teChunked.CopyTo(span[written..]); written += teChunked.Length;
            }
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
        await FlushWriterAsync(ct);
    }

    private async ValueTask WritePayloadAsync(ReadOnlyMemory<byte> payload, CancellationToken ct)
    {
        try
        {
            await _writer.WriteAsync(payload, ct).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is IOException or SocketException or OperationCanceledException)
        {
            OnWriteAborted?.Invoke();
            throw;
        }
    }

    private async ValueTask<FlushResult> FlushWriterAsync(CancellationToken ct)
    {
        try
        {
            FlushResult flush = await _writer.FlushAsync(ct).ConfigureAwait(false);
            if (flush.IsCompleted || flush.IsCanceled)
            {
                OnWriteAborted?.Invoke();
            }
            return flush;
        }
        catch (Exception ex) when (ex is IOException or SocketException or OperationCanceledException)
        {
            OnWriteAborted?.Invoke();
            throw;
        }
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
        await FlushWriterAsync(ct);
    }

    public async ValueTask WriteBytesAsync(ReadOnlyMemory<byte> bytes, CancellationToken ct = default)
    {
        await EnsureHeadersSentAsync(bytes.Length, ct);
        if (!bytes.IsEmpty)
        {
            await WritePayloadAsync(bytes, ct);
        }
        await FlushWriterAsync(ct);
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
                await WritePayloadAsync(payload, ct);
            }
        }
        finally
        {
            ReturnJsonBuffer(buffer);
        }

        await FlushWriterAsync(ct);
    }

    /// <summary>
    /// Writes a single NDJSON line directly into <see cref="BodyWriter"/> via <see cref="Utf8JsonWriter"/>
    /// with zero intermediate <c>byte[]</c> allocations, respecting <see cref="UseChunkedTransfer"/>.
    /// </summary>
    public async ValueTask WriteNdjsonLineAsync<T>(T value, JsonTypeInfo<T> jsonTypeInfo, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(jsonTypeInfo);
        if (UseChunkedTransfer)
        {
            var buffer = RentJsonBuffer();
            try
            {
                using (var jsonWriter = new Utf8JsonWriter(buffer))
                {
                    JsonSerializer.Serialize(jsonWriter, value, jsonTypeInfo);
                }

                Span<byte> nlSpan = buffer.GetSpan(1);
                nlSpan[0] = (byte)'\n';
                buffer.Advance(1);

                bool ok = await WriteChunkAsync(buffer.WrittenMemory, ct);
                if (!ok)
                    throw new OperationCanceledException(ct);
            }
            finally
            {
                ReturnJsonBuffer(buffer);
            }
            return;
        }

        await EnsureHeadersSentAsync(null, ct);
        using (var jsonWriter = new Utf8JsonWriter(_writer))
        {
            JsonSerializer.Serialize(jsonWriter, value, jsonTypeInfo);
        }

        Span<byte> nl = _writer.GetSpan(1);
        nl[0] = (byte)'\n';
        _writer.Advance(1);
        FlushResult flush = await FlushWriterAsync(ct);
        if (flush.IsCanceled)
            throw new OperationCanceledException(ct);
    }

    /// <summary>
    /// Writes a chunked HTTP/1.1 frame when <see cref="UseChunkedTransfer"/> is enabled.
    /// Returns <see langword="false"/> if the downstream reader completed or cancelled.
    /// </summary>
    public async ValueTask<bool> WriteChunkAsync(ReadOnlyMemory<byte> payload, CancellationToken ct = default)
    {
        UseChunkedTransfer = true;
        await EnsureHeadersSentAsync(null, ct);
        if (!payload.IsEmpty)
        {
            Span<byte> headerSpan = _writer.GetSpan(32);
            if (!Utf8Formatter.TryFormat(payload.Length, headerSpan, out int hexLen, new StandardFormat('X')))
                throw new InvalidOperationException("Failed to format HTTP chunk length.");
            headerSpan[hexLen++] = (byte)'\r';
            headerSpan[hexLen++] = (byte)'\n';
            _writer.Advance(hexLen);

            if (payload.Length <= 4096)
            {
                Span<byte> dst = _writer.GetSpan(payload.Length + 2);
                payload.Span.CopyTo(dst);
                dst[payload.Length] = (byte)'\r';
                dst[payload.Length + 1] = (byte)'\n';
                _writer.Advance(payload.Length + 2);
            }
            else
            {
                await WritePayloadAsync(payload, ct);
                Span<byte> tail = _writer.GetSpan(2);
                tail[0] = (byte)'\r';
                tail[1] = (byte)'\n';
                _writer.Advance(2);
            }
        }

        FlushResult flush = await FlushWriterAsync(ct);
        return !flush.IsCompleted && !flush.IsCanceled;
    }

    public async ValueTask CompleteChunkedAsync(CancellationToken ct = default)
    {
        if (!UseChunkedTransfer || _chunkedCompleted) return;
        _chunkedCompleted = true;
        await EnsureHeadersSentAsync(null, ct);
        Span<byte> term = _writer.GetSpan(5);
        "0\r\n\r\n"u8.CopyTo(term);
        _writer.Advance(5);
        await FlushWriterAsync(ct);
    }

    public async ValueTask<bool> WriteSseEventAsync(ReadOnlyMemory<byte> sseFrameBytes, CancellationToken ct = default)
    {
        if (!_headersSent && !Headers.ContainsKey("Content-Length"))
        {
            UseChunkedTransfer = true;
        }

        if (UseChunkedTransfer)
        {
            return await WriteChunkAsync(sseFrameBytes, ct);
        }

        await EnsureHeadersSentAsync(null, ct);
        if (!sseFrameBytes.IsEmpty)
        {
            await WritePayloadAsync(sseFrameBytes, ct);
        }
        FlushResult flush = await FlushWriterAsync(ct);
        return !flush.IsCompleted && !flush.IsCanceled;
    }
}

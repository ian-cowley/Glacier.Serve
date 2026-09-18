namespace Glacier.Serve.Core;

using System;
using System.Buffers.Text;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO.Pipelines;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization.Metadata;
using System.Threading.Tasks;

public sealed class HttpResponse
{
    private readonly PipeWriter _writer;
    private bool _headersSent = false;

    public int StatusCode { get; set; } = 200;
    public string ContentType { get; set; } = "text/plain; charset=utf-8";
    public Dictionary<string, string> Headers { get; } = new(StringComparer.OrdinalIgnoreCase);
    public PipeWriter BodyWriter => _writer;

    public HttpResponse(PipeWriter writer)
    {
        _writer = writer;
    }

    public async ValueTask EnsureHeadersSentAsync(long? contentLength = null)
    {
        if (_headersSent) return;
        _headersSent = true;

        string reason = StatusCode switch
        {
            200 => "OK",
            201 => "Created",
            204 => "No Content",
            400 => "Bad Request",
            404 => "Not Found",
            500 => "Internal Server Error",
            _ => "Status"
        };

        ReadOnlySpan<byte> httpPrefix = "HTTP/1.1 "u8;
        ReadOnlySpan<byte> ctPrefix = "\r\nContent-Type: "u8;
        ReadOnlySpan<byte> serverHeader = "\r\nServer: Glacier.Serve/1.0.0 (.NET 10)\r\n"u8;
        ReadOnlySpan<byte> clPrefix = "Content-Length: "u8;
        ReadOnlySpan<byte> crlf = "\r\n"u8;
        ReadOnlySpan<byte> colonSpace = ": "u8;

        var span = _writer.GetSpan(512);
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
        await _writer.FlushAsync();
    }

    public async ValueTask WriteUtf8Async(string text)
    {
        byte[] body = Encoding.UTF8.GetBytes(text);
        await EnsureHeadersSentAsync(body.Length);
        await _writer.WriteAsync(body);
        await _writer.FlushAsync();
    }

    public async ValueTask WriteBytesAsync(ReadOnlyMemory<byte> bytes)
    {
        await EnsureHeadersSentAsync(bytes.Length);
        await _writer.WriteAsync(bytes);
        await _writer.FlushAsync();
    }

    [RequiresUnreferencedCode("Use overload taking JsonTypeInfo for trim compatibility")]
    [RequiresDynamicCode("Use overload taking JsonTypeInfo for AOT compatibility")]
    public async ValueTask WriteJsonAsync<T>(T value)
    {
        ContentType = "application/json; charset=utf-8";
        byte[] json = JsonSerializer.SerializeToUtf8Bytes(value);
        await EnsureHeadersSentAsync(json.Length);
        await _writer.WriteAsync(json);
        await _writer.FlushAsync();
    }

    public async ValueTask WriteJsonAsync<T>(T value, JsonTypeInfo<T> jsonTypeInfo)
    {
        ContentType = "application/json; charset=utf-8";
        byte[] json = JsonSerializer.SerializeToUtf8Bytes(value, jsonTypeInfo);
        await EnsureHeadersSentAsync(json.Length);
        await _writer.WriteAsync(json);
        await _writer.FlushAsync();
    }
}

namespace Glacier.Serve.Core;

using System;
using System.Collections.Generic;
using System.IO.Pipelines;
using System.Text;
using System.Text.Json;
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

        var sb = new StringBuilder();
        sb.Append($"HTTP/1.1 {StatusCode} {reason}\r\n");
        sb.Append($"Content-Type: {ContentType}\r\n");
        sb.Append("Server: Glacier.Serve/1.0.0 (.NET 10)\r\n");

        if (contentLength.HasValue)
        {
            sb.Append($"Content-Length: {contentLength.Value}\r\n");
        }

        foreach (var (k, v) in Headers)
        {
            sb.Append($"{k}: {v}\r\n");
        }

        sb.Append("\r\n");
        byte[] headerBytes = Encoding.ASCII.GetBytes(sb.ToString());
        await _writer.WriteAsync(headerBytes);
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

    public async ValueTask WriteJsonAsync<T>(T value)
    {
        ContentType = "application/json; charset=utf-8";
        byte[] json = JsonSerializer.SerializeToUtf8Bytes(value);
        await EnsureHeadersSentAsync(json.Length);
        await _writer.WriteAsync(json);
        await _writer.FlushAsync();
    }
}

namespace Glacier.Serve.Tests;

using System;
using System.Text;
using Glacier.Serve.Parsing;
using Glacier.Serve.Routing;
using Xunit;

public class HttpSimdParserTests
{
    [Fact]
    public void TryParseRequestLine_ParsesStandardGet()
    {
        byte[] raw = "GET /api/v1/metrics HTTP/1.1\r\nHost: localhost\r\n\r\n"u8.ToArray();

        bool success = HttpSimdParser.TryParseRequestLine(raw, out var method, out var path, out int consumed);

        Assert.True(success);
        Assert.Equal("GET", Encoding.UTF8.GetString(method));
        Assert.Equal("/api/v1/metrics", Encoding.UTF8.GetString(path));
        Assert.Equal(30, consumed);
        Assert.Equal(HttpMethod.Get, HttpSimdParser.ParseMethod(method));
    }

    [Fact]
    public void TryParseHeader_TrimsLeadingAndTrailingOws()
    {
        byte[] headerLine = "Content-Type:   application/json \t \r\n"u8.ToArray();

        bool success = HttpSimdParser.TryParseHeader(headerLine, out var name, out var value, out int consumed);

        Assert.True(success);
        Assert.Equal("Content-Type", Encoding.UTF8.GetString(name));
        Assert.Equal("application/json", Encoding.UTF8.GetString(value));
        Assert.Equal(headerLine.Length, consumed);
    }

    [Fact]
    public void ParseMethod_IdentifiesAllStandardVerbs()
    {
        Assert.Equal(HttpMethod.Get, HttpSimdParser.ParseMethod("GET"u8));
        Assert.Equal(HttpMethod.Post, HttpSimdParser.ParseMethod("POST"u8));
        Assert.Equal(HttpMethod.Put, HttpSimdParser.ParseMethod("PUT"u8));
        Assert.Equal(HttpMethod.Delete, HttpSimdParser.ParseMethod("DELETE"u8));
        Assert.Equal(HttpMethod.Patch, HttpSimdParser.ParseMethod("PATCH"u8));
        Assert.Equal(HttpMethod.Head, HttpSimdParser.ParseMethod("HEAD"u8));
        Assert.Equal(HttpMethod.Options, HttpSimdParser.ParseMethod("OPTIONS"u8));
        Assert.Equal(HttpMethod.Unknown, HttpSimdParser.ParseMethod("INVALID"u8));
    }

    [Fact]
    public void TryParseRequestLine_RejectsIncompleteLine()
    {
        byte[] incomplete = "GET /api/v1/incomplete"u8.ToArray();
        bool success = HttpSimdParser.TryParseRequestLine(incomplete, out _, out _, out _);
        Assert.False(success);
    }

    [Fact]
    public void IndexOfByteVector256_FindsDelimitersInLongBuffers()
    {
        // 128-byte buffer with space at position 75
        byte[] buffer = new byte[128];
        buffer.AsSpan().Fill((byte)'A');
        buffer[75] = (byte)' ';

        int idx = HttpSimdParser.IndexOfByteVector256(buffer, (byte)' ');
        Assert.Equal(75, idx);
    }
}

namespace Glacier.Serve.Benchmarks;

using System;
using System.Text;
using BenchmarkDotNet.Attributes;
using Glacier.Serve.Parsing;
using Glacier.Serve.Routing;

[MemoryDiagnoser]
public class ParserBenchmarks
{
    private byte[] _requestLineBytes = null!;
    private byte[] _headerBytes = null!;
    private RadixTreeRouter _router = null!;
    private byte[] _pathBytes = null!;

    [GlobalSetup]
    public void Setup()
    {
        _requestLineBytes = "GET /api/v1/workspaces/ws_42/documents/doc_99 HTTP/1.1\r\n"u8.ToArray();
        _headerBytes = "Content-Type: application/json; charset=utf-8\r\n"u8.ToArray();

        _router = new RadixTreeRouter();
        _router.AddRoute(HttpMethod.Get, "/", ctx => ValueTask.CompletedTask);
        _router.AddRoute(HttpMethod.Get, "/api/health", ctx => ValueTask.CompletedTask);
        _router.AddRoute(HttpMethod.Get, "/api/v1/workspaces/{wsId}/documents/{docId}", ctx => ValueTask.CompletedTask);

        _pathBytes = "/api/v1/workspaces/ws_42/documents/doc_99"u8.ToArray();
    }

    [Benchmark(Description = "SIMD Parse Request-Line")]
    public bool ParseRequestLine()
    {
        return HttpSimdParser.TryParseRequestLine(_requestLineBytes, out _, out _, out _);
    }

    [Benchmark(Description = "SIMD Parse Header with OWS")]
    public bool ParseHeader()
    {
        return HttpSimdParser.TryParseHeader(_headerBytes, out _, out _, out _);
    }

    [Benchmark(Description = "Radix Tree Parameterized Match")]
    public bool MatchRoute()
    {
        return _router.TryMatch(HttpMethod.Get, _pathBytes, out _);
    }
}

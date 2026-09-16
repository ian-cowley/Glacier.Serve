namespace Glacier.Serve.Core;

using System;
using System.Collections.Generic;
using System.IO.Pipelines;
using Glacier.Serve.Routing;

public sealed class HttpRequest
{
    public HttpMethod Method { get; set; }
    public string Path { get; set; } = "/";
    public Dictionary<string, string> Headers { get; } = new(StringComparer.OrdinalIgnoreCase);
    public Dictionary<string, string> RouteParams { get; } = new(StringComparer.OrdinalIgnoreCase);
    public PipeReader BodyReader { get; set; } = null!;
    public ReadOnlyMemory<byte> Body { get; set; } = ReadOnlyMemory<byte>.Empty;

    public string? Param(string name) => RouteParams.TryGetValue(name, out var v) ? v : null;
    public string? Header(string name) => Headers.TryGetValue(name, out var v) ? v : null;
}

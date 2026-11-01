namespace Glacier.Serve.Core;

using System;
using System.Collections.Generic;
using System.IO.Pipelines;
using Glacier.Serve.Routing;

public sealed class HttpRequest
{
    public HttpMethod Method { get; set; }
    public string Path { get; set; } = "/";
    public string QueryString { get; set; } = string.Empty;
    public Dictionary<string, string> Headers { get; set; } = new(StringComparer.OrdinalIgnoreCase);
    public Dictionary<string, string> RouteParams { get; set; } = new(StringComparer.OrdinalIgnoreCase);
    public PipeReader BodyReader { get; set; } = null!;
    public ReadOnlyMemory<byte> Body { get; set; } = ReadOnlyMemory<byte>.Empty;

    public string? Param(string name) => RouteParams.TryGetValue(name, out var v) ? v : null;
    public string? Header(string name) => Headers.TryGetValue(name, out var v) ? v : null;

    public string? Query(string name)
    {
        if (string.IsNullOrEmpty(QueryString) || string.IsNullOrEmpty(name))
            return null;

        ReadOnlySpan<char> qs = QueryString.AsSpan();
        if (qs[0] == '?') qs = qs[1..];

        while (!qs.IsEmpty)
        {
            int amp = qs.IndexOf('&');
            ReadOnlySpan<char> pair = amp >= 0 ? qs[..amp] : qs;
            qs = amp >= 0 ? qs[(amp + 1)..] : ReadOnlySpan<char>.Empty;

            if (pair.IsEmpty) continue;
            int eq = pair.IndexOf('=');
            ReadOnlySpan<char> key = eq >= 0 ? pair[..eq] : pair;
            if (key.Equals(name.AsSpan(), StringComparison.OrdinalIgnoreCase))
            {
                ReadOnlySpan<char> rawVal = eq >= 0 ? pair[(eq + 1)..] : ReadOnlySpan<char>.Empty;
                string normalized = rawVal.IndexOf('+') >= 0
                    ? rawVal.ToString().Replace('+', ' ')
                    : rawVal.ToString();
                return Uri.UnescapeDataString(normalized);
            }
        }

        return null;
    }
}

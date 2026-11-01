namespace Glacier.Serve.Routing;

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;

/// <summary>
/// True compressed UTF-8 prefix &amp; parametric segment Radix Tree router.
/// Matches incoming HTTP paths directly over <see cref="ReadOnlySpan{T}"/> of <see cref="byte"/>
/// with zero intermediate string or segment-array heap allocations on static routes,
/// stripping query strings (<c>?...</c>) in place and decoding UTF-8 parameter slices only on match.
/// Supports both <c>{param}</c> and <c>:param</c> route parameter syntax.
/// </summary>
public sealed class RadixTreeRouter : IRouter
{
    private const int MethodCount = 8;

    private sealed class HandlerEntry
    {
        public required RequestDelegate Handler { get; init; }
        public required string[] ParamNames { get; init; }
    }

    private sealed class RadixNode
    {
        public byte[] Prefix;
        public RadixNode[] StaticChildren = [];
        public RadixNode? ParametricChild;
        public readonly HandlerEntry?[] Handlers = new HandlerEntry?[MethodCount];

        public RadixNode(byte[] prefix)
        {
            Prefix = prefix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public RadixNode? FindStaticChildByFirstByte(byte firstByte)
        {
            var children = StaticChildren;
            for (int i = 0; i < children.Length; i++)
            {
                if (children[i].Prefix[0] == firstByte)
                    return children[i];
            }
            return null;
        }

        public void AddOrReplaceStaticChild(RadixNode child)
        {
            byte first = child.Prefix[0];
            for (int i = 0; i < StaticChildren.Length; i++)
            {
                if (StaticChildren[i].Prefix[0] == first)
                {
                    StaticChildren[i] = child;
                    return;
                }
            }

            var expanded = new RadixNode[StaticChildren.Length + 1];
            Array.Copy(StaticChildren, expanded, StaticChildren.Length);
            expanded[^1] = child;
            StaticChildren = expanded;
        }
    }

    private readonly RadixNode _root = new([]);
    private int _maxParams;

    public void AddRoute(HttpMethod method, string routeTemplate, RequestDelegate handler)
    {
        ArgumentNullException.ThrowIfNull(handler);

        if (string.IsNullOrEmpty(routeTemplate))
            routeTemplate = "/";

        // Strip any accidental query string on template registration
        int qIdx = routeTemplate.IndexOf('?');
        if (qIdx >= 0)
            routeTemplate = routeTemplate[..qIdx];

        if (!routeTemplate.StartsWith('/'))
            routeTemplate = "/" + routeTemplate;

        while (routeTemplate.Length > 1 && routeTemplate.EndsWith('/'))
            routeTemplate = routeTemplate[..^1];

        var paramNames = new List<string>(4);
        RadixNode current = _root;

        int cursor = 0;
        int staticStart = 0;

        while (cursor < routeTemplate.Length)
        {
            // Detect parameter start right after '/' as either '{name}' or ':name'
            if (cursor > 0 && routeTemplate[cursor - 1] == '/' &&
                (routeTemplate[cursor] == '{' || routeTemplate[cursor] == ':'))
            {
                // Flush preceding static bytes (including the '/' right before the parameter)
                if (cursor > staticStart)
                {
                    byte[] staticBytes = Encoding.UTF8.GetBytes(routeTemplate.AsSpan(staticStart, cursor - staticStart).ToString());
                    current = InsertStaticPrefix(current, staticBytes);
                }

                string paramName;
                if (routeTemplate[cursor] == '{')
                {
                    int closeBrace = routeTemplate.IndexOf('}', cursor + 1);
                    if (closeBrace < 0)
                        throw new ArgumentException($"Unclosed route parameter brace in template '{routeTemplate}'.", nameof(routeTemplate));

                    paramName = routeTemplate[(cursor + 1)..closeBrace];
                    cursor = closeBrace + 1;
                }
                else
                {
                    int nextSlash = routeTemplate.IndexOf('/', cursor + 1);
                    int end = nextSlash >= 0 ? nextSlash : routeTemplate.Length;
                    paramName = routeTemplate[(cursor + 1)..end];
                    cursor = end;
                }

                if (string.IsNullOrEmpty(paramName))
                    throw new ArgumentException($"Empty route parameter name in template '{routeTemplate}'.", nameof(routeTemplate));

                paramNames.Add(paramName);
                current.ParametricChild ??= new RadixNode([]);
                current = current.ParametricChild;
                staticStart = cursor;
                continue;
            }

            cursor++;
        }

        if (cursor > staticStart)
        {
            byte[] trailingStatic = Encoding.UTF8.GetBytes(routeTemplate[staticStart..cursor]);
            current = InsertStaticPrefix(current, trailingStatic);
        }

        if (paramNames.Count > _maxParams)
            _maxParams = paramNames.Count;

        int methodIdx = (int)method;
        if ((uint)methodIdx >= MethodCount)
            throw new ArgumentOutOfRangeException(nameof(method));

        current.Handlers[methodIdx] = new HandlerEntry
        {
            Handler = handler,
            ParamNames = paramNames.Count == 0 ? [] : paramNames.ToArray()
        };
    }

    private static RadixNode InsertStaticPrefix(RadixNode parent, ReadOnlySpan<byte> prefix)
    {
        RadixNode current = parent;
        ReadOnlySpan<byte> remaining = prefix;

        while (!remaining.IsEmpty)
        {
            RadixNode? child = current.FindStaticChildByFirstByte(remaining[0]);
            if (child == null)
            {
                var leaf = new RadixNode(remaining.ToArray());
                current.AddOrReplaceStaticChild(leaf);
                return leaf;
            }

            ReadOnlySpan<byte> childPrefix = child.Prefix;
            int lcp = CommonPrefixLength(remaining, childPrefix);

            if (lcp < childPrefix.Length)
            {
                // Split existing edge at LCP
                var splitNode = new RadixNode(childPrefix[..lcp].ToArray());
                child.Prefix = childPrefix[lcp..].ToArray();
                splitNode.AddOrReplaceStaticChild(child);
                current.AddOrReplaceStaticChild(splitNode);
                child = splitNode;
            }

            remaining = remaining[lcp..];
            current = child;
        }

        return current;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int CommonPrefixLength(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b)
    {
        return a.CommonPrefixLength(b);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public bool TryMatch(HttpMethod method, ReadOnlySpan<byte> path, out RouteMatch match)
    {
        match = default;
        int methodIdx = (int)method;
        if ((uint)methodIdx >= MethodCount || path.IsEmpty)
            return false;

        // Strip query string (?...) directly on the UTF-8 byte span
        int qIdx = path.IndexOf((byte)'?');
        ReadOnlySpan<byte> cleanPath = qIdx >= 0 ? path[..qIdx] : path;
        if (cleanPath.IsEmpty)
            return false;

        // Strip trailing '/' when not root "/"
        while (cleanPath.Length > 1 && cleanPath[^1] == (byte)'/')
        {
            cleanPath = cleanPath[..^1];
        }

        if (_maxParams == 0)
        {
            return MatchStaticOnly(_root, methodIdx, cleanPath, out match);
        }

        int capCapacity = Math.Max(8, _maxParams);
        Span<(int Offset, int Length)> captures = capCapacity <= 16
            ? stackalloc (int Offset, int Length)[capCapacity]
            : new (int Offset, int Length)[capCapacity];

        return MatchNode(_root, methodIdx, cleanPath, 0, captures, 0, out match);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool MatchStaticOnly(RadixNode root, int methodIdx, ReadOnlySpan<byte> cleanPath, out RouteMatch match)
    {
        RadixNode current = root;
        int offset = 0;

        while (offset < cleanPath.Length)
        {
            RadixNode? child = current.FindStaticChildByFirstByte(cleanPath[offset]);
            if (child == null)
            {
                match = default;
                return false;
            }

            ReadOnlySpan<byte> edge = child.Prefix;
            if (!cleanPath[offset..].StartsWith(edge))
            {
                match = default;
                return false;
            }

            offset += edge.Length;
            current = child;
        }

        HandlerEntry? entry = current.Handlers[methodIdx];
        if (entry != null)
        {
            match = new RouteMatch(entry.Handler, []);
            return true;
        }

        match = default;
        return false;
    }

    private static bool MatchNode(
        RadixNode node,
        int methodIdx,
        ReadOnlySpan<byte> cleanPath,
        int offset,
        Span<(int Offset, int Length)> captures,
        int paramDepth,
        out RouteMatch match)
    {
        if (offset == cleanPath.Length)
        {
            HandlerEntry? entry = node.Handlers[methodIdx];
            if (entry != null)
            {
                if (paramDepth == 0 || entry.ParamNames.Length == 0)
                {
                    match = new RouteMatch(entry.Handler, []);
                }
                else
                {
                    int count = Math.Min(paramDepth, entry.ParamNames.Length);
                    var routeParams = new RouteParameter[count];
                    for (int i = 0; i < count; i++)
                    {
                        var (segOffset, segLen) = captures[i];
                        ReadOnlySpan<byte> rawSeg = cleanPath.Slice(segOffset, segLen);
                        string val = Encoding.UTF8.GetString(rawSeg);
                        if (rawSeg.IndexOf((byte)'%') >= 0)
                        {
                            try { val = Uri.UnescapeDataString(val); } catch { }
                        }
                        routeParams[i] = new RouteParameter(entry.ParamNames[i], val);
                    }
                    match = new RouteMatch(entry.Handler, routeParams);
                }
                return true;
            }

            match = default;
            return false;
        }

        // 1. Prioritize static prefix edge match
        RadixNode? staticChild = node.FindStaticChildByFirstByte(cleanPath[offset]);
        if (staticChild != null)
        {
            ReadOnlySpan<byte> edge = staticChild.Prefix;
            if (cleanPath[offset..].StartsWith(edge))
            {
                if (MatchNode(staticChild, methodIdx, cleanPath, offset + edge.Length, captures, paramDepth, out match))
                {
                    return true;
                }
            }
        }

        // 2. Fallback to parametric segment edge up to next '/'
        RadixNode? paramChild = node.ParametricChild;
        if (paramChild != null)
        {
            ReadOnlySpan<byte> slice = cleanPath[offset..];
            int slashIdx = slice.IndexOf((byte)'/');
            int segLen = slashIdx >= 0 ? slashIdx : slice.Length;
            if (segLen > 0)
            {
                captures[paramDepth] = (offset, segLen);
                if (MatchNode(paramChild, methodIdx, cleanPath, offset + segLen, captures, paramDepth + 1, out match))
                {
                    return true;
                }
            }
        }

        match = default;
        return false;
    }
}

namespace Glacier.Serve.Routing;

using System;
using System.Collections.Generic;
using System.Text;

public sealed class RadixTreeRouter : IRouter
{
    private class RouteEntry
    {
        public required HttpMethod Method { get; init; }
        public required string Template { get; init; }
        public required string[] Segments { get; init; }
        public required bool[] IsParam { get; init; }
        public required string[] ParamNames { get; init; }
        public required RequestDelegate Handler { get; init; }
    }

    private readonly List<RouteEntry> _routes = new();
    private readonly Dictionary<(HttpMethod, string), RequestDelegate> _exactRoutes = new();

    public void AddRoute(HttpMethod method, string routeTemplate, RequestDelegate handler)
    {
        if (string.IsNullOrEmpty(routeTemplate)) routeTemplate = "/";
        if (!routeTemplate.StartsWith('/')) routeTemplate = "/" + routeTemplate;

        // Strip trailing slash if not root
        if (routeTemplate.Length > 1 && routeTemplate.EndsWith('/'))
            routeTemplate = routeTemplate[..^1];

        // If route has no parameters, register in exact route fast map
        if (!routeTemplate.Contains('{'))
        {
            _exactRoutes[(method, routeTemplate)] = handler;
        }

        var rawSegments = routeTemplate.Split('/', StringSplitOptions.RemoveEmptyEntries);
        var isParam = new bool[rawSegments.Length];
        var paramNames = new string[rawSegments.Length];

        for (int i = 0; i < rawSegments.Length; i++)
        {
            var s = rawSegments[i];
            if (s.StartsWith('{') && s.EndsWith('}'))
            {
                isParam[i] = true;
                paramNames[i] = s[1..^1];
            }
            else
            {
                isParam[i] = false;
                paramNames[i] = string.Empty;
            }
        }

        _routes.Add(new RouteEntry
        {
            Method = method,
            Template = routeTemplate,
            Segments = rawSegments,
            IsParam = isParam,
            ParamNames = paramNames,
            Handler = handler
        });
    }

    public bool TryMatch(HttpMethod method, ReadOnlySpan<byte> path, out RouteMatch match)
    {
        match = default;
        if (path.IsEmpty) return false;

        // Fast path: Exact route match
        string pathStr = Encoding.UTF8.GetString(path);
        if (pathStr.Length > 1 && pathStr.EndsWith('/')) pathStr = pathStr[..^1];

        if (_exactRoutes.TryGetValue((method, pathStr), out var fastHandler))
        {
            match = new RouteMatch(fastHandler, []);
            return true;
        }

        var pathSegments = pathStr.Split('/', StringSplitOptions.RemoveEmptyEntries);

        foreach (var r in _routes)
        {
            if (r.Method != method) continue;
            if (r.Segments.Length != pathSegments.Length) continue;

            bool isMatch = true;
            var paramList = new List<RouteParameter>();

            for (int i = 0; i < r.Segments.Length; i++)
            {
                if (r.IsParam[i])
                {
                    paramList.Add(new RouteParameter(r.ParamNames[i], pathSegments[i]));
                }
                else
                {
                    if (!string.Equals(r.Segments[i], pathSegments[i], StringComparison.Ordinal))
                    {
                        isMatch = false;
                        break;
                    }
                }
            }

            if (isMatch)
            {
                match = new RouteMatch(r.Handler, paramList.ToArray());
                return true;
            }
        }

        return false;
    }
}

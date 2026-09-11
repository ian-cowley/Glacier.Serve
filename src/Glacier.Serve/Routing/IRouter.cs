namespace Glacier.Serve.Routing;

using System;
using System.Threading.Tasks;
using Glacier.Serve.Core;

public delegate ValueTask RequestDelegate(HttpContext context);

public readonly struct RouteParameter
{
    public readonly string Name;
    public readonly string Value;

    public RouteParameter(string name, string value)
    {
        Name = name;
        Value = value;
    }
}

public readonly struct RouteMatch
{
    public readonly RequestDelegate Handler;
    public readonly RouteParameter[] Parameters;

    public RouteMatch(RequestDelegate handler, RouteParameter[] parameters)
    {
        Handler = handler;
        Parameters = parameters;
    }

    public string? GetParam(string name)
    {
        if (Parameters != null)
        {
            foreach (var p in Parameters)
            {
                if (string.Equals(p.Name, name, StringComparison.OrdinalIgnoreCase))
                    return p.Value;
            }
        }
        return null;
    }
}

public interface IRouter
{
    void AddRoute(HttpMethod method, string routeTemplate, RequestDelegate handler);
    bool TryMatch(HttpMethod method, ReadOnlySpan<byte> path, out RouteMatch match);
}

namespace Glacier.Serve.Server;

using System.Net;
using Glacier.Serve.Routing;

public sealed class GlacierServeBuilder
{
    private int _port = 5000;
    private IPAddress _address = IPAddress.Loopback;
    private readonly RadixTreeRouter _router = new();

    public GlacierServeBuilder UsePort(int port)
    {
        _port = port;
        return this;
    }

    public GlacierServeBuilder UseHost(string host)
    {
        if (host == "0.0.0.0" || host == "*") _address = IPAddress.Any;
        else if (IPAddress.TryParse(host, out var ip)) _address = ip;
        else _address = IPAddress.Loopback;
        return this;
    }

    public GlacierServeBuilder MapGet(string pattern, RequestDelegate handler)
    {
        _router.AddRoute(HttpMethod.Get, pattern, handler);
        return this;
    }

    public GlacierServeBuilder MapPost(string pattern, RequestDelegate handler)
    {
        _router.AddRoute(HttpMethod.Post, pattern, handler);
        return this;
    }

    public GlacierServeApp Build()
    {
        return new GlacierServeApp(_address, _port, _router);
    }
}

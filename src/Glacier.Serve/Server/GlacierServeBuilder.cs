namespace Glacier.Serve.Server;

using System;
using System.Collections.Generic;
using System.Net;
using System.Security.Cryptography.X509Certificates;
using Glacier.Serve.Routing;

public sealed class GlacierServeBuilder
{
    private int _port = 5000;
    private IPAddress _address = IPAddress.Loopback;
    private long _maxRequestBodySize = GlacierServeApp.DefaultMaxRequestBodySize;
    private readonly RadixTreeRouter _router = new();
    private X509Certificate2? _tlsCertificate;
    private string? _bearerToken;
    private readonly HashSet<string> _authExemptPaths = new(StringComparer.Ordinal);
    private int _maxConcurrentConnections;
    private TimeSpan? _headerTimeout;

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

    public GlacierServeBuilder UseMaxRequestBodySize(long maxBytes)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxBytes, 0);
        _maxRequestBodySize = maxBytes;
        return this;
    }

    public GlacierServeBuilder UseTls(X509Certificate2 certificate)
    {
        ArgumentNullException.ThrowIfNull(certificate);
        if (!certificate.HasPrivateKey)
            throw new ArgumentException("TLS server certificate must include a private key.", nameof(certificate));
        _tlsCertificate = certificate;
        return this;
    }

    public GlacierServeBuilder UseBearerAuth(string bearerToken, params string[] exemptPaths)
    {
        ArgumentException.ThrowIfNullOrEmpty(bearerToken);
        _bearerToken = bearerToken;
        _authExemptPaths.Clear();
        if (exemptPaths != null)
        {
            foreach (string p in exemptPaths)
            {
                if (!string.IsNullOrEmpty(p))
                    _authExemptPaths.Add(GlacierServeApp.NormalizePath(p));
            }
        }
        return this;
    }

    public GlacierServeBuilder WithMaxConcurrentConnections(int maxConnections)
    {
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(maxConnections, 0);
        _maxConcurrentConnections = maxConnections;
        return this;
    }

    public GlacierServeBuilder WithHeaderTimeout(TimeSpan timeout)
    {
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(timeout, TimeSpan.Zero);
        _headerTimeout = timeout;
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

    public GlacierServeBuilder MapPut(string pattern, RequestDelegate handler)
    {
        _router.AddRoute(HttpMethod.Put, pattern, handler);
        return this;
    }

    public GlacierServeBuilder MapDelete(string pattern, RequestDelegate handler)
    {
        _router.AddRoute(HttpMethod.Delete, pattern, handler);
        return this;
    }

    public GlacierServeApp Build()
    {
        return new GlacierServeApp(
            _address,
            _port,
            _router,
            _maxRequestBodySize,
            _tlsCertificate,
            _bearerToken,
            _authExemptPaths,
            _maxConcurrentConnections,
            _headerTimeout);
    }
}

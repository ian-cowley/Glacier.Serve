namespace Glacier.Serve.Core;

using System;
using System.Threading;

public sealed class HttpContext
{
    public HttpRequest Request { get; }
    public HttpResponse Response { get; }
    public CancellationToken RequestAborted { get; }
    internal Func<bool>? ClientDisconnectProbe { get; init; }

    public bool IsClientDisconnected =>
        RequestAborted.IsCancellationRequested || (ClientDisconnectProbe?.Invoke() ?? false);

    public HttpContext(HttpRequest request, HttpResponse response, CancellationToken cancellationToken = default)
    {
        Request = request;
        Response = response;
        RequestAborted = cancellationToken;
    }
}

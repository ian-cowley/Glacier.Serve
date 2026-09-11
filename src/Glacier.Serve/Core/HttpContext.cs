namespace Glacier.Serve.Core;

using System.Threading;

public sealed class HttpContext
{
    public HttpRequest Request { get; }
    public HttpResponse Response { get; }
    public CancellationToken RequestAborted { get; }

    public HttpContext(HttpRequest request, HttpResponse response, CancellationToken cancellationToken = default)
    {
        Request = request;
        Response = response;
        RequestAborted = cancellationToken;
    }
}

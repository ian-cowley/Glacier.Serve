namespace Glacier.Serve.Tests;

using System.Threading.Tasks;
using Glacier.Serve.Core;
using Glacier.Serve.Routing;
using Xunit;

public class RadixTreeRouterTests
{
    [Fact]
    public void Router_MatchesExactStaticRoute()
    {
        var router = new RadixTreeRouter();
        bool handled = false;
        router.AddRoute(HttpMethod.Get, "/api/health", ctx =>
        {
            handled = true;
            return ValueTask.CompletedTask;
        });

        bool matched = router.TryMatch(HttpMethod.Get, "/api/health"u8, out var match);
        Assert.True(matched);
        match.Handler(new HttpContext(new HttpRequest(), new HttpResponse(null!)));
        Assert.True(handled);
    }

    [Fact]
    public void Router_MatchesParameterizedRoute_AndExtractsParam()
    {
        var router = new RadixTreeRouter();
        router.AddRoute(HttpMethod.Get, "/users/{id}/profile", ctx => ValueTask.CompletedTask);

        bool matched = router.TryMatch(HttpMethod.Get, "/users/1337/profile"u8, out var match);
        Assert.True(matched);
        Assert.Equal("1337", match.GetParam("id"));
    }

    [Fact]
    public void Router_MatchesDeeplyNestedMultiParams()
    {
        var router = new RadixTreeRouter();
        router.AddRoute(HttpMethod.Get, "/workspaces/{wsId}/docs/{docId}/chunks/{chunkId}", ctx => ValueTask.CompletedTask);

        bool matched = router.TryMatch(HttpMethod.Get, "/workspaces/ws_alpha/docs/doc_42/chunks/c_99"u8, out var match);
        Assert.True(matched);
        Assert.Equal("ws_alpha", match.GetParam("wsId"));
        Assert.Equal("doc_42", match.GetParam("docId"));
        Assert.Equal("c_99", match.GetParam("chunkId"));
    }

    [Fact]
    public void Router_RejectsWrongHttpMethod()
    {
        var router = new RadixTreeRouter();
        router.AddRoute(HttpMethod.Post, "/submit", ctx => ValueTask.CompletedTask);

        bool matched = router.TryMatch(HttpMethod.Get, "/submit"u8, out _);
        Assert.False(matched);
    }

    [Fact]
    public void Router_HandlesRootRoute()
    {
        var router = new RadixTreeRouter();
        router.AddRoute(HttpMethod.Get, "/", ctx => ValueTask.CompletedTask);

        bool matched = router.TryMatch(HttpMethod.Get, "/"u8, out _);
        Assert.True(matched);
    }
}

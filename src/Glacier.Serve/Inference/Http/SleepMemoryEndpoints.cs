namespace Glacier.Serve.Inference.Http;

using System;
using System.Threading.Tasks;
using Glacier.Serve.Daemons;
using Glacier.Serve.Serialization;
using Glacier.Serve.Server;

/// <summary>
/// First-class HTTP endpoints for episodic <see cref="SleepMemoryService"/> telemetry and triggers
/// (<c>GET /api/sleep/status</c> and <c>POST /api/sleep/trigger</c>).
/// </summary>
public static class SleepMemoryEndpoints
{
    public static void MapSleepMemory(this GlacierServeApp app, SleepMemoryService? sleepMemory)
    {
        ArgumentNullException.ThrowIfNull(app);

        app.MapGet("/api/sleep/status", async ctx =>
        {
            var status = sleepMemory?.GetStatus() ?? CreateDisabledStatus();
            await ctx.Response.WriteJsonAsync(
                status,
                ServeJsonContext.Default.SleepMemoryStatusResponse,
                ctx.RequestAborted).ConfigureAwait(false);
        });

        app.MapPost("/api/sleep/trigger", async ctx =>
        {
            if (sleepMemory == null)
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteJsonAsync(
                    CreateDisabledStatus(),
                    ServeJsonContext.Default.SleepMemoryStatusResponse,
                    ctx.RequestAborted).ConfigureAwait(false);
                return;
            }

            await sleepMemory.TriggerSleepAsync(ctx.RequestAborted).ConfigureAwait(false);
            await ctx.Response.WriteJsonAsync(
                sleepMemory.GetStatus(),
                ServeJsonContext.Default.SleepMemoryStatusResponse,
                ctx.RequestAborted).ConfigureAwait(false);
        });
    }

    private static SleepMemoryStatusResponse CreateDisabledStatus() => new()
    {
        Enabled = false,
        TokenBudget = 0,
        State = "Disabled"
    };
}

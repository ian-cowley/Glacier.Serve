namespace Glacier.Serve.Diagnostics;

using System;

/// <summary>
/// Silent no-op logger instance that discards all diagnostic entries.
/// </summary>
public sealed class NullGlacierLogger : IGlacierLogger
{
    public static NullGlacierLogger Instance { get; } = new();

    private NullGlacierLogger() { }

    public bool IsEnabled(LogLevel level) => false;

    public void Log(LogLevel level, string message, Exception? exception = null) { }
}

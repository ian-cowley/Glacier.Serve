namespace Glacier.Serve.Diagnostics;

using System;

/// <summary>
/// Routes diagnostic log messages to a delegate callback.
/// Ideal for unit testing, test output capture, or custom host adapters.
/// </summary>
public sealed class DelegateGlacierLogger : IGlacierLogger
{
    private readonly Action<LogLevel, string, Exception?> _logAction;
    private readonly LogLevel _minLevel;

    public DelegateGlacierLogger(Action<LogLevel, string, Exception?> logAction, LogLevel minLevel = LogLevel.Trace)
    {
        _logAction = logAction ?? throw new ArgumentNullException(nameof(logAction));
        _minLevel = minLevel;
    }

    public DelegateGlacierLogger(Action<string> logAction, LogLevel minLevel = LogLevel.Trace)
        : this((level, message, ex) =>
        {
            if (ex != null)
                logAction($"[{level}] {message}: {ex}");
            else
                logAction($"[{level}] {message}");
        }, minLevel)
    {
    }

    public bool IsEnabled(LogLevel level) => level != LogLevel.None && level >= _minLevel;

    public void Log(LogLevel level, string message, Exception? exception = null)
    {
        if (IsEnabled(level))
        {
            _logAction(level, message, exception);
        }
    }
}

namespace Glacier.Serve.Diagnostics;

using System;
using System.IO;
using System.Threading;

/// <summary>
/// Formats and routes diagnostic messages to console standard output and error streams.
/// </summary>
public sealed class ConsoleGlacierLogger : IGlacierLogger
{
    private readonly LogLevel _minLevel;
    private readonly bool _useColors;
    private static readonly Lock _syncLock = new();

    public ConsoleGlacierLogger(LogLevel minLevel = LogLevel.Information, bool useColors = true)
    {
        _minLevel = minLevel;
        _useColors = useColors;
    }

    public bool IsEnabled(LogLevel level) => level != LogLevel.None && level >= _minLevel;

    public void Log(LogLevel level, string message, Exception? exception = null)
    {
        if (!IsEnabled(level))
            return;

        string prefix = level switch
        {
            LogLevel.Trace => "[TRACE]",
            LogLevel.Debug => "[DEBUG]",
            LogLevel.Information => "[INFO]",
            LogLevel.Warning => "[WARN]",
            LogLevel.Error => "[ERROR]",
            LogLevel.Critical => "[CRIT]",
            _ => "[LOG]"
        };

        lock (_syncLock)
        {
            TextWriter writer = level >= LogLevel.Warning ? Console.Error : Console.Out;

            if (_useColors)
            {
                var prevColor = Console.ForegroundColor;
                try
                {
                    Console.ForegroundColor = level switch
                    {
                        LogLevel.Trace => ConsoleColor.DarkGray,
                        LogLevel.Debug => ConsoleColor.Gray,
                        LogLevel.Information => ConsoleColor.Cyan,
                        LogLevel.Warning => ConsoleColor.Yellow,
                        LogLevel.Error => ConsoleColor.Red,
                        LogLevel.Critical => ConsoleColor.DarkRed,
                        _ => prevColor
                    };

                    writer.WriteLine($"{prefix} {message}");
                    if (exception != null)
                    {
                        writer.WriteLine(exception.ToString());
                    }
                }
                finally
                {
                    Console.ForegroundColor = prevColor;
                }
            }
            else
            {
                writer.WriteLine($"{prefix} {message}");
                if (exception != null)
                {
                    writer.WriteLine(exception.ToString());
                }
            }
        }
    }
}

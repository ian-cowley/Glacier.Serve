namespace Glacier.Serve.Diagnostics;

using System;
using System.Runtime.CompilerServices;

/// <summary>
/// Static ambient diagnostic logger for Glacier.Serve.
/// Dispatches log messages to the configured ambient <see cref="IGlacierLogger"/>.
/// </summary>
public static class GlacierDiagnostics
{
    private static volatile IGlacierLogger _logger = NullGlacierLogger.Instance;

    /// <summary>
    /// Gets or sets the ambient logger. Defaults to <see cref="NullGlacierLogger.Instance"/>.
    /// </summary>
    public static IGlacierLogger Logger
    {
        get => _logger;
        set => _logger = value ?? NullGlacierLogger.Instance;
    }

    /// <summary>
    /// Sets the ambient logger. Passing null resets to <see cref="NullGlacierLogger.Instance"/>.
    /// </summary>
    public static void SetLogger(IGlacierLogger? logger) => _logger = logger ?? NullGlacierLogger.Instance;

    /// <summary>
    /// Resets the ambient logger back to the default <see cref="NullGlacierLogger.Instance"/>.
    /// </summary>
    public static void Reset() => _logger = NullGlacierLogger.Instance;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsEnabled(LogLevel level) => _logger.IsEnabled(level);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void LogTrace(string message)
    {
        if (_logger.IsEnabled(LogLevel.Trace))
            _logger.Log(LogLevel.Trace, message);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void LogDebug(string message)
    {
        if (_logger.IsEnabled(LogLevel.Debug))
            _logger.Log(LogLevel.Debug, message);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void LogInformation(string message)
    {
        if (_logger.IsEnabled(LogLevel.Information))
            _logger.Log(LogLevel.Information, message);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void LogWarning(string message, Exception? exception = null)
    {
        if (_logger.IsEnabled(LogLevel.Warning))
            _logger.Log(LogLevel.Warning, message, exception);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void LogError(string message, Exception? exception = null)
    {
        if (_logger.IsEnabled(LogLevel.Error))
            _logger.Log(LogLevel.Error, message, exception);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void LogCritical(string message, Exception? exception = null)
    {
        if (_logger.IsEnabled(LogLevel.Critical))
            _logger.Log(LogLevel.Critical, message, exception);
    }
}

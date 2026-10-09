[assembly: Xunit.CollectionBehavior(DisableTestParallelization = true)]

namespace Glacier.Serve.Tests;

using System;
using System.Collections.Generic;
using System.Linq;
using Glacier.Serve.Diagnostics;
using Xunit;

public class DiagnosticsTests
{
    private static readonly object s_diagLock = new();

    [Fact]
    public void GlacierDiagnostics_DefaultLogger_IsNullLogger()
    {
        lock (s_diagLock)
        {
            GlacierDiagnostics.Reset();
            Assert.Same(NullGlacierLogger.Instance, GlacierDiagnostics.Logger);
            Assert.False(GlacierDiagnostics.IsEnabled(LogLevel.Trace));
            Assert.False(GlacierDiagnostics.IsEnabled(LogLevel.Debug));
            Assert.False(GlacierDiagnostics.IsEnabled(LogLevel.Information));
            Assert.False(GlacierDiagnostics.IsEnabled(LogLevel.Warning));
            Assert.False(GlacierDiagnostics.IsEnabled(LogLevel.Error));
            Assert.False(GlacierDiagnostics.IsEnabled(LogLevel.Critical));
        }
    }

    [Fact]
    public void GlacierDiagnostics_SetLogger_And_Reset_Works()
    {
        lock (s_diagLock)
        {
            var messages = new List<(LogLevel Level, string Message, Exception? Ex)>();
            var testLogger = new DelegateGlacierLogger((level, msg, ex) =>
            {
                if (msg.StartsWith("[DiagnosticsTest]"))
                {
                    lock (messages)
                    {
                        messages.Add((level, msg, ex));
                    }
                }
            }, LogLevel.Trace);

            try
            {
                GlacierDiagnostics.SetLogger(testLogger);
                Assert.Same(testLogger, GlacierDiagnostics.Logger);
                Assert.True(GlacierDiagnostics.IsEnabled(LogLevel.Information));

                GlacierDiagnostics.LogInformation("[DiagnosticsTest] Server initialized");
                Assert.Single(messages);
                Assert.Equal(LogLevel.Information, messages[0].Level);
                Assert.Equal("[DiagnosticsTest] Server initialized", messages[0].Message);
                Assert.Null(messages[0].Ex);

                GlacierDiagnostics.Reset();
                Assert.Same(NullGlacierLogger.Instance, GlacierDiagnostics.Logger);

                GlacierDiagnostics.LogInformation("[DiagnosticsTest] Should not be recorded");
                Assert.Single(messages); // Count unchanged
            }
            finally
            {
                GlacierDiagnostics.Reset();
            }
        }
    }

    [Fact]
    public void GlacierDiagnostics_ConvenienceMethods_DispatchAllLevels()
    {
        lock (s_diagLock)
        {
            var messages = new List<(LogLevel Level, string Message, Exception? Ex)>();
            var testLogger = new DelegateGlacierLogger((level, msg, ex) =>
            {
                if (msg.StartsWith("[DiagnosticsTest]"))
                {
                    lock (messages)
                    {
                        messages.Add((level, msg, ex));
                    }
                }
            }, LogLevel.Trace);

            try
            {
                GlacierDiagnostics.Logger = testLogger;

                var testEx = new InvalidOperationException("Test socket failure");
                GlacierDiagnostics.LogTrace("[DiagnosticsTest] trace message");
                GlacierDiagnostics.LogDebug("[DiagnosticsTest] debug message");
                GlacierDiagnostics.LogInformation("[DiagnosticsTest] info message");
                GlacierDiagnostics.LogWarning("[DiagnosticsTest] warn message", testEx);
                GlacierDiagnostics.LogError("[DiagnosticsTest] error message", testEx);
                GlacierDiagnostics.LogCritical("[DiagnosticsTest] critical message", testEx);

                Assert.Equal(6, messages.Count);
                Assert.Equal(LogLevel.Trace, messages[0].Level);
                Assert.Equal("[DiagnosticsTest] trace message", messages[0].Message);

                Assert.Equal(LogLevel.Debug, messages[1].Level);
                Assert.Equal("[DiagnosticsTest] debug message", messages[1].Message);

                Assert.Equal(LogLevel.Information, messages[2].Level);
                Assert.Equal("[DiagnosticsTest] info message", messages[2].Message);

                Assert.Equal(LogLevel.Warning, messages[3].Level);
                Assert.Equal("[DiagnosticsTest] warn message", messages[3].Message);
                Assert.Same(testEx, messages[3].Ex);

                Assert.Equal(LogLevel.Error, messages[4].Level);
                Assert.Equal("[DiagnosticsTest] error message", messages[4].Message);
                Assert.Same(testEx, messages[4].Ex);

                Assert.Equal(LogLevel.Critical, messages[5].Level);
                Assert.Equal("[DiagnosticsTest] critical message", messages[5].Message);
                Assert.Same(testEx, messages[5].Ex);
            }
            finally
            {
                GlacierDiagnostics.Reset();
            }
        }
    }

    [Fact]
    public void ConsoleGlacierLogger_FiltersByLogLevel()
    {
        var logger = new ConsoleGlacierLogger(LogLevel.Warning, useColors: false);

        Assert.False(logger.IsEnabled(LogLevel.Trace));
        Assert.False(logger.IsEnabled(LogLevel.Debug));
        Assert.False(logger.IsEnabled(LogLevel.Information));
        Assert.True(logger.IsEnabled(LogLevel.Warning));
        Assert.True(logger.IsEnabled(LogLevel.Error));
        Assert.True(logger.IsEnabled(LogLevel.Critical));
        Assert.False(logger.IsEnabled(LogLevel.None));

        // Ensure logging executes safely without throwing
        logger.Log(LogLevel.Information, "Ignored info message");
        logger.Log(LogLevel.Warning, "Reported warning");
        logger.Log(LogLevel.Error, "Reported error", new Exception("Test"));
    }

    [Fact]
    public void DelegateGlacierLogger_StringOnlyConstructor_FormatsCorrectly()
    {
        var logEntries = new List<string>();
        var logger = new DelegateGlacierLogger(msg => logEntries.Add(msg), LogLevel.Debug);

        logger.Log(LogLevel.Debug, "Engine started");
        logger.Log(LogLevel.Error, "Socket failed", new Exception("Connection refused"));

        Assert.Equal(2, logEntries.Count);
        Assert.Equal("[Debug] Engine started", logEntries[0]);
        Assert.Contains("[Error] Socket failed:", logEntries[1]);
        Assert.Contains("Connection refused", logEntries[1]);
    }

    [Fact]
    public void DelegateGlacierLogger_RequiresNonNullAction()
    {
        Action<LogLevel, string, Exception?> nullAction = null!;
        Assert.Throws<ArgumentNullException>(() => new DelegateGlacierLogger(nullAction));
    }
}

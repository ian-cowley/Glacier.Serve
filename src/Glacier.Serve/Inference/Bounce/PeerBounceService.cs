namespace Glacier.Serve.Inference.Bounce;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Agent.Sandbox;
using Glacier.Serve.Diagnostics;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Serialization;

public enum BounceTaskKind
{
    Bypass,
    CSharp,
    Hlsl,
    HtmlJs,
    Python,
    ToolCall,
    FinalAnswer,
    General
}

public sealed record BounceVerificationResult(bool Success, string? ErrorMessage);

public interface IBounceVerifier
{
    Task<BounceVerificationResult> VerifyAsync(BounceTaskKind kind, string promptText, string completionText, CancellationToken ct);
}

public sealed record BounceCompletionResult(
    string Content,
    string FinishReason,
    string ModelId,
    int PromptTokens,
    int CompletionTokens,
    int RepairsUsed,
    bool CriticAssisted,
    bool Bypassed);

/// <summary>
/// OS-isolated compiler/syntax bounce verifier backed by <see cref="ISandboxExecutor"/>
/// (Windows Job Objects + restricted token, or Linux <c>clone3</c>/<c>unshare</c> namespaces).
/// </summary>
public sealed partial class SandboxedProcessBounceVerifier : IBounceVerifier
{
    private readonly string? _harnessExePath;
    private readonly ISandboxExecutor _sandbox;
    private readonly string _pythonExe;
    private readonly bool _isCustomPythonExe;
    private readonly string _nodeExe;
    private readonly TimeSpan _harnessTimeout;
    private readonly TimeSpan _scriptTimeout;
    private readonly long? _memoryLimitBytes;
    private readonly int? _maxProcesses;
    private readonly bool _allowNetwork;
    private readonly bool _enableNamespaceIsolation;

    [GeneratedRegex(@"```([^\r\n`]*)\r?\n(.*?)```", RegexOptions.Singleline)]
    private static partial Regex FencedCodeBlockRegex();

    [GeneratedRegex(@"\[numthreads\([^\]]+\)\]\s*(?:\[[^\]]+\]\s*)*void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", RegexOptions.Singleline)]
    private static partial Regex HlslEntryRegex();

    [GeneratedRegex(@"<tool_call>\s*(\{.*?\})\s*</tool_call>", RegexOptions.Singleline)]
    private static partial Regex ToolCallRegex();

    [GeneratedRegex(@"FINAL_ANSWER:\s*([^\r\n]+)")]
    private static partial Regex FinalAnswerRegex();

    [GeneratedRegex(@"<script\b([^>]*)>(.*?)</script>", RegexOptions.Singleline | RegexOptions.IgnoreCase)]
    private static partial Regex ScriptBlockRegex();

    [GeneratedRegex(@"(?:^|\s)type\s*=\s*(?:""([^""]*)""|'([^']*)'|([^\s""'>]+))", RegexOptions.IgnoreCase)]
    private static partial Regex ScriptTypeAttrRegex();

    public static string? ResolveDefaultHarnessPath(string? currentDirectory = null, string? baseDirectory = null)
    {
        string exeName = OperatingSystem.IsWindows() ? "Glacier.EvaluationHarness.exe" : "Glacier.EvaluationHarness";
        string baseDir = string.IsNullOrWhiteSpace(baseDirectory) ? AppContext.BaseDirectory : baseDirectory;
        string cwd = string.IsNullOrWhiteSpace(currentDirectory) ? Directory.GetCurrentDirectory() : currentDirectory;

        string directBase = Path.GetFullPath(Path.Combine(baseDir, exeName));
        if (File.Exists(directBase))
            return directBase;

        string directCwd = Path.GetFullPath(Path.Combine(cwd, exeName));
        if (File.Exists(directCwd))
            return directCwd;

        foreach (string startRoot in new[] { cwd, baseDir })
        {
            string? dir = startRoot;
            for (int depth = 0; depth < 6 && !string.IsNullOrEmpty(dir); depth++)
            {
                string releaseCandidate = Path.GetFullPath(Path.Combine(dir, "tests", "Glacier.EvaluationHarness", "bin", "Release", "net10.0", exeName));
                if (File.Exists(releaseCandidate))
                    return releaseCandidate;

                string debugCandidate = Path.GetFullPath(Path.Combine(dir, "tests", "Glacier.EvaluationHarness", "bin", "Debug", "net10.0", exeName));
                if (File.Exists(debugCandidate))
                    return debugCandidate;

                dir = Path.GetDirectoryName(dir);
            }
        }

        return null;
    }

    public ISandboxExecutor SandboxExecutor => _sandbox;
    public string? LastIsolationMechanism { get; private set; }
    public TimeSpan? LastStartupLatency { get; private set; }

    public SandboxedProcessBounceVerifier(
        string? harnessExePath,
        ISandboxExecutor? sandboxExecutor = null,
        string pythonExe = "python",
        string nodeExe = "node",
        TimeSpan? harnessTimeout = null,
        TimeSpan? scriptTimeout = null,
        long? memoryLimitBytes = null,
        int? maxProcesses = null,
        bool allowNetwork = false,
        bool enableNamespaceIsolation = true)
    {
        _harnessExePath = string.IsNullOrWhiteSpace(harnessExePath) ? null : harnessExePath;
        _sandbox = sandboxExecutor ?? new OsProcessSandbox();
        _pythonExe = string.IsNullOrWhiteSpace(pythonExe) ? "python" : pythonExe;
        _isCustomPythonExe = !string.Equals(_pythonExe, "python", StringComparison.OrdinalIgnoreCase) &&
                             !string.Equals(_pythonExe, "python3", StringComparison.OrdinalIgnoreCase);
        _nodeExe = string.IsNullOrWhiteSpace(nodeExe) ? "node" : nodeExe;
        _harnessTimeout = harnessTimeout ?? TimeSpan.FromSeconds(15);
        _scriptTimeout = scriptTimeout ?? TimeSpan.FromSeconds(8);
        _memoryLimitBytes = memoryLimitBytes;
        _maxProcesses = maxProcesses;
        _allowNetwork = allowNetwork;
        _enableNamespaceIsolation = enableNamespaceIsolation;
    }

    public async Task<BounceVerificationResult> VerifyAsync(
        BounceTaskKind kind,
        string promptText,
        string completionText,
        CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(completionText))
            return new BounceVerificationResult(false, "Model returned an empty response.");

        switch (kind)
        {
            case BounceTaskKind.ToolCall:
                return VerifyToolCall(completionText);

            case BounceTaskKind.FinalAnswer:
                return VerifyFinalAnswer(completionText);

            case BounceTaskKind.CSharp:
            {
                string? code = ExtractCodeBlock(completionText, "csharp", "cs");
                if (string.IsNullOrWhiteSpace(code))
                    return new BounceVerificationResult(false, "Missing ```csharp code block.");
                if (string.IsNullOrWhiteSpace(_harnessExePath))
                    return new BounceVerificationResult(false, "C# compiler verification harness is not configured.");
                if (!File.Exists(_harnessExePath))
                    return new BounceVerificationResult(false, $"C# compiler verification harness was not found at '{_harnessExePath}'.");
                return await RunHarnessAsync("--verify-csharp", code, ".cs", extraArgs: null, ct).ConfigureAwait(false);
            }

            case BounceTaskKind.Hlsl:
            {
                string? code = ExtractCodeBlock(completionText, "hlsl", "fx");
                if (string.IsNullOrWhiteSpace(code))
                    return new BounceVerificationResult(false, "Missing ```hlsl code block.");
                if (string.IsNullOrWhiteSpace(_harnessExePath))
                    return new BounceVerificationResult(false, "HLSL compiler verification harness is not configured.");
                if (!File.Exists(_harnessExePath))
                    return new BounceVerificationResult(false, $"HLSL compiler verification harness was not found at '{_harnessExePath}'.");
                string entry = DetectHlslEntry(code);
                return await RunHarnessAsync("--verify-hlsl", code, ".hlsl", ["cs_6_6", entry], ct).ConfigureAwait(false);
            }

            case BounceTaskKind.HtmlJs:
                return await VerifyHtmlJsAsync(completionText, ct).ConfigureAwait(false);

            case BounceTaskKind.Python:
                return await VerifyPythonSyntaxAsync(completionText, ct).ConfigureAwait(false);

            default:
                return new BounceVerificationResult(true, null);
        }
    }

    public static string? ExtractCodeBlock(string text, params string[] langs)
    {
        var matches = FencedCodeBlockRegex().Matches(text);
        if (matches.Count == 0)
            return null;

        foreach (string lang in langs)
        {
            for (int i = matches.Count - 1; i >= 0; i--)
            {
                string info = matches[i].Groups[1].Value.Trim();
                int spaceIdx = info.IndexOfAny([' ', '\t']);
                if (spaceIdx >= 0)
                    info = info[..spaceIdx];

                if (string.Equals(info, lang, StringComparison.OrdinalIgnoreCase))
                    return matches[i].Groups[2].Value.Trim();
            }
        }

        return matches[^1].Groups[2].Value.Trim();
    }

    private static string DetectHlslEntry(string code)
    {
        var m = HlslEntryRegex().Match(code);
        if (m.Success) return m.Groups[1].Value;
        return code.Contains("CSMain", StringComparison.Ordinal) ? "CSMain" : "main";
    }

    private static BounceVerificationResult VerifyToolCall(string text)
    {
        var m = ToolCallRegex().Match(text);
        if (!m.Success)
            return new BounceVerificationResult(false, "Response must wrap valid JSON inside <tool_call>{...}</tool_call> tags.");
        try
        {
            string json = m.Groups[1].Value;
            int maxBytes = Encoding.UTF8.GetMaxByteCount(json.Length);
            byte[]? rented = null;
            Span<byte> utf8 = maxBytes <= 1024
                ? stackalloc byte[1024]
                : (rented = ArrayPool<byte>.Shared.Rent(maxBytes)).AsSpan(0, maxBytes);
            try
            {
                int written = Encoding.UTF8.GetBytes(json, utf8);
                var reader = new Utf8JsonReader(utf8[..written]);
                if (!reader.Read() || reader.TokenType != JsonTokenType.StartObject)
                    return new BounceVerificationResult(false, "Invalid JSON inside <tool_call>: expected JSON object.");

                bool hasValidName = false;
                bool hasObjectArgs = false;

                while (reader.Read())
                {
                    if (reader.TokenType == JsonTokenType.EndObject)
                        break;

                    if (reader.TokenType != JsonTokenType.PropertyName)
                        continue;

                    bool isName = reader.ValueTextEquals("name"u8);
                    bool isArgs = reader.ValueTextEquals("arguments"u8);

                    if (!reader.Read())
                        break;

                    if (isName)
                    {
                        hasValidName = reader.TokenType == JsonTokenType.String && !string.IsNullOrWhiteSpace(reader.GetString());
                    }
                    else if (isArgs)
                    {
                        hasObjectArgs = reader.TokenType == JsonTokenType.StartObject;
                    }

                    if (reader.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray)
                        reader.Skip();
                }

                if (!hasValidName)
                    return new BounceVerificationResult(false, "<tool_call> JSON must contain a non-empty string \"name\" field.");
                if (!hasObjectArgs)
                    return new BounceVerificationResult(false, "<tool_call> JSON must contain an object \"arguments\" field.");
                return new BounceVerificationResult(true, null);
            }
            finally
            {
                if (rented != null)
                    ArrayPool<byte>.Shared.Return(rented);
            }
        }
        catch (JsonException ex)
        {
            return new BounceVerificationResult(false, $"Invalid JSON inside <tool_call>: {ex.Message}");
        }
    }

    private static BounceVerificationResult VerifyFinalAnswer(string text)
    {
        var m = FinalAnswerRegex().Match(text);
        if (!m.Success || string.IsNullOrWhiteSpace(m.Groups[1].Value))
            return new BounceVerificationResult(false, "Response truncated or missing required `FINAL_ANSWER: <value>` line.");
        return new BounceVerificationResult(true, null);
    }

    private static bool IsJavaScriptScriptTag(string attributes, out bool isModule)
    {
        isModule = false;
        if (string.IsNullOrWhiteSpace(attributes))
            return true;

        var typeMatch = ScriptTypeAttrRegex().Match(attributes);
        if (!typeMatch.Success)
            return true;

        string rawType = typeMatch.Groups[1].Success
            ? typeMatch.Groups[1].Value
            : typeMatch.Groups[2].Success
                ? typeMatch.Groups[2].Value
                : typeMatch.Groups[3].Value;

        string mediaType = rawType.Trim();
        int semiIdx = mediaType.IndexOf(';');
        if (semiIdx >= 0)
            mediaType = mediaType[..semiIdx].Trim();

        if (mediaType.Length == 0)
            return true;

        if (string.Equals(mediaType, "text/javascript", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(mediaType, "application/javascript", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        if (string.Equals(mediaType, "module", StringComparison.OrdinalIgnoreCase))
        {
            isModule = true;
            return true;
        }

        return false;
    }

    private async Task<BounceVerificationResult> VerifyHtmlJsAsync(string text, CancellationToken ct)
    {
        string? html = ExtractCodeBlock(text, "html", "htm");
        if (string.IsNullOrWhiteSpace(html))
        {
            if (text.Contains("<!DOCTYPE html", StringComparison.OrdinalIgnoreCase) || text.Contains("<html", StringComparison.OrdinalIgnoreCase))
                html = text;
            else
                return new BounceVerificationResult(false, "Missing ```html single-file document.");
        }

        if (!html.Contains("</html>", StringComparison.OrdinalIgnoreCase))
            return new BounceVerificationResult(false, "HTML output appears truncated (missing closing </html> tag). Make the implementation more concise so it finishes completely.");

        var scriptMatches = ScriptBlockRegex().Matches(html);
        if (scriptMatches.Count == 0)
            return new BounceVerificationResult(false, "HTML document is missing a closed <script>...</script> block.");

        var sb = new StringBuilder();
        int jsBlockCount = 0;
        bool hasModule = false;
        foreach (Match sm in scriptMatches)
        {
            string attrs = sm.Groups[1].Value;
            if (!IsJavaScriptScriptTag(attrs, out bool isModule))
                continue;

            jsBlockCount++;
            hasModule |= isModule;
            sb.AppendLine(sm.Groups[2].Value);
        }

        if (jsBlockCount == 0)
            return new BounceVerificationResult(true, null);

        string tempDir = CreateIsolatedTempDir();
        string tmpJs = Path.Combine(tempDir, hasModule ? "check.mjs" : "check.js");
        try
        {
            await File.WriteAllTextAsync(tmpJs, sb.ToString(), Encoding.UTF8, ct).ConfigureAwait(false);
            var res = await RunSandboxedAsync(_nodeExe, ["--check", tmpJs], tempDir, _scriptTimeout, ct).ConfigureAwait(false);
            if (!OperatingSystem.IsWindows()
                && _enableNamespaceIsolation
                && !res.TimedOut
                && res.ExitCode != 0
                && res.StandardError.Contains("Native stack trace", StringComparison.OrdinalIgnoreCase))
            {
                // Node.js 22/24 on some Linux kernels/CI containers aborts in node::Start under CLONE_NEWPID
                // when /proc is not remounted. Retry --check (parse-only, no execution) inside the same
                // OsProcessSandbox with namespace isolation disabled (retaining NO_NEW_PRIVS, rlimits, and timeout).
                res = await RunSandboxedAsync(_nodeExe, ["--check", tmpJs], tempDir, _scriptTimeout, ct, enableNamespaceIsolationOverride: false).ConfigureAwait(false);
            }
            if (res.TimedOut)
            {
                return new BounceVerificationResult(false, $"Node --check timed out after {_scriptTimeout.TotalMilliseconds:F0} ms.");
            }
            if (res.ExitCode != 0)
            {
                string err = string.IsNullOrWhiteSpace(res.StandardError) ? "JavaScript syntax error in <script> block." : res.StandardError.Trim();
                if (err.Length > 400) err = err[..400];
                return new BounceVerificationResult(false, $"Node --check failed: {err}");
            }
            return new BounceVerificationResult(true, null);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            return new BounceVerificationResult(false, $"JavaScript syntax verifier '{_nodeExe}' unavailable or failed to launch: {ex.Message}");
        }
        finally
        {
            CleanupTempDir(tempDir);
        }
    }

    private async Task<BounceVerificationResult> VerifyPythonSyntaxAsync(string text, CancellationToken ct)
    {
        string? code = ExtractCodeBlock(text, "python", "py");
        if (string.IsNullOrWhiteSpace(code))
            return new BounceVerificationResult(false, "Missing ```python code block.");

        string tempDir = CreateIsolatedTempDir();
        string tmpPy = Path.Combine(tempDir, "check.py");
        try
        {
            await File.WriteAllTextAsync(tmpPy, code, Encoding.UTF8, ct).ConfigureAwait(false);
            SandboxExecutionResult res;
            try
            {
                res = await RunSandboxedAsync(_pythonExe, ["-m", "py_compile", tmpPy], tempDir, _scriptTimeout, ct).ConfigureAwait(false);
            }
            catch (Exception ex) when (!OperatingSystem.IsWindows()
                                       && string.Equals(_pythonExe, "python", StringComparison.Ordinal)
                                       && (ex is Win32Exception or FileNotFoundException or InvalidOperationException))
            {
                res = await RunSandboxedAsync("python3", ["-m", "py_compile", tmpPy], tempDir, _scriptTimeout, ct).ConfigureAwait(false);
            }

            if (res.TimedOut)
            {
                return new BounceVerificationResult(false, $"Python py_compile timed out after {_scriptTimeout.TotalMilliseconds:F0} ms.");
            }
            if (res.ExitCode != 0)
            {
                string err = string.IsNullOrWhiteSpace(res.StandardError) ? "Python syntax error." : res.StandardError.Trim();
                if (err.Length > 400) err = err[..400];
                return new BounceVerificationResult(false, $"Python py_compile failed: {err}");
            }
            return new BounceVerificationResult(true, null);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex) when (!_isCustomPythonExe && ex is Win32Exception or FileNotFoundException or InvalidOperationException)
        {
            GlacierDiagnostics.LogDebug($"[PeerBounceService] Python binary '{_pythonExe}' unavailable; using pure-C# Python syntax verifier: {ex.Message}");
            return VerifyPythonSyntaxPureCSharp(code);
        }
        catch (Exception ex)
        {
            return new BounceVerificationResult(false, $"Python syntax verifier '{_pythonExe}' unavailable or failed to launch: {ex.Message}");
        }
        finally
        {
            CleanupTempDir(tempDir);
        }
    }

    internal static BounceVerificationResult VerifyPythonSyntaxPureCSharp(string code)
    {
        if (string.IsNullOrWhiteSpace(code))
            return new BounceVerificationResult(false, "Python py_compile failed (lexical): empty code block.");

        var delimiterStack = new Stack<(char Open, int Line)>();
        var indentStack = new List<int> { 0 };
        bool expectingIndentBlock = false;

        bool inTripleSingle = false;
        bool inTripleDouble = false;
        bool inSingle = false;
        bool inDouble = false;
        bool lineContinuation = false;
        int lineNumber = 0;

        string normalized = code.Replace("\r\n", "\n", StringComparison.Ordinal).Replace("\r", "\n", StringComparison.Ordinal);
        string[] lines = normalized.Split('\n');
        int lineCount = (normalized.EndsWith('\n') && lines.Length > 0 && lines[^1].Length == 0)
            ? lines.Length - 1
            : lines.Length;
        for (int li = 0; li < lineCount; li++)
        {
            lineNumber = li + 1;
            string line = lines[li];

            // Compute leading indentation only when not continuing a string, open bracket expression, or trailing '\'
            if (!inTripleSingle && !inTripleDouble && !inSingle && !inDouble && delimiterStack.Count == 0 && !lineContinuation)
            {
                int indent = 0;
                bool hasSpace = false;
                bool hasTab = false;
                int p = 0;
                while (p < line.Length && (line[p] == ' ' || line[p] == '\t'))
                {
                    if (line[p] == ' ')
                    {
                        hasSpace = true;
                        indent++;
                    }
                    else
                    {
                        hasTab = true;
                        indent += 8;
                    }
                    p++;
                }

                bool isBlankOrComment = p >= line.Length || line[p] == '#';
                if (!isBlankOrComment)
                {
                    if (hasSpace && hasTab)
                        return new BounceVerificationResult(false, $"Python py_compile failed (lexical): inconsistent use of tabs and spaces in indentation at line {lineNumber}.");

                    if (expectingIndentBlock)
                    {
                        if (indent <= indentStack[^1])
                            return new BounceVerificationResult(false, $"Python py_compile failed (lexical): expected an indented block at line {lineNumber}.");
                        indentStack.Add(indent);
                        expectingIndentBlock = false;
                    }
                    else if (indent > indentStack[^1])
                    {
                        return new BounceVerificationResult(false, $"Python py_compile failed (lexical): unexpected indent at line {lineNumber}.");
                    }
                    else if (indent < indentStack[^1])
                    {
                        while (indentStack.Count > 1 && indent < indentStack[^1])
                        {
                            indentStack.RemoveAt(indentStack.Count - 1);
                        }
                        if (indent != indentStack[^1])
                            return new BounceVerificationResult(false, $"Python py_compile failed (lexical): unindent does not match any outer indentation level at line {lineNumber}.");
                    }
                }
            }

            bool nextLineContinuation = false;
            char lastNonWsChar = '\0';
            int codeStartIdx = -1;

            for (int ci = 0; ci < line.Length; ci++)
            {
                char ch = line[ci];

                if (inTripleSingle)
                {
                    if (ch == '\\' && ci + 1 < line.Length) { ci++; continue; }
                    if (ch == '\'' && ci + 2 < line.Length && line[ci + 1] == '\'' && line[ci + 2] == '\'')
                    {
                        inTripleSingle = false;
                        ci += 2;
                        lastNonWsChar = '\'';
                    }
                    continue;
                }

                if (inTripleDouble)
                {
                    if (ch == '\\' && ci + 1 < line.Length) { ci++; continue; }
                    if (ch == '"' && ci + 2 < line.Length && line[ci + 1] == '"' && line[ci + 2] == '"')
                    {
                        inTripleDouble = false;
                        ci += 2;
                        lastNonWsChar = '"';
                    }
                    continue;
                }

                if (inSingle)
                {
                    if (ch == '\\')
                    {
                        if (ci + 1 < line.Length) { ci++; }
                        else { nextLineContinuation = true; }
                        continue;
                    }
                    if (ch == '\'')
                    {
                        inSingle = false;
                        lastNonWsChar = '\'';
                    }
                    continue;
                }

                if (inDouble)
                {
                    if (ch == '\\')
                    {
                        if (ci + 1 < line.Length) { ci++; }
                        else { nextLineContinuation = true; }
                        continue;
                    }
                    if (ch == '"')
                    {
                        inDouble = false;
                        lastNonWsChar = '"';
                    }
                    continue;
                }

                if (ch == '#')
                    break;

                if (ch is ' ' or '\t')
                    continue;

                if (codeStartIdx < 0)
                    codeStartIdx = ci;

                if (ch == '\'' && ci + 2 < line.Length && line[ci + 1] == '\'' && line[ci + 2] == '\'')
                {
                    inTripleSingle = true;
                    ci += 2;
                    continue;
                }

                if (ch == '"' && ci + 2 < line.Length && line[ci + 1] == '"' && line[ci + 2] == '"')
                {
                    inTripleDouble = true;
                    ci += 2;
                    continue;
                }

                if (ch == '\'')
                {
                    inSingle = true;
                    continue;
                }

                if (ch == '"')
                {
                    inDouble = true;
                    continue;
                }

                if (ch is '(' or '[' or '{')
                {
                    delimiterStack.Push((ch, lineNumber));
                }
                else if (ch is ')' or ']' or '}')
                {
                    if (delimiterStack.Count == 0)
                        return new BounceVerificationResult(false, $"Python py_compile failed (lexical): unmatched '{ch}' at line {lineNumber}.");

                    var (open, _) = delimiterStack.Pop();
                    bool matches = (open == '(' && ch == ')') || (open == '[' && ch == ']') || (open == '{' && ch == '}');
                    if (!matches)
                        return new BounceVerificationResult(false, $"Python py_compile failed (lexical): closing '{ch}' does not match '{open}' at line {lineNumber}.");
                }

                if (ch == '\\')
                {
                    if (ci != line.Length - 1)
                        return new BounceVerificationResult(false, $"Python py_compile failed (lexical): unexpected character after line continuation character at line {lineNumber}.");

                    nextLineContinuation = true;
                    lastNonWsChar = '\\';
                    continue;
                }

                lastNonWsChar = ch;
            }

            if ((inSingle || inDouble) && !nextLineContinuation)
                return new BounceVerificationResult(false, $"Python py_compile failed (lexical): unterminated string literal at line {lineNumber}.");

            lineContinuation = nextLineContinuation;

            if (!inTripleSingle && !inTripleDouble && !inSingle && !inDouble && delimiterStack.Count == 0 && codeStartIdx >= 0)
            {
                if (lastNonWsChar == ':')
                {
                    expectingIndentBlock = true;
                }
                else if (StartsWithPythonBlockKeyword(line.AsSpan(codeStartIdx)) && lastNonWsChar != '\\')
                {
                    // Multi-line header only valid if brackets were left open (already checked Count == 0)
                    // or an inline suite follows ':' on the same line
                    if (!ContainsColonOutsideQuotes(line.AsSpan(codeStartIdx)))
                        return new BounceVerificationResult(false, $"Python py_compile failed (lexical): expected ':' at end of block statement on line {lineNumber}.");
                }
            }
        }

        if (inTripleSingle || inTripleDouble)
            return new BounceVerificationResult(false, "Python py_compile failed (lexical): unterminated triple-quoted string literal.");

        if (inSingle || inDouble)
            return new BounceVerificationResult(false, $"Python py_compile failed (lexical): unterminated string literal at line {lineNumber}.");

        if (lineContinuation)
            return new BounceVerificationResult(false, $"Python py_compile failed (lexical): unexpected EOF after line continuation at line {lineNumber}.");

        if (delimiterStack.Count > 0)
        {
            var (open, openLine) = delimiterStack.Peek();
            return new BounceVerificationResult(false, $"Python py_compile failed (lexical): '{open}' at line {openLine} was never closed.");
        }

        if (expectingIndentBlock)
            return new BounceVerificationResult(false, $"Python py_compile failed (lexical): expected an indented block after line {lineNumber}.");

        return new BounceVerificationResult(true, null);
    }

    private static bool StartsWithPythonBlockKeyword(ReadOnlySpan<char> trimmed)
    {
        return MatchesKeyword(trimmed, "def") ||
               MatchesKeyword(trimmed, "class") ||
               MatchesKeyword(trimmed, "if") ||
               MatchesKeyword(trimmed, "elif") ||
               MatchesKeyword(trimmed, "else") ||
               MatchesKeyword(trimmed, "for") ||
               MatchesKeyword(trimmed, "while") ||
               MatchesKeyword(trimmed, "try") ||
               MatchesKeyword(trimmed, "except") ||
               MatchesKeyword(trimmed, "finally") ||
               MatchesKeyword(trimmed, "with") ||
               MatchesKeyword(trimmed, "async");
    }

    private static bool MatchesKeyword(ReadOnlySpan<char> span, string kw)
    {
        if (!span.StartsWith(kw, StringComparison.Ordinal))
            return false;
        if (span.Length == kw.Length)
            return true;
        char next = span[kw.Length];
        return next is ' ' or '\t' or '(' or ':';
    }

    private static bool ContainsColonOutsideQuotes(ReadOnlySpan<char> span)
    {
        bool inS = false, inD = false;
        for (int i = 0; i < span.Length; i++)
        {
            char c = span[i];
            if (inS) { if (c == '\\') i++; else if (c == '\'') inS = false; continue; }
            if (inD) { if (c == '\\') i++; else if (c == '"') inD = false; continue; }
            if (c == '#') break;
            if (c == '\'') inS = true;
            else if (c == '"') inD = true;
            else if (c == ':') return true;
        }
        return false;
    }

    private async Task<BounceVerificationResult> RunHarnessAsync(
        string flag,
        string code,
        string ext,
        string[]? extraArgs,
        CancellationToken ct)
    {
        string tempDir = CreateIsolatedTempDir();
        string tmp = Path.Combine(tempDir, $"verify{ext}");
        try
        {
            await File.WriteAllTextAsync(tmp, code, Encoding.UTF8, ct).ConfigureAwait(false);
            var args = new List<string> { flag, tmp };
            if (extraArgs != null) args.AddRange(extraArgs);

            var res = await RunSandboxedAsync(_harnessExePath!, args, tempDir, _harnessTimeout, ct).ConfigureAwait(false);
            if (res.TimedOut)
            {
                return new BounceVerificationResult(false, $"Compiler verification timed out after {_harnessTimeout.TotalMilliseconds:F0} ms.");
            }
            if (res.ExitCode == 0)
                return new BounceVerificationResult(true, null);

            string detail = res.StandardOutput;
            try
            {
                var diagPayload = JsonSerializer.Deserialize(res.StandardOutput, ServeJsonContext.Default.HarnessDiagnosticPayload);
                if (diagPayload != null && !string.IsNullOrWhiteSpace(diagPayload.Error))
                {
                    detail = string.IsNullOrWhiteSpace(diagPayload.DiagnosticId)
                        ? diagPayload.Error
                        : $"{diagPayload.DiagnosticId}: {diagPayload.Error}";
                }
            }
            catch (JsonException ex)
            {
                GlacierDiagnostics.LogDebug($"[PeerBounceService] Harness diagnostic output was not JSON: {ex.Message}");
                if (string.IsNullOrWhiteSpace(detail)) detail = res.StandardError;
            }
            if (string.IsNullOrWhiteSpace(detail)) detail = "Compiler verification failed.";
            if (detail.Length > 500) detail = detail[..500];
            return new BounceVerificationResult(false, detail.Trim());
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            return new BounceVerificationResult(false, $"Failed to launch compiler verification harness '{_harnessExePath}': {ex.Message}");
        }
        finally
        {
            CleanupTempDir(tempDir);
        }
    }

    private async Task<SandboxExecutionResult> RunSandboxedAsync(
        string executable,
        IReadOnlyList<string> args,
        string workingDirectory,
        TimeSpan timeout,
        CancellationToken ct,
        bool? enableNamespaceIsolationOverride = null)
    {
        var options = new SandboxOptions
        {
            ExecutablePath = executable,
            Arguments = args,
            WorkingDirectory = workingDirectory,
            Timeout = timeout,
            MemoryLimitBytes = _memoryLimitBytes,
            AllowNetwork = _allowNetwork,
            MaxProcesses = _maxProcesses,
            EnableNamespaceIsolation = enableNamespaceIsolationOverride ?? _enableNamespaceIsolation
        };

        var result = await _sandbox.ExecuteAsync(options, ct).ConfigureAwait(false);
        LastIsolationMechanism = result.IsolationMechanism;
        LastStartupLatency = result.StartupLatency;
        return result;
    }

    private static string CreateIsolatedTempDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"glacier_bounce_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void CleanupTempDir(string dir)
    {
        try
        {
            if (Directory.Exists(dir))
            {
                Directory.Delete(dir, recursive: true);
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            GlacierDiagnostics.LogDebug($"[PeerBounceService] Temp directory cleanup skipped for '{dir}': {ex.Message}");
        }
    }
}

/// <summary>
/// Compatibility bounce verifier delegating all process execution to <see cref="SandboxedProcessBounceVerifier"/>
/// (zero unsandboxed <c>Process.Start</c> calls).
/// </summary>
public sealed class LocalProcessBounceVerifier : IBounceVerifier
{
    private readonly SandboxedProcessBounceVerifier _inner;

    public LocalProcessBounceVerifier(string? harnessExePath, string pythonExe = "python", string nodeExe = "node")
        : this(harnessExePath, sandboxExecutor: null, pythonExe, nodeExe)
    {
    }

    public LocalProcessBounceVerifier(
        string? harnessExePath,
        ISandboxExecutor? sandboxExecutor,
        string pythonExe = "python",
        string nodeExe = "node")
    {
        _inner = new SandboxedProcessBounceVerifier(harnessExePath, sandboxExecutor, pythonExe, nodeExe);
    }

    public ISandboxExecutor SandboxExecutor => _inner.SandboxExecutor;
    public string? LastIsolationMechanism => _inner.LastIsolationMechanism;
    public TimeSpan? LastStartupLatency => _inner.LastStartupLatency;

    public static string? ExtractCodeBlock(string text, params string[] langs)
        => SandboxedProcessBounceVerifier.ExtractCodeBlock(text, langs);

    public Task<BounceVerificationResult> VerifyAsync(
        BounceTaskKind kind,
        string promptText,
        string completionText,
        CancellationToken ct)
        => _inner.VerifyAsync(kind, promptText, completionText, ct);
}

/// <summary>
/// Compiler- and peer-critic-backed auto-repair completion router.
/// </summary>
public sealed partial class PeerBounceService
{
    [GeneratedRegex(@"\bdef\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(")]
    private static partial Regex PythonDefRegex();

    [GeneratedRegex(@"<think>.*?</think>\s*", RegexOptions.Singleline)]
    private static partial Regex ThinkTagRegex();

    private readonly HttpClient _http;
    private readonly IBounceVerifier _verifier;
    private readonly string _primaryEndpoint;
    private readonly string? _criticEndpoint;
    private readonly bool _compilerBounceEnabled;
    private readonly int _maxRepairs;
    private readonly TimeSpan _criticTimeout;

    private string? _cachedPrimaryModel;
    private string? _cachedCriticModel;

    private long _totalRequests;
    private long _bypassedRequests;
    private long _criticAssistedRequests;
    private long _criticFallbacks;
    private long _verifierChecks;
    private long _verifierFirstPassOk;
    private long _repairAttempts;
    private long _repairsSucceeded;

    public PeerBounceService(
        HttpClient httpClient,
        IBounceVerifier verifier,
        string primaryEndpoint,
        string? criticEndpoint = null,
        bool compilerBounceEnabled = true,
        int maxRepairs = 2,
        TimeSpan? criticTimeout = null)
    {
        ArgumentNullException.ThrowIfNull(httpClient);
        ArgumentNullException.ThrowIfNull(verifier);
        ArgumentException.ThrowIfNullOrWhiteSpace(primaryEndpoint);

        _http = httpClient;
        _verifier = verifier;
        _primaryEndpoint = primaryEndpoint.TrimEnd('/');
        _criticEndpoint = string.IsNullOrWhiteSpace(criticEndpoint) ? null : criticEndpoint.TrimEnd('/');
        _compilerBounceEnabled = compilerBounceEnabled;
        _maxRepairs = Math.Clamp(maxRepairs, 0, 3);
        _criticTimeout = criticTimeout ?? TimeSpan.FromSeconds(18);
    }

    public BounceStatusResponse GetStatus() => new()
    {
        PrimaryEndpoint = _primaryEndpoint,
        CriticEndpoint = _criticEndpoint,
        CompilerBounceEnabled = _compilerBounceEnabled,
        MaxRepairs = _maxRepairs,
        TotalRequests = Interlocked.Read(ref _totalRequests),
        BypassedRequests = Interlocked.Read(ref _bypassedRequests),
        CriticAssistedRequests = Interlocked.Read(ref _criticAssistedRequests),
        CriticFallbacks = Interlocked.Read(ref _criticFallbacks),
        VerifierChecks = Interlocked.Read(ref _verifierChecks),
        VerifierFirstPassOk = Interlocked.Read(ref _verifierFirstPassOk),
        RepairAttempts = Interlocked.Read(ref _repairAttempts),
        RepairsSucceeded = Interlocked.Read(ref _repairsSucceeded),
    };

    public static BounceTaskKind ClassifyRequest(IReadOnlyList<ChatMessage> messages, int maxTokens)
    {
        ArgumentNullException.ThrowIfNull(messages);

        string userText = "";
        var fullSb = new StringBuilder();
        foreach (var m in messages)
        {
            fullSb.Append('\n').Append(m.Content);
            if (string.Equals(m.Role, "user", StringComparison.OrdinalIgnoreCase))
                userText = m.Content ?? "";
        }

        string lower = userText.ToLowerInvariant();
        string fullLower = fullSb.ToString().ToLowerInvariant();

        // Bypass direct retrieval / sleep-memory / NIAH haystack prompts (zero latency overhead)
        if (fullLower.Contains("consolidated episodic memory", StringComparison.Ordinal) ||
            fullLower.Contains("sleep memory", StringComparison.Ordinal) ||
            fullLower.Contains("recall the exact", StringComparison.Ordinal) ||
            (userText.Length > 3200 && !lower.Contains("```", StringComparison.Ordinal) && !lower.Contains("csharp", StringComparison.Ordinal) && !lower.Contains("hlsl", StringComparison.Ordinal) && !lower.Contains("python", StringComparison.Ordinal) && !lower.Contains("html", StringComparison.Ordinal)))
        {
            return BounceTaskKind.Bypass;
        }

        if (userText.Contains("<tool_call>", StringComparison.OrdinalIgnoreCase) || lower.Contains("tool_call", StringComparison.Ordinal))
            return BounceTaskKind.ToolCall;

        if (lower.Contains("hlsl", StringComparison.Ordinal) || lower.Contains("cs_6_6", StringComparison.Ordinal) || lower.Contains("groupshared", StringComparison.Ordinal) || lower.Contains("numthreads", StringComparison.Ordinal) || lower.Contains("compute shader", StringComparison.Ordinal))
            return BounceTaskKind.Hlsl;

        if (lower.Contains("csharp", StringComparison.Ordinal) || lower.Contains("c#", StringComparison.Ordinal) || lower.Contains(".net 10", StringComparison.Ordinal) || lower.Contains("vector256", StringComparison.Ordinal) || lower.Contains("roslyn", StringComparison.Ordinal) || lower.Contains("readonlyspan", StringComparison.Ordinal))
            return BounceTaskKind.CSharp;

        if (lower.Contains("single-file html", StringComparison.Ordinal) || lower.Contains("html5", StringComparison.Ordinal) || lower.Contains("<!doctype html", StringComparison.Ordinal) || (lower.Contains("html", StringComparison.Ordinal) && lower.Contains("javascript", StringComparison.Ordinal)))
            return BounceTaskKind.HtmlJs;

        if (lower.Contains("python", StringComparison.Ordinal) || PythonDefRegex().IsMatch(userText))
            return BounceTaskKind.Python;

        if (userText.Contains("FINAL_ANSWER:", StringComparison.Ordinal))
            return BounceTaskKind.FinalAnswer;

        if (maxTokens <= 96)
            return BounceTaskKind.Bypass;

        return BounceTaskKind.General;
    }

    public async Task<BounceCompletionResult> CompleteAsync(
        IReadOnlyList<ChatMessage> messages,
        string? requestedModel,
        double temperature,
        int maxTokens,
        CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(messages);

        ChatMessage[] messageArray = messages as ChatMessage[] ?? new List<ChatMessage>(messages).ToArray();

        Interlocked.Increment(ref _totalRequests);
        string primaryModel = await ResolveModelIdAsync(_primaryEndpoint, isCritic: false, requestedModel, ct).ConfigureAwait(false);
        var kind = ClassifyRequest(messageArray, maxTokens);

        if (kind == BounceTaskKind.Bypass)
        {
            Interlocked.Increment(ref _bypassedRequests);
            var direct = await CallEndpointAsync(_primaryEndpoint, primaryModel, messageArray, temperature, maxTokens, disableThinking: false, timeout: null, ct).ConfigureAwait(false);
            return new BounceCompletionResult(direct.Content, direct.FinishReason, primaryModel, direct.PromptTokens, direct.CompletionTokens, 0, false, true);
        }

        string userPrompt = "";
        for (int i = messageArray.Length - 1; i >= 0; i--)
        {
            if (string.Equals(messageArray[i].Role, "user", StringComparison.OrdinalIgnoreCase))
            {
                userPrompt = messageArray[i].Content ?? "";
                break;
            }
        }

        // Stage 1: Optional 35B Peer Architect Contract Brief (compact, graceful offline fallback)
        bool criticAssisted = false;
        string? peerBrief = null;
        if (_criticEndpoint != null && ShouldConsultCritic(kind, userPrompt))
        {
            peerBrief = await TryFetchCriticBriefAsync(kind, userPrompt, ct).ConfigureAwait(false);
            if (!string.IsNullOrWhiteSpace(peerBrief))
            {
                criticAssisted = true;
                Interlocked.Increment(ref _criticAssistedRequests);
            }
        }

        var workingMessages = BuildPrimaryMessages(messageArray, peerBrief);
        var primaryResp = await CallEndpointAsync(_primaryEndpoint, primaryModel, workingMessages, temperature, maxTokens, disableThinking: false, timeout: null, ct).ConfigureAwait(false);
        int totalPromptTokens = primaryResp.PromptTokens;
        int totalCompletionTokens = primaryResp.CompletionTokens;
        string currentText = primaryResp.Content;
        string currentFinish = primaryResp.FinishReason;

        if (!_compilerBounceEnabled || _maxRepairs == 0 || kind == BounceTaskKind.General)
        {
            return new BounceCompletionResult(currentText, currentFinish, primaryModel, totalPromptTokens, totalCompletionTokens, 0, criticAssisted, false);
        }

        // Stage 2: Local Generic Verifier + Up to _maxRepairs Auto-Repair Turns
        int effectiveMaxRepairs = (kind == BounceTaskKind.HtmlJs && (maxTokens >= 1500 || messageArray.Length > 2))
            ? Math.Min(1, _maxRepairs)
            : _maxRepairs;
        int repairsUsed = 0;
        for (int attempt = 0; attempt <= effectiveMaxRepairs; attempt++)
        {
            Interlocked.Increment(ref _verifierChecks);
            var verify = await _verifier.VerifyAsync(kind, userPrompt, currentText, ct).ConfigureAwait(false);
            if (verify.Success)
            {
                if (attempt == 0)
                    Interlocked.Increment(ref _verifierFirstPassOk);
                else
                    Interlocked.Increment(ref _repairsSucceeded);
                break;
            }

            if (attempt == effectiveMaxRepairs)
                break;

            repairsUsed++;
            Interlocked.Increment(ref _repairAttempts);

            string? criticHint = null;
            if (attempt == 1 && _criticEndpoint != null)
            {
                criticHint = await TryFetchCriticRepairHintAsync(kind, verify.ErrorMessage ?? "Verification failed", ct).ConfigureAwait(false);
            }

            var repairPrompt = new StringBuilder();
            repairPrompt.AppendLine($"[LOCAL VERIFIER ERROR] {verify.ErrorMessage}");
            if (!string.IsNullOrWhiteSpace(criticHint))
                repairPrompt.AppendLine($"[ARCHITECT HINT] {criticHint}");
            repairPrompt.AppendLine("Fix ONLY this error and return the complete, corrected, concise final response in the exact required format.");

            var repairMessages = new List<ChatMessage>(workingMessages)
            {
                new() { Role = "assistant", Content = currentText.Length > 2400 ? currentText[..2400] : currentText },
                new() { Role = "user", Content = repairPrompt.ToString() }
            };

            try
            {
                var repaired = await CallEndpointAsync(_primaryEndpoint, primaryModel, repairMessages.ToArray(), 0.0, maxTokens, disableThinking: false, timeout: null, ct).ConfigureAwait(false);
                totalPromptTokens += repaired.PromptTokens;
                totalCompletionTokens += repaired.CompletionTokens;
                if (!string.IsNullOrWhiteSpace(repaired.Content))
                {
                    currentText = repaired.Content;
                    currentFinish = repaired.FinishReason;
                }
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                throw;
            }
            catch
            {
                // Keep best primary response gracefully if a repair turn overflows context or times out.
                break;
            }
        }

        return new BounceCompletionResult(currentText, currentFinish, primaryModel, totalPromptTokens, totalCompletionTokens, repairsUsed, criticAssisted, false);
    }

    private static bool ShouldConsultCritic(BounceTaskKind kind, string userPrompt)
    {
        return kind switch
        {
            BounceTaskKind.CSharp => true,
            BounceTaskKind.Hlsl => true,
            BounceTaskKind.HtmlJs => true,
            BounceTaskKind.Python => userPrompt.Length >= 100,
            BounceTaskKind.FinalAnswer => userPrompt.Length >= 120,
            _ => false
        };
    }

    private async Task<string?> TryFetchCriticBriefAsync(BounceTaskKind kind, string userPrompt, CancellationToken ct)
    {
        try
        {
            string criticModel = await ResolveModelIdAsync(_criticEndpoint!, isCritic: true, null, ct).ConfigureAwait(false);
            string sysPrompt = kind == BounceTaskKind.FinalAnswer
                ? "You are a concise quantitative reviewer. Give at most 4 short bullet points outlining the exact equation/method and pitfalls to avoid. No LaTeX bloat, max 120 tokens."
                : "You are a principal systems architect. List at most 5 terse bullet points of critical compiler rules, API signatures, and edge cases to satisfy for this task. Do NOT write full code. Max 150 tokens.";

            var msgs = new ChatMessage[]
            {
                new() { Role = "system", Content = sysPrompt },
                new() { Role = "user", Content = userPrompt.Length > 2500 ? userPrompt[..2500] : userPrompt }
            };

            var resp = await CallEndpointAsync(_criticEndpoint!, criticModel, msgs, 0.0, 180, disableThinking: true, timeout: _criticTimeout, ct).ConfigureAwait(false);
            string clean = StripThinkTags(resp.Content);
            return string.IsNullOrWhiteSpace(clean) ? null : clean;
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            throw;
        }
        catch
        {
            Interlocked.Increment(ref _criticFallbacks);
            return null;
        }
    }

    private async Task<string?> TryFetchCriticRepairHintAsync(BounceTaskKind kind, string errorMessage, CancellationToken ct)
    {
        try
        {
            string criticModel = await ResolveModelIdAsync(_criticEndpoint!, isCritic: true, null, ct).ConfigureAwait(false);
            var msgs = new ChatMessage[]
            {
                new() { Role = "system", Content = "In 2 concise sentences, explain the exact compiler/syntax fix needed for this error. No filler." },
                new() { Role = "user", Content = $"Task kind: {kind}\nError: {errorMessage}" }
            };
            var resp = await CallEndpointAsync(_criticEndpoint!, criticModel, msgs, 0.0, 100, disableThinking: true, timeout: _criticTimeout, ct).ConfigureAwait(false);
            string clean = StripThinkTags(resp.Content);
            return string.IsNullOrWhiteSpace(clean) ? null : clean;
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            throw;
        }
        catch
        {
            Interlocked.Increment(ref _criticFallbacks);
            return null;
        }
    }

    private static ChatMessage[] BuildPrimaryMessages(ChatMessage[] original, string? peerBrief)
    {
        if (string.IsNullOrWhiteSpace(peerBrief))
            return original;

        var list = new List<ChatMessage>(original.Length + 1);
        string note = $"\n\n[Peer Architect Guardrails — apply concisely]:\n{peerBrief}";
        bool mergedIntoSystem = false;
        foreach (var m in original)
        {
            if (!mergedIntoSystem && string.Equals(m.Role, "system", StringComparison.OrdinalIgnoreCase))
            {
                list.Add(new ChatMessage { Role = m.Role, Content = m.Content + note });
                mergedIntoSystem = true;
            }
            else
            {
                list.Add(m);
            }
        }
        if (!mergedIntoSystem)
        {
            list.Insert(0, new ChatMessage
            {
                Role = "system",
                Content = "Follow these concise engineering guardrails carefully:" + note
            });
        }
        return list.ToArray();
    }

    private static string StripThinkTags(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return "";
        return ThinkTagRegex().Replace(text, "").Trim();
    }

    private async Task<string> ResolveModelIdAsync(string endpoint, bool isCritic, string? fallback, CancellationToken ct)
    {
        string? cached = isCritic
            ? Volatile.Read(ref _cachedCriticModel)
            : Volatile.Read(ref _cachedPrimaryModel);
        if (!string.IsNullOrWhiteSpace(cached))
            return cached;

        try
        {
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            cts.CancelAfter(TimeSpan.FromSeconds(5));
            string json = await _http.GetStringAsync($"{endpoint}/v1/models", cts.Token).ConfigureAwait(false);
            var models = JsonSerializer.Deserialize(json, ServeJsonContext.Default.ModelListResponse);
            if (models?.Data is { Count: > 0 })
            {
                string? id = models.Data[0].Id;
                if (!string.IsNullOrWhiteSpace(id))
                {
                    if (isCritic) Volatile.Write(ref _cachedCriticModel, id);
                    else Volatile.Write(ref _cachedPrimaryModel, id);
                    return id;
                }
            }
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            GlacierDiagnostics.LogDebug($"[PeerBounceService] Model resolution fallback for '{endpoint}': {ex.Message}");
        }

        return string.IsNullOrWhiteSpace(fallback) ? "glacier-primary" : fallback;
    }

    private async Task<(string Content, string FinishReason, int PromptTokens, int CompletionTokens)> CallEndpointAsync(
        string endpoint,
        string modelId,
        ChatMessage[] messages,
        double temperature,
        int maxTokens,
        bool disableThinking,
        TimeSpan? timeout,
        CancellationToken ct)
    {
        var payload = new BounceChatPayload
        {
            Model = modelId,
            Messages = messages,
            Temperature = temperature,
            MaxTokens = maxTokens > 0 ? maxTokens : 1024,
            Stream = false,
            ChatTemplateKwargs = disableThinking ? new ChatTemplateKwargs { EnableThinking = false } : null
        };

        string reqJson = JsonSerializer.Serialize(payload, ServeJsonContext.Default.BounceChatPayload);
        using var req = new HttpRequestMessage(HttpMethod.Post, $"{endpoint}/v1/chat/completions")
        {
            Content = new StringContent(reqJson, Encoding.UTF8, "application/json")
        };

        using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        if (timeout.HasValue) cts.CancelAfter(timeout.Value);

        using var resp = await _http.SendAsync(req, cts.Token).ConfigureAwait(false);
        string body = await resp.Content.ReadAsStringAsync(cts.Token).ConfigureAwait(false);
        resp.EnsureSuccessStatusCode();

        var completion = JsonSerializer.Deserialize(body, ServeJsonContext.Default.ChatCompletionResponse);
        string content = "";
        string finish = "stop";
        int pTok = 0, cTok = 0;

        if (completion?.Choices is { Count: > 0 })
        {
            var ch = completion.Choices[0];
            content = ch.Message?.Content ?? "";
            finish = string.IsNullOrEmpty(ch.FinishReason) ? "stop" : ch.FinishReason;
        }
        if (completion?.Usage != null)
        {
            pTok = completion.Usage.PromptTokens;
            cTok = completion.Usage.CompletionTokens;
        }

        return (content, finish, pTok, cTok);
    }
}

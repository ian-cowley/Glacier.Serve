using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Agent.Sandbox;
using Glacier.Serve.Daemons;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Inference.Bounce;
using Glacier.Serve.Inference.Http;
using Glacier.Serve.Serialization;
using Glacier.Serve.Server;
using Xunit;

namespace Glacier.Serve.Tests;

public sealed class SleepMemoryAndPeerBounceTests
{
    private static int GetAvailablePort()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        return ((IPEndPoint)listener.LocalEndpoint).Port;
    }

    private static string CreateMinimalGgufFile(bool includeAsciiTokens = false)
    {
        string path = Path.Combine(Path.GetTempPath(), $"glacier_serve_m5a_{Guid.NewGuid():N}.gguf");
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: true);

        static void WriteGgufString(BinaryWriter writer, string value)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(value);
            writer.Write((ulong)bytes.Length);
            writer.Write(bytes);
        }

        static void WriteKvUInt32(BinaryWriter writer, string key, uint value)
        {
            WriteGgufString(writer, key);
            writer.Write(4u); // GGUF_TYPE_UINT32
            writer.Write(value);
        }

        static void WriteKvString(BinaryWriter writer, string key, string value)
        {
            WriteGgufString(writer, key);
            writer.Write(8u); // GGUF_TYPE_STRING
            WriteGgufString(writer, value);
        }

        static void WriteKvStringArray(BinaryWriter writer, string key, string[] values)
        {
            WriteGgufString(writer, key);
            writer.Write(9u); // GGUF_TYPE_ARRAY
            writer.Write(8u); // element type = STRING
            writer.Write((ulong)values.Length);
            foreach (string v in values)
                WriteGgufString(writer, v);
        }

        static void WriteTensorInfo(BinaryWriter writer, string name, ulong[] dims, ulong offset)
        {
            WriteGgufString(writer, name);
            writer.Write((uint)dims.Length);
            foreach (ulong d in dims) writer.Write(d);
            writer.Write(0u); // GGML_TYPE_F32
            writer.Write(offset);
        }

        string[] tokens;
        if (includeAsciiTokens)
        {
            var tokenList = new List<string>(128) { "<s>", "</s>", "hello", "world", "<think>", "</think>", "\n", "\r", "\t" };
            for (int c = 32; c < 127 && tokenList.Count < 128; c++)
            {
                tokenList.Add(((char)c).ToString());
            }
            while (tokenList.Count < 128)
            {
                tokenList.Add($"<unk_{tokenList.Count}>");
            }
            tokens = tokenList.ToArray();
        }
        else
        {
            tokens = ["<s>", "</s>", "hello", "world", "<think>", "</think>"];
        }

        const int dim = 4;
        const int ffn = 4;
        int vocab = tokens.Length;
        var tensors = new (string Name, ulong[] Dims, int FloatCount)[]
        {
            ("token_embd.weight", [(ulong)dim, (ulong)vocab], dim * vocab),
            ("output_norm.weight", [(ulong)dim], dim),
            ("output.weight", [(ulong)dim, (ulong)vocab], dim * vocab),
            ("blk.0.attn_norm.weight", [(ulong)dim], dim),
            ("blk.0.attn_q.weight", [(ulong)dim, (ulong)dim], dim * dim),
            ("blk.0.attn_k.weight", [(ulong)dim, (ulong)dim], dim * dim),
            ("blk.0.attn_v.weight", [(ulong)dim, (ulong)dim], dim * dim),
            ("blk.0.attn_output.weight", [(ulong)dim, (ulong)dim], dim * dim),
            ("blk.0.ffn_norm.weight", [(ulong)dim], dim),
            ("blk.0.ffn_gate.weight", [(ulong)dim, (ulong)ffn], dim * ffn),
            ("blk.0.ffn_up.weight", [(ulong)dim, (ulong)ffn], dim * ffn),
            ("blk.0.ffn_down.weight", [(ulong)ffn, (ulong)dim], ffn * dim),
        };

        bw.Write(0x46554747u); // "GGUF"
        bw.Write(3u);          // version 3
        bw.Write((ulong)tensors.Length); // tensor_count
        bw.Write(9UL);         // metadata_kv_count

        WriteKvString(bw, "general.architecture", "llama");
        WriteKvUInt32(bw, "llama.block_count", 1u);
        WriteKvUInt32(bw, "llama.embedding_length", (uint)dim);
        WriteKvUInt32(bw, "llama.feed_forward_length", (uint)ffn);
        WriteKvUInt32(bw, "llama.attention.head_count", 1u);
        WriteKvUInt32(bw, "llama.attention.head_count_kv", 1u);
        WriteKvUInt32(bw, "tokenizer.ggml.bos_token_id", 0u);
        WriteKvUInt32(bw, "tokenizer.ggml.eos_token_id", 1u);
        WriteKvStringArray(bw, "tokenizer.ggml.tokens", tokens);

        ulong curOffset = 0;
        foreach (var t in tensors)
        {
            WriteTensorInfo(bw, t.Name, t.Dims, curOffset);
            ulong byteSize = (ulong)t.FloatCount * sizeof(float);
            ulong aligned = (byteSize + 31UL) & ~31UL;
            curOffset += aligned;
        }
        bw.Flush();

        long rem = fs.Position % 32;
        if (rem != 0)
        {
            fs.Write(new byte[32 - rem]);
        }

        foreach (var t in tensors)
        {
            byte[] data = new byte[((t.FloatCount * sizeof(float)) + 31) & ~31];
            fs.Write(data);
        }
        return path;
    }

    [Fact]
    public async Task SleepMemoryService_ConsolidatesLongPrompt_RecordsTurns_AndTriggersSleep()
    {
        await using var sleep = new SleepMemoryService(
            tokenBudget: 40,
            idleThreshold: TimeSpan.FromMinutes(10),
            pollingInterval: TimeSpan.FromMilliseconds(100));

        string shortPrompt = "Hello Glacier";
        string shortOut = await sleep.ConsolidatePromptAsync(shortPrompt);
        Assert.Equal(shortPrompt, shortOut);

        var sb = new StringBuilder();
        for (int i = 0; i < 12; i++)
        {
            sb.AppendLine($"FACT_{i:D2}: Subsystem MODULE_{i:D2} operates at {100 + i} MHz with CRC_TOKEN_{i:D2}.");
            sb.AppendLine();
        }
        sb.Append("What frequency does MODULE_07 operate at?");

        string consolidated = await sleep.ConsolidatePromptAsync(sb.ToString());
        Assert.Contains("Consolidated Context:", consolidated, StringComparison.Ordinal);
        Assert.Contains("Question:", consolidated, StringComparison.Ordinal);
        Assert.True(consolidated.Length < sb.Length, "Consolidated prompt should be shorter than raw haystack.");

        sleep.RecordTurnCompletion("What is MODULE_07?", "<think>internal chain of thought</think>MODULE_07 operates at 107 MHz.");

        var consolidationResult = await sleep.TriggerSleepAsync();
        Assert.NotNull(consolidationResult);

        var status = sleep.GetStatus();
        Assert.True(status.Enabled);
        Assert.Equal(40, status.TokenBudget);
        Assert.True(status.RawChunks >= 12);
        Assert.True(status.RetainedTraces >= 12);
        Assert.True(status.SleepCycles >= 1);
        Assert.True(status.DreamsConsolidated >= 1);
        Assert.True(status.LastSelectedTraces >= 1);
        Assert.True(status.LastPackedTokens >= 1);
    }

    [Fact]
    public async Task MapSleepMemory_HandlesDisabledAndEnabledEndpoints_WhilePreservingLegacyV1SleepRoutes()
    {
        int disabledPort = GetAvailablePort();
        using (var disabledApp = GlacierServeApp.CreateBuilder().UsePort(disabledPort).Build())
        {
            disabledApp.MapSleepMemory(sleepMemory: null);
            await disabledApp.StartAsync();

            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{disabledPort}") };
            using var statusResp = await client.GetAsync("/api/sleep/status");
            Assert.Equal(HttpStatusCode.OK, statusResp.StatusCode);
            var disabledStatus = JsonSerializer.Deserialize(
                await statusResp.Content.ReadAsStringAsync(),
                ServeJsonContext.Default.SleepMemoryStatusResponse);
            Assert.NotNull(disabledStatus);
            Assert.False(disabledStatus.Enabled);
            Assert.Equal(0, disabledStatus.TokenBudget);
            Assert.Equal("Disabled", disabledStatus.State);

            using var triggerResp = await client.PostAsync("/api/sleep/trigger", content: null);
            Assert.Equal(HttpStatusCode.BadRequest, triggerResp.StatusCode);
            var triggerPayload = JsonSerializer.Deserialize(
                await triggerResp.Content.ReadAsStringAsync(),
                ServeJsonContext.Default.SleepMemoryStatusResponse);
            Assert.NotNull(triggerPayload);
            Assert.False(triggerPayload.Enabled);
            Assert.Equal("Disabled", triggerPayload.State);

            disabledApp.Stop();
        }

        string ggufPath = CreateMinimalGgufFile();
        try
        {
            await using var sleep = new SleepMemoryService(
                tokenBudget: 80,
                idleThreshold: TimeSpan.FromMinutes(10),
                pollingInterval: TimeSpan.FromMilliseconds(100));
            sleep.RecordTurnCompletion("ALPHA_KEY_42", "ALPHA_VALUE_42 is stored cleanly.");

            using var engine = new ContinuousBatchEngine(ggufPath, totalKvBlocks: 16, maxBatchSize: 2, maxSeqLen: 64, useGpu: false);
            engine.AttachSleepMemory(sleep);
            Assert.Same(sleep, engine.SleepMemory);
            Assert.Same(sleep.Daemon, engine.SleepDaemon);

            int port = GetAvailablePort();
            using var app = GlacierServeApp.CreateBuilder().UsePort(port).Build();
            app.MapInference(engine, "test-qwen", sleep);
            await app.StartAsync();

            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

            // New /api/sleep/status & /api/sleep/trigger
            using var apiTrigger = await client.PostAsync("/api/sleep/trigger", content: null);
            Assert.Equal(HttpStatusCode.OK, apiTrigger.StatusCode);
            var apiStatus = JsonSerializer.Deserialize(
                await apiTrigger.Content.ReadAsStringAsync(),
                ServeJsonContext.Default.SleepMemoryStatusResponse);
            Assert.NotNull(apiStatus);
            Assert.True(apiStatus.Enabled);
            Assert.Equal(80, apiStatus.TokenBudget);
            Assert.True(apiStatus.SleepCycles >= 1);

            using var apiGet = await client.GetAsync("/api/sleep/status");
            Assert.Equal(HttpStatusCode.OK, apiGet.StatusCode);
            string rawApiJson = await apiGet.Content.ReadAsStringAsync();
            using (var doc = JsonDocument.Parse(rawApiJson))
            {
                Assert.True(doc.RootElement.TryGetProperty("token_budget", out _));
                Assert.True(doc.RootElement.TryGetProperty("raw_chunks", out _));
                Assert.True(doc.RootElement.TryGetProperty("retained_traces", out _));
                Assert.True(doc.RootElement.TryGetProperty("sleep_cycles", out _));
                Assert.True(doc.RootElement.TryGetProperty("dreams_consolidated", out _));
            }

            // Legacy /v1/sleep/status & /v1/sleep/trigger (Glacier.Studio compatibility)
            using var v1StatusResp = await client.GetAsync("/v1/sleep/status");
            Assert.Equal(HttpStatusCode.OK, v1StatusResp.StatusCode);
            var v1Status = JsonSerializer.Deserialize(
                await v1StatusResp.Content.ReadAsStringAsync(),
                ServeJsonContext.Default.SleepStatusResponse);
            Assert.NotNull(v1Status);
            Assert.True(v1Status.Enabled);
            Assert.True(v1Status.CompletedCycles >= 1);

            using var v1TriggerResp = await client.PostAsync("/v1/sleep/trigger", content: null);
            Assert.Equal(HttpStatusCode.OK, v1TriggerResp.StatusCode);
            var v1Trigger = JsonSerializer.Deserialize(
                await v1TriggerResp.Content.ReadAsStringAsync(),
                ServeJsonContext.Default.SleepTriggerResponse);
            Assert.NotNull(v1Trigger);
            Assert.Equal("Awake", v1Trigger.State);

            app.Stop();
        }
        finally
        {
            if (File.Exists(ggufPath)) File.Delete(ggufPath);
        }
    }

    [Fact]
    public async Task SandboxedProcessBounceVerifier_UsesOsProcessSandbox_AndCatchesSyntaxErrors()
    {
        var verifier = new SandboxedProcessBounceVerifier(harnessExePath: null);
        Assert.IsType<OsProcessSandbox>(verifier.SandboxExecutor);

        string validPy = "```python\ndef add(a, b):\n    return a + b\n```";
        var okRes = await verifier.VerifyAsync(BounceTaskKind.Python, "write add", validPy, CancellationToken.None);
        Assert.True(okRes.Success, okRes.ErrorMessage);
        Assert.False(string.IsNullOrWhiteSpace(verifier.LastIsolationMechanism));
        Assert.NotNull(verifier.LastStartupLatency);

        string brokenPy = "```python\ndef add(a, b:\n    return a + b\n```";
        var badRes = await verifier.VerifyAsync(BounceTaskKind.Python, "write add", brokenPy, CancellationToken.None);
        Assert.False(badRes.Success);
        Assert.Contains("py_compile failed", badRes.ErrorMessage ?? string.Empty, StringComparison.OrdinalIgnoreCase);

        // LocalProcessBounceVerifier compatibility wrapper delegates through SandboxedProcessBounceVerifier
        var compat = new LocalProcessBounceVerifier(harnessExePath: null);
        Assert.IsType<OsProcessSandbox>(compat.SandboxExecutor);
        var compatRes = await compat.VerifyAsync(BounceTaskKind.Python, "write add", brokenPy, CancellationToken.None);
        Assert.False(compatRes.Success);
        Assert.False(string.IsNullOrWhiteSpace(compat.LastIsolationMechanism));
    }

    [Fact]
    public async Task SandboxedProcessBounceVerifier_HandlesTimeouts_Cancellation_AndMissingOptionalTools()
    {
        var timeoutSandbox = new StubSandboxExecutor(_ => new SandboxExecutionResult
        {
            ExitCode = -1,
            StandardOutput = "",
            StandardError = "",
            StartupLatency = TimeSpan.FromMilliseconds(2),
            TotalDuration = TimeSpan.FromMilliseconds(50),
            TimedOut = true,
            IsolationMechanism = "StubSandbox"
        });

        var verifier = new SandboxedProcessBounceVerifier(harnessExePath: null, sandboxExecutor: timeoutSandbox);
        var res = await verifier.VerifyAsync(BounceTaskKind.Python, "p", "```python\nx = 1\n```", CancellationToken.None);
        Assert.False(res.Success);
        Assert.Contains("timed out", res.ErrorMessage ?? string.Empty, StringComparison.OrdinalIgnoreCase);
        Assert.NotNull(timeoutSandbox.LastOptions);
        Assert.False(timeoutSandbox.LastOptions!.AllowNetwork);
        Assert.True(timeoutSandbox.LastOptions.EnableNamespaceIsolation);
        Assert.Null(timeoutSandbox.LastOptions.MaxProcesses);
        Assert.Null(timeoutSandbox.LastOptions.MemoryLimitBytes);
        Assert.False(Directory.Exists(timeoutSandbox.LastOptions.WorkingDirectory));

        // Missing optional python/node/harness tools fail explicitly with diagnostic error (never silent Success=true)
        var missingTool = new SandboxedProcessBounceVerifier(
            harnessExePath: null,
            pythonExe: "nonexistent_python_binary_xyz_12345",
            nodeExe: "nonexistent_node_binary_xyz_12345");
        var missingPyRes = await missingTool.VerifyAsync(BounceTaskKind.Python, "p", "```python\nx = 1\n```", CancellationToken.None);
        Assert.False(missingPyRes.Success);
        Assert.False(string.IsNullOrWhiteSpace(missingPyRes.ErrorMessage));

        var missingNodeRes = await missingTool.VerifyAsync(
            BounceTaskKind.HtmlJs,
            "html",
            "```html\n<!DOCTYPE html><html><body><script>const a = 1;</script></body></html>\n```",
            CancellationToken.None);
        Assert.False(missingNodeRes.Success);
        Assert.False(string.IsNullOrWhiteSpace(missingNodeRes.ErrorMessage));

        var nullCSharpRes = await missingTool.VerifyAsync(BounceTaskKind.CSharp, "cs", "```csharp\npublic class C {}\n```", CancellationToken.None);
        Assert.False(nullCSharpRes.Success);
        Assert.Contains("not configured", nullCSharpRes.ErrorMessage ?? string.Empty, StringComparison.OrdinalIgnoreCase);

        var nullHlslRes = await missingTool.VerifyAsync(BounceTaskKind.Hlsl, "hlsl", "```hlsl\n[numthreads(1,1,1)] void CSMain() {}\n```", CancellationToken.None);
        Assert.False(nullHlslRes.Success);
        Assert.Contains("not configured", nullHlslRes.ErrorMessage ?? string.Empty, StringComparison.OrdinalIgnoreCase);

        // Caller cancellation propagates OperationCanceledException
        var cancelSandbox = new StubSandboxExecutor(ct => throw new OperationCanceledException(ct));
        var cancelVerifier = new SandboxedProcessBounceVerifier(harnessExePath: null, sandboxExecutor: cancelSandbox);
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            cancelVerifier.VerifyAsync(BounceTaskKind.Python, "p", "```python\nx = 1\n```", cts.Token));
    }

    [Fact]
    public async Task MapPeerBounce_StandaloneAndNonCollidingRoutes_VerifyRepairAndStream()
    {
        int primaryCalls = 0;
        var handler = new StubHttpMessageHandler(async (req, ct) =>
        {
            if (req.Method == HttpMethod.Get && req.RequestUri!.AbsolutePath == "/v1/models")
            {
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent("{\"object\":\"list\",\"data\":[{\"id\":\"mock-primary\"}]}", Encoding.UTF8, "application/json")
                };
            }

            if (req.Method == HttpMethod.Post && req.RequestUri!.AbsolutePath == "/v1/chat/completions")
            {
                string reqBody = await req.Content!.ReadAsStringAsync(ct);
                int callNum = Interlocked.Increment(ref primaryCalls);
                string reply = reqBody.Contains("FINAL_ANSWER:", StringComparison.Ordinal)
                    ? "FINAL_ANSWER: 42"
                    : callNum == 1
                        ? "```python\ndef broken(:\n    pass\n```"
                        : "```python\ndef fixed(x):\n    return x * 2\n```";

                var completion = new ChatCompletionResponse
                {
                    Id = $"mock-{callNum}",
                    Created = 1710000000,
                    Model = "mock-primary",
                    Choices =
                    [
                        new ChatChoice
                        {
                            Index = 0,
                            Message = new ChatMessage { Role = "assistant", Content = reply },
                            FinishReason = "stop"
                        }
                    ],
                    Usage = new ChatUsage { PromptTokens = 12, CompletionTokens = 8, TotalTokens = 20 }
                };

                string json = JsonSerializer.Serialize(completion, ServeJsonContext.Default.ChatCompletionResponse);
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(json, Encoding.UTF8, "application/json")
                };
            }

            return new HttpResponseMessage(HttpStatusCode.NotFound);
        });

        using var outboundHttp = new HttpClient(handler);
        var verifier = new SandboxedProcessBounceVerifier(harnessExePath: null);
        var bounceService = new PeerBounceService(
            outboundHttp,
            verifier,
            primaryEndpoint: "http://mock-primary.local:18080",
            criticEndpoint: null,
            compilerBounceEnabled: true,
            maxRepairs: 2);

        int port = GetAvailablePort();
        using var app = GlacierServeApp.CreateBuilder().UsePort(port).Build();
        app.MapPeerBounce(bounceService, mapOpenAiAndHealthRoutes: true, modelAlias: "Glacier-Bounce-Test");
        await app.StartAsync();

        using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

        using var healthResp = await client.GetAsync("/health");
        Assert.Equal(HttpStatusCode.OK, healthResp.StatusCode);
        Assert.Equal("application/json", healthResp.Content.Headers.ContentType?.MediaType);
        Assert.Equal("{\"status\":\"ok\"}", await healthResp.Content.ReadAsStringAsync());

        using var modelsResp = await client.GetAsync("/v1/models");
        Assert.Equal(HttpStatusCode.OK, modelsResp.StatusCode);
        var models = JsonSerializer.Deserialize(await modelsResp.Content.ReadAsStringAsync(), ServeJsonContext.Default.ModelListResponse);
        Assert.NotNull(models);
        Assert.Contains(models.Data, m => m.Id == "Glacier-Bounce-Test");

        // Non-streaming completion triggers syntax check + auto-repair
        string chatReqJson = JsonSerializer.Serialize(new BounceChatRequest
        {
            Model = "Glacier-Bounce-Test",
            Messages = [new ChatMessage { Role = "user", Content = "Write a Python function fixed(x)." }],
            Stream = false
        }, ServeJsonContext.Default.BounceChatRequest);

        using var chatResp = await client.PostAsync("/v1/chat/completions", new StringContent(chatReqJson, Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.OK, chatResp.StatusCode);
        var chatObj = JsonSerializer.Deserialize(await chatResp.Content.ReadAsStringAsync(), ServeJsonContext.Default.ChatCompletionResponse);
        Assert.NotNull(chatObj);
        Assert.Contains("def fixed(x):", chatObj.Choices[0].Message.Content, StringComparison.Ordinal);

        // Dedicated /api/bounce/chat/completions route also works with streaming SSE
        string streamReqJson = JsonSerializer.Serialize(new BounceChatRequest
        {
            Model = "Glacier-Bounce-Test",
            Messages = [new ChatMessage { Role = "user", Content = "Return FINAL_ANSWER: 42" }],
            Stream = true
        }, ServeJsonContext.Default.BounceChatRequest);

        using var streamResp = await client.PostAsync("/api/bounce/chat/completions", new StringContent(streamReqJson, Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.OK, streamResp.StatusCode);
        Assert.Equal("text/event-stream", streamResp.Content.Headers.ContentType?.MediaType);
        string sseText = await streamResp.Content.ReadAsStringAsync();
        Assert.Contains("data: [DONE]", sseText, StringComparison.Ordinal);

        using var bounceStatusResp = await client.GetAsync("/api/bounce/status");
        Assert.Equal(HttpStatusCode.OK, bounceStatusResp.StatusCode);
        var bounceStatus = JsonSerializer.Deserialize(await bounceStatusResp.Content.ReadAsStringAsync(), ServeJsonContext.Default.BounceStatusResponse);
        Assert.NotNull(bounceStatus);
        Assert.Equal("http://mock-primary.local:18080", bounceStatus.PrimaryEndpoint);
        Assert.Equal(2, bounceStatus.TotalRequests);
        Assert.Equal(1, bounceStatus.RepairAttempts);
        Assert.Equal(1, bounceStatus.RepairsSucceeded);

        app.Stop();
    }

    [Fact]
    public async Task MapPeerBounce_WithMapOpenAiAndHealthRoutesFalse_DoesNotOverwriteOpenAiOrHealthRoutes()
    {
        string ggufPath = CreateMinimalGgufFile();
        try
        {
            using var engine = new ContinuousBatchEngine(ggufPath, totalKvBlocks: 16, maxBatchSize: 2, maxSeqLen: 64, useGpu: false);
            using var outboundHttp = new HttpClient(new StubHttpMessageHandler((_, _) =>
                Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent("{}", Encoding.UTF8, "application/json")
                })));
            var bounceService = new PeerBounceService(outboundHttp, new SandboxedProcessBounceVerifier(null), "http://mock.local:18080");

            int port = GetAvailablePort();
            using var app = GlacierServeApp.CreateBuilder().UsePort(port).Build();
            app.MapInference(engine, "base-inference-model");
            app.MapPeerBounce(bounceService, mapOpenAiAndHealthRoutes: false, modelAlias: "should-not-overwrite");
            await app.StartAsync();

            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

            using var modelsResp = await client.GetAsync("/v1/models");
            Assert.Equal(HttpStatusCode.OK, modelsResp.StatusCode);
            var models = JsonSerializer.Deserialize(await modelsResp.Content.ReadAsStringAsync(), ServeJsonContext.Default.ModelListResponse);
            Assert.NotNull(models);
            Assert.Contains(models.Data, m => m.Id == "base-inference-model");
            Assert.DoesNotContain(models.Data, m => m.Id == "should-not-overwrite");

            using var healthResp = await client.GetAsync("/health");
            Assert.Equal(HttpStatusCode.OK, healthResp.StatusCode);
            var health = JsonSerializer.Deserialize(await healthResp.Content.ReadAsStringAsync(), ServeJsonContext.Default.HealthCheckResponse);
            Assert.NotNull(health);
            Assert.Equal("ContinuousBatchEngine", health.Engine);

            using var bounceStatusResp = await client.GetAsync("/api/bounce/status");
            Assert.Equal(HttpStatusCode.OK, bounceStatusResp.StatusCode);

            app.Stop();
        }
        finally
        {
            if (File.Exists(ggufPath)) File.Delete(ggufPath);
        }
    }

    [Fact]
    public async Task MapPeerBounce_RegisteredBeforeMapInference_RoutesAllInferenceSleepAndBounceEndpoints()
    {
        string ggufPath = CreateMinimalGgufFile(includeAsciiTokens: true);
        try
        {
            await using var sleep = new SleepMemoryService(
                tokenBudget: 80,
                idleThreshold: TimeSpan.FromMinutes(10),
                pollingInterval: TimeSpan.FromMilliseconds(100));
            sleep.RecordTurnCompletion("ORDER_TEST_KEY", "ORDER_TEST_VALUE is preserved across radix splits.");

            using var engine = new ContinuousBatchEngine(ggufPath, totalKvBlocks: 32, maxBatchSize: 2, maxSeqLen: 256, useGpu: false);
            engine.AttachSleepMemory(sleep);

            var mockHandler = new StubHttpMessageHandler((req, _) =>
            {
                if (req.Method == HttpMethod.Get && req.RequestUri!.AbsolutePath == "/v1/models")
                {
                    return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                    {
                        Content = new StringContent("{\"object\":\"list\",\"data\":[{\"id\":\"mock-primary\"}]}", Encoding.UTF8, "application/json")
                    });
                }

                if (req.Method == HttpMethod.Post && req.RequestUri!.AbsolutePath == "/v1/chat/completions")
                {
                    var completion = new ChatCompletionResponse
                    {
                        Id = "mock-bounce-reply",
                        Created = 1710000000,
                        Model = "mock-primary",
                        Choices =
                        [
                            new ChatChoice
                            {
                                Index = 0,
                                Message = new ChatMessage { Role = "assistant", Content = "Bounce OK" },
                                FinishReason = "stop"
                            }
                        ],
                        Usage = new ChatUsage { PromptTokens = 5, CompletionTokens = 2, TotalTokens = 7 }
                    };
                    string json = JsonSerializer.Serialize(completion, ServeJsonContext.Default.ChatCompletionResponse);
                    return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                    {
                        Content = new StringContent(json, Encoding.UTF8, "application/json")
                    });
                }

                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
            });

            using var outboundHttp = new HttpClient(mockHandler);
            var bounceService = new PeerBounceService(outboundHttp, new SandboxedProcessBounceVerifier(null), "http://mock.local:18080");

            int port = GetAvailablePort();
            using var app = GlacierServeApp.CreateBuilder().UsePort(port).Build();
            // Intentionally register MapPeerBounce BEFORE MapInference to exercise RadixTreeRouter edge splitting.
            app.MapPeerBounce(bounceService, mapOpenAiAndHealthRoutes: false, modelAlias: "should-not-overwrite");
            app.MapInference(engine, "base-inference-model", sleep);
            await app.StartAsync();

            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

            using var healthResp = await client.GetAsync("/health");
            Assert.Equal(HttpStatusCode.OK, healthResp.StatusCode);
            var health = JsonSerializer.Deserialize(await healthResp.Content.ReadAsStringAsync(), ServeJsonContext.Default.HealthCheckResponse);
            Assert.NotNull(health);
            Assert.Equal("ContinuousBatchEngine", health.Engine);

            using var modelsResp = await client.GetAsync("/v1/models");
            Assert.Equal(HttpStatusCode.OK, modelsResp.StatusCode);
            var models = JsonSerializer.Deserialize(await modelsResp.Content.ReadAsStringAsync(), ServeJsonContext.Default.ModelListResponse);
            Assert.NotNull(models);
            Assert.Contains(models.Data, m => m.Id == "base-inference-model");
            Assert.DoesNotContain(models.Data, m => m.Id == "should-not-overwrite");

            const string inferReqJson = "{\"model\":\"base-inference-model\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":4,\"stream\":false}";
            using var inferResp = await client.PostAsync("/v1/chat/completions", new StringContent(inferReqJson, Encoding.UTF8, "application/json"));
            Assert.Equal(HttpStatusCode.OK, inferResp.StatusCode);
            var inferObj = JsonSerializer.Deserialize(await inferResp.Content.ReadAsStringAsync(), ServeJsonContext.Default.ChatCompletionResponse);
            Assert.NotNull(inferObj);
            Assert.Equal("base-inference-model", inferObj.Model);

            using var v1SleepStatusResp = await client.GetAsync("/v1/sleep/status");
            Assert.Equal(HttpStatusCode.OK, v1SleepStatusResp.StatusCode);
            var v1SleepStatus = JsonSerializer.Deserialize(await v1SleepStatusResp.Content.ReadAsStringAsync(), ServeJsonContext.Default.SleepStatusResponse);
            Assert.NotNull(v1SleepStatus);
            Assert.True(v1SleepStatus.Enabled);

            using var v1SleepTriggerResp = await client.PostAsync("/v1/sleep/trigger", content: null);
            Assert.Equal(HttpStatusCode.OK, v1SleepTriggerResp.StatusCode);
            var v1SleepTrigger = JsonSerializer.Deserialize(await v1SleepTriggerResp.Content.ReadAsStringAsync(), ServeJsonContext.Default.SleepTriggerResponse);
            Assert.NotNull(v1SleepTrigger);
            Assert.Equal("Awake", v1SleepTrigger.State);

            using var apiSleepStatusResp = await client.GetAsync("/api/sleep/status");
            Assert.Equal(HttpStatusCode.OK, apiSleepStatusResp.StatusCode);
            var apiSleepStatus = JsonSerializer.Deserialize(await apiSleepStatusResp.Content.ReadAsStringAsync(), ServeJsonContext.Default.SleepMemoryStatusResponse);
            Assert.NotNull(apiSleepStatus);
            Assert.True(apiSleepStatus.Enabled);

            using var apiSleepTriggerResp = await client.PostAsync("/api/sleep/trigger", content: null);
            Assert.Equal(HttpStatusCode.OK, apiSleepTriggerResp.StatusCode);
            var apiSleepTrigger = JsonSerializer.Deserialize(await apiSleepTriggerResp.Content.ReadAsStringAsync(), ServeJsonContext.Default.SleepMemoryStatusResponse);
            Assert.NotNull(apiSleepTrigger);
            Assert.True(apiSleepTrigger.SleepCycles >= 1);

            using var bounceStatusResp = await client.GetAsync("/api/bounce/status");
            Assert.Equal(HttpStatusCode.OK, bounceStatusResp.StatusCode);
            var bounceStatus = JsonSerializer.Deserialize(await bounceStatusResp.Content.ReadAsStringAsync(), ServeJsonContext.Default.BounceStatusResponse);
            Assert.NotNull(bounceStatus);
            Assert.Equal("http://mock.local:18080", bounceStatus.PrimaryEndpoint);

            const string bounceReqJson = "{\"model\":\"Glacier-3Machine-Bounce\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":32,\"stream\":false}";
            using var bounceChatResp = await client.PostAsync("/api/bounce/chat/completions", new StringContent(bounceReqJson, Encoding.UTF8, "application/json"));
            Assert.Equal(HttpStatusCode.OK, bounceChatResp.StatusCode);
            var bounceChatObj = JsonSerializer.Deserialize(await bounceChatResp.Content.ReadAsStringAsync(), ServeJsonContext.Default.ChatCompletionResponse);
            Assert.NotNull(bounceChatObj);
            Assert.StartsWith("chatcmpl-bounce-", bounceChatObj.Id, StringComparison.Ordinal);

            app.Stop();
        }
        finally
        {
            if (File.Exists(ggufPath)) File.Delete(ggufPath);
        }
    }

    [Fact]
    public async Task SleepMemoryService_SurvivesTenPlusSequentialCycles_AndConcurrentConsolidationStress()
    {
        await using var sleep = new SleepMemoryService(
            tokenBudget: 60,
            idleThreshold: TimeSpan.FromMinutes(10),
            pollingInterval: TimeSpan.FromMilliseconds(100));

        int port = GetAvailablePort();
        using var app = GlacierServeApp.CreateBuilder().UsePort(port).Build();
        app.MapSleepMemory(sleep);
        await app.StartAsync();

        using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

        // Part A: 10 sequential POST /api/sleep/trigger cycles (far exceeding 4 expert slots)
        for (int cycle = 1; cycle <= 10; cycle++)
        {
            sleep.RecordTurnCompletion(
                $"Cycle {cycle} query about TOKEN_SEQ_{cycle:D2}",
                $"Cycle {cycle} answer confirming TOKEN_SEQ_{cycle:D2} operates at {200 + cycle} Hz.");

            using var resp = await client.PostAsync("/api/sleep/trigger", content: null);
            Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
            var status = JsonSerializer.Deserialize(
                await resp.Content.ReadAsStringAsync(),
                ServeJsonContext.Default.SleepMemoryStatusResponse);
            Assert.NotNull(status);
            Assert.True(status.Enabled);
            Assert.Equal("Awake", status.State);
            Assert.True(status.SleepCycles >= cycle, $"Expected SleepCycles >= {cycle}, got {status.SleepCycles}.");
            Assert.True(status.DreamsConsolidated >= 1);
        }

        Assert.Equal(sleep.ExpansionBank.MaxExperts, sleep.ExpansionBank.ActiveExpertCount);

        // Part B: 16 concurrent workers stressing ConsolidateMessagesAsync + RecordTurnCompletion + TriggerSleepAsync
        var workers = new Task[16];
        for (int w = 0; w < workers.Length; w++)
        {
            int workerId = w;
            workers[w] = Task.Run(async () =>
            {
                for (int iter = 0; iter < 6; iter++)
                {
                    var sb = new StringBuilder();
                    for (int f = 0; f < 10; f++)
                    {
                        sb.AppendLine($"W{workerId}_I{iter}_FACT_{f}: Subsystem NODE_{workerId}_{iter}_{f} uses CRC_{workerId * 100 + iter * 10 + f:D4}.");
                        sb.AppendLine();
                    }
                    sb.Append($"What CRC does NODE_{workerId}_{iter}_3 use?");

                    var consolidated = await sleep.ConsolidateMessagesAsync(
                        [new ChatMessage { Role = "user", Content = sb.ToString() }]);
                    Assert.NotEmpty(consolidated);

                    sleep.RecordTurnCompletion(
                        $"Worker {workerId} iter {iter}",
                        $"Worker {workerId} iter {iter} resolved NODE_{workerId}_{iter}_3.");

                    await sleep.TriggerSleepAsync();
                    var s = sleep.GetStatus();
                    Assert.True(s.Enabled);
                }
            });
        }

        await Task.WhenAll(workers);

        using var finalResp = await client.GetAsync("/api/sleep/status");
        Assert.Equal(HttpStatusCode.OK, finalResp.StatusCode);
        var finalStatus = JsonSerializer.Deserialize(
            await finalResp.Content.ReadAsStringAsync(),
            ServeJsonContext.Default.SleepMemoryStatusResponse);
        Assert.NotNull(finalStatus);
        Assert.True(finalStatus.SleepCycles > 10);
        Assert.True(finalStatus.DreamsConsolidated > 10);

        app.Stop();
    }

    [Fact]
    public async Task SandboxedProcessBounceVerifier_Tier5_HtmlJs_CSharp_Hlsl_And_BogusHarnessHttpNo500()
    {
        // 1. HtmlJs: skips non-JS script types (application/json, ld+json, importmap, speculationrules),
        // catches broken JS, and fails cleanly when node is missing.
        var htmlVerifier = new SandboxedProcessBounceVerifier(harnessExePath: null);
        const string validHtmlWithDataScripts = """
            ```html
            <!DOCTYPE html>
            <html>
            <head>
              <script type="application/json">{"port":8080,"mode":"prod"}</script>
              <script type="application/ld+json">{"@context":"https://schema.org","@type":"SoftwareApplication"}</script>
              <script type="importmap">{"imports":{"app":"./app.js"}}</script>
              <script type="speculationrules">{"prefetch":[{"source":"list","urls":["/next"]}]}</script>
            </head>
            <body>
              <script type="text/javascript">
                function greet(name) { return "hello " + name; }
              </script>
              <script>const x = greet("glacier");</script>
            </body>
            </html>
            ```
            """;
        var validHtmlRes = await htmlVerifier.VerifyAsync(BounceTaskKind.HtmlJs, "html5", validHtmlWithDataScripts, CancellationToken.None);
        bool nodeUnavailableOnHost = !validHtmlRes.Success &&
            !OperatingSystem.IsWindows() &&
            ((validHtmlRes.ErrorMessage?.Contains("unavailable", StringComparison.OrdinalIgnoreCase) ?? false) ||
             (validHtmlRes.ErrorMessage?.Contains("No such file", StringComparison.OrdinalIgnoreCase) ?? false));

        if (!nodeUnavailableOnHost)
        {
            Assert.True(validHtmlRes.Success, validHtmlRes.ErrorMessage);
        }

        const string dataOnlyHtml = """
            ```html
            <!DOCTYPE html>
            <html><body><script type="application/json">{"only":"json"}</script></body></html>
            ```
            """;
        var dataOnlyRes = await htmlVerifier.VerifyAsync(BounceTaskKind.HtmlJs, "html5", dataOnlyHtml, CancellationToken.None);
        Assert.True(dataOnlyRes.Success, dataOnlyRes.ErrorMessage);

        const string brokenJsHtml = """
            ```html
            <!DOCTYPE html>
            <html><body>
              <script type="application/json">{"port":8080}</script>
              <script>const broken = ;</script>
            </body></html>
            ```
            """;
        var brokenJsRes = await htmlVerifier.VerifyAsync(BounceTaskKind.HtmlJs, "html5", brokenJsHtml, CancellationToken.None);
        Assert.False(brokenJsRes.Success);
        Assert.False(string.IsNullOrWhiteSpace(brokenJsRes.ErrorMessage));

        // 2. Bogus non-executable harness path returns Success=false (no Win32Exception) and HTTP endpoint never returns 500
        string tempDir = Path.Combine(Path.GetTempPath(), $"glacier_bounce_tier5_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        try
        {
            string bogusHarness = Path.Combine(tempDir, OperatingSystem.IsWindows() ? "not_an_exe.exe" : "not_an_exe");
            await File.WriteAllTextAsync(bogusHarness, "NOT_A_VALID_EXECUTABLE");

            var bogusVerifier = new SandboxedProcessBounceVerifier(bogusHarness);
            var bogusRes = await bogusVerifier.VerifyAsync(
                BounceTaskKind.CSharp,
                "write csharp",
                "```csharp\npublic static class Demo { public static int Add(int a, int b) => a + b; }\n```",
                CancellationToken.None);
            Assert.False(bogusRes.Success);
            Assert.False(string.IsNullOrWhiteSpace(bogusRes.ErrorMessage));

            using var primaryHttp = new HttpClient(new StubHttpMessageHandler((req, _) =>
            {
                if (req.Method == HttpMethod.Get && req.RequestUri!.AbsolutePath == "/v1/models")
                {
                    return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                    {
                        Content = new StringContent("{\"object\":\"list\",\"data\":[{\"id\":\"mock-primary\"}]}", Encoding.UTF8, "application/json")
                    });
                }

                var completion = new ChatCompletionResponse
                {
                    Id = "mock-cs-1",
                    Created = 1710000000,
                    Model = "mock-primary",
                    Choices =
                    [
                        new ChatChoice
                        {
                            Index = 0,
                            Message = new ChatMessage { Role = "assistant", Content = "```csharp\npublic static class Ok { public static int X() => 1; }\n```" },
                            FinishReason = "stop"
                        }
                    ],
                    Usage = new ChatUsage { PromptTokens = 5, CompletionTokens = 5, TotalTokens = 10 }
                };
                string json = JsonSerializer.Serialize(completion, ServeJsonContext.Default.ChatCompletionResponse);
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(json, Encoding.UTF8, "application/json")
                });
            }));

            var bogusService = new PeerBounceService(primaryHttp, bogusVerifier, "http://mock-primary.local:18080", maxRepairs: 1);
            int port = GetAvailablePort();
            using var app = GlacierServeApp.CreateBuilder().UsePort(port).Build();
            app.MapPeerBounce(bogusService, mapOpenAiAndHealthRoutes: true, modelAlias: "bogus-harness-test");
            await app.StartAsync();

            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };
            string reqJson = JsonSerializer.Serialize(new BounceChatRequest
            {
                Model = "bogus-harness-test",
                Messages = [new ChatMessage { Role = "user", Content = "Write a C# helper class" }],
                Stream = false
            }, ServeJsonContext.Default.BounceChatRequest);

            using var resp = await client.PostAsync("/v1/chat/completions", new StringContent(reqJson, Encoding.UTF8, "application/json"));
            Assert.NotEqual(HttpStatusCode.InternalServerError, resp.StatusCode);
            Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
            app.Stop();

            // 3. CSharp & Hlsl against real Glacier.EvaluationHarness (if built on host)
            string? harnessPath = SandboxedProcessBounceVerifier.ResolveDefaultHarnessPath();
            if (!string.IsNullOrEmpty(harnessPath))
            {
                var realVerifier = new SandboxedProcessBounceVerifier(harnessPath);

                var csOk = await realVerifier.VerifyAsync(
                    BounceTaskKind.CSharp,
                    "write csharp",
                    "```csharp\npublic static class Demo { public static int Add(int a, int b) => a + b; }\n```",
                    CancellationToken.None);
                Assert.True(csOk.Success, csOk.ErrorMessage);

                var csBad = await realVerifier.VerifyAsync(
                    BounceTaskKind.CSharp,
                    "write csharp",
                    "```csharp\npublic static class Demo { public static int Bad() => \"oops\"; }\n```",
                    CancellationToken.None);
                Assert.False(csBad.Success);
                Assert.Contains("CS0029", csBad.ErrorMessage ?? string.Empty, StringComparison.Ordinal);

                var hlslOk = await realVerifier.VerifyAsync(
                    BounceTaskKind.Hlsl,
                    "write hlsl",
                    "```hlsl\nRWStructuredBuffer<float> Out : register(u0);\n[numthreads(1,1,1)]\nvoid CSMain(uint3 id : SV_DispatchThreadID) { Out[id.x] = 1.0f; }\n```",
                    CancellationToken.None);

                bool dxcUnavailableOnHost = !hlslOk.Success &&
                    !OperatingSystem.IsWindows() &&
                    ((hlslOk.ErrorMessage?.Contains("dxc", StringComparison.OrdinalIgnoreCase) ?? false) ||
                     (hlslOk.ErrorMessage?.Contains("No such file", StringComparison.OrdinalIgnoreCase) ?? false));

                if (!dxcUnavailableOnHost)
                {
                    Assert.True(hlslOk.Success, hlslOk.ErrorMessage);

                    var hlslBad = await realVerifier.VerifyAsync(
                        BounceTaskKind.Hlsl,
                        "write hlsl",
                        "```hlsl\n[numthreads(1,1,1)]\nvoid CSMain(uint3 id : SV_DispatchThreadID) { broken_undeclared_sym = 1; }\n```",
                        CancellationToken.None);
                    Assert.False(hlslBad.Success);
                    Assert.False(string.IsNullOrWhiteSpace(hlslBad.ErrorMessage));
                }
            }
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }

    private sealed class StubSandboxExecutor : ISandboxExecutor
    {
        private readonly Func<CancellationToken, SandboxExecutionResult> _handler;
        public SandboxOptions? LastOptions { get; private set; }

        public StubSandboxExecutor(Func<CancellationToken, SandboxExecutionResult> handler)
        {
            _handler = handler;
        }

        public ValueTask<SandboxExecutionResult> ExecuteAsync(SandboxOptions options, CancellationToken ct = default)
        {
            LastOptions = options;
            return ValueTask.FromResult(_handler(ct));
        }
    }

    private sealed class StubHttpMessageHandler : HttpMessageHandler
    {
        private readonly Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> _handler;

        public StubHttpMessageHandler(Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> handler)
        {
            _handler = handler;
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            => _handler(request, cancellationToken);
    }
}

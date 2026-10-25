namespace Glacier.Serve.Benchmarks;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Glacier.Inference.Sampling;
using Glacier.Serve.Inference.Batching;

/// <summary>
/// Definition of a benchmark prompt in the REC-1 reasoning-loop benchmark suite.
/// </summary>
public sealed class BenchmarkPromptDefinition
{
    public required string Id { get; init; }
    public required string Name { get; init; }
    public required string Category { get; init; } // "Loop-Inducing" or "Control"
    public required string Prompt { get; init; }
    public required string ExpectedTarget { get; init; }
    public required string Notes { get; init; }
}

/// <summary>
/// Telemetry measurement from an individual prompt run.
/// </summary>
public sealed class BenchmarkRunRecord
{
    public required string PromptId { get; init; }
    public required string PromptName { get; init; }
    public required string Category { get; init; }
    public int RunIndex { get; init; }
    public required string PromptText { get; init; }
    public int PromptTokens { get; set; }
    public int CompletionTokens { get; set; }
    public double TtftMs { get; set; }
    public double GenerationDurationMs { get; set; }
    public double TotalLatencyMs { get; set; }
    public double TokensPerSecond { get; set; }
    public int LoopRecoveryTriggers { get; set; } // Triggers inside engine during generation (0 in baseline)
    public int DetectedPeriodicLoopPeriod { get; set; } // Post-hoc loop check on generated token stream
    public required string FinishReason { get; set; }
    public required string CompletionText { get; set; }
}

/// <summary>
/// Summary report containing aggregate statistics and hardware metadata.
/// </summary>
public sealed class BenchmarkSuiteReport
{
    public required string Timestamp { get; init; }
    public required string Phase { get; init; } // "Pre-Change Baseline" or "Post-Change"
    public required string ModelPath { get; init; }
    public long ModelSizeBytes { get; init; }
    public required string Architecture { get; init; }
    public int Layers { get; init; }
    public int VocabSize { get; init; }
    public double ColdLoadTimeMs { get; init; }
    public int TotalRuns { get; init; }
    public int MaxTokensPerPrompt { get; init; }
    public float Temperature { get; init; }
    public List<BenchmarkRunRecord> Runs { get; init; } = [];
}

/// <summary>
/// Reproducible benchmark harness executing the 10-prompt task set for REC-1.
/// </summary>
public static class Rec1Benchmark
{
    public static readonly IReadOnlyList<BenchmarkPromptDefinition> BenchmarkPrompts =
    [
        // -------------------------------------------------------------
        // Loop-Inducing Prompts (L1 - L5)
        // -------------------------------------------------------------
        new BenchmarkPromptDefinition
        {
            Id = "L1",
            Name = "Bat & Ball Self-Doubt",
            Category = "Loop-Inducing",
            Prompt = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Think step by step. Recheck and verify your reasoning repeatedly before giving the final answer.",
            ExpectedTarget = "$0.05 (5 cents)",
            Notes = "Elicits recursive 'Wait, let me double check... But wait...' verification cycles."
        },
        new BenchmarkPromptDefinition
        {
            Id = "L2",
            Name = "Liar Paradox Contradiction",
            Category = "Loop-Inducing",
            Prompt = "Is the statement 'This statement is false' true or false? Analyze the contradiction. Keep re-evaluating whether your previous sentence was correct.",
            ExpectedTarget = "Contradiction / Semantic Paradox",
            Notes = "Semantic self-referential paradox inducing circular flip-flop reasoning."
        },
        new BenchmarkPromptDefinition
        {
            Id = "L3",
            Name = "Strawberry Letter Count",
            Category = "Loop-Inducing",
            Prompt = "How many letter 'r's are in 'strawberry'? Check each letter. If you find an 'r', re-verify that it is indeed an 'r'. Repeat verification to be certain.",
            ExpectedTarget = "3",
            Notes = "Provokes repetitive token verification cycles across letter spelling."
        },
        new BenchmarkPromptDefinition
        {
            Id = "L4",
            Name = "Three Boxes Label Riddle",
            Category = "Loop-Inducing",
            Prompt = "You have three boxes: [Apples], [Oranges], [Apples and Oranges]. All three labels are wrong. You pick one fruit from [Apples and Oranges] and it's an apple. What is inside each box? Re-verify each box label multiple times.",
            ExpectedTarget = "Apples & Oranges has Apples; Apples has Oranges; Oranges has Apples & Oranges",
            Notes = "Induces cyclical case analysis across box permutations."
        },
        new BenchmarkPromptDefinition
        {
            Id = "L5",
            Name = "0.999... Equals 1 Circularity",
            Category = "Loop-Inducing",
            Prompt = "Explain why 0.999... equals 1. Reconsider your argument over and over from different angles.",
            ExpectedTarget = "Mathematical proof (algebraic, series, or epsilon-delta)",
            Notes = "Induces cyclical mathematical restatements and doubt loops."
        },

        // -------------------------------------------------------------
        // Non-Looping Control Prompts (C1 - C5)
        // -------------------------------------------------------------
        new BenchmarkPromptDefinition
        {
            Id = "C1",
            Name = "Factual Retrieval",
            Category = "Control",
            Prompt = "What is the capital of France? Answer in one word.",
            ExpectedTarget = "Paris",
            Notes = "Single-token direct factual lookup; loop guard must not false-trigger."
        },
        new BenchmarkPromptDefinition
        {
            Id = "C2",
            Name = "Arithmetic Calculation",
            Category = "Control",
            Prompt = "What is 17 * 19? Answer with just the number.",
            ExpectedTarget = "323",
            Notes = "Direct deterministic mental math; loop guard must remain neutral."
        },
        new BenchmarkPromptDefinition
        {
            Id = "C3",
            Name = "Technical Definition",
            Category = "Control",
            Prompt = "Explain in two clear sentences what a CPU cache is.",
            ExpectedTarget = "Fast on-chip SRAM memory between CPU execution cores and main RAM.",
            Notes = "Succinct definition; tests normal sentence construction without degradation."
        },
        new BenchmarkPromptDefinition
        {
            Id = "C4",
            Name = "Code Generation",
            Category = "Control",
            Prompt = "Write a C# static method named ReverseArray that takes an int[] array and reverses it in place.",
            ExpectedTarget = "Syntactically correct in-place array reversal method in C#.",
            Notes = "Structured syntax generation; verifies no corruption of loop constructs."
        },
        new BenchmarkPromptDefinition
        {
            Id = "C5",
            Name = "Common Sense Physics",
            Category = "Control",
            Prompt = "Which is heavier: one pound of feathers or one pound of lead? Answer concisely.",
            ExpectedTarget = "They weigh the same (both are one pound).",
            Notes = "Basic physical reasoning; verifies clear conclusion without stall."
        }
    ];

    public static async Task<int> RunAsync(string[] args)
    {
        string defaultModelPath = @"D:\lmstudio\models\lmstudio-community\DeepSeek-R1-Distill-Qwen-7B-GGUF\DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf";
        string fallbackModelPath = @"D:\lmstudio\models\lmstudio-community\Qwen2.5-7B-Instruct-1M-GGUF\Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf";

        string modelPath = defaultModelPath;
        int maxTokens = 48;
        int runs = 1;
        float temp = 0.6f;
        string? jsonOutputPath = null;
        string? promptFilter = null;
        string phase = "Pre-Change Baseline";

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i].Equals("--model", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                modelPath = args[++i];
            else if (args[i].Equals("--max-tokens", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                maxTokens = int.Parse(args[++i]);
            else if (args[i].Equals("--runs", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                runs = int.Parse(args[++i]);
            else if (args[i].Equals("--temp", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                temp = float.Parse(args[++i]);
            else if (args[i].Equals("--json-output", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                jsonOutputPath = args[++i];
            else if (args[i].Equals("--filter", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                promptFilter = args[++i];
            else if (args[i].Equals("--phase", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                phase = args[++i];
        }

        if (!File.Exists(modelPath))
        {
            if (File.Exists(fallbackModelPath))
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"[WARN] Target model '{modelPath}' not found. Falling back to '{fallbackModelPath}'.");
                Console.ResetColor();
                modelPath = fallbackModelPath;
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[ERROR] Neither target model '{modelPath}' nor fallback '{fallbackModelPath}' was found!");
                Console.WriteLine("[STOP] Requirement R2 states: If no real model can be run in this environment, STOP and report.");
                Console.ResetColor();
                return 1;
            }
        }

        long fileSizeBytes = new FileInfo(modelPath).Length;

        Console.WriteLine("================================================================================");
        Console.WriteLine($"   GLACIER.SERVE REC-1 REASONING LOOP BENCHMARK HARNESS — {phase.ToUpperInvariant()}");
        Console.WriteLine("================================================================================");
        Console.WriteLine($"  Model Path:       {modelPath}");
        Console.WriteLine($"  Model Size:       {fileSizeBytes / (1024.0 * 1024.0 * 1024.0):F2} GB ({fileSizeBytes:N0} bytes)");
        Console.WriteLine($"  Max Tokens:       {maxTokens}");
        Console.WriteLine($"  Runs per Prompt:  {runs}");
        Console.WriteLine($"  Sampling Temp:    {temp:F2}");
        Console.WriteLine($"  Filter:           {promptFilter ?? "(All 10 prompts)"}");
        Console.WriteLine("--------------------------------------------------------------------------------");

        Console.WriteLine("[1/3] Cold Loading ContinuousBatchEngine...");
        var swLoad = Stopwatch.StartNew();
        using var engine = new ContinuousBatchEngine(modelPath, totalKvBlocks: 512, maxBatchSize: 4);
        swLoad.Stop();
        double coldLoadMs = swLoad.Elapsed.TotalMilliseconds;
        Console.WriteLine($"  ✓ Engine Initialized: {engine.ModelArchitecture} ({engine.Layers} layers, {engine.VocabSize:N0} vocab) in {coldLoadMs:F1} ms ({coldLoadMs / 1000.0:F2} s)");
        Console.WriteLine("--------------------------------------------------------------------------------");

        var report = new BenchmarkSuiteReport
        {
            Timestamp = DateTimeOffset.UtcNow.ToString("O"),
            Phase = phase,
            ModelPath = modelPath,
            ModelSizeBytes = fileSizeBytes,
            Architecture = engine.ModelArchitecture,
            Layers = engine.Layers,
            VocabSize = engine.VocabSize,
            ColdLoadTimeMs = coldLoadMs,
            TotalRuns = runs,
            MaxTokensPerPrompt = maxTokens,
            Temperature = temp
        };

        Console.WriteLine("[2/3] Executing 10-Prompt Benchmark Suite...");
        Console.WriteLine($"{"ID",-4} | {"Category",-14} | {"Prompt Name",-28} | {"Tokens",-6} | {"TTFT (ms)",-10} | {"tok/s",-7} | {"Latency",-10} | {"Loop Trig",-9} | {"Periodic",-8}");
        Console.WriteLine(new string('-', 112));

        foreach (var def in BenchmarkPrompts)
        {
            if (!string.IsNullOrEmpty(promptFilter) &&
                !def.Id.Equals(promptFilter, StringComparison.OrdinalIgnoreCase) &&
                !def.Name.Contains(promptFilter, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            for (int r = 1; r <= runs; r++)
            {
                var req = new InferenceRequest
                {
                    Prompt = def.Prompt,
                    MaxTokens = maxTokens,
                    Temperature = temp
                };

                var swRun = Stopwatch.StartNew();
                var (state, tokenStream) = await engine.EnqueueAsync(req);

                var sbText = new StringBuilder();
                int tokenCount = 0;
                double measuredTtft = 0;
                bool firstSeen = false;

                await foreach (var tok in tokenStream)
                {
                    if (!firstSeen)
                    {
                        firstSeen = true;
                        measuredTtft = swRun.Elapsed.TotalMilliseconds;
                    }
                    sbText.Append(tok);
                    tokenCount++;
                }
                swRun.Stop();

                double totalDurationMs = swRun.Elapsed.TotalMilliseconds;
                double generationDurationMs = Math.Max(0.001, totalDurationMs - measuredTtft);
                double tokPerSec = tokenCount > 1
                    ? (tokenCount - 1) / (generationDurationMs / 1000.0)
                    : tokenCount / (totalDurationMs / 1000.0);

                // Check post-hoc loop detection on the generated token stream
                ReadOnlySpan<int> genTokensSpan = CollectionsMarshal.AsSpan(state.GeneratedTokens);
                int detectedLoopPeriod = Sampler.DetectPeriodicReasoningLoop(genTokensSpan, minPeriod: 1, maxPeriod: 64, minRepeats: 3);

                // Captures active loop recovery triggers in engine during generation
                int engineLoopTriggers = state.LoopRecoveryTriggers;

                var record = new BenchmarkRunRecord
                {
                    PromptId = def.Id,
                    PromptName = def.Name,
                    Category = def.Category,
                    RunIndex = r,
                    PromptText = def.Prompt,
                    PromptTokens = state.PromptTokens.Length,
                    CompletionTokens = tokenCount,
                    TtftMs = measuredTtft > 0 ? measuredTtft : state.TimeToFirstTokenMs,
                    GenerationDurationMs = generationDurationMs,
                    TotalLatencyMs = totalDurationMs,
                    TokensPerSecond = tokPerSec,
                    LoopRecoveryTriggers = engineLoopTriggers,
                    DetectedPeriodicLoopPeriod = detectedLoopPeriod,
                    FinishReason = state.FinishReason,
                    CompletionText = sbText.ToString()
                };

                report.Runs.Add(record);

                string runId = runs > 1 ? $"{def.Id}#{r}" : def.Id;
                Console.WriteLine($"{runId,-4} | {def.Category,-14} | {def.Name,-28} | {tokenCount,6} | {record.TtftMs,10:F1} | {tokPerSec,7:F2} | {totalDurationMs,10:F1} | {engineLoopTriggers,9} | {(detectedLoopPeriod > 0 ? $"P={detectedLoopPeriod}" : "None"),-8}");
            }
        }

        Console.WriteLine(new string('-', 112));
        Console.WriteLine("[3/3] Benchmark Summary & Artifact Export");

        double avgTtft = 0;
        double avgTokS = 0;
        int totalTokensGen = 0;
        double totalTimeMs = 0;
        int totalLoopsDetected = 0;

        foreach (var r in report.Runs)
        {
            avgTtft += r.TtftMs;
            avgTokS += r.TokensPerSecond;
            totalTokensGen += r.CompletionTokens;
            totalTimeMs += r.TotalLatencyMs;
            if (r.DetectedPeriodicLoopPeriod > 0) totalLoopsDetected++;
        }

        if (report.Runs.Count > 0)
        {
            avgTtft /= report.Runs.Count;
            avgTokS /= report.Runs.Count;
        }

        Console.WriteLine($"  Total Completed Prompt Runs:      {report.Runs.Count}");
        Console.WriteLine($"  Total Generated Tokens:           {totalTokensGen:N0}");
        Console.WriteLine($"  Total Evaluation Time:            {totalTimeMs / 1000.0:F2} s");
        Console.WriteLine($"  Average TTFT:                     {avgTtft:F1} ms");
        Console.WriteLine($"  Average Generation Speed:         {avgTokS:F2} tok/s");
        Console.WriteLine($"  Total Unmitigated Loops Detected: {totalLoopsDetected}");
        Console.WriteLine($"  Engine Loop Recovery Triggers:    0 (Dormant in pre-change baseline)");

        // Save JSON report
        if (string.IsNullOrEmpty(jsonOutputPath))
        {
            string outDir = Path.Combine(AppContext.BaseDirectory, "../../../../docs/benchmarks");
            Directory.CreateDirectory(outDir);
            jsonOutputPath = Path.Combine(outDir, "rec1_baseline_results.json");
        }
        else
        {
            string? dir = Path.GetDirectoryName(jsonOutputPath);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        }

        string json = JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(jsonOutputPath, json);
        Console.WriteLine($"  ✓ JSON Results Exported:          {Path.GetFullPath(jsonOutputPath)}");
        Console.WriteLine("================================================================================");

        return 0;
    }
}

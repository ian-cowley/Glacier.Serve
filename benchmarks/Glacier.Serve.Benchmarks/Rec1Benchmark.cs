namespace Glacier.Serve.Benchmarks;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
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
    public required string Category { get; init; } // "Loop-Inducing", "Reasoning-Budget", "Control", "Mixed-Batch"
    public required string Prompt { get; init; }
    public int MaxTokens { get; init; } = 256;
    public int MaxReasoningTokens { get; init; } = 0;
    public bool IsControl { get; init; } = false;
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
    public int MaxTokens { get; set; }
    public int MaxReasoningTokens { get; set; }
    public double TtftMs { get; set; }
    public double GenerationDurationMs { get; set; }
    public double TotalLatencyMs { get; set; }
    public double TokensPerSecond { get; set; }
    public double DecodeTokensPerSecond { get; set; }
    public int LoopRecoveryTriggers { get; set; }
    public int TriggersPeriodic { get; set; }
    public int TriggersBudget { get; set; }
    public int DetectedPeriodicLoopPeriod { get; set; }
    public required string FinishReason { get; set; }
    public bool IsControl { get; set; }
    public bool? SemanticEquality { get; set; }
    public int TokensSaved { get; set; }
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
    public float Temperature { get; init; }
    public int Seed { get; init; }
    public int TotalGeneratedTokens { get; set; }
    public double TotalEvaluationTimeMs { get; set; }
    public double AverageTtftMs { get; set; }
    public double AverageDecodeTokPerSec { get; set; }
    public int TotalPeriodicTriggers { get; set; }
    public int TotalBudgetTriggers { get; set; }
    public int TotalTokensSaved { get; set; }
    public int FalsePositiveCount { get; set; }
    public List<BenchmarkRunRecord> Runs { get; init; } = [];
}

[JsonSourceGenerationOptions(WriteIndented = true)]
[JsonSerializable(typeof(BenchmarkPromptDefinition))]
[JsonSerializable(typeof(BenchmarkRunRecord))]
[JsonSerializable(typeof(BenchmarkSuiteReport))]
internal partial class BenchmarkJsonContext : JsonSerializerContext
{
}

/// <summary>
/// Reproducible benchmark harness executing the hardened 6-task benchmark suite for REC-1.
/// </summary>
public static class Rec1Benchmark
{
    public static readonly IReadOnlyList<BenchmarkPromptDefinition> BenchmarkPrompts =
    [
        // --------------------------------------------------------------------------------------------------
        // Task 1: Deep Epistemic Paradox & Doubt Loop Trap (L-PARADOX)
        // --------------------------------------------------------------------------------------------------
        new BenchmarkPromptDefinition
        {
            Id = "L-PARADOX",
            Name = "Epistemic Paradox Doubt Loop",
            Category = "Loop-Inducing",
            Prompt = "<｜User｜>Analyze the following logical situation in rigorous depth:\n" +
                     "Premise 1: Sentence B is true.\n" +
                     "Premise 2: Sentence A is false.\n" +
                     "Analyze whether Sentence A and Sentence B can simultaneously be assigned consistent truth values.\n" +
                     "For every assignment you try, you must immediately hypothesize the opposite assignment and search for doubts.\n" +
                     "Every time you reach a tentative conclusion, challenge your premise, state \"Wait, let me rethink that assumption...\", and re-analyze from the beginning.\n" +
                     "Do not stop until you have completely eliminated all contradiction.<｜Assistant｜><think>\n",
            MaxTokens = 2000,
            MaxReasoningTokens = 0,
            IsControl = false,
            ExpectedTarget = "Contradiction / Undecidable Paradox (escapes self-doubt loop)",
            Notes = "Exercises periodic loop detection across deep chain-of-thought horizon (up to 2,000 tokens)."
        },

        // --------------------------------------------------------------------------------------------------
        // Task 2: Dynamic Reasoning Token Budget Exhaustion (L-BUDGET)
        // --------------------------------------------------------------------------------------------------
        new BenchmarkPromptDefinition
        {
            Id = "L-BUDGET",
            Name = "Irrationality Proof Budget Clamp",
            Category = "Reasoning-Budget",
            Prompt = "<｜User｜>Prove that the square root of 2 plus the square root of 3 is an irrational number.\n" +
                     "Provide an extensive, exhaustive proof examining all algebraic, field-theoretic, and polynomial properties.<｜Assistant｜><think>\n",
            MaxTokens = 1500,
            MaxReasoningTokens = 400,
            IsControl = false,
            ExpectedTarget = "Forced </think> transition at token 400, followed by concise proof synthesis",
            Notes = "Exercises Sampler.ClampReasoningLogits forcing transition out of <think> at token budget."
        },

        // --------------------------------------------------------------------------------------------------
        // Task 3: False-Positive Control — 30-Row Markdown Table (C-TABLE)
        // --------------------------------------------------------------------------------------------------
        new BenchmarkPromptDefinition
        {
            Id = "C-TABLE",
            Name = "30-Row Virtual Switch Table",
            Category = "Control",
            Prompt = "<｜User｜>Generate a detailed Markdown table listing 30 virtual network switches with columns: ID, Interface Name, MAC Address, Status, Duplex, MTU, VLAN, and Errors. Use standard enterprise values where Status is 'UP', Duplex is 'FULL', MTU is '1500', and Errors is '0' for every row.<｜Assistant｜><think>\n" +
                     "I need to generate a 30-row markdown table of network interfaces.\n" +
                     "</think>\n" +
                     "| ID | Interface | MAC Address | Status | Duplex | MTU | VLAN | Errors |\n" +
                     "|---|---|---|---|---|---|---|---|\n",
            MaxTokens = 500,
            MaxReasoningTokens = 0,
            IsControl = true,
            ExpectedTarget = "30-row Markdown table; 0 loop triggers; 100% semantic and token equality with baseline.",
            Notes = "False-positive control with high structural delimiter repetition."
        },

        // --------------------------------------------------------------------------------------------------
        // Task 4: False-Positive Control — Repetitive C# Switch Statement (C-CODE)
        // --------------------------------------------------------------------------------------------------
        new BenchmarkPromptDefinition
        {
            Id = "C-CODE",
            Name = "C# Switch Statement 200-230",
            Category = "Control",
            Prompt = "<｜User｜>Write a C# static method named GetHttpCodeMessage that contains a complete switch statement mapping HTTP status codes 200 through 230 to their descriptive string (e.g. case 200: return \"OK\"; case 201: return \"Created\"; ...). Return \"Unknown\" as default. Output only the C# method code.<｜Assistant｜><think>\n" +
                     "Writing a clean C# switch mapping codes 200 to 230.\n" +
                     "</think>\n" +
                     "public static string GetHttpCodeMessage(int statusCode)\n" +
                     "{\n" +
                     "    switch (statusCode)\n" +
                     "    {\n",
            MaxTokens = 450,
            MaxReasoningTokens = 0,
            IsControl = true,
            ExpectedTarget = "Complete C# switch method; 0 loop triggers; 100% equality with baseline.",
            Notes = "False-positive control with legitimate repeating code syntax patterns."
        },

        // --------------------------------------------------------------------------------------------------
        // Task 5: Control — Direct Factual Retrieval (C-FACT)
        // --------------------------------------------------------------------------------------------------
        new BenchmarkPromptDefinition
        {
            Id = "C-FACT",
            Name = "Capital of France Factual Retrieval",
            Category = "Control",
            Prompt = "<｜User｜>What is the capital of France? Answer in one word.<｜Assistant｜><think>\nThe capital of France is Paris.\n</think>\n",
            MaxTokens = 30,
            MaxReasoningTokens = 0,
            IsControl = true,
            ExpectedTarget = "Paris; 0 loop triggers; finishes with stop token.",
            Notes = "Succinct factual retrieval control."
        }
    ];

    public static async Task<int> RunAsync(string[] args)
    {
        string? modelPath = Environment.GetEnvironmentVariable("GLACIER_MODEL_PATH");
        int? globalMaxTokens = null;
        int runs = 1;
        float temp = 0.0f; // Deterministic greedy by default
        int seed = 42;
        int minPeriod = 2; // Explicit opt-in for REC-1 benchmark to ignore period-1 punctuation
        bool disableLoopGuard = false;
        bool skipMixedBatch = false;
        string? jsonOutputPath = null;
        string? baselineJsonPath = null;
        string? promptFilter = null;
        string? phase = null;

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i].Equals("--model", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                modelPath = args[++i];
            else if (args[i].Equals("--max-tokens", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                globalMaxTokens = int.Parse(args[++i]);
            else if (args[i].Equals("--runs", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                runs = int.Parse(args[++i]);
            else if (args[i].Equals("--temp", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                temp = float.Parse(args[++i]);
            else if (args[i].Equals("--seed", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                seed = int.Parse(args[++i]);
            else if (args[i].Equals("--min-period", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                minPeriod = int.Parse(args[++i]);
            else if (args[i].Equals("--disable-loop-guard", StringComparison.OrdinalIgnoreCase))
                disableLoopGuard = true;
            else if (args[i].Equals("--skip-batch", StringComparison.OrdinalIgnoreCase))
                skipMixedBatch = true;
            else if (args[i].Equals("--json-output", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                jsonOutputPath = args[++i];
            else if (args[i].Equals("--baseline-json", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                baselineJsonPath = args[++i];
            else if (args[i].Equals("--filter", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                promptFilter = args[++i];
            else if (args[i].Equals("--phase", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                phase = args[++i];
        }

        phase ??= disableLoopGuard ? "Pre-Change Baseline" : "Post-Change";

        if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"[ERROR] Model file was not found (resolved path: '{modelPath ?? "<unset>"}').");
            Console.WriteLine("        Provide --model <path-to-gguf> or set the GLACIER_MODEL_PATH environment variable.");
            Console.ResetColor();
            return 1;
        }

        long fileSizeBytes = new FileInfo(modelPath).Length;

        // Load baseline records if provided
        Dictionary<string, BenchmarkRunRecord>? baselineLookup = null;
        if (!string.IsNullOrEmpty(baselineJsonPath) && File.Exists(baselineJsonPath))
        {
            try
            {
                string baseJson = await File.ReadAllTextAsync(baselineJsonPath);
                var baseReport = JsonSerializer.Deserialize(baseJson, BenchmarkJsonContext.Default.BenchmarkSuiteReport);
                if (baseReport?.Runs != null)
                {
                    baselineLookup = new Dictionary<string, BenchmarkRunRecord>(StringComparer.OrdinalIgnoreCase);
                    foreach (var r in baseReport.Runs)
                    {
                        baselineLookup[r.PromptId] = r;
                    }
                    Console.WriteLine($"[INFO] Loaded baseline reference with {baselineLookup.Count} runs from '{baselineJsonPath}'.");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WARN] Failed to load baseline JSON from '{baselineJsonPath}': {ex.Message}");
            }
        }

        Console.WriteLine("==========================================================================================");
        Console.WriteLine($"   GLACIER.SERVE REC-1 HARDENED BENCHMARK HARNESS — {phase.ToUpperInvariant()}");
        Console.WriteLine("==========================================================================================");
        Console.WriteLine($"  Model Path:          {modelPath}");
        Console.WriteLine($"  Model Size:          {fileSizeBytes / (1024.0 * 1024.0 * 1024.0):F2} GB ({fileSizeBytes:N0} bytes)");
        Console.WriteLine($"  Decoding Mode:       {(temp <= 0.001f ? "Deterministic Greedy (temp = 0.0, seed = " + seed + ")" : "Stochastic (temp = " + temp.ToString("F2") + ")")}");
        Console.WriteLine($"  Loop Guard Active:   {!disableLoopGuard}");
        Console.WriteLine($"  Runs per Prompt:     {runs}");
        Console.WriteLine($"  Filter:              {promptFilter ?? "(All 6 focused tasks)"}");
        Console.WriteLine("------------------------------------------------------------------------------------------");

        Console.WriteLine("[1/3] Cold Loading ContinuousBatchEngine...");
        var swLoad = Stopwatch.StartNew();
        using var engine = new ContinuousBatchEngine(modelPath, totalKvBlocks: 1024, maxBatchSize: 4, maxSeqLen: 4096, seed: seed);
        swLoad.Stop();
        double coldLoadMs = swLoad.Elapsed.TotalMilliseconds;
        Console.WriteLine($"  ✓ Engine Initialized: {engine.ModelArchitecture} ({engine.Layers} layers, {engine.VocabSize:N0} vocab) in {coldLoadMs:F1} ms ({coldLoadMs / 1000.0:F2} s)");
        Console.WriteLine("------------------------------------------------------------------------------------------");

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
            Temperature = temp,
            Seed = seed
        };

        var suiteTimer = Stopwatch.StartNew();

        Console.WriteLine("[2/3] Executing Benchmark Tasks...");
        Console.WriteLine($"{"ID",-10} | {"Category",-16} | {"Tokens",-6} | {"TTFT(ms)",-9} | {"Decode tok/s",-12} | {"Elapsed",-9} | {"Periodic",-8} | {"Budget",-6} | {"Finish",-8} | {"Saved",-6} | {"Notes"}");
        Console.WriteLine(new string('-', 124));

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
                int maxTokens = globalMaxTokens ?? def.MaxTokens;
                int maxReasoning = disableLoopGuard ? 0 : def.MaxReasoningTokens;

                var req = new InferenceRequest
                {
                    Prompt = def.Prompt,
                    MaxTokens = maxTokens,
                    Temperature = temp,
                    MaxReasoningTokens = maxReasoning,
                    EnableReasoningLoopGuard = !disableLoopGuard,
                    ReasoningLoopMinPeriod = minPeriod
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
                double overallTokPerSec = totalDurationMs > 0 ? tokenCount / (totalDurationMs / 1000.0) : 0;
                double decodeTokPerSec = tokenCount > 1
                    ? (tokenCount - 1) / (generationDurationMs / 1000.0)
                    : overallTokPerSec;

                ReadOnlySpan<int> genTokensSpan = CollectionsMarshal.AsSpan(state.GeneratedTokens);
                int detectedLoopPeriod = Sampler.DetectPeriodicReasoningLoop(genTokensSpan, minPeriod: minPeriod, maxPeriod: 64, minRepeats: 3);

                int periodicTriggers = state.PeriodicLoopTriggers;
                int budgetTriggers = state.BudgetClampTriggers;
                int totalTriggers = state.LoopRecoveryTriggers;

                int tokensSaved = 0;
                bool? semanticEq = null;

                if (baselineLookup != null && baselineLookup.TryGetValue(def.Id, out var baseRec))
                {
                    tokensSaved = Math.Max(0, baseRec.CompletionTokens - tokenCount);
                    if (def.IsControl)
                    {
                        semanticEq = string.Equals(sbText.ToString().Trim(), baseRec.CompletionText.Trim(), StringComparison.Ordinal);
                    }
                }

                var record = new BenchmarkRunRecord
                {
                    PromptId = def.Id,
                    PromptName = def.Name,
                    Category = def.Category,
                    RunIndex = r,
                    PromptText = def.Prompt,
                    PromptTokens = state.PromptTokens.Length,
                    CompletionTokens = tokenCount,
                    MaxTokens = maxTokens,
                    MaxReasoningTokens = maxReasoning,
                    TtftMs = measuredTtft > 0 ? measuredTtft : state.TimeToFirstTokenMs,
                    GenerationDurationMs = generationDurationMs,
                    TotalLatencyMs = totalDurationMs,
                    TokensPerSecond = overallTokPerSec,
                    DecodeTokensPerSecond = decodeTokPerSec,
                    LoopRecoveryTriggers = totalTriggers,
                    TriggersPeriodic = periodicTriggers,
                    TriggersBudget = budgetTriggers,
                    DetectedPeriodicLoopPeriod = detectedLoopPeriod,
                    FinishReason = state.FinishReason,
                    IsControl = def.IsControl,
                    SemanticEquality = semanticEq,
                    TokensSaved = tokensSaved,
                    CompletionText = sbText.ToString()
                };

                report.Runs.Add(record);

                string runId = runs > 1 ? $"{def.Id}#{r}" : def.Id;
                string note = "";
                if (semanticEq.HasValue) note = semanticEq.Value ? "✓ Match" : "⚠ Diverged";
                else if (tokensSaved > 0) note = $"Saved {tokensSaved} tok";
                else if (totalTriggers > 0) note = $"Recovered ({totalTriggers} trig)";

                Console.WriteLine($"{runId,-10} | {def.Category,-16} | {tokenCount,6} | {record.TtftMs,9:F1} | {decodeTokPerSec,12:F2} | {totalDurationMs / 1000.0,8:F1}s | {periodicTriggers,8} | {budgetTriggers,6} | {state.FinishReason,-8} | {tokensSaved,6} | {note}");
            }
        }

        // --------------------------------------------------------------------------------------------------
        // Task 6: Mixed-Batch Concurrency Run (M-BATCH)
        // --------------------------------------------------------------------------------------------------
        if (!skipMixedBatch && (string.IsNullOrEmpty(promptFilter) || promptFilter.Contains("BATCH", StringComparison.OrdinalIgnoreCase) || promptFilter.Contains("M-", StringComparison.OrdinalIgnoreCase)))
        {
            Console.WriteLine(new string('-', 124));
            Console.WriteLine("[Task 6] Executing Mixed-Batch Concurrency: L-PARADOX (MaxTokens=800) + C-FACT (MaxTokens=30)...");

            var reqA = new InferenceRequest
            {
                Prompt = "<｜User｜>Analyze the following logical situation in rigorous depth:\n" +
                         "Premise 1: Sentence B is true.\n" +
                         "Premise 2: Sentence A is false.\n" +
                         "Analyze whether Sentence A and Sentence B can simultaneously be assigned consistent truth values.\n" +
                         "For every assignment you try, challenge your premise, state \"Wait, let me rethink that assumption...\", and re-analyze from the beginning.<｜Assistant｜><think>\n",
                MaxTokens = 800,
                Temperature = temp,
                EnableReasoningLoopGuard = !disableLoopGuard,
                ReasoningLoopMinPeriod = minPeriod
            };

            var reqB = new InferenceRequest
            {
                Prompt = "<｜User｜>What is the capital of France? Answer in one word.<｜Assistant｜><think>\nThe capital of France is Paris.\n</think>\n",
                MaxTokens = 30,
                Temperature = temp,
                EnableReasoningLoopGuard = !disableLoopGuard,
                ReasoningLoopMinPeriod = minPeriod
            };

            var swBatch = Stopwatch.StartNew();
            var (stateA, streamA) = await engine.EnqueueAsync(reqA);
            var (stateB, streamB) = await engine.EnqueueAsync(reqB);

            var sbA = new StringBuilder();
            var sbB = new StringBuilder();
            int countA = 0, countB = 0;
            double ttftA = 0, ttftB = 0;
            double latA = 0, latB = 0;

            var taskA = Task.Run(async () =>
            {
                var swA = Stopwatch.StartNew();
                await foreach (var tok in streamA)
                {
                    if (countA == 0) ttftA = swA.Elapsed.TotalMilliseconds;
                    sbA.Append(tok);
                    countA++;
                }
                swA.Stop();
                latA = swA.Elapsed.TotalMilliseconds;
            });

            var taskB = Task.Run(async () =>
            {
                var swB = Stopwatch.StartNew();
                await foreach (var tok in streamB)
                {
                    if (countB == 0) ttftB = swB.Elapsed.TotalMilliseconds;
                    sbB.Append(tok);
                    countB++;
                }
                swB.Stop();
                latB = swB.Elapsed.TotalMilliseconds;
            });

            await Task.WhenAll(taskA, taskB);
            swBatch.Stop();

            double genDurA = Math.Max(0.001, latA - ttftA);
            double genDurB = Math.Max(0.001, latB - ttftB);
            double decA = countA > 1 ? (countA - 1) / (genDurA / 1000.0) : 0;
            double decB = countB > 1 ? (countB - 1) / (genDurB / 1000.0) : 0;

            var recBatchA = new BenchmarkRunRecord
            {
                PromptId = "M-BATCH-A",
                PromptName = "Mixed Batch Sequence A (Loop Trap)",
                Category = "Mixed-Batch",
                RunIndex = 1,
                PromptText = reqA.Prompt,
                PromptTokens = stateA.PromptTokens.Length,
                CompletionTokens = countA,
                MaxTokens = 800,
                MaxReasoningTokens = 0,
                TtftMs = ttftA,
                GenerationDurationMs = genDurA,
                TotalLatencyMs = latA,
                TokensPerSecond = countA / (latA / 1000.0),
                DecodeTokensPerSecond = decA,
                LoopRecoveryTriggers = stateA.LoopRecoveryTriggers,
                TriggersPeriodic = stateA.PeriodicLoopTriggers,
                TriggersBudget = stateA.BudgetClampTriggers,
                DetectedPeriodicLoopPeriod = 0,
                FinishReason = stateA.FinishReason,
                IsControl = false,
                CompletionText = sbA.ToString()
            };

            var recBatchB = new BenchmarkRunRecord
            {
                PromptId = "M-BATCH-B",
                PromptName = "Mixed Batch Sequence B (Fact Control)",
                Category = "Mixed-Batch",
                RunIndex = 1,
                PromptText = reqB.Prompt,
                PromptTokens = stateB.PromptTokens.Length,
                CompletionTokens = countB,
                MaxTokens = 30,
                MaxReasoningTokens = 0,
                TtftMs = ttftB,
                GenerationDurationMs = genDurB,
                TotalLatencyMs = latB,
                TokensPerSecond = countB / (latB / 1000.0),
                DecodeTokensPerSecond = decB,
                LoopRecoveryTriggers = stateB.LoopRecoveryTriggers,
                TriggersPeriodic = stateB.PeriodicLoopTriggers,
                TriggersBudget = stateB.BudgetClampTriggers,
                DetectedPeriodicLoopPeriod = 0,
                FinishReason = stateB.FinishReason,
                IsControl = true,
                CompletionText = sbB.ToString()
            };

            report.Runs.Add(recBatchA);
            report.Runs.Add(recBatchB);

            Console.WriteLine($"{"M-BATCH-A",-10} | {"Mixed-Batch (A)",-16} | {countA,6} | {ttftA,9:F1} | {decA,12:F2} | {latA / 1000.0,8:F1}s | {stateA.PeriodicLoopTriggers,8} | {stateA.BudgetClampTriggers,6} | {stateA.FinishReason,-8} | {"—",6} | Concurrent Loop Trap");
            Console.WriteLine($"{"M-BATCH-B",-10} | {"Mixed-Batch (B)",-16} | {countB,6} | {ttftB,9:F1} | {decB,12:F2} | {latB / 1000.0,8:F1}s | {stateB.PeriodicLoopTriggers,8} | {stateB.BudgetClampTriggers,6} | {stateB.FinishReason,-8} | {"—",6} | Concurr Fact: '{sbB.ToString().Trim()}'");
        }

        suiteTimer.Stop();
        Console.WriteLine(new string('-', 124));
        Console.WriteLine("[3/3] Aggregate Statistics & Telemetry Summary");

        double totalEvalTimeMs = suiteTimer.Elapsed.TotalMilliseconds;
        int totalTokensGen = 0;
        double sumTtft = 0;
        double sumDecode = 0;
        int totalPeriodic = 0;
        int totalBudget = 0;
        int totalSaved = 0;
        int falsePositives = 0;

        foreach (var r in report.Runs)
        {
            totalTokensGen += r.CompletionTokens;
            sumTtft += r.TtftMs;
            sumDecode += r.DecodeTokensPerSecond;
            totalPeriodic += r.TriggersPeriodic;
            totalBudget += r.TriggersBudget;
            totalSaved += r.TokensSaved;
            if (r.IsControl)
            {
                falsePositives += r.LoopRecoveryTriggers;
            }
        }

        double avgTtft = report.Runs.Count > 0 ? sumTtft / report.Runs.Count : 0;
        double avgDecode = report.Runs.Count > 0 ? sumDecode / report.Runs.Count : 0;

        report.TotalGeneratedTokens = totalTokensGen;
        report.TotalEvaluationTimeMs = totalEvalTimeMs;
        report.AverageTtftMs = avgTtft;
        report.AverageDecodeTokPerSec = avgDecode;
        report.TotalPeriodicTriggers = totalPeriodic;
        report.TotalBudgetTriggers = totalBudget;
        report.TotalTokensSaved = totalSaved;
        report.FalsePositiveCount = falsePositives;

        Console.WriteLine($"  Total Completed Runs:             {report.Runs.Count}");
        Console.WriteLine($"  Total Generated Tokens:           {totalTokensGen:N0}");
        Console.WriteLine($"  Total Suite Runtime:              {totalEvalTimeMs / 1000.0:F1} s ({totalEvalTimeMs / 60000.0:F2} min)");
        Console.WriteLine($"  Average TTFT:                     {avgTtft:F1} ms");
        Console.WriteLine($"  Average Decode Speed:             {avgDecode:F2} tok/s");
        Console.WriteLine($"  Periodic Loop Triggers:           {totalPeriodic}");
        Console.WriteLine($"  Budget Clamp Triggers:            {totalBudget}");
        Console.WriteLine($"  Total Tokens Saved vs Baseline:   {totalSaved:N0}");
        Console.WriteLine($"  False-Positive Triggers:          {falsePositives} (on controls C-TABLE, C-CODE, C-FACT, M-BATCH-B)");

        // Save JSON report
        if (string.IsNullOrEmpty(jsonOutputPath))
        {
            string outDir = Path.Combine(AppContext.BaseDirectory, "../../../../docs/benchmarks");
            Directory.CreateDirectory(outDir);
            jsonOutputPath = Path.Combine(outDir, disableLoopGuard ? "rec1_hardened_baseline_results.json" : "rec1_hardened_postchange_results.json");
        }
        else
        {
            string? dir = Path.GetDirectoryName(jsonOutputPath);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        }

        string json = JsonSerializer.Serialize(report, BenchmarkJsonContext.Default.BenchmarkSuiteReport);
        await File.WriteAllTextAsync(jsonOutputPath, json);
        Console.WriteLine($"  ✓ Exported Full JSON Results:     {Path.GetFullPath(jsonOutputPath)}");
        Console.WriteLine("==========================================================================================");

        return 0;
    }
}

namespace Glacier.Serve.Tests;

using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Serve.Daemons;
using Glacier.Serve.Inference.Batching;
using Glacier.Tune.Expansion;
using Glacier.Tune.Sleep;
using Xunit;

public class AutonomousSleepDaemonTests
{
    [Fact]
    public void AutonomousSleepDaemon_InitialState_IsAwakeWithZeroCycles()
    {
        using var bank = new PreallocatedMoeExpansionBank(dimension: 16, rank: 4, maxExperts: 2);
        using var trainer = new SleepConsolidationTrainer(bank);
        using var daemon = new AutonomousSleepDaemon(trainer);

        Assert.Equal(SleepLifecycleState.Awake, daemon.CurrentState);
        Assert.Equal(0, daemon.CompletedSleepCycles);
        Assert.Equal(0, daemon.TotalDreamsConsolidated);
        Assert.Equal(0, daemon.PendingDreamCount);
    }

    [Fact]
    public async Task AutonomousSleepDaemon_ImmediateSleep_ExecutesConsolidationAndStateTransitions()
    {
        const int dim = 32;
        const int rank = 8;
        using var bank = new PreallocatedMoeExpansionBank(dim, rank, maxExperts: 4);

        bool pruningCalled = false;
        using var trainer = new SleepConsolidationTrainer(bank, synapticPruningCallback: () =>
        {
            pruningCalled = true;
        });

        var options = new SleepDaemonOptions(
            IdleThreshold: TimeSpan.FromSeconds(10),
            PollingInterval: TimeSpan.FromMilliseconds(50),
            DistillationSteps: 80);

        using var daemon = new AutonomousSleepDaemon(trainer, options);

        var stateHistory = new List<SleepLifecycleState>();
        daemon.StateChanged += state => stateHistory.Add(state);

        Random rng = new Random(101);
        float[] centroid = CreateRandomUnitVector(rng, dim);

        // Queue 4 dream candidates (3 valid, 1 hallucinated)
        for (int i = 0; i < 4; i++)
        {
            bool isHallucinated = (i == 3);
            float[] q = (float[])centroid.Clone();
            float[] target = CreateRandomUnitVector(rng, dim);
            string prompt = $"Concept_{i}";
            string dream = isHallucinated ? "Noise_Fail" : prompt;
            float score = isHallucinated ? 0.10f : 0.90f;

            daemon.QueueDream(new DreamCandidate(prompt, dream, q, target, score, score, score));
        }

        Assert.Equal(4, daemon.PendingDreamCount);

        ConsolidationResult? result = await daemon.TriggerImmediateSleepAsync(centroid);

        Assert.NotNull(result);
        Assert.Equal(0, result.ExpertSlotIndex);
        Assert.Equal(4, result.TotalDreamsGenerated);
        Assert.Equal(3, result.TotalDreamsAccepted);
        Assert.Equal(1, daemon.CompletedSleepCycles);
        Assert.Equal(3, daemon.TotalDreamsConsolidated);
        Assert.Equal(0, daemon.PendingDreamCount);
        Assert.Equal(SleepLifecycleState.Awake, daemon.CurrentState);
        Assert.True(pruningCalled);

        // Verify state transitions: EnteringSleep -> Sleeping -> Waking -> Awake
        Assert.Contains(SleepLifecycleState.EnteringSleep, stateHistory);
        Assert.Contains(SleepLifecycleState.Sleeping, stateHistory);
        Assert.Contains(SleepLifecycleState.Waking, stateHistory);
        Assert.Equal(SleepLifecycleState.Awake, daemon.CurrentState);
    }

    [Fact]
    public async Task AutonomousSleepDaemon_IdleWatcher_AutomaticallyTriggersSleepWhenQuiescent()
    {
        const int dim = 16;
        const int rank = 4;
        using var bank = new PreallocatedMoeExpansionBank(dim, rank, maxExperts: 2);
        using var trainer = new SleepConsolidationTrainer(bank);

        var options = new SleepDaemonOptions(
            IdleThreshold: TimeSpan.FromMilliseconds(60),
            PollingInterval: TimeSpan.FromMilliseconds(10),
            DistillationSteps: 30,
            MinDreamsToTriggerSleep: 1);

        using var daemon = new AutonomousSleepDaemon(trainer, options);

        var tcs = new TaskCompletionSource<ConsolidationResult>();
        daemon.ConsolidationCompleted += res => tcs.TrySetResult(res);

        Random rng = new Random(202);
        float[] q = CreateRandomUnitVector(rng, dim);
        float[] target = CreateRandomUnitVector(rng, dim);

        daemon.QueueDream(new DreamCandidate("Prompt", "Prompt", q, target, 0.95f, 0.95f, 0.95f));

        daemon.Start();

        // Wait for autonomous sleep to trigger on idle timeout
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        cts.Token.Register(() => tcs.TrySetCanceled());

        ConsolidationResult consolidated = await tcs.Task;
        daemon.Stop();

        Assert.NotNull(consolidated);
        Assert.Equal(1, daemon.CompletedSleepCycles);
        Assert.Equal(1, daemon.TotalDreamsConsolidated);
        Assert.Equal(SleepLifecycleState.Awake, daemon.CurrentState);
    }

    [Fact]
    public async Task AutonomousSleepDaemon_MultiTaskContinualLearning_ZeroCatastrophicForgetting()
    {
        const int dim = 32;
        const int rank = 8;
        const int numTasks = 3;
        const int factsPerTask = 8;
        Random rng = new Random(303);

        using var bank = new PreallocatedMoeExpansionBank(dim, rank, maxExperts: 4);
        using var trainer = new SleepConsolidationTrainer(bank);
        using var daemon = new AutonomousSleepDaemon(trainer, new SleepDaemonOptions(
            IdleThreshold: TimeSpan.FromSeconds(10),
            PollingInterval: TimeSpan.FromMilliseconds(50),
            DistillationSteps: 120,
            LearningRate: 0.25f));

        // Generate a single unified orthogonal basis so sharedKeys and taskSignatures are mutually orthogonal
        float[][] basis = CreateOrthogonalBasis(rng, dim, factsPerTask + 1 + numTasks);
        float[][] sharedKeys = new float[factsPerTask][];
        for (int f = 0; f < factsPerTask; f++) sharedKeys[f] = basis[f];

        float[][] taskSignatures = new float[1 + numTasks][];
        for (int t = 0; t <= numTasks; t++) taskSignatures[t] = basis[factsPerTask + t];

        int totalFacts = (1 + numTasks) * factsPerTask;
        float[][] allKeys = new float[totalFacts][];
        float[][] allVals = new float[totalFacts][];

        for (int task = 0; task <= numTasks; task++)
        {
            for (int f = 0; f < factsPerTask; f++)
            {
                int idx = task * factsPerTask + f;
                allKeys[idx] = new float[dim];
                for (int d = 0; d < dim; d++)
                {
                    allKeys[idx][d] = 0.80f * sharedKeys[f][d] + 0.60f * taskSignatures[task][d];
                }
                NormalizeInPlace(allKeys[idx]);
                allVals[idx] = CreateRandomUnitVector(rng, dim);
            }
        }

        // Base foundation model W0 trained on Task 0
        float[] W0 = new float[dim * dim];
        float[] predBuf = new float[dim];
        for (int step = 0; step < 80; step++)
        {
            for (int i = 0; i < factsPerTask; i++)
            {
                MatVecMul(W0, allKeys[i], predBuf, dim);
                for (int r = 0; r < dim; r++)
                {
                    float err = predBuf[r] - allVals[i][r];
                    for (int c = 0; c < dim; c++)
                    {
                        W0[r * dim + c] -= 0.25f * err * allKeys[i][c];
                    }
                }
            }
        }

        // Sequential Wake/Sleep cycles for Task 1, Task 2, Task 3
        for (int task = 1; task <= numTasks; task++)
        {
            int startIdx = task * factsPerTask;
            int endIdx = startIdx + factsPerTask;

            // Generate synthetic dream rollouts for this task
            for (int i = startIdx; i < endIdx; i++)
            {
                // Target residual: v_residual = v_target - W0 * k
                MatVecMul(W0, allKeys[i], predBuf, dim);
                float[] residualTarget = new float[dim];
                for (int d = 0; d < dim; d++) residualTarget[d] = allVals[i][d] - predBuf[d];

                daemon.QueueDream(new DreamCandidate(
                    $"Task{task}_Item{i}",
                    $"Task{task}_Item{i}",
                    allKeys[i],
                    residualTarget,
                    0.95f,
                    0.95f,
                    0.95f));
            }

            // Sleep phase trigger: consolidates knowledge into unlocked MoE expert
            await daemon.TriggerImmediateSleepAsync(taskSignatures[task]);
        }

        Assert.Equal(3, daemon.CompletedSleepCycles);
        Assert.Equal(3, bank.ActiveExpertCount);

        // Verification across all 4 tasks:
        // Base foundation weights W0 were NEVER mutated, so Task 0 has 1.0 retention!
        float[] taskSims = new float[1 + numTasks];
        float[] outBuf = new float[dim];
        float[] deltaBuf = new float[dim];
        float[] rankScratch = new float[rank];

        for (int task = 0; task <= numTasks; task++)
        {
            int startIdx = task * factsPerTask;
            int endIdx = startIdx + factsPerTask;
            float sumSim = 0.0f;

            for (int i = startIdx; i < endIdx; i++)
            {
                // Base model contribution
                MatVecMul(W0, allKeys[i], outBuf, dim);

                // MoE expert contribution
                bank.ForwardDelta(allKeys[i], deltaBuf, rankScratch);
                for (int d = 0; d < dim; d++) outBuf[d] += deltaBuf[d];

                sumSim += CosineSimilarity(outBuf, allVals[i]);
            }

            taskSims[task] = sumSim / factsPerTask;
        }

        Assert.True(taskSims[0] >= 0.999f, $"Expected Task 0 retention >= 0.999 (zero catastrophic forgetting), got {taskSims[0]:F4}");
        Assert.True(taskSims[1] >= 0.980f, $"Expected Task 1 retention >= 0.980, got {taskSims[1]:F4}");
        Assert.True(taskSims[2] >= 0.980f, $"Expected Task 2 retention >= 0.980, got {taskSims[2]:F4}");
        Assert.True(taskSims[3] >= 0.980f, $"Expected Task 3 retention >= 0.980, got {taskSims[3]:F4}");
    }

    private static void MatVecMul(ReadOnlySpan<float> M, ReadOnlySpan<float> x, Span<float> y, int dim)
    {
        for (int i = 0; i < dim; i++)
        {
            float sum = 0f;
            int offset = i * dim;
            for (int j = 0; j < dim; j++) sum += M[offset + j] * x[j];
            y[i] = sum;
        }
    }

    private static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float dot = 0f, magA = 0f, magB = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            magA += a[i] * a[i];
            magB += b[i] * b[i];
        }
        return (magA > 1e-12f && magB > 1e-12f) ? dot / (MathF.Sqrt(magA) * MathF.Sqrt(magB)) : 0f;
    }

    private static float[] CreateRandomUnitVector(Random rng, int dim)
    {
        float[] v = new float[dim];
        for (int i = 0; i < dim; i++) v[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        NormalizeInPlace(v);
        return v;
    }

    private static void NormalizeInPlace(Span<float> v)
    {
        float mag = 0f;
        for (int i = 0; i < v.Length; i++) mag += v[i] * v[i];
        mag = MathF.Sqrt(mag);
        if (mag > 1e-12f)
        {
            for (int i = 0; i < v.Length; i++) v[i] /= mag;
        }
    }

    private static float[][] CreateOrthogonalBasis(Random rng, int dim, int count)
    {
        float[][] basis = new float[count][];
        for (int i = 0; i < count; i++)
        {
            float[] v = CreateRandomUnitVector(rng, dim);
            for (int pass = 0; pass < 2; pass++)
            {
                for (int j = 0; j < i; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < dim; d++) dot += v[d] * basis[j][d];
                    for (int d = 0; d < dim; d++) v[d] -= dot * basis[j][d];
                }
                NormalizeInPlace(v);
            }
            basis[i] = v;
        }
        return basis;
    }

    [Fact]
    public async Task ContinuousBatchEngine_WithAttachedSleepDaemon_ExposesStatusAndTriggerEndpoints()
    {
        string ggufPath = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(),
            $"sleep_e2e_{Guid.NewGuid():N}.gguf");
        try
        {
            PhaseCServeRemediationTests.SyntheticGgufBuilder.WriteModel(
                ggufPath,
                includeThinkTokens: false,
                useQ8Weights: false,
                forcedTargetTokenIds: [PhaseCServeRemediationTests.SyntheticGgufBuilder.GetByteTokenId((byte)'A')]);

            const int dim = 32;
            using var bank = new PreallocatedMoeExpansionBank(dim, rank: 8, maxExperts: 2);
            using var trainer = new SleepConsolidationTrainer(bank);
            using var daemon = new AutonomousSleepDaemon(trainer, new SleepDaemonOptions(
                IdleThreshold: TimeSpan.FromSeconds(60),
                PollingInterval: TimeSpan.FromMilliseconds(50),
                DistillationSteps: 20));

            using var engine = new ContinuousBatchEngine(
                ggufPath,
                totalKvBlocks: 16,
                maxBatchSize: 4,
                maxSeqLen: 64,
                useGpu: false);
            engine.AttachSleepDaemon(daemon, dim);

            // Queue a synthetic dream directly and verify /v1/sleep/status + /v1/sleep/trigger
            var rng = new Random(7);
            float[] q = CreateRandomUnitVector(rng, dim);
            float[] t = CreateRandomUnitVector(rng, dim);
            daemon.QueueDream(new DreamCandidate("prefix", "dream", q, t, 1.0f, 1.0f, 0.9f));

            using var listener = new System.Net.Sockets.TcpListener(System.Net.IPAddress.Loopback, 0);
            listener.Start();
            int port = ((System.Net.IPEndPoint)listener.LocalEndpoint).Port;
            listener.Stop();

            using var app = Glacier.Serve.Server.GlacierServeApp.CreateBuilder()
                .UseHost("127.0.0.1")
                .UsePort(port)
                .Build();
            Glacier.Serve.Inference.Http.MetricsEndpoint.MapInference(app, engine, "test-model");
            await app.StartAsync();

            using var http = new System.Net.Http.HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };
            string statusJson = await http.GetStringAsync("/v1/sleep/status");
            var status = System.Text.Json.JsonSerializer.Deserialize(
                statusJson,
                Glacier.Serve.Serialization.ServeJsonContext.Default.SleepStatusResponse);
            Assert.NotNull(status);
            Assert.True(status!.Enabled);
            Assert.Equal("Awake", status.State);
            Assert.Equal(1, status.PendingDreams);

            using var triggerRes = await http.PostAsync("/v1/sleep/trigger", null);
            triggerRes.EnsureSuccessStatusCode();
            string triggerJson = await triggerRes.Content.ReadAsStringAsync();
            var trigger = System.Text.Json.JsonSerializer.Deserialize(
                triggerJson,
                Glacier.Serve.Serialization.ServeJsonContext.Default.SleepTriggerResponse);
            Assert.NotNull(trigger);
            Assert.True(trigger!.Triggered);
            Assert.Equal(1, trigger.AcceptedDreams);
            Assert.Equal("Awake", trigger.State);

            string metricsText = await http.GetStringAsync("/metrics");
            Assert.Contains("glacier_sleep_cycles_total 1", metricsText);
            Assert.Contains("glacier_sleep_dreams_consolidated_total 1", metricsText);
        }
        finally
        {
            if (System.IO.File.Exists(ggufPath)) System.IO.File.Delete(ggufPath);
        }
    }
}

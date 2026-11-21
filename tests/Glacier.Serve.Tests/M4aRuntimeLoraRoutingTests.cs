using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Inference.Http;
using Glacier.Serve.Inference.PagedAttention;
using Glacier.Serve.Server;
using Xunit;

namespace Glacier.Serve.Tests;

public sealed class M4aRuntimeLoraRoutingTests
{
    private sealed class LegacyOnlyMockRunner : IPagedModelRunner
    {
        public bool IsGpuAccelerated => false;
        public string BackendName => "LegacyMock";
        public int VocabSize => 8;
        public int MaxSeqLen => 64;
        public int MaxBatchSize => 4;

        public bool ForwardToken(int token, int pos, BlockTable blockTable, PagedBlockPool pool, Span<float> logits, bool computeLogits = true) => true;
        public bool ForwardPrefillChunk(ReadOnlySpan<int> tokens, int startPos, BlockTable blockTable, PagedBlockPool pool, Span<float> logits, bool computeLogits = true) => true;
        public bool ForwardBatch(ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions, IReadOnlyList<BlockTable> blockTables, PagedBlockPool pool, Span<float> logitsBatch, Span<bool> successFlags, bool computeLogits = true)
        {
            successFlags.Fill(true);
            return true;
        }
        public void Dispose() { }
    }

    [Fact]
    public void IPagedModelRunner_DefaultInterfaceMethods_PreserveLegacyImplementations()
    {
        IPagedModelRunner runner = new LegacyOnlyMockRunner();
        using var pool = new PagedBlockPool(4, 1, 2, 8);
        var bt = new BlockTable(pool);
        Span<float> logits = stackalloc float[8];
        Span<bool> flags = stackalloc bool[1];

        Assert.True(runner.ForwardToken(0, 0, bt, pool, logits, true, lora: null));
        Assert.True(runner.ForwardPrefillChunk([0, 1], 0, bt, pool, logits, true, lora: null));
        Assert.True(runner.ForwardBatch([0], [0], [bt], pool, logits, flags, true, [null]));

        var dummyLora = new LoraAdapterWeights(1, 2, 4.0f);
        Assert.Throws<NotSupportedException>(() =>
        {
            Span<float> l = stackalloc float[8];
            runner.ForwardToken(0, 0, bt, pool, l, true, dummyLora);
        });
        Assert.Throws<NotSupportedException>(() =>
        {
            Span<float> l = stackalloc float[8];
            runner.ForwardPrefillChunk([0, 1], 0, bt, pool, l, true, dummyLora);
        });
        Assert.Throws<NotSupportedException>(() =>
        {
            Span<float> l = stackalloc float[8];
            Span<bool> f = stackalloc bool[1];
            runner.ForwardBatch([0], [0], [bt], pool, l, f, true, [dummyLora]);
        });
    }

    [Fact]
    public void PagedTransformerRunner_AppliesPerSequenceLoraDeltas_InTokenPrefillAndHeterogeneousBatch()
    {
        string tempDir = M4aTestHelpers.CreateTempDirectory("m4a_r2_math");
        try
        {
            const int dim = 16;
            const int ffnDim = 32;
            string[] vocab = ["<bos>", "<eos>", "t0", "t1", "t2", "t3", "t4", "t5"];
            string modelPath = Path.Combine(tempDir, "base.gguf");
            M4aTestHelpers.WriteSyntheticGguf(
                modelPath, vocab, eosTokenId: 1, bosTokenId: 0,
                dim: dim, ffnDim: ffnDim, nHeads: 2, nHeadsKv: 2, layerCount: 1,
                nonZeroLayerWeights: true);

            // Create Glacier-native adapter A (all 7 projections) and PEFT-transposed adapter B (large rank > 256)
            string loraAPath = Path.Combine(tempDir, "adapterA.bin");
            M4aTestHelpers.WriteSyntheticLoraBinary(
                loraAPath, layerCount: 1, rank: 4, alpha: 8.0f,
                entries: BuildFullLayerEntries(0, dim, ffnDim, rank: 4, seed: 11, peftTransposed: false),
                peftTransposed: false);

            string loraBDir = Path.Combine(tempDir, "adapterB_dir");
            Directory.CreateDirectory(loraBDir);
            string loraBPath = Path.Combine(loraBDir, "adapter_model.bin");
            M4aTestHelpers.WriteSyntheticLoraBinary(
                loraBPath, layerCount: 1, rank: 300, alpha: 150.0f,
                entries: BuildFullLayerEntries(0, dim, ffnDim, rank: 300, seed: 29, peftTransposed: true),
                peftTransposed: true);

            var loraA = LoraAdapterWeights.Load(loraAPath);
            var loraB = LoraAdapterWeights.Load(loraBPath);

            using var gguf = GgufFile.Open(modelPath);
            using var weights = new ModelWeights(gguf);
            using var runner = new PagedTransformerRunner(weights, maxSeqLen: 64, maxBatchSize: 8);
            using var pool = new PagedBlockPool(32, weights.BlockCount, weights.HeadCountKv, weights.HeadDim);

            int vocabSize = weights.VocabSize;
            float[] baseLogitsTok = new float[vocabSize];
            float[] loraALogitsTok = new float[vocabSize];
            float[] loraBLogitsTok = new float[vocabSize];

            // 1. Single-token path (base vs adapterA vs adapterB)
            var btBase = new BlockTable(pool);
            var btA = new BlockTable(pool);
            var btB = new BlockTable(pool);
            Assert.True(runner.ForwardToken(2, 0, btBase, pool, baseLogitsTok, true, lora: null));
            Assert.True(runner.ForwardToken(2, 0, btA, pool, loraALogitsTok, true, lora: loraA));
            Assert.True(runner.ForwardToken(2, 0, btB, pool, loraBLogitsTok, true, lora: loraB));

            AssertDistinctLogits(baseLogitsTok, loraALogitsTok);
            AssertDistinctLogits(baseLogitsTok, loraBLogitsTok);
            AssertDistinctLogits(loraALogitsTok, loraBLogitsTok);

            // 2. Prefill chunk matches step-by-step ForwardToken for adapterA
            float[] prefillA = new float[vocabSize];
            float[] stepA = new float[vocabSize];
            var btPrefillA = new BlockTable(pool);
            var btStepA = new BlockTable(pool);
            Assert.True(runner.ForwardPrefillChunk([2, 3, 4], 0, btPrefillA, pool, prefillA, true, loraA));
            Assert.True(runner.ForwardToken(2, 0, btStepA, pool, Span<float>.Empty, false, loraA));
            Assert.True(runner.ForwardToken(3, 1, btStepA, pool, Span<float>.Empty, false, loraA));
            Assert.True(runner.ForwardToken(4, 2, btStepA, pool, stepA, true, loraA));
            AssertCloseLogits(stepA, prefillA, tolerance: 1e-4f);

            // 3. Heterogeneous ForwardBatch: [null, loraA, loraB] in a single batch matches individual ForwardToken calls
            float[] indBase = new float[vocabSize];
            float[] indA = new float[vocabSize];
            float[] indB = new float[vocabSize];
            Assert.True(runner.ForwardToken(5, 1, btBase, pool, indBase, true, null));
            Assert.True(runner.ForwardToken(6, 1, btA, pool, indA, true, loraA));
            Assert.True(runner.ForwardToken(7, 1, btB, pool, indB, true, loraB));

            var btBatchBase = new BlockTable(pool);
            var btBatchA = new BlockTable(pool);
            var btBatchB = new BlockTable(pool);
            Assert.True(runner.ForwardToken(2, 0, btBatchBase, pool, Span<float>.Empty, false, null));
            Assert.True(runner.ForwardToken(2, 0, btBatchA, pool, Span<float>.Empty, false, loraA));
            Assert.True(runner.ForwardToken(2, 0, btBatchB, pool, Span<float>.Empty, false, loraB));

            float[] batchLogits = new float[3 * vocabSize];
            bool[] successFlags = new bool[3];
            Assert.True(runner.ForwardBatch(
                [5, 6, 7],
                [1, 1, 1],
                [btBatchBase, btBatchA, btBatchB],
                pool,
                batchLogits,
                successFlags,
                computeLogits: true,
                [null, loraA, loraB]));

            Assert.All(successFlags, Assert.True);
            AssertCloseLogits(indBase, batchLogits.AsSpan(0, vocabSize), tolerance: 1e-4f);
            AssertCloseLogits(indA, batchLogits.AsSpan(vocabSize, vocabSize), tolerance: 1e-4f);
            AssertCloseLogits(indB, batchLogits.AsSpan(2 * vocabSize, vocabSize), tolerance: 1e-4f);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public async Task CudaPagedTransformerRunner_WhenAvailable_SupportsHeterogeneousLoraAndEngineMixedBatch()
    {
        string tempDir = M4aTestHelpers.CreateTempDirectory("m4a_r2_cuda");
        try
        {
            const int dim = 16;
            const int ffnDim = 32;
            string[] vocab = ["<bos>", "<eos>", "p", "q", "a", "b", "c", "d"];
            string modelPath = Path.Combine(tempDir, "cuda_base.gguf");
            M4aTestHelpers.WriteSyntheticGguf(
                modelPath, vocab, eosTokenId: 1, bosTokenId: 0,
                dim: dim, ffnDim: ffnDim, nHeads: 2, nHeadsKv: 2, layerCount: 1,
                nonZeroLayerWeights: true);

            string loraPath = Path.Combine(tempDir, "cuda_lora.bin");
            M4aTestHelpers.WriteSyntheticLoraBinary(
                loraPath, layerCount: 1, rank: 4, alpha: 8.0f,
                entries: BuildFullLayerEntries(0, dim, ffnDim, rank: 4, seed: 17, peftTransposed: false));

            using var gguf = GgufFile.Open(modelPath);
            using var weights = new ModelWeights(gguf);
            if (!CudaPagedTransformerRunner.TryCreate(weights, totalBlocks: 32, blockSize: PagedBlockPool.DefaultBlockSize, maxSeqLen: 64, maxBatchSize: 8, out var cudaRunner))
            {
                // Skip CUDA-only portion cleanly on hosts without CUDA hardware
                return;
            }

            using (cudaRunner)
            using (var cpuRunner = new PagedTransformerRunner(weights, maxSeqLen: 64, maxBatchSize: 8))
            using (var gpuPool = new PagedBlockPool(32, weights.BlockCount, weights.HeadCountKv, weights.HeadDim, allocateHostBuffers: false))
            using (var cpuPool = new PagedBlockPool(32, weights.BlockCount, weights.HeadCountKv, weights.HeadDim, allocateHostBuffers: true))
            {
                var lora = LoraAdapterWeights.Load(loraPath);
                int vocabSize = weights.VocabSize;

                var btGpuBase = new BlockTable(gpuPool);
                var btGpuLora = new BlockTable(gpuPool);
                var btCpuLora = new BlockTable(cpuPool);

                float[] gpuBaseLogits = new float[vocabSize];
                float[] gpuLoraLogits = new float[vocabSize];
                float[] cpuLoraLogits = new float[vocabSize];

                Assert.True(cudaRunner.ForwardToken(2, 0, btGpuBase, gpuPool, gpuBaseLogits, true, null));
                Assert.True(cudaRunner.ForwardToken(2, 0, btGpuLora, gpuPool, gpuLoraLogits, true, lora));
                Assert.True(cpuRunner.ForwardToken(2, 0, btCpuLora, cpuPool, cpuLoraLogits, true, lora));

                AssertDistinctLogits(gpuBaseLogits, gpuLoraLogits);
                AssertCloseLogits(cpuLoraLogits, gpuLoraLogits, tolerance: 1e-4f);

                // Heterogeneous batch on CudaPagedTransformerRunner: [null, lora]
                float[] mixedLogits = new float[2 * vocabSize];
                bool[] mixedSuccess = new bool[2];
                Assert.True(cudaRunner.ForwardBatch(
                    [3, 3],
                    [1, 1],
                    [btGpuBase, btGpuLora],
                    gpuPool,
                    mixedLogits,
                    mixedSuccess,
                    computeLogits: true,
                    [null, lora]));
                Assert.All(mixedSuccess, Assert.True);
                AssertDistinctLogits(mixedLogits.AsSpan(0, vocabSize), mixedLogits.AsSpan(vocabSize, vocabSize));
            }

            // Also verify ContinuousBatchEngine with useGpu: true routes base + LoRA concurrently
            using var engine = new ContinuousBatchEngine(modelPath, totalKvBlocks: 64, maxBatchSize: 8, maxSeqLen: 64, seed: 3, useGpu: true);
            engine.LoadLoraAdapter("gpu-lora", loraPath);

            var tBase = engine.EnqueueAsync(new InferenceRequest { Prompt = "p", MaxTokens = 3, Temperature = 0f });
            var tLora = engine.EnqueueAsync(new InferenceRequest { Prompt = "p", Lora = "gpu-lora", MaxTokens = 3, Temperature = 0f });
            var (sBase, stBase) = await tBase;
            var (sLora, stLora) = await tLora;
            await foreach (var _ in stBase) { }
            await foreach (var _ in stLora) { }
            await sBase.Completion.Task;
            await sLora.Completion.Task;
            Assert.Equal(3, sBase.GeneratedTokens.Count);
            Assert.Equal(3, sLora.GeneratedTokens.Count);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public async Task ContinuousBatchEngine_HotLoadUnloadReload_PrefixCacheIsolation_And_InFlightSafety()
    {
        string tempDir = M4aTestHelpers.CreateTempDirectory("m4a_r2_engine");
        try
        {
            const int dim = 16;
            const int ffnDim = 32;
            // Create tokens so prompt >= 16 tokens (1+ full KV block) to exercise prefix caching
            string[] vocab = ["<bos>", "<eos>", "w", "\u0120w", "outA", "outB"];
            float[] outW = M4aTestHelpers.CreateDeterministicMatrix(vocab.Length * dim, 0.05f, 2);
            // Zero <eos> (token 1) row and enforce an opposite pair on two non-EOS tokens so <eos> is never argmax when normX != 0
            for (int d = 0; d < dim; d++)
            {
                outW[1 * dim + d] = 0.0f;
                outW[5 * dim + d] = -outW[4 * dim + d];
            }
            string modelPath = Path.Combine(tempDir, "prefix_base.gguf");
            M4aTestHelpers.WriteSyntheticGguf(
                modelPath, vocab, eosTokenId: 1, bosTokenId: 0,
                dim: dim, ffnDim: ffnDim, nHeads: 2, nHeadsKv: 2, layerCount: 1,
                outputWeights: outW,
                nonZeroLayerWeights: true);

            string loraADir = Path.Combine(tempDir, "roslyn");
            Directory.CreateDirectory(loraADir);
            string loraAPath = Path.Combine(loraADir, "adapter_model.bin");
            M4aTestHelpers.WriteSyntheticLoraBinary(
                loraAPath, layerCount: 1, rank: 2, alpha: 4.0f,
                entries: BuildFullLayerEntries(0, dim, ffnDim, rank: 2, seed: 5));

            string loraBPath = Path.Combine(tempDir, "pi.bin");
            M4aTestHelpers.WriteSyntheticLoraBinary(
                loraBPath, layerCount: 1, rank: 2, alpha: 4.0f,
                entries: BuildFullLayerEntries(0, dim, ffnDim, rank: 2, seed: 19));

            using var engine = new ContinuousBatchEngine(modelPath, totalKvBlocks: 256, maxBatchSize: 8, maxSeqLen: 512, seed: 42, useGpu: false);

            // Validation errors at load time
            Assert.Throws<ArgumentException>(() => engine.LoadLoraAdapter("bad:name", loraAPath));
            Assert.Throws<FileNotFoundException>(() => engine.LoadLoraAdapter("missing", Path.Combine(tempDir, "does_not_exist.bin")));

            string badDimLora = Path.Combine(tempDir, "bad_dim.bin");
            M4aTestHelpers.WriteSyntheticLoraBinary(
                badDimLora, layerCount: 1, rank: 2, alpha: 2.0f,
                entries: [(0, "attn_q", dim + 4, dim, new float[(dim + 4) * 2], new float[2 * dim])]);
            Assert.Throws<InvalidDataException>(() => engine.LoadLoraAdapter("bad-dim", badDimLora));

            // Hot-load directory & file adapters
            engine.LoadLoraAdapter("roslyn", loraADir);
            engine.LoadLoraAdapter("pi-adapted", loraBPath);
            Assert.Equal(new[] { "pi-adapted", "roslyn" }, engine.GetLoadedLoraAdapters());

            string longPrompt = string.Join(" ", Enumerable.Repeat("w", 20));

            using var holdCts = new System.Threading.CancellationTokenSource();

            // Keep base and roslyn sequences in-flight (long MaxTokens, cancelled cleanly via holdCts) while testing prefix cache isolation & hits
            // 1. Base request populates base prefix cache (wait until prefill done so block 0 is bound & held)
            var (sBase1, stBase1) = await engine.EnqueueAsync(new InferenceRequest { Prompt = longPrompt, MaxTokens = 400, Temperature = 0f }, holdCts.Token);
            Assert.True(sBase1.PromptTokens.Length >= 16);
            Assert.Equal(0, sBase1.BoundPrefixBlocks);
            var enumBase1 = stBase1.GetAsyncEnumerator();
            Assert.True(await enumBase1.MoveNextAsync()); // prefill is done; block 0 is bound and held

            // 2. First roslyn request MUST NOT reuse base's cached KV blocks while sBase1 holds them
            var (sRos1, stRos1) = await engine.EnqueueAsync(new InferenceRequest { Model = "qwen-14b:roslyn", Prompt = longPrompt, MaxTokens = 400, Temperature = 0f }, holdCts.Token);
            Assert.Equal("roslyn", sRos1.LoraAdapterName);
            Assert.Equal(0, sRos1.BoundPrefixBlocks);
            var enumRos1 = stRos1.GetAsyncEnumerator();
            Assert.True(await enumRos1.MoveNextAsync()); // prefill is done; roslyn block 0 is now bound and held

            // 3. Second roslyn request with same prompt MUST hit roslyn's live cached KV blocks
            var (sRos2, stRos2) = await engine.EnqueueAsync(new InferenceRequest { Lora = "roslyn", Prompt = longPrompt, MaxTokens = 2, Temperature = 0f });
            Assert.True(sRos2.BoundPrefixBlocks >= 1);
            await foreach (var _ in stRos2) { }
            await sRos2.Completion.Task;

            // 4. First pi-adapted request MUST NOT reuse base or roslyn's cached KV blocks
            var (sPi1, stPi1) = await engine.EnqueueAsync(new InferenceRequest { Model = "qwen-14b:pi-adapted", Prompt = longPrompt, MaxTokens = 2, Temperature = 0f });
            Assert.Equal("pi-adapted", sPi1.LoraAdapterName);
            Assert.Equal(0, sPi1.BoundPrefixBlocks);
            await foreach (var _ in stPi1) { }
            await sPi1.Completion.Task;

            // 5. Hot-reload roslyn with updated weights while sRos1 is STILL holding v1 cached blocks -> invalidates old roslyn prefix cache automatically
            M4aTestHelpers.WriteSyntheticLoraBinary(
                loraAPath, layerCount: 1, rank: 2, alpha: 6.0f,
                entries: BuildFullLayerEntries(0, dim, ffnDim, rank: 2, seed: 31));
            engine.LoadLoraAdapter("roslyn", loraAPath);
            var (sRosReloaded, stRosReloaded) = await engine.EnqueueAsync(new InferenceRequest { Lora = "roslyn", Prompt = longPrompt, MaxTokens = 2, Temperature = 0f });
            Assert.Equal(0, sRosReloaded.BoundPrefixBlocks);
            Assert.NotEqual(sRos1.LoraAdapterInstanceId, sRosReloaded.LoraAdapterInstanceId);
            await foreach (var _ in stRosReloaded) { }
            await sRosReloaded.Completion.Task;

            // Release held sequences cleanly
            holdCts.Cancel();
            await enumBase1.DisposeAsync();
            await enumRos1.DisposeAsync();
            try { await sBase1.Completion.Task; } catch (OperationCanceledException) { }
            try { await sRos1.Completion.Task; } catch (OperationCanceledException) { }

            // 6. Enqueue in-flight request, immediately UnloadLoraAdapter -> in-flight completes, subsequent fails
            var (sInFlight, stInFlight) = await engine.EnqueueAsync(new InferenceRequest { Lora = "roslyn", Prompt = longPrompt, MaxTokens = 4, Temperature = 0f });
            Assert.True(engine.UnloadLoraAdapter("roslyn"));
            Assert.False(engine.UnloadLoraAdapter("roslyn"));
            Assert.Equal(new[] { "pi-adapted" }, engine.GetLoadedLoraAdapters());

            await foreach (var _ in stInFlight) { }
            await sInFlight.Completion.Task;
            Assert.Equal(4, sInFlight.GeneratedTokens.Count);

            await Assert.ThrowsAsync<KeyNotFoundException>(() =>
                engine.EnqueueAsync(new InferenceRequest { Model = "qwen-14b:roslyn", Prompt = "w", MaxTokens = 2 }));
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public async Task OpenAiAndOllamaEndpoints_RouteLoraAdapters_ListModels_AndReturnClean400And404Errors()
    {
        string tempDir = M4aTestHelpers.CreateTempDirectory("m4a_r2_http");
        try
        {
            const int dim = 16;
            const int ffnDim = 32;
            string[] vocab = ["<bos>", "<eos>", "h", "o", "<|im_start|>", "<|im_end|>", "\u010A"];
            float[] embd = new float[vocab.Length * dim];
            float[] outW = new float[vocab.Length * dim];
            embd[2 * dim + 0] = 1.0f;
            embd[6 * dim + 0] = 1.0f;
            outW[3 * dim + 0] = 10.0f;
            embd[3 * dim + 1] = 1.0f;
            outW[1 * dim + 1] = 10.0f;

            string modelPath = Path.Combine(tempDir, "http_lora_base.gguf");
            M4aTestHelpers.WriteSyntheticGguf(
                modelPath, vocab, eosTokenId: 1, bosTokenId: 0,
                dim: dim, ffnDim: ffnDim, nHeads: 2, nHeadsKv: 2, layerCount: 1,
                tokenEmbeddings: embd, outputWeights: outW);

            string loraPath = Path.Combine(tempDir, "roslyn.bin");
            M4aTestHelpers.WriteSyntheticLoraBinary(
                loraPath, layerCount: 1, rank: 2, alpha: 4.0f,
                entries: BuildFullLayerEntries(0, dim, ffnDim, rank: 2, seed: 7));

            using var engine = new ContinuousBatchEngine(modelPath, totalKvBlocks: 64, maxBatchSize: 8, maxSeqLen: 64, seed: 1, useGpu: false);
            engine.LoadLoraAdapter("roslyn", loraPath);

            int port = M4aTestHelpers.GetFreeTcpPort();
            var builder = GlacierServeApp.CreateBuilder();
            builder.UsePort(port);
            using var app = builder.Build();
            // Deliberately use a defaultModelName containing a colon ("qwen2.5:7b") to verify LastIndexOf(':') handling
            app.MapOpenAiEndpoints(engine, defaultModelName: "qwen2.5:7b");
            app.MapOllamaEndpoints(engine, defaultModelName: "qwen2.5:7b");
            await app.StartAsync();

            try
            {
                using var http = M4aTestHelpers.CreateHttpClient(port);

                // 1. GET /v1/models and GET /api/tags include base + qwen2.5:7b:roslyn
                using (var v1Doc = JsonDocument.Parse(await http.GetStringAsync("/v1/models")))
                {
                    var ids = v1Doc.RootElement.GetProperty("data").EnumerateArray().Select(e => e.GetProperty("id").GetString()).ToArray();
                    Assert.Equal(new[] { "qwen2.5:7b", "qwen2.5:7b:roslyn" }, ids);
                }
                using (var tagsDoc = JsonDocument.Parse(await http.GetStringAsync("/api/tags")))
                {
                    var names = tagsDoc.RootElement.GetProperty("models").EnumerateArray().Select(e => e.GetProperty("name").GetString()).ToArray();
                    Assert.Equal(new[] { "qwen2.5:7b", "qwen2.5:7b:roslyn" }, names);
                }

                // 2. Base model containing colon ("qwen2.5:7b") routes to base without treating "7b" as LoRA
                using (var baseResp = await http.PostAsync("/v1/completions", JsonBody("""{"model":"qwen2.5:7b","prompt":"h","max_tokens":2,"temperature":0}""")))
                {
                    Assert.Equal(HttpStatusCode.OK, baseResp.StatusCode);
                }

                // 3. Composite "qwen2.5:7b:roslyn" routes to roslyn
                using (var compResp = await http.PostAsync("/v1/chat/completions", JsonBody("""{"model":"qwen2.5:7b:roslyn","messages":[{"role":"user","content":"h"}],"max_tokens":2,"temperature":0}""")))
                {
                    Assert.Equal(HttpStatusCode.OK, compResp.StatusCode);
                    using var doc = JsonDocument.Parse(await compResp.Content.ReadAsStringAsync());
                    Assert.Equal("qwen2.5:7b:roslyn", doc.RootElement.GetProperty("model").GetString());
                }

                // 4. Explicit "lora": "roslyn" on /api/generate (non-streaming)
                using (var loraResp = await http.PostAsync("/api/generate", JsonBody("""{"model":"qwen2.5:7b","lora":"roslyn","prompt":"h","max_tokens":2,"temperature":0,"stream":false}""")))
                {
                    Assert.Equal(HttpStatusCode.OK, loraResp.StatusCode);
                }

                // 5. Unknown LoRA returns HTTP 404 JSON ApiErrorResponse
                using (var notFoundResp = await http.PostAsync("/v1/chat/completions", JsonBody("""{"model":"qwen2.5:7b:does_not_exist","messages":[{"role":"user","content":"h"}],"max_tokens":2}""")))
                {
                    Assert.Equal(HttpStatusCode.NotFound, notFoundResp.StatusCode);
                    using var errDoc = JsonDocument.Parse(await notFoundResp.Content.ReadAsStringAsync());
                    Assert.Equal("model_not_found", errDoc.RootElement.GetProperty("error").GetProperty("type").GetString());
                }

                // 6. Conflicting model:adapter + lora returns HTTP 400 JSON ApiErrorResponse
                using (var conflictResp = await http.PostAsync("/v1/completions", JsonBody("""{"model":"qwen2.5:7b:roslyn","lora":"other","prompt":"h","max_tokens":2}""")))
                {
                    Assert.Equal(HttpStatusCode.BadRequest, conflictResp.StatusCode);
                    using var errDoc = JsonDocument.Parse(await conflictResp.Content.ReadAsStringAsync());
                    Assert.Equal("invalid_request_error", errDoc.RootElement.GetProperty("error").GetProperty("type").GetString());
                }

                // 7. Malformed trailing colon returns HTTP 400
                using (var malformedResp = await http.PostAsync("/api/chat", JsonBody("""{"model":"qwen2.5:7b:","messages":[{"role":"user","content":"h"}],"stream":false}""")))
                {
                    Assert.Equal(HttpStatusCode.BadRequest, malformedResp.StatusCode);
                }

                // 8. Unload removes adapter from GET /v1/models dynamically
                Assert.True(engine.UnloadLoraAdapter("roslyn"));
                using (var v1After = JsonDocument.Parse(await http.GetStringAsync("/v1/models")))
                {
                    var idsAfter = v1After.RootElement.GetProperty("data").EnumerateArray().Select(e => e.GetProperty("id").GetString()).ToArray();
                    Assert.Equal(new[] { "qwen2.5:7b" }, idsAfter);
                }
            }
            finally
            {
                app.Stop();
            }
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    private static StringContent JsonBody(string json) => new(json, Encoding.UTF8, "application/json");

    private static List<(int Layer, string ProjName, int InDim, int OutDim, float[] A, float[] B)> BuildFullLayerEntries(
        int layer, int dim, int ffnDim, int rank, int seed, bool peftTransposed = false)
    {
        _ = peftTransposed;
        return
        [
            (layer, "attn_q", dim, dim, M4aTestHelpers.CreateDeterministicMatrix(dim * rank, 0.05f, seed + 1), M4aTestHelpers.CreateDeterministicMatrix(rank * dim, 0.05f, seed + 2)),
            (layer, "attn_k", dim, dim, M4aTestHelpers.CreateDeterministicMatrix(dim * rank, 0.05f, seed + 3), M4aTestHelpers.CreateDeterministicMatrix(rank * dim, 0.05f, seed + 4)),
            (layer, "attn_v", dim, dim, M4aTestHelpers.CreateDeterministicMatrix(dim * rank, 0.05f, seed + 5), M4aTestHelpers.CreateDeterministicMatrix(rank * dim, 0.05f, seed + 6)),
            (layer, "attn_output", dim, dim, M4aTestHelpers.CreateDeterministicMatrix(dim * rank, 0.05f, seed + 7), M4aTestHelpers.CreateDeterministicMatrix(rank * dim, 0.05f, seed + 8)),
            (layer, "ffn_gate", dim, ffnDim, M4aTestHelpers.CreateDeterministicMatrix(dim * rank, 0.05f, seed + 9), M4aTestHelpers.CreateDeterministicMatrix(rank * ffnDim, 0.05f, seed + 10)),
            (layer, "ffn_up", dim, ffnDim, M4aTestHelpers.CreateDeterministicMatrix(dim * rank, 0.05f, seed + 11), M4aTestHelpers.CreateDeterministicMatrix(rank * ffnDim, 0.05f, seed + 12)),
            (layer, "ffn_down", ffnDim, dim, M4aTestHelpers.CreateDeterministicMatrix(ffnDim * rank, 0.05f, seed + 13), M4aTestHelpers.CreateDeterministicMatrix(rank * dim, 0.05f, seed + 14))
        ];
    }

    private static void AssertDistinctLogits(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float maxDiff = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            float d = MathF.Abs(a[i] - b[i]);
            if (d > maxDiff) maxDiff = d;
        }
        Assert.True(maxDiff > 1e-3f, $"Expected distinct logits, but maxDiff was {maxDiff}");
    }

    private static void AssertCloseLogits(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            Assert.True(diff <= tolerance, $"Logit[{i}] mismatch: expected {expected[i]}, actual {actual[i]} (diff={diff})");
        }
    }
}

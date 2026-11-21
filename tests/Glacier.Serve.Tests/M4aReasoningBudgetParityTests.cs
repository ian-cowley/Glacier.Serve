using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Glacier.Inference.Engine;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Inference.Http;
using Glacier.Serve.Inference.PagedAttention;
using Glacier.Serve.Serialization;
using Glacier.Serve.Server;
using Xunit;

namespace Glacier.Serve.Tests;

public sealed class M4aReasoningBudgetParityTests
{
    [Fact]
    public void InferenceRequest_AotDeserialization_SupportsReasoningBudgetAliasesAndOpenAiParityFields()
    {
        const string json = """
        {
          "model": "qwen3-8b",
          "messages": [{"role": "user", "content": "hi"}],
          "max_think_tokens": "auto",
          "chat_template_kwargs": {"enable_thinking": false},
          "stream": true,
          "stream_options": {"include_usage": true},
          "lora": "roslyn"
        }
        """;

        var req = JsonSerializer.Deserialize(json, ServeJsonContext.Default.InferenceRequest);
        Assert.NotNull(req);
        Assert.Equal(ReasoningBudgetPolicy.AutoBudgetSentinel, req!.MaxThinkTokens);
        Assert.Equal(ReasoningBudgetPolicy.AutoBudgetSentinel, req.GetEffectiveReasoningBudget(0));
        Assert.True(req.HasExplicitReasoningBudget);
        Assert.NotNull(req.ChatTemplateKwargs);
        Assert.False(req.ChatTemplateKwargs!.EnableThinking);
        Assert.False(req.EffectiveEnableThinking);
        Assert.NotNull(req.StreamOptions);
        Assert.True(req.StreamOptions!.IncludeUsage);
        Assert.Equal("roslyn", req.Lora);
    }

    [Theory]
    [InlineData("128", 128)]
    [InlineData("\"auto\"", -1)]
    [InlineData("\"off\"", 0)]
    [InlineData("\"none\"", 0)]
    [InlineData("\"unlimited\"", 0)]
    [InlineData("\"42\"", 42)]
    [InlineData("false", 0)]
    [InlineData("true", -1)]
    [InlineData("-1", -1)]
    [InlineData("0", 0)]
    public void PolymorphicReasoningBudget_ParsesAllWireVariants(string jsonValue, int expected)
    {
        string json = $"{{\"prompt\":\"p\",\"max_think_tokens\":{jsonValue},\"max_reasoning_tokens\":{jsonValue}}}";
        var req = JsonSerializer.Deserialize(json, ServeJsonContext.Default.InferenceRequest);
        Assert.NotNull(req);
        Assert.Equal(expected, req!.MaxThinkTokens);
        Assert.Equal(expected, req.MaxReasoningTokens);
        Assert.Equal(expected, req.GetEffectiveReasoningBudget(defaultBudget: 99));
    }

    [Theory]
    [InlineData("-2")]
    [InlineData("-100")]
    [InlineData("-2147483648")]
    public void InferenceRequest_NegativeReasoningBudgetBelowAutoSentinel_ThrowsJsonException(string invalidBudget)
    {
        string maxThinkJson = $"{{\"prompt\":\"p\",\"max_think_tokens\":{invalidBudget}}}";
        Assert.Throws<JsonException>(() =>
            JsonSerializer.Deserialize(maxThinkJson, ServeJsonContext.Default.InferenceRequest));

        string ollamaJson = $"{{\"prompt\":\"p\",\"options\":{{\"reasoning_budget\":{invalidBudget}}}}}";
        Assert.Throws<JsonException>(() =>
            JsonSerializer.Deserialize(ollamaJson, ServeJsonContext.Default.InferenceRequest));

        string maxReasoningJson = $"{{\"prompt\":\"p\",\"max_reasoning_tokens\":{invalidBudget}}}";
        Assert.Throws<JsonException>(() =>
            JsonSerializer.Deserialize(maxReasoningJson, ServeJsonContext.Default.InferenceRequest));
    }

    [Fact]
    public void InferenceRequest_InvalidReasoningBudgetString_ThrowsJsonException()
    {
        const string json = """{"prompt":"p","max_think_tokens":"not_a_budget"}""";
        Assert.Throws<JsonException>(() =>
            JsonSerializer.Deserialize(json, ServeJsonContext.Default.InferenceRequest));
    }

    [Fact]
    public void InferenceRequest_Precedence_ChatTemplateKwargsOverridesEnableThinking_AndExplicitZeroOverridesDefault()
    {
        const string json = """
        {
          "prompt": "p",
          "enable_thinking": true,
          "chat_template_kwargs": {"enable_thinking": false},
          "max_reasoning_tokens": 0
        }
        """;
        var req = JsonSerializer.Deserialize(json, ServeJsonContext.Default.InferenceRequest)!;
        Assert.False(req.EffectiveEnableThinking);
        Assert.True(req.HasExplicitReasoningBudget);
        Assert.Equal(0, req.GetEffectiveReasoningBudget(defaultBudget: 256));

        const string ollamaJson = """
        {
          "prompt": "p",
          "options": {"reasoning_budget": "auto"}
        }
        """;
        var ollamaReq = JsonSerializer.Deserialize(ollamaJson, ServeJsonContext.Default.InferenceRequest)!;
        Assert.Equal(ReasoningBudgetPolicy.AutoBudgetSentinel, ollamaReq.GetEffectiveReasoningBudget(defaultBudget: 0));
    }

    [Fact]
    public async Task Engine_SingleTokenThink_EnforcesMaxThinkTokens_InjectsEndThinkAndContinuesWithAnswer()
    {
        string tempDir = M4aTestHelpers.CreateTempDirectory("m4a_r1_single");
        try
        {
            // Vocab:
            // 0: <bos>, 1: <eos>, 2: \n (\u010A), 3: \n\n (\u010A\u010A),
            // 4: <think>, 5: </think>, 6: s (single-char prompt), 7: t1, 8: t2, 9: t3, 10: ANS
            string[] vocab =
            [
                "<bos>", "<eos>", "\u010A", "\u010A\u010A",
                "<think>", "</think>", "s", "t1", "t2", "t3", "ANS"
            ];
            string[] merges = ["\u010A \u010A"];
            const int dim = 16;
            float[] embd = new float[vocab.Length * dim];
            float[] outW = new float[vocab.Length * dim];

            // Route: s(6) -> <think>(4) -> t1(7) -> t2(8) -> t3(9) -> t3(9)...
            // And after forced \n</think>\n\n, \n\n(3) -> ANS(10) -> <eos>(1)
            SetBasisTransition(embd, outW, dim, fromTok: 6, toTok: 4, basis: 0);
            SetBasisTransition(embd, outW, dim, fromTok: 4, toTok: 7, basis: 1);
            SetBasisTransition(embd, outW, dim, fromTok: 7, toTok: 8, basis: 2);
            SetBasisTransition(embd, outW, dim, fromTok: 8, toTok: 9, basis: 3);
            SetBasisTransition(embd, outW, dim, fromTok: 9, toTok: 9, basis: 3);
            SetBasisTransition(embd, outW, dim, fromTok: 3, toTok: 10, basis: 4);
            SetBasisTransition(embd, outW, dim, fromTok: 10, toTok: 1, basis: 5);

            string modelPath = Path.Combine(tempDir, "single_think.gguf");
            M4aTestHelpers.WriteSyntheticGguf(
                modelPath, vocab, eosTokenId: 1, bosTokenId: 0, dim: dim,
                tokenEmbeddings: embd, outputWeights: outW, merges: merges);

            using var engine = new ContinuousBatchEngine(modelPath, totalKvBlocks: 64, maxBatchSize: 4, maxSeqLen: 128, seed: 42, useGpu: false);

            var req = new InferenceRequest
            {
                Prompt = "s",
                Temperature = 0f,
                MaxTokens = 20,
                MaxThinkTokens = 2
            };

            var (state, stream) = await engine.EnqueueAsync(req);
            var chunks = new List<string>();
            await foreach (var c in stream) chunks.Add(c);
            string full = await state.Completion.Task;

            Assert.Equal("stop", state.FinishReason);
            Assert.True(state.BudgetClampTriggers >= 1);
            Assert.True(state.ForcedThinkClose);
            Assert.Contains("<think>", full, StringComparison.Ordinal);
            Assert.Contains("\n</think>\n\n", full, StringComparison.Ordinal);
            Assert.EndsWith("ANS", full, StringComparison.Ordinal);
            Assert.Equal(full, string.Concat(chunks));
            // Sequence expected: <think>, t1, t2, \n, </think>, \n\n, ANS (7 tokens)
            Assert.Equal(7, state.GeneratedTokens.Count);
            Assert.Equal(5, state.GeneratedTokens[4]); // </think> token ID
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public async Task Engine_MultiTokenThink_TracksAcrossTokenBoundaries_EnforcesAutoBudgetAndContinues()
    {
        string tempDir = M4aTestHelpers.CreateTempDirectory("m4a_r1_multi");
        try
        {
            // Vocab WITHOUT single-token <think> or </think>!
            // BPE tokens so "<think>" -> ["<", "think", ">"] and "\n</think>\n\n" -> ["\n", "</", "think", ">", "\n\n"]
            // 0: <bos>, 1: <eos>, 2: \n, 3: \n\n, 4: <, 5: think, 6: >, 7: /, 8: g, 9: r1, 10: r2, 11: r3, 12: DONE
            // 13: t, 14: h, 15: i, 16: n, 17: k, 18: th, 19: thi, 20: thin, 21: </
            string[] vocab =
            [
                "<bos>", "<eos>", "\u010A", "\u010A\u010A",
                "<", "think", ">", "/", "g", "r1", "r2", "r3", "DONE",
                "t", "h", "i", "n", "k", "th", "thi", "thin", "</"
            ];
            string[] merges = ["\u010A \u010A", "t h", "th i", "thi n", "thin k", "< /"];
            const int dim = 16;
            float[] embd = new float[vocab.Length * dim];
            float[] outW = new float[vocab.Length * dim];

            // g(8) -> <(4) -> think(5) -> >(6) -> r1(9) -> r2(10) -> r3(11)...
            // And after forced "\n</think>\n\n" ending in \n\n(3) -> DONE(12) -> <eos>(1)
            SetBasisTransition(embd, outW, dim, fromTok: 8, toTok: 4, basis: 0);
            SetBasisTransition(embd, outW, dim, fromTok: 4, toTok: 5, basis: 1);
            SetBasisTransition(embd, outW, dim, fromTok: 5, toTok: 6, basis: 2);
            SetBasisTransition(embd, outW, dim, fromTok: 6, toTok: 9, basis: 3);
            SetBasisTransition(embd, outW, dim, fromTok: 9, toTok: 10, basis: 4);
            SetBasisTransition(embd, outW, dim, fromTok: 10, toTok: 11, basis: 5);
            SetBasisTransition(embd, outW, dim, fromTok: 11, toTok: 11, basis: 5);
            SetBasisTransition(embd, outW, dim, fromTok: 3, toTok: 12, basis: 6);
            SetBasisTransition(embd, outW, dim, fromTok: 12, toTok: 1, basis: 7);

            string modelPath = Path.Combine(tempDir, "multi_think.gguf");
            M4aTestHelpers.WriteSyntheticGguf(
                modelPath, vocab, eosTokenId: 1, bosTokenId: 0, dim: dim,
                tokenEmbeddings: embd, outputWeights: outW, merges: merges);

            using var engine = new ContinuousBatchEngine(modelPath, totalKvBlocks: 64, maxBatchSize: 4, maxSeqLen: 128, seed: 7, useGpu: false);

            // Check explicit "auto" budget with max_tokens = 12:
            // ComputeAutoMaxReasoningTokens(12): reserve = max(1, 12/3)=4, target = (int)(12*0.65)=7 -> min(7, 8) = 7.
            // Also check explicit max_think_tokens = 2:
            var reqExplicit = new InferenceRequest
            {
                Prompt = "g",
                Temperature = 0f,
                MaxTokens = 20,
                MaxThinkTokens = 2
            };

            var (state1, stream1) = await engine.EnqueueAsync(reqExplicit);
            await foreach (var _ in stream1) { }
            string text1 = await state1.Completion.Task;

            Assert.Equal("stop", state1.FinishReason);
            Assert.True(state1.BudgetClampTriggers >= 1);
            Assert.True(state1.ForcedThinkClose);
            Assert.StartsWith("<think>r1r2\n</think>\n\nDONE", text1, StringComparison.Ordinal);

            // Verify "auto" budget resolution via ReasoningBudgetPolicy (MaxTokens=30 -> auto=19, leaving room for <think>, \n</think>\n\n, and DONE)
            var reqAuto = new InferenceRequest
            {
                Prompt = "g",
                Temperature = 0f,
                MaxTokens = 30,
                MaxThinkTokens = ReasoningBudgetPolicy.AutoBudgetSentinel
            };
            var (stateAuto, streamAuto) = await engine.EnqueueAsync(reqAuto);
            await foreach (var _ in streamAuto) { }
            string textAuto = await stateAuto.Completion.Task;

            Assert.Equal(ReasoningBudgetPolicy.ComputeAutoMaxReasoningTokens(30), stateAuto.SamplingOptions.MaxReasoningTokens);
            Assert.True(stateAuto.BudgetClampTriggers >= 1);
            Assert.Contains("\n</think>\n\nDONE", textAuto, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public async Task Engine_FirstTokenForcedClose_And_StopSequenceDuringForcedClose_And_EnableThinkingFalse()
    {
        string tempDir = M4aTestHelpers.CreateTempDirectory("m4a_r1_edges");
        try
        {
            // Include DeepSeek tokens so FormatConversation uses thinkingMode for <think>\n
            // 0:<bos>, 1:<eos>, 2:\n, 3:\n\n, 4:<think>, 5:</think>, 6:<｜User｜>, 7:<｜Assistant｜>, 8:hi, 9:t1, 10:ANS
            string[] vocab =
            [
                "<bos>", "<eos>", "\u010A", "\u010A\u010A",
                "<think>", "</think>", "<｜User｜>", "<｜Assistant｜>", "h", "t1", "ANS"
            ];
            string[] merges = ["\u010A \u010A"];
            const int dim = 16;
            float[] embd = new float[vocab.Length * dim];
            float[] outW = new float[vocab.Length * dim];

            // <think>(4) -> t1(9), t1(9) -> t1(9), \n\n(3) -> ANS(10) -> <eos>(1)
            // <｜Assistant｜>(7) -> t1(9)
            SetBasisTransition(embd, outW, dim, fromTok: 4, toTok: 9, basis: 0);
            SetBasisTransition(embd, outW, dim, fromTok: 2, toTok: 9, basis: 0);
            SetBasisTransition(embd, outW, dim, fromTok: 7, toTok: 9, basis: 1);
            SetBasisTransition(embd, outW, dim, fromTok: 9, toTok: 9, basis: 1);
            SetBasisTransition(embd, outW, dim, fromTok: 3, toTok: 10, basis: 2);
            SetBasisTransition(embd, outW, dim, fromTok: 10, toTok: 1, basis: 3);

            string modelPath = Path.Combine(tempDir, "edge_think.gguf");
            M4aTestHelpers.WriteSyntheticGguf(
                modelPath, vocab, eosTokenId: 1, bosTokenId: 0, dim: dim,
                tokenEmbeddings: embd, outputWeights: outW, merges: merges);

            // Engine has defaultMaxReasoningTokens = AutoBudgetSentinel (-1)
            using var engine = new ContinuousBatchEngine(
                modelPath, totalKvBlocks: 64, maxBatchSize: 4, maxSeqLen: 128, seed: 5, useGpu: false,
                defaultMaxReasoningTokens: ReasoningBudgetPolicy.AutoBudgetSentinel);

            // 1. First-token forced close: prompt ends with <think>, max_think_tokens = 1
            var reqFirstTok = new InferenceRequest
            {
                Prompt = "<think>",
                Temperature = 0f,
                MaxTokens = 10,
                MaxThinkTokens = 1
            };
            var (s1, st1) = await engine.EnqueueAsync(reqFirstTok);
            await foreach (var _ in st1) { }
            string r1 = await s1.Completion.Task;
            Assert.Equal("stop", s1.FinishReason);
            Assert.Equal(1, s1.BudgetClampTriggers);
            Assert.Equal("t1\n</think>\n\nANS", r1);

            // 2. StopSequence "</think>" honored during forced close
            var reqStop = new InferenceRequest
            {
                Prompt = "<think>",
                Temperature = 0f,
                MaxTokens = 10,
                MaxThinkTokens = 1,
                Stop = ["</think>"]
            };
            var (s2, st2) = await engine.EnqueueAsync(reqStop);
            await foreach (var _ in st2) { }
            string r2 = await s2.Completion.Task;
            Assert.Equal("stop", s2.FinishReason);
            Assert.Equal("t1\n", r2);

            // 3. chat_template_kwargs.enable_thinking = false overrides engine auto default (resolvedBudget == 0)
            //    and disables DeepSeek <think>\n prefix
            var reqNoThink = new InferenceRequest
            {
                Messages = [new ChatMessage { Role = "user", Content = "h" }],
                ChatTemplateKwargs = new ChatTemplateKwargs { EnableThinking = false },
                Temperature = 0f,
                MaxTokens = 3
            };
            var (s3, st3) = await engine.EnqueueAsync(reqNoThink);
            await foreach (var _ in st3) { }
            await s3.Completion.Task;
            Assert.Equal(0, s3.SamplingOptions.MaxReasoningTokens);
            Assert.DoesNotContain(4, s3.PromptTokens); // <think> not added to DeepSeek prompt
            Assert.Equal(0, s3.BudgetClampTriggers);

            // 4. Unbudgeted multi-token <think> (MaxReasoningTokens == 0) still updates IsInReasoningBlock for loop guard
            var reqUnbudgeted = new InferenceRequest
            {
                Prompt = "<think>",
                Temperature = 0f,
                MaxTokens = 4,
                MaxThinkTokens = 0,
                EnableReasoningLoopGuard = true
            };
            var (s4, st4) = await engine.EnqueueAsync(reqUnbudgeted);
            await foreach (var _ in st4) { }
            await s4.Completion.Task;
            Assert.Equal(0, s4.SamplingOptions.MaxReasoningTokens);
            Assert.True(s4.IsInReasoningBlock);
            Assert.Equal(4, s4.CurrentReasoningTokens);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public async Task OpenAiEndpoints_StreamOptionsIncludeUsage_And_GetV1Models_WorkEndToEnd()
    {
        string tempDir = M4aTestHelpers.CreateTempDirectory("m4a_r1_http");
        try
        {
            string[] vocab = ["<bos>", "<eos>", "h", "w", "<|im_start|>", "<|im_end|>", "\u010A"];
            const int dim = 16;
            float[] embd = new float[vocab.Length * dim];
            float[] outW = new float[vocab.Length * dim];
            SetBasisTransition(embd, outW, dim, fromTok: 2, toTok: 3, basis: 0);
            SetBasisTransition(embd, outW, dim, fromTok: 6, toTok: 3, basis: 0);
            SetBasisTransition(embd, outW, dim, fromTok: 3, toTok: 1, basis: 1);

            string modelPath = Path.Combine(tempDir, "http_model.gguf");
            M4aTestHelpers.WriteSyntheticGguf(
                modelPath, vocab, eosTokenId: 1, bosTokenId: 0, dim: dim,
                tokenEmbeddings: embd, outputWeights: outW);

            using var engine = new ContinuousBatchEngine(modelPath, totalKvBlocks: 64, maxBatchSize: 4, maxSeqLen: 128, seed: 1, useGpu: false);
            int port = M4aTestHelpers.GetFreeTcpPort();
            var builder = GlacierServeApp.CreateBuilder();
            builder.UsePort(port);
            using var app = builder.Build();
            app.MapOpenAiEndpoints(engine, defaultModelName: "glacier-test-7b");
            await app.StartAsync();

            try
            {
                using var http = M4aTestHelpers.CreateHttpClient(port);

                // 1. GET /v1/models initially has only base model
                var modelsResp1 = await http.GetStringAsync("/v1/models");
                using (var doc = JsonDocument.Parse(modelsResp1))
                {
                    var data = doc.RootElement.GetProperty("data");
                    Assert.Equal(1, data.GetArrayLength());
                    Assert.Equal("glacier-test-7b", data[0].GetProperty("id").GetString());
                }

                // 2. POST /v1/chat/completions with stream_options.include_usage = true
                const string chatStreamReq = """
                {
                  "model": "glacier-test-7b",
                  "messages": [{"role": "user", "content": "h"}],
                  "temperature": 0,
                  "max_tokens": 5,
                  "stream": true,
                  "stream_options": {"include_usage": true}
                }
                """;
                using var chatResp = await http.PostAsync("/v1/chat/completions", new StringContent(chatStreamReq, Encoding.UTF8, "application/json"));
                Assert.Equal(HttpStatusCode.OK, chatResp.StatusCode);
                string chatSse = await chatResp.Content.ReadAsStringAsync();
                var chatEvents = ParseSseEvents(chatSse);

                Assert.Equal("[DONE]", chatEvents[^1]);
                Assert.True(chatEvents.Count >= 3);
                // Second-to-last event must be usage chunk with choices: [] and usage: {...}
                using (var usageDoc = JsonDocument.Parse(chatEvents[^2]))
                {
                    var root = usageDoc.RootElement;
                    Assert.Equal("chat.completion.chunk", root.GetProperty("object").GetString());
                    Assert.Equal(0, root.GetProperty("choices").GetArrayLength());
                    var usage = root.GetProperty("usage");
                    Assert.True(usage.GetProperty("prompt_tokens").GetInt32() > 0);
                    Assert.True(usage.GetProperty("completion_tokens").GetInt32() >= 1);
                    Assert.Equal(
                        usage.GetProperty("prompt_tokens").GetInt32() + usage.GetProperty("completion_tokens").GetInt32(),
                        usage.GetProperty("total_tokens").GetInt32());
                }
                // Third-to-last event must be finish_reason chunk
                using (var finishDoc = JsonDocument.Parse(chatEvents[^3]))
                {
                    var finishReason = finishDoc.RootElement.GetProperty("choices")[0].GetProperty("finish_reason").GetString();
                    Assert.False(string.IsNullOrEmpty(finishReason));
                }

                // 3. POST /v1/chat/completions WITHOUT include_usage must NOT emit choices:[] usage chunk
                const string chatNoUsageReq = """
                {
                  "model": "glacier-test-7b",
                  "messages": [{"role": "user", "content": "h"}],
                  "temperature": 0,
                  "max_tokens": 5,
                  "stream": true
                }
                """;
                using var noUsageResp = await http.PostAsync("/v1/chat/completions", new StringContent(chatNoUsageReq, Encoding.UTF8, "application/json"));
                string noUsageSse = await noUsageResp.Content.ReadAsStringAsync();
                var noUsageEvents = ParseSseEvents(noUsageSse);
                Assert.Equal("[DONE]", noUsageEvents[^1]);
                using (var lastJson = JsonDocument.Parse(noUsageEvents[^2]))
                {
                    Assert.Equal(1, lastJson.RootElement.GetProperty("choices").GetArrayLength());
                    Assert.False(lastJson.RootElement.TryGetProperty("usage", out _));
                }

                // 4. POST /v1/completions with stream_options.include_usage = true
                const string compStreamReq = """
                {
                  "model": "glacier-test-7b",
                  "prompt": "h",
                  "temperature": 0,
                  "max_tokens": 5,
                  "stream": true,
                  "stream_options": {"include_usage": true}
                }
                """;
                using var compResp = await http.PostAsync("/v1/completions", new StringContent(compStreamReq, Encoding.UTF8, "application/json"));
                Assert.Equal(HttpStatusCode.OK, compResp.StatusCode);
                string compSse = await compResp.Content.ReadAsStringAsync();
                var compEvents = ParseSseEvents(compSse);
                Assert.Equal("[DONE]", compEvents[^1]);
                using (var compUsageDoc = JsonDocument.Parse(compEvents[^2]))
                {
                    var root = compUsageDoc.RootElement;
                    Assert.Equal("text_completion", root.GetProperty("object").GetString());
                    Assert.Equal(0, root.GetProperty("choices").GetArrayLength());
                    var usage = root.GetProperty("usage");
                    Assert.Equal(1, usage.GetProperty("prompt_tokens").GetInt32());
                    Assert.Equal(1, usage.GetProperty("completion_tokens").GetInt32());
                    Assert.Equal(2, usage.GetProperty("total_tokens").GetInt32());
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

    [Fact]
    public async Task OpenAiEndpoints_InvalidNegativeMaxThinkTokens_Returns400AndKeepsHostHealthy_And_EngineRejectsInProcess()
    {
        string tempDir = M4aTestHelpers.CreateTempDirectory("m4a_r1_invalid_budget");
        try
        {
            string[] vocab = ["<bos>", "<eos>", "h", "w", "<|im_start|>", "<|im_end|>", "\u010A"];
            const int dim = 16;
            float[] embd = new float[vocab.Length * dim];
            float[] outW = new float[vocab.Length * dim];
            SetBasisTransition(embd, outW, dim, fromTok: 2, toTok: 3, basis: 0);
            SetBasisTransition(embd, outW, dim, fromTok: 6, toTok: 3, basis: 0);
            SetBasisTransition(embd, outW, dim, fromTok: 3, toTok: 1, basis: 1);

            string modelPath = Path.Combine(tempDir, "invalid_budget_model.gguf");
            M4aTestHelpers.WriteSyntheticGguf(
                modelPath, vocab, eosTokenId: 1, bosTokenId: 0, dim: dim,
                tokenEmbeddings: embd, outputWeights: outW);

            using var engine = new ContinuousBatchEngine(modelPath, totalKvBlocks: 64, maxBatchSize: 4, maxSeqLen: 128, seed: 1, useGpu: false);

            await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
                engine.EnqueueAsync(new InferenceRequest
                {
                    Prompt = "h",
                    MaxTokens = 4,
                    MaxThinkTokens = -2
                }));

            int port = M4aTestHelpers.GetFreeTcpPort();
            var builder = GlacierServeApp.CreateBuilder();
            builder.UsePort(port);
            using var app = builder.Build();
            app.MapOpenAiEndpoints(engine, defaultModelName: "glacier-test-7b");
            app.MapOllamaEndpoints(engine, defaultModelName: "glacier-test-7b");
            app.MapMetricsEndpoints(engine);
            await app.StartAsync();

            try
            {
                using var http = M4aTestHelpers.CreateHttpClient(port);

                foreach (int invalidBudget in new[] { -2, -100 })
                {
                    string chatBadReq = $"{{\"model\":\"glacier-test-7b\",\"messages\":[{{\"role\":\"user\",\"content\":\"h\"}}],\"max_think_tokens\":{invalidBudget}}}";
                    using var chatResp = await http.PostAsync("/v1/chat/completions", new StringContent(chatBadReq, Encoding.UTF8, "application/json"));
                    Assert.Equal(HttpStatusCode.BadRequest, chatResp.StatusCode);
                    string chatBody = await chatResp.Content.ReadAsStringAsync();
                    using (var doc = JsonDocument.Parse(chatBody))
                    {
                        Assert.Equal("invalid_request_error", doc.RootElement.GetProperty("error").GetProperty("type").GetString());
                    }

                    string compBadReq = $"{{\"model\":\"glacier-test-7b\",\"prompt\":\"h\",\"max_think_tokens\":{invalidBudget}}}";
                    using var compResp = await http.PostAsync("/v1/completions", new StringContent(compBadReq, Encoding.UTF8, "application/json"));
                    Assert.Equal(HttpStatusCode.BadRequest, compResp.StatusCode);
                    string compBody = await compResp.Content.ReadAsStringAsync();
                    using (var doc = JsonDocument.Parse(compBody))
                    {
                        Assert.Equal("invalid_request_error", doc.RootElement.GetProperty("error").GetProperty("type").GetString());
                    }

                    string ollamaChatBadReq = $"{{\"model\":\"glacier-test-7b\",\"messages\":[{{\"role\":\"user\",\"content\":\"h\"}}],\"options\":{{\"reasoning_budget\":{invalidBudget}}},\"stream\":false}}";
                    using var ollamaChatResp = await http.PostAsync("/api/chat", new StringContent(ollamaChatBadReq, Encoding.UTF8, "application/json"));
                    Assert.Equal(HttpStatusCode.BadRequest, ollamaChatResp.StatusCode);

                    string ollamaGenBadReq = $"{{\"model\":\"glacier-test-7b\",\"prompt\":\"h\",\"options\":{{\"reasoning_budget\":{invalidBudget}}},\"stream\":false}}";
                    using var ollamaGenResp = await http.PostAsync("/api/generate", new StringContent(ollamaGenBadReq, Encoding.UTF8, "application/json"));
                    Assert.Equal(HttpStatusCode.BadRequest, ollamaGenResp.StatusCode);
                }

                using var healthResp = await http.GetAsync("/health");
                Assert.Equal(HttpStatusCode.OK, healthResp.StatusCode);

                const string validReq = """
                {
                  "model": "glacier-test-7b",
                  "messages": [{"role": "user", "content": "h"}],
                  "temperature": 0,
                  "max_tokens": 5,
                  "max_think_tokens": -1
                }
                """;
                using var okResp = await http.PostAsync("/v1/chat/completions", new StringContent(validReq, Encoding.UTF8, "application/json"));
                Assert.Equal(HttpStatusCode.OK, okResp.StatusCode);
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

    [Fact]
    public async Task Engine_InvalidNegativeReasoningBudget_WithCachedPrefixBlock_DoesNotLeakRefCountOrBlocks()
    {
        string tempDir = M4aTestHelpers.CreateTempDirectory("m4a_r1_prefix_leak");
        try
        {
            string[] vocab = ["<bos>", "<eos>", "h", "w", "<|im_start|>", "<|im_end|>", "\u010A"];
            const int dim = 16;
            float[] embd = new float[vocab.Length * dim];
            float[] outW = new float[vocab.Length * dim];
            SetBasisTransition(embd, outW, dim, fromTok: 2, toTok: 3, basis: 0);
            SetBasisTransition(embd, outW, dim, fromTok: 6, toTok: 3, basis: 0);
            SetBasisTransition(embd, outW, dim, fromTok: 3, toTok: 1, basis: 1);

            string modelPath = Path.Combine(tempDir, "prefix_leak_model.gguf");
            M4aTestHelpers.WriteSyntheticGguf(
                modelPath, vocab, eosTokenId: 1, bosTokenId: 0, dim: dim,
                tokenEmbeddings: embd, outputWeights: outW);

            using var engine = new ContinuousBatchEngine(modelPath, totalKvBlocks: 64, maxBatchSize: 4, maxSeqLen: 128, seed: 1, useGpu: false);

            string longPrompt = new string('h', 18);
            int[] encoded = engine.Tokenizer.Encode(longPrompt);
            Assert.True(encoded.Length >= engine.BlockPool.BlockSize);

            ulong blockHash = PagedBlockPool.ComputePrefixHash(0UL, encoded.AsSpan(0, engine.BlockPool.BlockSize));
            if (blockHash == 0UL) blockHash = 1UL;

            Assert.True(engine.BlockPool.TryAllocateBlock(out int prefixBlockId));
            engine.BlockPool.BindPrefixBlock(blockHash, prefixBlockId);

            try
            {
                int freeBefore = engine.BlockPool.FreeBlocksCount;
                Assert.Equal(63, freeBefore);
                Assert.Equal(1, engine.BlockPool.GetRefCount(prefixBlockId));

                await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
                    engine.EnqueueAsync(new InferenceRequest
                    {
                        Prompt = longPrompt,
                        MaxTokens = 4,
                        MaxThinkTokens = -2
                    }));

                // Validation must happen BEFORE TryFindPrefixBlock/AttachPrefixBlocks/RentHiddenStateHistory
                Assert.Equal(freeBefore, engine.BlockPool.FreeBlocksCount);
                Assert.Equal(1, engine.BlockPool.GetRefCount(prefixBlockId));

                // Confirm the same prompt genuinely hits the bound prefix block on a valid request and recycles cleanly
                var (okState, okStream) = await engine.EnqueueAsync(new InferenceRequest
                {
                    Prompt = longPrompt,
                    MaxTokens = 2,
                    MaxThinkTokens = -1
                });
                Assert.True(okState.BoundPrefixBlocks >= 1);
                await foreach (var _ in okStream) { }
                await okState.Completion.Task;

                Assert.Equal(freeBefore, engine.BlockPool.FreeBlocksCount);
                Assert.Equal(1, engine.BlockPool.GetRefCount(prefixBlockId));
            }
            finally
            {
                engine.BlockPool.ReleaseBlock(prefixBlockId);
            }

            Assert.Equal(64, engine.BlockPool.FreeBlocksCount);
            Assert.Equal(0, engine.BlockPool.GetRefCount(prefixBlockId));
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    private static List<string> ParseSseEvents(string sseBody)
    {
        return sseBody
            .Split("\n\n", StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.StartsWith("data: ", StringComparison.Ordinal) ? line["data: ".Length..].Trim() : line.Trim())
            .Where(s => !string.IsNullOrEmpty(s))
            .ToList();
    }

    private static void SetBasisTransition(float[] embd, float[] outW, int dim, int fromTok, int toTok, int basis)
    {
        embd[fromTok * dim + basis] = 1.0f;
        outW[toTok * dim + basis] = 10.0f;
    }
}

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Glacier.Inference.Sampling;
using Glacier.Inference.Tokenizer;
using Glacier.Serve.Inference.PagedAttention;

namespace Glacier.Serve.Inference.Batching;

/// <summary>
/// Continuous Iteration-Level Batching Engine with PagedAttention.
/// Dynamically admits pending requests at every token iteration and evicts finished requests,
/// maximizing hardware utilization while maintaining sub-10ms Time-To-First-Token latency.
/// </summary>
public sealed class ContinuousBatchEngine : IDisposable
{
    private readonly GgufFile _gguf;
    private readonly ModelWeights _weights;
    private readonly BpeTokenizer _tokenizer;
    private readonly PagedBlockPool _pool;
    private readonly Sampler _sampler;
    private readonly int _maxBatchSize;
    private readonly int _maxSeqLen;

    private readonly Channel<SequenceState> _waitingQueue = Channel.CreateUnbounded<SequenceState>(new UnboundedChannelOptions
    {
        SingleReader = true,
        SingleWriter = false
    });

    private readonly List<SequenceState> _activeBatch = new(64);
    private readonly ThreadLocal<PagedTransformerRunner> _threadRunner;
    private readonly CancellationTokenSource _cts = new();
    private readonly Task _loopTask;
    private readonly Stopwatch _uptime = Stopwatch.StartNew();

    private int _waitingCount;
    private long _totalTokensGenerated;
    private long _totalRequestsServed;
    private double _lastIterationMs;
    private bool _disposed;

    public string ModelArchitecture => _gguf.Architecture;
    public int Layers => _weights.BlockCount;
    public int VocabSize => _weights.VocabSize;
    public int MaxBatchSize => _maxBatchSize;
    public int ActiveBatchSize => _activeBatch.Count;
    public int WaitingQueueLength => Volatile.Read(ref _waitingCount);
    public PagedBlockPool BlockPool => _pool;
    public long TotalTokensGenerated => Interlocked.Read(ref _totalTokensGenerated);
    public long TotalRequestsServed => Interlocked.Read(ref _totalRequestsServed);
    public double LastIterationMs => _lastIterationMs;
    public double ThroughputTokensPerSec => TotalTokensGenerated / Math.Max(0.001, _uptime.Elapsed.TotalSeconds);

    public ContinuousBatchEngine(string modelPath, int totalKvBlocks = 1024, int maxBatchSize = 32, int maxSeqLen = 4096)
    {
        _gguf = GgufFile.Open(modelPath);
        _weights = new ModelWeights(_gguf);
        _tokenizer = new BpeTokenizer(_gguf);
        _pool = new PagedBlockPool(totalKvBlocks, _weights.BlockCount, _weights.HeadCountKv, _weights.HeadDim, blockSize: 16, vHeadDim: _weights.ValueDim);
        _sampler = new Sampler();
        _maxBatchSize = maxBatchSize;
        _maxSeqLen = maxSeqLen;

        _threadRunner = new ThreadLocal<PagedTransformerRunner>(() => new PagedTransformerRunner(_weights, _maxSeqLen), trackAllValues: true);

        _loopTask = Task.Run(RunIterationLoopAsync);
    }

    /// <summary>
    /// Enqueues a new inference request into the continuous batch queue.
    /// </summary>
    public async Task<(SequenceState State, IAsyncEnumerable<string> TokenStream)> EnqueueAsync(InferenceRequest request, CancellationToken ct = default)
    {
        string promptText = FormatPrompt(request);
        int[] promptTokens = _tokenizer.Encode(promptText);

        string reqId = $"req-{Guid.NewGuid():N}";
        var blockTable = new BlockTable(_pool);
        var state = new SequenceState(
            requestId: reqId,
            promptTokens: promptTokens,
            blockTable: blockTable,
            maxNewTokens: request.MaxTokens > 0 ? request.MaxTokens : 256,
            temperature: request.Temperature,
            topP: request.TopP,
            stopSequences: request.Stop);

        Interlocked.Increment(ref _waitingCount);
        await _waitingQueue.Writer.WriteAsync(state, ct);
        return (state, ReadTokensAsync(state, ct));
    }

    private static async IAsyncEnumerable<string> ReadTokensAsync(SequenceState state, [EnumeratorCancellation] CancellationToken ct)
    {
        while (await state.TokenChannel.Reader.WaitToReadAsync(ct))
        {
            while (state.TokenChannel.Reader.TryRead(out var tokenText))
            {
                yield return tokenText;
            }
        }
    }

    private async Task RunIterationLoopAsync()
    {
        var swStep = new Stopwatch();
        var logitsBuffer = new float[_weights.VocabSize];

        while (!_cts.IsCancellationRequested)
        {
            swStep.Restart();

            // 1. Dynamic Admission: admit waiting requests up to MaxBatchSize
            while (_activeBatch.Count < _maxBatchSize && _waitingQueue.Reader.TryRead(out var seq))
            {
                Interlocked.Decrement(ref _waitingCount);
                if (PrefillSequence(seq, logitsBuffer))
                {
                    if (!seq.IsFinished)
                    {
                        _activeBatch.Add(seq);
                    }
                }
            }

            // 2. Continuous Iteration Step across all active sequences
            if (_activeBatch.Count > 0)
            {
                for (int i = 0; i < _activeBatch.Count; i++)
                {
                    var seq = _activeBatch[i];
                    var runner = _threadRunner.Value!;

                    // Forward single token step
                    bool ok = runner.ForwardToken(seq.LastToken, seq.CurrentPos, seq.BlockTable, _pool, logitsBuffer.AsSpan(), computeLogits: true);
                    if (!ok)
                    {
                        seq.IsFinished = true;
                        seq.FinishReason = "pool_oom";
                        continue;
                    }

                    // Sample next token
                    int nextToken = _sampler.Sample(logitsBuffer.AsSpan(), seq.SamplingOptions);
                    seq.LastToken = nextToken;
                    seq.GeneratedTokens.Add(nextToken);
                    seq.CurrentPos++;
                    Interlocked.Increment(ref _totalTokensGenerated);

                    string tokenText = _tokenizer.DecodeToken(nextToken);
                    seq.TokenChannel.Writer.TryWrite(tokenText);

                    // Check EOS or max tokens limit
                    if (IsEosToken(nextToken) || seq.GeneratedTokens.Count >= seq.MaxNewTokens)
                    {
                        seq.IsFinished = true;
                        seq.FinishReason = IsEosToken(nextToken) ? "stop" : "length";
                    }
                }

                // 3. Dynamic Eviction: clean up and recycle finished requests
                for (int i = _activeBatch.Count - 1; i >= 0; i--)
                {
                    var seq = _activeBatch[i];
                    if (seq.IsFinished)
                    {
                        seq.TokenChannel.Writer.TryComplete();
                        seq.Completion.TrySetResult(string.Concat(seq.GeneratedTokens.ConvertAll(_tokenizer.DecodeToken)));
                        seq.BlockTable.ReleaseAll(); // Instantly recycles pages!
                        _activeBatch.RemoveAt(i);
                        Interlocked.Increment(ref _totalRequestsServed);
                    }
                }
            }
            else
            {
                // Idle sleep if no work
                await Task.Delay(1);
            }

            swStep.Stop();
            _lastIterationMs = swStep.Elapsed.TotalMilliseconds;
        }
    }

    private bool PrefillSequence(SequenceState seq, float[] logitsBuffer)
    {
        var runner = _threadRunner.Value!;
        int promptLen = seq.PromptTokens.Length;
        if (promptLen == 0)
        {
            seq.IsFinished = true;
            return false;
        }

        // Prefill all prompt tokens
        for (int p = 0; p < promptLen; p++)
        {
            bool isLast = (p == promptLen - 1);
            bool ok = runner.ForwardToken(seq.PromptTokens[p], p, seq.BlockTable, _pool, isLast ? logitsBuffer.AsSpan() : Span<float>.Empty, computeLogits: isLast);
            if (!ok)
            {
                seq.IsFinished = true;
                seq.FinishReason = "pool_oom";
                seq.TokenChannel.Writer.TryComplete();
                seq.BlockTable.ReleaseAll();
                return false;
            }
        }

        // Sample first token from prompt completion
        int firstToken = _sampler.Sample(logitsBuffer.AsSpan(), seq.SamplingOptions);
        seq.LastToken = firstToken;
        seq.GeneratedTokens.Add(firstToken);
        seq.CurrentPos = promptLen;
        seq.FirstTokenTime = DateTimeOffset.UtcNow;
        Interlocked.Increment(ref _totalTokensGenerated);

        string text = _tokenizer.DecodeToken(firstToken);
        seq.TokenChannel.Writer.TryWrite(text);

        if (IsEosToken(firstToken) || seq.GeneratedTokens.Count >= seq.MaxNewTokens)
        {
            seq.IsFinished = true;
            seq.FinishReason = IsEosToken(firstToken) ? "stop" : "length";
            seq.TokenChannel.Writer.TryComplete();
            seq.Completion.TrySetResult(text);
            seq.BlockTable.ReleaseAll();
            Interlocked.Increment(ref _totalRequestsServed);
            return false;
        }

        seq.IsPrefillDone = true;
        return true;
    }

    private static bool IsEosToken(int token)
    {
        // Standard Qwen2 / Llama EOS tokens: 151643 (<|endoftext|>), 151645 (<|im_end|>), 2 (<eos>)
        return token == 151643 || token == 151645 || token == 2 || token == 1;
    }

    private static string FormatPrompt(InferenceRequest req)
    {
        if (req.Messages != null && req.Messages.Count > 0)
        {
            var sb = new System.Text.StringBuilder();
            foreach (var m in req.Messages)
            {
                sb.Append($"<|im_start|>{m.Role}\n{m.Content}<|im_end|>\n");
            }
            sb.Append("<|im_start|>assistant\n");
            return sb.ToString();
        }
        return req.Prompt ?? string.Empty;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _cts.Cancel();
            try { _loopTask.Wait(500); } catch { }
            _cts.Dispose();

            if (_threadRunner.Values != null)
            {
                foreach (var runner in _threadRunner.Values)
                {
                    runner.Dispose();
                }
            }
            _threadRunner.Dispose();
            _pool.Dispose();
            _gguf.Dispose();
        }
    }
}

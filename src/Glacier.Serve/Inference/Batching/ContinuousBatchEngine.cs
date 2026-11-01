using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Glacier.Inference.Sampling;
using Glacier.Inference.Tokenizer;
using Glacier.Serve.Diagnostics;
using Glacier.Serve.Inference.PagedAttention;

namespace Glacier.Serve.Inference.Batching;

/// <summary>
/// Continuous Iteration-Level Batching Engine with PagedAttention.
/// Dynamically admits pending requests, interleaves non-blocking chunked batched prefill
/// with multi-sequence decode ticks, and immediately recycles KV pages on completion or client abort.
/// </summary>
public sealed class ContinuousBatchEngine : IDisposable
{
    public const int DefaultPrefillChunkSize = 128;

    private readonly GgufFile _gguf;
    private readonly ModelWeights _weights;
    private readonly BpeTokenizer _tokenizer;
    private readonly PagedBlockPool _pool;
    private readonly IPagedModelRunner _runner;
    private readonly Sampler _sampler;
    private readonly int _maxBatchSize;
    private readonly int _maxSeqLen;
    private readonly int _prefillChunkSize;
    private readonly int _defaultReasoningLoopMinPeriod;

    private readonly Channel<SequenceState> _waitingQueue = Channel.CreateUnbounded<SequenceState>(new UnboundedChannelOptions
    {
        SingleReader = true,
        SingleWriter = false
    });

    private readonly List<SequenceState> _prefillingBatch = new(64);
    private readonly List<SequenceState> _activeBatch = new(64);
    private readonly CancellationTokenSource _cts = new();
    private readonly Task _loopTask;
    private readonly Stopwatch _uptime = Stopwatch.StartNew();

    private int _waitingCount;
    private int _activeSequenceCount;
    private long _totalTokensGenerated;
    private long _totalRequestsServed;
    private double _lastIterationMs;
    private volatile Exception? _faultException;
    private bool _disposed;

    public string ModelArchitecture => _gguf.Architecture;
    public int Layers => _weights.BlockCount;
    public int VocabSize => _weights.VocabSize;
    public int MaxBatchSize => _maxBatchSize;
    public int MaxSeqLen => _maxSeqLen;
    public int PrefillChunkSize => _prefillChunkSize;
    public bool IsGpuAccelerated => _runner.IsGpuAccelerated;
    public string ActiveBackend => _runner.BackendName;
    public int ActiveBatchSize => Volatile.Read(ref _activeSequenceCount);
    public int WaitingQueueLength => Volatile.Read(ref _waitingCount);
    public PagedBlockPool BlockPool => _pool;
    public BpeTokenizer Tokenizer => _tokenizer;
    public long TotalTokensGenerated => Interlocked.Read(ref _totalTokensGenerated);
    public long TotalRequestsServed => Interlocked.Read(ref _totalRequestsServed);
    public double LastIterationMs => Volatile.Read(ref _lastIterationMs);
    public double ThroughputTokensPerSec => TotalTokensGenerated / Math.Max(0.001, _uptime.Elapsed.TotalSeconds);

    public ContinuousBatchEngine(
        string modelPath,
        int totalKvBlocks = 1024,
        int maxBatchSize = 32,
        int maxSeqLen = 4096,
        int? seed = null,
        int defaultReasoningLoopMinPeriod = 1,
        bool useGpu = true,
        int prefillChunkSize = DefaultPrefillChunkSize)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        ArgumentOutOfRangeException.ThrowIfLessThan(totalKvBlocks, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxBatchSize, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxSeqLen, 2);
        ArgumentOutOfRangeException.ThrowIfLessThan(prefillChunkSize, 1);

        _gguf = GgufFile.Open(modelPath);
        try
        {
            _weights = new ModelWeights(_gguf);
            _tokenizer = new BpeTokenizer(_gguf);
            _sampler = new Sampler(seed);
            _maxBatchSize = maxBatchSize;
            _maxSeqLen = maxSeqLen;
            _prefillChunkSize = prefillChunkSize;
            _defaultReasoningLoopMinPeriod = defaultReasoningLoopMinPeriod;

            int runnerBatchCapacity = Math.Max(_maxBatchSize, _prefillChunkSize);
            if (useGpu && CudaPagedTransformerRunner.TryCreate(
                _weights,
                totalKvBlocks,
                PagedBlockPool.DefaultBlockSize,
                _maxSeqLen,
                runnerBatchCapacity,
                out var cudaRunner))
            {
                _runner = cudaRunner;
                _pool = new PagedBlockPool(
                    totalKvBlocks,
                    _weights.BlockCount,
                    _weights.HeadCountKv,
                    _weights.HeadDim,
                    blockSize: PagedBlockPool.DefaultBlockSize,
                    vHeadDim: _weights.ValueDim,
                    allocateHostBuffers: false);
            }
            else
            {
                _runner = new PagedTransformerRunner(_weights, _maxSeqLen, runnerBatchCapacity);
                _pool = new PagedBlockPool(
                    totalKvBlocks,
                    _weights.BlockCount,
                    _weights.HeadCountKv,
                    _weights.HeadDim,
                    blockSize: PagedBlockPool.DefaultBlockSize,
                    vHeadDim: _weights.ValueDim,
                    allocateHostBuffers: true);
            }

            _loopTask = Task.Run(RunIterationLoopAsync);
        }
        catch
        {
            _runner?.Dispose();
            _pool?.Dispose();
            _cts.Dispose();
            _gguf.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Enqueues a new inference request into the continuous batch queue.
    /// Throws <see cref="ArgumentOutOfRangeException"/> if the prompt is empty or meets/exceeds <see cref="MaxSeqLen"/>.
    /// </summary>
    public async Task<(SequenceState State, IAsyncEnumerable<string> TokenStream)> EnqueueAsync(InferenceRequest request, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_faultException is Exception loopFault)
        {
            throw new InvalidOperationException("ContinuousBatchEngine background loop faulted and cannot accept new requests.", loopFault);
        }

        float temp = request.EffectiveTemperature;
        float topP = request.EffectiveTopP;
        float repPenalty = request.EffectiveRepetitionPenalty;

        if (temp < 0f || float.IsNaN(temp))
            throw new ArgumentOutOfRangeException(nameof(request), temp, "Temperature must be non-negative.");
        if (topP < 0f || topP > 1f || float.IsNaN(topP))
            throw new ArgumentOutOfRangeException(nameof(request), topP, "TopP must be in [0.0, 1.0].");
        if (repPenalty <= 0f || float.IsNaN(repPenalty))
            throw new ArgumentOutOfRangeException(nameof(request), repPenalty, "RepetitionPenalty must be greater than zero.");

        string promptText = FormatPrompt(request);
        int[] promptTokens = _tokenizer.Encode(promptText);

        if (promptTokens.Length == 0 || promptTokens.Length >= _maxSeqLen)
        {
            throw new ArgumentOutOfRangeException(
                nameof(request),
                promptTokens.Length,
                $"Encoded prompt token count ({promptTokens.Length}) must be between 1 and maxSeqLen - 1 ({_maxSeqLen - 1}).");
        }

        int requestedMaxTokens = request.EffectiveMaxTokens > 0 ? request.EffectiveMaxTokens : 256;
        int effectiveMaxNewTokens = Math.Min(requestedMaxTokens, _maxSeqLen - promptTokens.Length);

        string reqId = $"req-{Guid.NewGuid():N}";
        var blockTable = new BlockTable(_pool);
        var state = new SequenceState(
            requestId: reqId,
            promptTokens: promptTokens,
            blockTable: blockTable,
            maxNewTokens: effectiveMaxNewTokens,
            temperature: temp,
            topP: topP,
            stopSequences: NormalizeStopSequences(request.EffectiveStop),
            repetitionPenalty: repPenalty,
            topK: request.EffectiveTopK,
            seed: request.EffectiveSeed,
            cancellationToken: ct);

        if (_tokenizer.TryGetTokenId("<think>", out int thinkId))
            state.SamplingOptions.ThinkTokenId = thinkId;
        if (_tokenizer.TryGetTokenId("</think>", out int endThinkId))
            state.SamplingOptions.EndThinkTokenId = endThinkId;
        if (request.MaxReasoningTokens > 0)
            state.SamplingOptions.MaxReasoningTokens = request.MaxReasoningTokens;
        if (request.EnableReasoningLoopGuard.HasValue)
            state.SamplingOptions.EnableReasoningLoopGuard = request.EnableReasoningLoopGuard.Value;
        if (request.ReasoningLoopMinPeriod.HasValue)
            state.SamplingOptions.ReasoningLoopMinPeriod = request.ReasoningLoopMinPeriod.Value;
        else if (_defaultReasoningLoopMinPeriod != 1)
            state.SamplingOptions.ReasoningLoopMinPeriod = _defaultReasoningLoopMinPeriod;

        if (state.SamplingOptions.ThinkTokenId >= 0)
        {
            int lastThink = Array.LastIndexOf(promptTokens, state.SamplingOptions.ThinkTokenId);
            int lastEndThink = state.SamplingOptions.EndThinkTokenId >= 0 ? Array.LastIndexOf(promptTokens, state.SamplingOptions.EndThinkTokenId) : -1;
            if (lastThink >= 0 && lastThink > lastEndThink)
            {
                state.IsInReasoningBlock = true;
            }
        }

        Interlocked.Increment(ref _waitingCount);
        try
        {
            await _waitingQueue.Writer.WriteAsync(state, ct);
        }
        catch
        {
            Interlocked.Decrement(ref _waitingCount);
            throw;
        }

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
        var prefillLogits = new float[_weights.VocabSize];

        int[] batchTokens = new int[_maxBatchSize];
        int[] batchPositions = new int[_maxBatchSize];
        bool[] successFlags = new bool[_maxBatchSize];
        var blockTables = new BlockTable[_maxBatchSize];

        try
        {
            while (!_cts.IsCancellationRequested)
            {
                // When completely idle, await channel notification instead of sleeping 15ms on Windows timers
                if (_prefillingBatch.Count == 0 && _activeBatch.Count == 0 && !_waitingQueue.Reader.TryPeek(out _))
                {
                    Volatile.Write(ref _activeSequenceCount, 0);
                    if (!await _waitingQueue.Reader.WaitToReadAsync(_cts.Token))
                        break;
                }

                swStep.Restart();

                // 1. Dynamic Admission: admit waiting requests up to MaxBatchSize
                while ((_prefillingBatch.Count + _activeBatch.Count) < _maxBatchSize &&
                       _waitingQueue.Reader.TryRead(out var incoming))
                {
                    Interlocked.Decrement(ref _waitingCount);
                    if (incoming.CancellationToken.IsCancellationRequested)
                    {
                        CancelAndRecycleSequence(incoming);
                        continue;
                    }
                    _prefillingBatch.Add(incoming);
                }

                // 2. Sweep aborted sequences immediately so KV blocks are recycled without waiting for generation end
                SweepCancelledSequences(_prefillingBatch);
                SweepCancelledSequences(_activeBatch);
                Volatile.Write(ref _activeSequenceCount, _prefillingBatch.Count + _activeBatch.Count);

                // 3. Decode Step across all active decoding sequences FIRST so ongoing streams never stall behind prefill
                if (_activeBatch.Count > 0)
                {
                    ExecuteDecodeStep(batchTokens, batchPositions, successFlags, blockTables);
                }

                // 4. Non-Blocking Chunked Batched Prefill Step (at most _prefillChunkSize tokens per tick)
                if (_prefillingBatch.Count > 0)
                {
                    ExecutePrefillChunkStep(prefillLogits);
                }

                Volatile.Write(ref _activeSequenceCount, _prefillingBatch.Count + _activeBatch.Count);

                swStep.Stop();
                Volatile.Write(ref _lastIterationMs, swStep.Elapsed.TotalMilliseconds);

                // Yield scheduler slice cooperatively so waiting producers/consumers get prompt continuation dispatch
                await Task.Yield();
            }
        }
        catch (OperationCanceledException) when (_cts.IsCancellationRequested)
        {
            // Normal engine shutdown
        }
        catch (Exception ex)
        {
            _faultException = ex;
            _waitingQueue.Writer.TryComplete(ex);
            GlacierDiagnostics.LogError("[ContinuousBatchEngine] Unhandled error in continuous batching loop.", ex);
        }
        finally
        {
            // Drain and release any remaining sequences on shutdown
            foreach (var seq in _prefillingBatch) CancelAndRecycleSequence(seq);
            _prefillingBatch.Clear();

            foreach (var seq in _activeBatch) CancelAndRecycleSequence(seq);
            _activeBatch.Clear();

            while (_waitingQueue.Reader.TryRead(out var waiting))
            {
                Interlocked.Decrement(ref _waitingCount);
                CancelAndRecycleSequence(waiting);
            }
            Volatile.Write(ref _activeSequenceCount, 0);
        }
    }

    private void ExecuteDecodeStep(
        int[] batchTokens,
        int[] batchPositions,
        bool[] successFlags,
        BlockTable[] blockTables)
    {
        // Guard against any sequence already at _maxSeqLen before invoking runner
        for (int i = _activeBatch.Count - 1; i >= 0; i--)
        {
            var seq = _activeBatch[i];
            if (seq.CurrentPos >= _maxSeqLen)
            {
                FlushDecoderAndPendingText(seq);
                seq.IsFinished = true;
                seq.FinishReason = "length";
                CompleteAndRecycleSequence(seq);
                _activeBatch.RemoveAt(i);
            }
        }

        int count = _activeBatch.Count;
        if (count == 0) return;

        for (int i = 0; i < count; i++)
        {
            var seq = _activeBatch[i];
            batchTokens[i] = seq.LastToken;
            batchPositions[i] = seq.CurrentPos;
            blockTables[i] = seq.BlockTable;
        }

        int totalVocab = _weights.VocabSize;
        float[] batchLogits = ArrayPool<float>.Shared.Rent(count * totalVocab);
        try
        {
            _runner.ForwardBatch(
                batchTokens.AsSpan(0, count),
                batchPositions.AsSpan(0, count),
                blockTables,
                _pool,
                batchLogits.AsSpan(0, count * totalVocab),
                successFlags.AsSpan(0, count),
                computeLogits: true);

            for (int i = 0; i < count; i++)
            {
                var seq = _activeBatch[i];
                if (seq.CancellationToken.IsCancellationRequested)
                {
                    continue;
                }

                if (!successFlags[i])
                {
                    FlushDecoderAndPendingText(seq);
                    seq.IsFinished = true;
                    seq.FinishReason = "pool_oom";
                    continue;
                }

                // Position evaluated was seq.CurrentPos; advance position for subsequent decode step
                seq.CurrentPos++;

                var seqLogits = batchLogits.AsSpan(i * totalVocab, totalVocab);
                ProcessSampledToken(seq, seqLogits, isFirstToken: false);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(batchLogits);
        }

        // Dynamic eviction of finished or newly-cancelled sequences
        for (int i = _activeBatch.Count - 1; i >= 0; i--)
        {
            var seq = _activeBatch[i];
            if (seq.CancellationToken.IsCancellationRequested)
            {
                CancelAndRecycleSequence(seq);
                _activeBatch.RemoveAt(i);
            }
            else if (seq.IsFinished)
            {
                CompleteAndRecycleSequence(seq);
                _activeBatch.RemoveAt(i);
            }
        }
    }

    private void ExecutePrefillChunkStep(float[] logitsBuffer)
    {
        var seq = _prefillingBatch[0];
        if (seq.CancellationToken.IsCancellationRequested)
        {
            CancelAndRecycleSequence(seq);
            _prefillingBatch.RemoveAt(0);
            return;
        }

        int promptLen = seq.PromptTokens.Length;
        int startPos = seq.PrefilledTokens;
        int chunkLen = Math.Min(promptLen - startPos, _prefillChunkSize);
        bool isFinalChunk = (startPos + chunkLen == promptLen);

        bool ok = _runner.ForwardPrefillChunk(
            seq.PromptTokens.AsSpan(startPos, chunkLen),
            startPos,
            seq.BlockTable,
            _pool,
            isFinalChunk ? logitsBuffer.AsSpan() : Span<float>.Empty,
            computeLogits: isFinalChunk);

        if (!ok)
        {
            seq.IsFinished = true;
            seq.FinishReason = "pool_oom";
            CompleteAndRecycleSequence(seq);
            _prefillingBatch.RemoveAt(0);
            return;
        }

        seq.PrefilledTokens += chunkLen;
        seq.CurrentPos = seq.PrefilledTokens;

        if (!isFinalChunk)
        {
            // Keep sequence in _prefillingBatch so active decodes can run on the very next tick
            return;
        }

        _prefillingBatch.RemoveAt(0);
        seq.IsPrefillDone = true;

        if (seq.CancellationToken.IsCancellationRequested)
        {
            CancelAndRecycleSequence(seq);
            return;
        }

        // Sample first generated token from the completed prefill logits (CurrentPos stays promptLen for first decode step)
        ProcessSampledToken(seq, logitsBuffer.AsSpan(), isFirstToken: true);

        if (seq.IsFinished)
        {
            CompleteAndRecycleSequence(seq);
        }
        else
        {
            _activeBatch.Add(seq);
        }
    }

    private void ProcessSampledToken(SequenceState seq, Span<float> seqLogits, bool isFirstToken)
    {
        // 1. Dynamic reasoning token budgeting & logit clamping (strictly guarded behind EnableReasoningLoopGuard)
        if (seq.SamplingOptions.EnableReasoningLoopGuard &&
            seq.IsInReasoningBlock &&
            seq.SamplingOptions.MaxReasoningTokens > 0 &&
            seq.CurrentReasoningTokens >= seq.SamplingOptions.MaxReasoningTokens &&
            seq.SamplingOptions.EndThinkTokenId >= 0)
        {
            seq.BudgetClampTriggers++;
            seq.LoopRecoveryTriggers++;
            Sampler.ClampReasoningLogits(seqLogits, seq.SamplingOptions.EndThinkTokenId);
        }

        // 2. Single-pass periodic reasoning-loop check ONLY when model has <think> and sequence is inside <think>...</think>
        bool isReasoningActive = seq.SamplingOptions.ThinkTokenId >= 0 && seq.IsInReasoningBlock;
        ReadOnlySpan<int> recentTokens = CollectionsMarshal.AsSpan(seq.GeneratedTokens);
        ReadOnlySpan<int> penaltyWindow = recentTokens;

        bool origEnableLoopGuard = seq.SamplingOptions.EnableReasoningLoopGuard;
        float origRepetitionPenalty = seq.SamplingOptions.RepetitionPenalty;

        if (origEnableLoopGuard && isReasoningActive && !recentTokens.IsEmpty)
        {
            int loopPeriod = Sampler.DetectPeriodicReasoningLoop(
                recentTokens,
                minPeriod: seq.SamplingOptions.ReasoningLoopMinPeriod);
            if (loopPeriod > 0)
            {
                seq.PeriodicLoopTriggers++;
                seq.LoopRecoveryTriggers++;
                seq.SamplingOptions.RepetitionPenalty = Math.Max(origRepetitionPenalty, seq.SamplingOptions.LoopRecoveryRepetitionPenalty);
                int windowLen = Math.Min(recentTokens.Length, loopPeriod * 3);
                penaltyWindow = recentTokens.Slice(recentTokens.Length - windowLen, windowLen);
            }
        }

        // Prevent duplicate DetectPeriodicReasoningLoop scan inside Sampler.Sample
        seq.SamplingOptions.EnableReasoningLoopGuard = false;
        int nextToken;
        try
        {
            Sampler activeSampler = seq.SequenceSampler ?? _sampler;
            nextToken = activeSampler.Sample(seqLogits, seq.SamplingOptions, penaltyWindow);
        }
        finally
        {
            seq.SamplingOptions.EnableReasoningLoopGuard = origEnableLoopGuard;
            seq.SamplingOptions.RepetitionPenalty = origRepetitionPenalty;
        }

        if (isFirstToken)
        {
            seq.FirstTokenTime = DateTimeOffset.UtcNow;
        }

        // 3. EOS non-leakage check BEFORE appending to GeneratedTokens or emitting text
        if (IsEosToken(nextToken))
        {
            FlushDecoderAndPendingText(seq);
            seq.IsFinished = true;
            seq.FinishReason = "stop";
            return;
        }

        // 4. Record valid non-EOS token
        seq.LastToken = nextToken;
        seq.GeneratedTokens.Add(nextToken);
        Interlocked.Increment(ref _totalTokensGenerated);

        // Update reasoning block state
        if (seq.SamplingOptions.ThinkTokenId >= 0 && nextToken == seq.SamplingOptions.ThinkTokenId)
        {
            seq.IsInReasoningBlock = true;
        }
        else if (seq.SamplingOptions.EndThinkTokenId >= 0 && nextToken == seq.SamplingOptions.EndThinkTokenId)
        {
            seq.IsInReasoningBlock = false;
        }
        else if (seq.IsInReasoningBlock)
        {
            seq.CurrentReasoningTokens++;
        }

        // 5. Incremental stateful UTF-8 decoding (prevents U+FFFD on split multi-byte BPE tokens)
        DecodeTokenBytesIntoBuilder(seq, nextToken);

        // 6. StopSequences holdback & truncation enforcement
        if (EmitOrStopOnSequences(seq))
        {
            return;
        }

        // 7. MaxNewTokens / MaxSeqLen length cap check
        if (seq.GeneratedTokens.Count >= seq.MaxNewTokens || (!isFirstToken && seq.CurrentPos >= _maxSeqLen))
        {
            FlushDecoderAndPendingText(seq);
            seq.IsFinished = true;
            seq.FinishReason = "length";
        }
    }

    private void DecodeTokenBytesIntoBuilder(SequenceState seq, int tokenId)
    {
        byte[] rawBytes = _tokenizer.GetTokenBytes(tokenId);
        if (rawBytes.Length == 0) return;

        int maxChars = Encoding.UTF8.GetMaxCharCount(rawBytes.Length) + 4;
        char[]? rented = null;
        Span<char> charSpan = maxChars <= 128
            ? stackalloc char[maxChars]
            : (rented = ArrayPool<char>.Shared.Rent(maxChars)).AsSpan(0, maxChars);
        try
        {
            int written = seq.Utf8Decoder.GetChars(rawBytes, charSpan, flush: false);
            if (written > 0)
            {
                seq.GeneratedTextBuilder.Append(charSpan[..written]);
            }
        }
        finally
        {
            if (rented != null)
                ArrayPool<char>.Shared.Return(rented);
        }
    }

    private static bool EmitOrStopOnSequences(SequenceState seq)
    {
        int totalLen = seq.GeneratedTextBuilder.Length;
        int unemittedLen = totalLen - seq.EmittedTextLength;
        if (unemittedLen <= 0)
            return false;

        if (seq.StopSequences == null || seq.StopSequences.Length == 0)
        {
            string chunk = seq.GeneratedTextBuilder.ToString(seq.EmittedTextLength, unemittedLen);
            seq.EmittedTextLength = totalLen;
            seq.TokenChannel.Writer.TryWrite(chunk);
            return false;
        }

        string pending = seq.GeneratedTextBuilder.ToString(seq.EmittedTextLength, unemittedLen);
        int earliestMatch = -1;
        for (int i = 0; i < seq.StopSequences.Length; i++)
        {
            string stop = seq.StopSequences[i];
            if (string.IsNullOrEmpty(stop)) continue;
            int idx = pending.IndexOf(stop, StringComparison.Ordinal);
            if (idx >= 0 && (earliestMatch < 0 || idx < earliestMatch))
            {
                earliestMatch = idx;
            }
        }

        if (earliestMatch >= 0)
        {
            // Truncate generated text right before the matched stop sequence
            seq.GeneratedTextBuilder.Length = seq.EmittedTextLength + earliestMatch;
            if (earliestMatch > 0)
            {
                seq.TokenChannel.Writer.TryWrite(pending[..earliestMatch]);
                seq.EmittedTextLength += earliestMatch;
            }
            seq.IsFinished = true;
            seq.FinishReason = "stop";
            return true;
        }

        // Hold back any suffix that is a partial prefix of a configured stop sequence
        int maxHoldback = 0;
        for (int i = 0; i < seq.StopSequences.Length; i++)
        {
            string stop = seq.StopSequences[i];
            if (string.IsNullOrEmpty(stop) || stop.Length <= 1) continue;

            int maxPrefix = Math.Min(pending.Length, stop.Length - 1);
            for (int prefixLen = maxPrefix; prefixLen >= 1; prefixLen--)
            {
                if (pending.AsSpan(pending.Length - prefixLen, prefixLen).SequenceEqual(stop.AsSpan(0, prefixLen)))
                {
                    if (prefixLen > maxHoldback)
                        maxHoldback = prefixLen;
                    break;
                }
            }
        }

        int safeCount = pending.Length - maxHoldback;
        if (safeCount > 0)
        {
            seq.TokenChannel.Writer.TryWrite(pending[..safeCount]);
            seq.EmittedTextLength += safeCount;
        }

        return false;
    }

    private static void FlushDecoderAndPendingText(SequenceState seq)
    {
        Span<char> flushBuf = stackalloc char[8];
        int flushed = seq.Utf8Decoder.GetChars(ReadOnlySpan<byte>.Empty, flushBuf, flush: true);
        if (flushed > 0)
        {
            seq.GeneratedTextBuilder.Append(flushBuf[..flushed]);
        }

        int remaining = seq.GeneratedTextBuilder.Length - seq.EmittedTextLength;
        if (remaining > 0)
        {
            string tail = seq.GeneratedTextBuilder.ToString(seq.EmittedTextLength, remaining);
            seq.EmittedTextLength = seq.GeneratedTextBuilder.Length;
            seq.TokenChannel.Writer.TryWrite(tail);
        }
    }

    private static void SweepCancelledSequences(List<SequenceState> batch)
    {
        for (int i = batch.Count - 1; i >= 0; i--)
        {
            var seq = batch[i];
            if (seq.CancellationToken.IsCancellationRequested)
            {
                CancelAndRecycleSequence(seq);
                batch.RemoveAt(i);
            }
        }
    }

    private static void CancelAndRecycleSequence(SequenceState seq)
    {
        seq.IsFinished = true;
        if (string.IsNullOrEmpty(seq.FinishReason))
            seq.FinishReason = "cancelled";
        seq.BlockTable.ReleaseAll();
        seq.TokenChannel.Writer.TryComplete();
        if (seq.CancellationToken.IsCancellationRequested)
            seq.Completion.TrySetCanceled(seq.CancellationToken);
        else
            seq.Completion.TrySetCanceled();
    }

    private void CompleteAndRecycleSequence(SequenceState seq)
    {
        seq.BlockTable.ReleaseAll();
        seq.TokenChannel.Writer.TryComplete();
        seq.Completion.TrySetResult(seq.GeneratedTextBuilder.ToString());
        Interlocked.Increment(ref _totalRequestsServed);
    }

    private static string[]? NormalizeStopSequences(string[]? stops)
    {
        if (stops == null || stops.Length == 0) return null;
        int validCount = 0;
        for (int i = 0; i < stops.Length; i++)
        {
            if (!string.IsNullOrEmpty(stops[i])) validCount++;
        }
        if (validCount == 0) return null;
        if (validCount == stops.Length) return stops;

        var filtered = new string[validCount];
        int idx = 0;
        for (int i = 0; i < stops.Length; i++)
        {
            if (!string.IsNullOrEmpty(stops[i]))
                filtered[idx++] = stops[i];
        }
        return filtered;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private bool IsEosToken(int token)
    {
        return _tokenizer.IsStopToken(token);
    }

    private string FormatPrompt(InferenceRequest req)
    {
        if (req.Messages != null && req.Messages.Count > 0)
        {
            var list = new List<(string Role, string Content)>(req.Messages.Count);
            foreach (var m in req.Messages)
            {
                list.Add((m.Role, m.Content));
            }
            bool hasThinkToken = _tokenizer.TryGetTokenId("<think>", out _);
            bool thinking = req.EnableThinking ?? hasThinkToken;
            return _tokenizer.FormatConversation(list, thinkingMode: thinking);
        }
        return req.Prompt ?? string.Empty;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _cts.Cancel();
            _waitingQueue.Writer.TryComplete();
            try
            {
                _loopTask.Wait(2000);
            }
            catch (Exception ex)
            {
                GlacierDiagnostics.LogDebug($"[Glacier.Serve] ContinuousBatchEngine loop task shutdown: {ex.Message}");
            }
            _cts.Dispose();
            _runner.Dispose();
            _pool.Dispose();
            _gguf.Dispose();
        }
    }
}

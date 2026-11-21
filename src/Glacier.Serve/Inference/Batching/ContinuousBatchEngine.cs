using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Hashing;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Sampling;
using Glacier.Inference.Tokenizer;
using Glacier.Serve.Daemons;
using Glacier.Serve.Diagnostics;
using Glacier.Serve.Inference.PagedAttention;
using Glacier.Tune.Sleep;

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
    private readonly int _defaultMaxReasoningTokens;
    private readonly bool _enableSpeculativeDecoding;
    private readonly int _maxDraftTokens;
    private readonly int _speculativeBatchThreshold;
    private readonly string? _mtpSidecarPath;
    private readonly bool _mtpConfigured;
    private readonly PromptLookupDraftProvider _promptLookupProvider;
    private readonly CostGatedDraftPolicy _draftPolicy;
    private readonly ConcurrentStack<MtpDraftProvider> _mtpPool = new();
    private readonly ConcurrentStack<float[]> _hiddenHistoryPool = new();

    private sealed record LoadedLoraEntry(LoraAdapterWeights Weights, ulong InstanceId, string SourcePath);
    private readonly ConcurrentDictionary<string, LoadedLoraEntry> _loraAdapters = new(StringComparer.Ordinal);
    private long _loraInstanceEpoch;

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
    private long _speculativeSteps;
    private long _speculativeDraftTokensProposed;
    private long _speculativeDraftTokensAccepted;
    private long _speculativePromptLookupRounds;
    private long _speculativeMtpRounds;
    private long _speculativeSerialFallbackRounds;
    private double _lastIterationMs;
    private volatile Exception? _faultException;
    private AutonomousSleepDaemon? _sleepDaemon;
    private SleepMemoryService? _sleepMemory;
    private ISleepExpansionRouter? _expansionRouter;
    private float _expansionResidualScale = 1.0f;
    private int _sleepVectorDimension = 64;
    private int _disposed;

    public AutonomousSleepDaemon? SleepDaemon => _sleepDaemon;
    public SleepMemoryService? SleepMemory => _sleepMemory;
    public ISleepExpansionRouter? ExpansionRouter => _expansionRouter;
    public int SleepVectorDimension => _sleepVectorDimension;

    public void AttachSleepMemory(
        SleepMemoryService sleepMemory,
        bool attachDaemonAndExpansion = true,
        float expansionResidualScale = 1.0f)
    {
        ArgumentNullException.ThrowIfNull(sleepMemory);
        _sleepMemory = sleepMemory;
        if (attachDaemonAndExpansion)
        {
            AttachSleepDaemon(sleepMemory.Daemon, vectorDimension: 64, sleepMemory.ExpansionBank, expansionResidualScale);
        }
    }

    public void AttachSleepDaemon(
        AutonomousSleepDaemon daemon,
        int vectorDimension = 64,
        ISleepExpansionRouter? expansionRouter = null,
        float expansionResidualScale = 1.0f)
    {
        ArgumentNullException.ThrowIfNull(daemon);
        if (vectorDimension <= 0) throw new ArgumentOutOfRangeException(nameof(vectorDimension));
        _sleepDaemon = daemon;
        _sleepVectorDimension = vectorDimension;
        if (expansionRouter != null)
        {
            _expansionRouter = expansionRouter;
            _expansionResidualScale = expansionResidualScale;
        }
    }

    public void AttachSleepExpansionRouter(ISleepExpansionRouter router, float residualScale = 1.0f)
    {
        ArgumentNullException.ThrowIfNull(router);
        _expansionRouter = router;
        _expansionResidualScale = residualScale;
    }

    public string ModelArchitecture => _gguf.Architecture;
    public int Layers => _weights.BlockCount;
    public int VocabSize => _weights.VocabSize;
    public int MaxBatchSize => _maxBatchSize;
    public int MaxSeqLen => _maxSeqLen;
    public int PrefillChunkSize => _prefillChunkSize;
    public bool IsGpuAccelerated => _runner.IsGpuAccelerated;
    public string ActiveBackend => _runner.BackendName;
    public KvCachePrecision KvPrecision => _pool.Precision;
    public int ActiveBatchSize => Volatile.Read(ref _activeSequenceCount);
    public int WaitingQueueLength => Volatile.Read(ref _waitingCount);
    public PagedBlockPool BlockPool => _pool;
    internal int HiddenHistoryPoolCount => _hiddenHistoryPool.Count;
    public BpeTokenizer Tokenizer => _tokenizer;
    public long TotalTokensGenerated => Interlocked.Read(ref _totalTokensGenerated);
    public long TotalRequestsServed => Interlocked.Read(ref _totalRequestsServed);
    public bool IsSpeculativeDecodingActive => _enableSpeculativeDecoding && _maxDraftTokens > 0 && ActiveBatchSize <= _speculativeBatchThreshold;
    public long SpeculativeSteps => Interlocked.Read(ref _speculativeSteps);
    public long SpeculativeDraftTokensProposed => Interlocked.Read(ref _speculativeDraftTokensProposed);
    public long SpeculativeDraftTokensAccepted => Interlocked.Read(ref _speculativeDraftTokensAccepted);
    public long SpeculativePromptLookupRounds => Interlocked.Read(ref _speculativePromptLookupRounds);
    public long SpeculativeMtpRounds => Interlocked.Read(ref _speculativeMtpRounds);
    public long SpeculativeSerialFallbackRounds => Interlocked.Read(ref _speculativeSerialFallbackRounds);
    public CostGatedDraftPolicy DraftPolicy => _draftPolicy;
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
        int prefillChunkSize = DefaultPrefillChunkSize,
        int defaultMaxReasoningTokens = 0,
        KvCachePrecision kvCachePrecision = KvCachePrecision.Auto,
        bool enableSpeculativeDecoding = true,
        int maxDraftTokens = 4,
        int speculativeBatchThreshold = 4,
        string? mtpSidecarPath = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        ArgumentOutOfRangeException.ThrowIfLessThan(totalKvBlocks, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxBatchSize, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxSeqLen, 2);
        ArgumentOutOfRangeException.ThrowIfLessThan(prefillChunkSize, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(maxDraftTokens);
        ArgumentOutOfRangeException.ThrowIfLessThan(speculativeBatchThreshold, 1);

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
            _defaultMaxReasoningTokens = defaultMaxReasoningTokens;
            _enableSpeculativeDecoding = enableSpeculativeDecoding;
            _maxDraftTokens = maxDraftTokens;
            _speculativeBatchThreshold = speculativeBatchThreshold;
            _mtpSidecarPath = string.IsNullOrWhiteSpace(mtpSidecarPath) ? null : mtpSidecarPath;
            _promptLookupProvider = new PromptLookupDraftProvider(minMatchLength: 2, maxMatchLength: 4);
            _draftPolicy = new CostGatedDraftPolicy();

            if (_mtpSidecarPath != null && !File.Exists(_mtpSidecarPath))
            {
                throw new FileNotFoundException($"MTP sidecar GGUF not found: '{_mtpSidecarPath}'.", _mtpSidecarPath);
            }

            int runnerBatchCapacity = Math.Max(_maxBatchSize, _prefillChunkSize);
            if (useGpu && CudaPagedTransformerRunner.TryCreate(
                _weights,
                totalKvBlocks,
                PagedBlockPool.DefaultBlockSize,
                _maxSeqLen,
                runnerBatchCapacity,
                kvCachePrecision,
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
                    allocateHostBuffers: false,
                    kvPrecision: cudaRunner.KvPrecision);
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
                    allocateHostBuffers: true,
                    kvPrecision: kvCachePrecision);
            }

            bool hasMtpSource = _weights.EmbeddedMtpBlockIndex >= 0 || _mtpSidecarPath != null;
            if (_enableSpeculativeDecoding && _maxDraftTokens > 0 && hasMtpSource && _runner is IHiddenStateSource trunkSource)
            {
                trunkSource.EnablePositionHistory();
                if (_runner is PagedTransformerRunner)
                {
                    MtpDraftProvider? initialMtp = CreateMtpProvider(trunkSource);
                    if (initialMtp != null)
                    {
                        _mtpPool.Push(initialMtp);
                        _mtpConfigured = true;
                    }
                }
                else
                {
                    _mtpConfigured = true;
                }
            }

            _loopTask = Task.Run(RunIterationLoopAsync);
        }
        catch
        {
            while (_mtpPool.TryPop(out var mtp))
                mtp.Dispose();
            _promptLookupProvider?.Dispose();
            _runner?.Dispose();
            _pool?.Dispose();
            _cts.Dispose();
            _gguf.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Hot-loads or atomically replaces a <c>0x41524F4C</c> LoRA adapter without reloading the base model.
    /// </summary>
    public void LoadLoraAdapter(string name, string path)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        string trimmedName = name.Trim();
        if (trimmedName.Contains(':', StringComparison.Ordinal))
            throw new ArgumentException("LoRA adapter name cannot contain ':'.", nameof(name));

        string resolvedPath = Directory.Exists(path)
            ? Path.Combine(path, "adapter_model.bin")
            : path;

        var weights = LoraAdapterWeights.Load(resolvedPath);
        ValidateLoraDimensions(weights);

        ulong instanceId = (ulong)Interlocked.Increment(ref _loraInstanceEpoch);
        _loraAdapters[trimmedName] = new LoadedLoraEntry(weights, instanceId, resolvedPath);
    }

    /// <summary>
    /// Unloads a previously registered LoRA adapter. Any in-flight sequences keep their captured adapter weights until completion.
    /// </summary>
    public bool UnloadLoraAdapter(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            return false;

        return _loraAdapters.TryRemove(name.Trim(), out _);
    }

    /// <summary>
    /// Returns a deterministic snapshot of currently loaded LoRA adapter names.
    /// </summary>
    public IReadOnlyList<string> GetLoadedLoraAdapters()
    {
        var keys = new List<string>(_loraAdapters.Keys);
        keys.Sort(StringComparer.Ordinal);
        return keys.ToArray();
    }

    private void ValidateLoraDimensions(LoraAdapterWeights weights)
    {
        if (weights.LayerCount > _weights.BlockCount)
        {
            throw new InvalidDataException(
                $"LoRA adapter layer count ({weights.LayerCount}) exceeds model layer count ({_weights.BlockCount}).");
        }

        int dim = _weights.EmbeddingLength;
        int ffnDim = _weights.FeedForwardLength;
        int qDim = _weights.HeadCount * _weights.HeadDim;
        int kvDim = _weights.HeadCountKv * _weights.HeadDim;
        int vHeadDim = _weights.ValueDim > 0 ? _weights.ValueDim : _weights.HeadDim;
        int vDim = _weights.HeadCountKv * vHeadDim;
        int outDim = _weights.HeadCount * vHeadDim;

        int layersToCheck = Math.Min(weights.LayerCount, _weights.BlockCount);
        for (int l = 0; l < layersToCheck; l++)
        {
            var lp = weights.GetProjections(l);
            ValidateLoraSlot(lp.Q, l, LoraProjection.Q, dim, qDim);
            ValidateLoraSlot(lp.K, l, LoraProjection.K, dim, kvDim);
            ValidateLoraSlot(lp.V, l, LoraProjection.V, dim, vDim);
            ValidateLoraSlot(lp.AttnOut, l, LoraProjection.AttnOut, outDim, dim);
            ValidateLoraSlot(lp.Gate, l, LoraProjection.Gate, dim, ffnDim);
            ValidateLoraSlot(lp.Up, l, LoraProjection.Up, dim, ffnDim);
            ValidateLoraSlot(lp.Down, l, LoraProjection.Down, ffnDim, dim);
        }
    }

    private static void ValidateLoraSlot(
        (float[]? A, float[]? B, int inDim, int outDim) slot,
        int layer,
        LoraProjection proj,
        int expectedIn,
        int expectedOut)
    {
        if (slot.A == null && slot.B == null)
            return;

        if ((slot.inDim > 0 && slot.inDim != expectedIn) ||
            (slot.outDim > 0 && slot.outDim != expectedOut))
        {
            throw new InvalidDataException(
                $"LoRA projection {proj} at layer {layer} dimension mismatch: expected [{expectedIn}, {expectedOut}], got [{slot.inDim}, {slot.outDim}].");
        }
    }

    private (string? Name, LoadedLoraEntry? Entry) ResolveLoraAdapter(InferenceRequest request, string? defaultModelName)
    {
        string? explicitLora = null;
        if (request.Lora != null)
        {
            if (string.IsNullOrWhiteSpace(request.Lora))
                throw new ArgumentException("LoRA adapter name cannot be empty or whitespace.", nameof(request));
            explicitLora = request.Lora.Trim();
        }

        string? modelLora = null;
        string rawModel = request.Model?.Trim() ?? string.Empty;
        if (!string.IsNullOrEmpty(rawModel))
        {
            bool isExactDefaultModel = defaultModelName != null &&
                string.Equals(rawModel, defaultModelName, StringComparison.OrdinalIgnoreCase);

            if (!isExactDefaultModel)
            {
                int colonIdx = rawModel.LastIndexOf(':');
                if (colonIdx >= 0)
                {
                    string basePart = rawModel[..colonIdx].Trim();
                    string adapterPart = rawModel[(colonIdx + 1)..].Trim();
                    if (string.IsNullOrEmpty(basePart) || string.IsNullOrEmpty(adapterPart))
                    {
                        throw new ArgumentException($"Invalid model:adapter specifier '{rawModel}'.", nameof(request));
                    }
                    modelLora = adapterPart;
                }
            }
        }

        if (modelLora != null && explicitLora != null &&
            !string.Equals(modelLora, explicitLora, StringComparison.Ordinal))
        {
            throw new ArgumentException(
                $"Conflicting LoRA adapter selection between model '{rawModel}' ('{modelLora}') and lora '{explicitLora}'.",
                nameof(request));
        }

        string? selectedName = modelLora ?? explicitLora;
        if (selectedName == null)
            return (null, null);

        if (!_loraAdapters.TryGetValue(selectedName, out var entry))
            throw new KeyNotFoundException($"Unknown LoRA adapter '{selectedName}'.");

        return (selectedName, entry);
    }

    private static ulong ComputeAdapterSeedHash(ulong instanceId, string adapterName)
    {
        ulong hash = XxHash3.HashToUInt64(MemoryMarshal.AsBytes(adapterName.AsSpan()), (long)instanceId);
        return hash == 0UL ? (instanceId | 1UL) : hash;
    }

    /// <summary>
    /// Enqueues a new inference request into the continuous batch queue.
    /// Throws <see cref="ArgumentOutOfRangeException"/> if the prompt is empty or meets/exceeds <see cref="MaxSeqLen"/>.
    /// </summary>
    public Task<(SequenceState State, IAsyncEnumerable<string> TokenStream)> EnqueueAsync(InferenceRequest request, CancellationToken ct = default) =>
        EnqueueAsync(request, defaultModelName: null, ct);

    /// <summary>
    /// Enqueues a new inference request into the continuous batch queue using an optional <paramref name="defaultModelName"/>
    /// to disambiguate base model names that contain colons (e.g. <c>qwen2.5:7b</c>).
    /// </summary>
    public async Task<(SequenceState State, IAsyncEnumerable<string> TokenStream)> EnqueueAsync(
        InferenceRequest request,
        string? defaultModelName,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ct.ThrowIfCancellationRequested();
        if (_faultException is Exception loopFault)
        {
            throw new InvalidOperationException("ContinuousBatchEngine background loop faulted and cannot accept new requests.", loopFault);
        }

        var (loraName, loraEntry) = ResolveLoraAdapter(request, defaultModelName);

        float temp = request.EffectiveTemperature;
        float topP = request.EffectiveTopP;
        float repPenalty = request.EffectiveRepetitionPenalty;

        if (temp < 0f || float.IsNaN(temp))
            throw new ArgumentOutOfRangeException(nameof(request), temp, "Temperature must be non-negative.");
        if (topP < 0f || topP > 1f || float.IsNaN(topP))
            throw new ArgumentOutOfRangeException(nameof(request), topP, "TopP must be in [0.0, 1.0].");
        if (repPenalty <= 0f || float.IsNaN(repPenalty))
            throw new ArgumentOutOfRangeException(nameof(request), repPenalty, "RepetitionPenalty must be greater than zero.");

        int configuredBudget = request.GetEffectiveReasoningBudget(_defaultMaxReasoningTokens);
        if (configuredBudget < ReasoningBudgetPolicy.AutoBudgetSentinel)
        {
            throw new ArgumentOutOfRangeException(
                nameof(request),
                configuredBudget,
                $"Reasoning budget ({configuredBudget}) must be >= {ReasoningBudgetPolicy.AutoBudgetSentinel}.");
        }

        string? rawUserQueryForSleep = null;
        string promptText;
        var sleepMemory = _sleepMemory;
        if (sleepMemory != null)
        {
            if (request.Messages != null && request.Messages.Count > 0)
            {
                for (int i = request.Messages.Count - 1; i >= 0; i--)
                {
                    if (string.Equals(request.Messages[i].Role, "user", StringComparison.OrdinalIgnoreCase))
                    {
                        rawUserQueryForSleep = request.Messages[i].Content;
                        break;
                    }
                }
                var consolidated = await sleepMemory.ConsolidateMessagesAsync(request.Messages, ct).ConfigureAwait(false);
                promptText = FormatMessages(consolidated, request.EffectiveEnableThinking);
            }
            else
            {
                rawUserQueryForSleep = request.Prompt;
                promptText = await sleepMemory.ConsolidatePromptAsync(request.Prompt ?? string.Empty, ct).ConfigureAwait(false);
            }
        }
        else
        {
            promptText = FormatPrompt(request);
        }

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
        bool captureHiddenStates = _mtpConfigured && RunnerSupportsHiddenStateForSequence(loraEntry?.Weights);
        SequenceState? state = null;
        float[]? rentedHidden = null;

        try
        {
            AttachMatchedPrefixBlocks(
                blockTable,
                promptTokens,
                loraEntry,
                loraName,
                captureHiddenStates,
                out int matchedBlocks,
                out ulong runningHash);

            int blockSize = _pool.BlockSize;
            rentedHidden = captureHiddenStates ? RentHiddenStateHistory() : null;

            state = new SequenceState(
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
                cancellationToken: ct)
            {
                PrefilledTokens = matchedBlocks * blockSize,
                CurrentPos = matchedBlocks * blockSize,
                BoundPrefixBlocks = matchedBlocks,
                LastPrefixHash = runningHash,
                LoraAdapterName = loraName,
                LoraAdapter = loraEntry?.Weights,
                LoraAdapterInstanceId = loraEntry?.InstanceId ?? 0UL,
                RawUserQueryForSleep = rawUserQueryForSleep,
                HiddenStateHistory = rentedHidden
            };
            rentedHidden = null;

            if (_tokenizer.TryGetTokenId("<think>", out int thinkId))
                state.SamplingOptions.ThinkTokenId = thinkId;
            if (_tokenizer.TryGetTokenId("</think>", out int endThinkId))
                state.SamplingOptions.EndThinkTokenId = endThinkId;

            int resolvedBudget = (!request.HasExplicitReasoningBudget && request.EffectiveEnableThinking == false)
                ? 0
                : ReasoningBudgetPolicy.ResolveBudget(configuredBudget, effectiveMaxNewTokens);
            state.SamplingOptions.MaxReasoningTokens = resolvedBudget;

            if (request.EnableReasoningLoopGuard.HasValue)
                state.SamplingOptions.EnableReasoningLoopGuard = request.EnableReasoningLoopGuard.Value;
            if (request.ReasoningLoopMinPeriod.HasValue)
                state.SamplingOptions.ReasoningLoopMinPeriod = request.ReasoningLoopMinPeriod.Value;
            else if (_defaultReasoningLoopMinPeriod != 1)
                state.SamplingOptions.ReasoningLoopMinPeriod = _defaultReasoningLoopMinPeriod;

            state.InitializeReasoningTracker(
                state.SamplingOptions.ThinkTokenId,
                state.SamplingOptions.EndThinkTokenId,
                resolvedBudget,
                promptTokens,
                promptText);

            Interlocked.Increment(ref _waitingCount);
            _sleepDaemon?.RecordActivity();
            try
            {
                await _waitingQueue.Writer.WriteAsync(state, ct);
            }
            catch
            {
                Interlocked.Decrement(ref _waitingCount);
                throw;
            }
        }
        catch
        {
            if (state != null)
            {
                CancelAndRecycleSequence(state);
            }
            else
            {
                if (rentedHidden != null)
                    _hiddenHistoryPool.Push(rentedHidden);
                blockTable.ReleaseAll();
            }
            throw;
        }

        return (state, ReadTokensAsync(state, ct));
    }

    private void AttachMatchedPrefixBlocks(
        BlockTable blockTable,
        int[] promptTokens,
        LoadedLoraEntry? loraEntry,
        string? loraName,
        bool captureHiddenStates,
        out int matchedBlocks,
        out ulong runningHash)
    {
        int blockSize = _pool.BlockSize;
        int fullBlocks = promptTokens.Length / blockSize;
        matchedBlocks = 0;
        runningHash = loraEntry != null ? ComputeAdapterSeedHash(loraEntry.InstanceId, loraName!) : 0UL;

        if (captureHiddenStates || fullBlocks <= 0)
            return;

        Span<int> matchedBlockIds = stackalloc int[Math.Min(fullBlocks, 128)];
        List<int>? dynamicMatched = null;
        int pendingBlockId = -1;
        try
        {
            for (int b = 0; b < fullBlocks; b++)
            {
                ReadOnlySpan<int> blockTokens = promptTokens.AsSpan(b * blockSize, blockSize);
                ulong blockHash = PagedBlockPool.ComputePrefixHash(runningHash, blockTokens);
                if (blockHash == 0UL) blockHash = 1UL;
                if (_pool.TryFindPrefixBlock(blockHash, out int pBlockId))
                {
                    pendingBlockId = pBlockId;
                    runningHash = blockHash;
                    if (matchedBlocks < matchedBlockIds.Length)
                    {
                        matchedBlockIds[matchedBlocks] = pBlockId;
                    }
                    else
                    {
                        if (dynamicMatched == null)
                        {
                            var list = new List<int>(fullBlocks);
                            foreach (int id in matchedBlockIds)
                                list.Add(id);
                            dynamicMatched = list;
                        }
                        dynamicMatched.Add(pBlockId);
                    }
                    pendingBlockId = -1;
                    matchedBlocks++;
                }
                else
                {
                    break;
                }
            }

            if (matchedBlocks > 0)
            {
                if (dynamicMatched != null)
                {
                    blockTable.AttachPrefixBlocks(CollectionsMarshal.AsSpan(dynamicMatched), matchedBlocks * blockSize);
                }
                else
                {
                    blockTable.AttachPrefixBlocks(matchedBlockIds.Slice(0, matchedBlocks), matchedBlocks * blockSize);
                }
            }
        }
        catch
        {
            int attached = blockTable.BlockCount;
            if (matchedBlocks > attached)
            {
                if (dynamicMatched != null)
                {
                    _pool.FreeBlocks(CollectionsMarshal.AsSpan(dynamicMatched).Slice(attached));
                }
                else
                {
                    _pool.FreeBlocks(matchedBlockIds.Slice(attached, matchedBlocks - attached));
                }
            }
            if (pendingBlockId >= 0)
            {
                _pool.FreeBlock(pendingBlockId);
            }
            throw;
        }
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
        var batchLoras = new LoraAdapterWeights?[_maxBatchSize];

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
                    _sleepDaemon?.RecordActivity();
                    ExecuteDecodeStep(batchTokens, batchPositions, successFlags, blockTables, batchLoras);
                }

                // 4. Non-Blocking Chunked Batched Prefill Step (at most _prefillChunkSize tokens per tick)
                if (_prefillingBatch.Count > 0)
                {
                    _sleepDaemon?.RecordActivity();
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
        BlockTable[] blockTables,
        LoraAdapterWeights?[] batchLoras)
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

        int totalActiveCount = _prefillingBatch.Count + _activeBatch.Count;
        Volatile.Write(ref _activeSequenceCount, totalActiveCount);

        int count = _activeBatch.Count;
        if (count == 0) return;

        bool allowSpeculation = _enableSpeculativeDecoding &&
                                _maxDraftTokens > 0 &&
                                totalActiveCount <= _speculativeBatchThreshold;

        if (!allowSpeculation)
        {
            ExecuteSerialForwardBatch(_activeBatch, batchTokens, batchPositions, successFlags, blockTables, batchLoras);
        }
        else
        {
            ExecuteSpeculativeOrSerialBatch(batchTokens, batchPositions, successFlags, blockTables, batchLoras);
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

    private void ExecuteSerialForwardBatch(
        List<SequenceState> sequences,
        int[] batchTokens,
        int[] batchPositions,
        bool[] successFlags,
        BlockTable[] blockTables,
        LoraAdapterWeights?[] batchLoras)
    {
        int count = sequences.Count;
        if (count == 0) return;

        float[]?[]? batchHistories = _mtpConfigured ? new float[]?[count] : null;
        for (int i = 0; i < count; i++)
        {
            var seq = sequences[i];
            batchTokens[i] = seq.LastToken;
            batchPositions[i] = seq.CurrentPos;
            blockTables[i] = seq.BlockTable;
            batchLoras[i] = seq.LoraAdapter;
            if (batchHistories != null)
                batchHistories[i] = seq.HiddenStateHistory;
        }

        var hiddenCtx = _runner as IPagedHiddenStateContext;
        hiddenCtx?.BindBatchHiddenStateHistories(batchHistories);

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
                computeLogits: true,
                batchLoras);

            for (int i = 0; i < count; i++)
            {
                var seq = sequences[i];
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
            hiddenCtx?.BindBatchHiddenStateHistories(null);
            ArrayPool<float>.Shared.Return(batchLogits);
        }
    }

    private void ExecuteSpeculativeOrSerialBatch(
        int[] batchTokens,
        int[] batchPositions,
        bool[] successFlags,
        BlockTable[] blockTables,
        LoraAdapterWeights?[] batchLoras)
    {
        var serialBatch = new List<SequenceState>(_activeBatch.Count);
        var hiddenCtx = _runner as IPagedHiddenStateContext;
        int totalVocab = _weights.VocabSize;
        int maxVerifyLen = _maxDraftTokens + 1;

        int[] draftScratch = new int[_maxDraftTokens];
        int[] verifyTokens = new int[maxVerifyLen];
        var tokenHistoryScratch = new List<int>(256);

        float[] verifyLogits = ArrayPool<float>.Shared.Rent(maxVerifyLen * totalVocab);
        float[]? targetProbsArr = null;
        float[]? residualProbsArr = null;
        float[]? draftProbsArr = null;

        try
        {
            for (int idx = 0; idx < _activeBatch.Count; idx++)
            {
                var seq = _activeBatch[idx];
                if (seq.CancellationToken.IsCancellationRequested)
                    continue;

                int remainingBudget = seq.MaxNewTokens - seq.GeneratedTokens.Count;
                int maxPosBudget = _maxSeqLen - 1 - seq.CurrentPos;
                int maxDraftForSeq = Math.Min(_maxDraftTokens, Math.Min(remainingBudget - 1, maxPosBudget));

                if (seq.HasPendingForcedTokens || maxDraftForSeq <= 0)
                {
                    serialBatch.Add(seq);
                    continue;
                }

                tokenHistoryScratch.Clear();
                tokenHistoryScratch.AddRange(seq.PromptTokens);
                tokenHistoryScratch.AddRange(seq.GeneratedTokens);
                ReadOnlySpan<int> allTokens = CollectionsMarshal.AsSpan(tokenHistoryScratch);

                int lookupCandidateCount = _promptLookupProvider.DraftWithMatchLength(
                    allTokens,
                    maxDraftForSeq,
                    draftScratch,
                    out int matchLen);

                bool hasMtp = false;
                if (_mtpConfigured &&
                    allTokens.Length >= 2 &&
                    seq.HiddenStateHistory != null &&
                    RunnerSupportsHiddenStateForSequence(seq.LoraAdapter))
                {
                    seq.MtpDrafter ??= RentOrCreateMtpProvider();
                    hasMtp = seq.MtpDrafter != null;
                }

                DraftDecision decision = _draftPolicy.SelectRoute(maxDraftForSeq, hasMtp, lookupCandidateCount, matchLen);
                if (decision.Route == DraftRoute.Serial || decision.DraftCount <= 0)
                {
                    Interlocked.Increment(ref _speculativeSerialFallbackRounds);
                    serialBatch.Add(seq);
                    continue;
                }

                bool isGreedy = seq.SamplingOptions.Temperature <= Sampler.GreedyTemperatureThreshold ||
                                seq.SamplingOptions.TopK == 1;

                int draftedCount;
                if (decision.Route == DraftRoute.PromptLookup)
                {
                    draftedCount = Math.Min(decision.DraftCount, lookupCandidateCount);
                    if (!isGreedy && draftedCount > 0)
                    {
                        draftProbsArr ??= ArrayPool<float>.Shared.Rent(_maxDraftTokens * totalVocab);
                        Span<float> draftProbsSpan = draftProbsArr.AsSpan(0, draftedCount * totalVocab);
                        draftProbsSpan.Clear();
                        for (int d = 0; d < draftedCount; d++)
                        {
                            int tok = draftScratch[d];
                            if ((uint)tok < (uint)totalVocab)
                                draftProbsSpan[d * totalVocab + tok] = 1.0f;
                        }
                    }
                }
                else if (decision.Route == DraftRoute.Mtp && seq.MtpDrafter != null)
                {
                    hiddenCtx?.BindHiddenStateHistory(seq.HiddenStateHistory);
                    if (isGreedy)
                    {
                        draftedCount = seq.MtpDrafter.Draft(allTokens, decision.DraftCount, draftScratch);
                    }
                    else
                    {
                        draftProbsArr ??= ArrayPool<float>.Shared.Rent(_maxDraftTokens * totalVocab);
                        Sampler draftSampler = seq.SequenceSampler ?? _sampler;
                        draftedCount = seq.MtpDrafter.DraftWithProbabilities(
                            allTokens,
                            CollectionsMarshal.AsSpan(seq.GeneratedTokens),
                            decision.DraftCount,
                            seq.SamplingOptions,
                            draftSampler,
                            totalVocab,
                            draftScratch,
                            draftProbsArr.AsSpan(0, decision.DraftCount * totalVocab));
                    }
                }
                else
                {
                    Interlocked.Increment(ref _speculativeSerialFallbackRounds);
                    serialBatch.Add(seq);
                    continue;
                }

                if (draftedCount <= 0)
                {
                    Interlocked.Increment(ref _speculativeSerialFallbackRounds);
                    serialBatch.Add(seq);
                    continue;
                }

                verifyTokens[0] = seq.LastToken;
                draftScratch.AsSpan(0, draftedCount).CopyTo(verifyTokens.AsSpan(1, draftedCount));

                int startPos = seq.CurrentPos;
                int verifyLen = draftedCount + 1;
                long roundStartTs = Stopwatch.GetTimestamp();

                Interlocked.Increment(ref _speculativeSteps);
                Interlocked.Add(ref _speculativeDraftTokensProposed, draftedCount);
                if (decision.Route == DraftRoute.Mtp)
                    Interlocked.Increment(ref _speculativeMtpRounds);
                else
                    Interlocked.Increment(ref _speculativePromptLookupRounds);

                hiddenCtx?.BindHiddenStateHistory(seq.HiddenStateHistory);
                bool verified = _runner.VerifyDraftChunk(
                    verifyTokens.AsSpan(0, verifyLen),
                    startPos,
                    seq.BlockTable,
                    _pool,
                    verifyLogits.AsSpan(0, verifyLen * totalVocab),
                    seq.LoraAdapter);
                hiddenCtx?.BindHiddenStateHistory(null);

                if (!verified)
                {
                    FlushDecoderAndPendingText(seq);
                    seq.IsFinished = true;
                    seq.FinishReason = "pool_oom";
                    continue;
                }

                if (!isGreedy)
                {
                    targetProbsArr ??= ArrayPool<float>.Shared.Rent(totalVocab);
                    residualProbsArr ??= ArrayPool<float>.Shared.Rent(totalVocab);
                }

                Sampler activeSampler = seq.SequenceSampler ?? _sampler;
                int acceptedInStep = 0;
                bool stopped = false;

                for (int i = 0; i < draftedCount; i++)
                {
                    int candidate = draftScratch[i];
                    Span<float> posLogits = verifyLogits.AsSpan(i * totalVocab, totalVocab);

                    if (isGreedy)
                    {
                        int targetPred = SampleWithSequencePolicy(
                            seq,
                            posLogits,
                            activeSampler,
                            isGreedy: true,
                            targetProbs: Span<float>.Empty,
                            out bool periodicTriggered);

                        if (candidate == targetPred)
                        {
                            acceptedInStep++;
                            seq.CurrentPos = startPos + i + 1;
                            CommitDecodedToken(seq, candidate, isFirstToken: false, periodicTriggered, out bool shouldForceCloseThink);
                            if (seq.IsFinished || shouldForceCloseThink)
                            {
                                seq.BlockTable.TruncateTo(startPos + i + 1);
                                stopped = true;
                                break;
                            }
                        }
                        else
                        {
                            seq.BlockTable.TruncateTo(startPos + i + 1);
                            seq.CurrentPos = startPos + i + 1;
                            CommitDecodedToken(seq, targetPred, isFirstToken: false, periodicTriggered, out _);
                            stopped = true;
                            break;
                        }
                    }
                    else
                    {
                        Span<float> targetProbs = targetProbsArr.AsSpan(0, totalVocab);
                        SampleWithSequencePolicy(
                            seq,
                            posLogits,
                            activeSampler,
                            isGreedy: false,
                            targetProbs,
                            out bool periodicTriggered);

                        ReadOnlySpan<float> qRow = draftProbsArr!.AsSpan(i * totalVocab, totalVocab);
                        float pCandidate = (uint)candidate < (uint)totalVocab ? targetProbs[candidate] : 0f;
                        float qCandidate = (uint)candidate < (uint)totalVocab ? qRow[candidate] : 0f;

                        bool accept = qCandidate > 0f &&
                                      (pCandidate >= qCandidate || activeSampler.Rng.NextDouble() < (pCandidate / qCandidate));

                        if (accept)
                        {
                            acceptedInStep++;
                            seq.CurrentPos = startPos + i + 1;
                            CommitDecodedToken(seq, candidate, isFirstToken: false, periodicTriggered, out bool shouldForceCloseThink);
                            if (seq.IsFinished || shouldForceCloseThink)
                            {
                                seq.BlockTable.TruncateTo(startPos + i + 1);
                                stopped = true;
                                break;
                            }
                        }
                        else
                        {
                            int correctionToken = SampleResidualDistribution(
                                targetProbs,
                                qRow,
                                residualProbsArr.AsSpan(0, totalVocab),
                                activeSampler.Rng);
                            seq.BlockTable.TruncateTo(startPos + i + 1);
                            seq.CurrentPos = startPos + i + 1;
                            CommitDecodedToken(seq, correctionToken, isFirstToken: false, periodicTriggered, out _);
                            stopped = true;
                            break;
                        }
                    }
                }

                if (!stopped && acceptedInStep == draftedCount)
                {
                    Span<float> bonusLogits = verifyLogits.AsSpan(draftedCount * totalVocab, totalVocab);
                    int bonusToken;
                    bool bonusPeriodic;
                    if (isGreedy)
                    {
                        bonusToken = SampleWithSequencePolicy(
                            seq,
                            bonusLogits,
                            activeSampler,
                            isGreedy: true,
                            targetProbs: Span<float>.Empty,
                            out bonusPeriodic);
                    }
                    else
                    {
                        Span<float> targetProbs = targetProbsArr.AsSpan(0, totalVocab);
                        SampleWithSequencePolicy(
                            seq,
                            bonusLogits,
                            activeSampler,
                            isGreedy: false,
                            targetProbs,
                            out bonusPeriodic);
                        bonusToken = Sampler.SampleFromDistribution(targetProbs, activeSampler.Rng);
                    }

                    seq.CurrentPos = startPos + draftedCount + 1;
                    CommitDecodedToken(seq, bonusToken, isFirstToken: false, bonusPeriodic, out _);
                }

                double roundElapsedMs = Stopwatch.GetElapsedTime(roundStartTs).TotalMilliseconds;
                Interlocked.Add(ref _speculativeDraftTokensAccepted, acceptedInStep);
                _draftPolicy.RecordOutcome(decision.Route, matchLen, draftedCount, acceptedInStep, roundElapsedMs);
            }

            if (serialBatch.Count > 0)
            {
                ExecuteSerialForwardBatch(serialBatch, batchTokens, batchPositions, successFlags, blockTables, batchLoras);
            }
        }
        finally
        {
            hiddenCtx?.BindHiddenStateHistory(null);
            ArrayPool<float>.Shared.Return(verifyLogits);
            if (targetProbsArr != null) ArrayPool<float>.Shared.Return(targetProbsArr);
            if (residualProbsArr != null) ArrayPool<float>.Shared.Return(residualProbsArr);
            if (draftProbsArr != null) ArrayPool<float>.Shared.Return(draftProbsArr);
        }
    }

    private int SampleWithSequencePolicy(
        SequenceState seq,
        Span<float> seqLogits,
        Sampler activeSampler,
        bool isGreedy,
        Span<float> targetProbs,
        out bool periodicTriggeredPreSample)
    {
        bool isReasoningActive = seq.IsInReasoningBlock;
        ReadOnlySpan<int> recentTokens = CollectionsMarshal.AsSpan(seq.GeneratedTokens);
        ReadOnlySpan<int> penaltyWindow = recentTokens;

        bool origEnableLoopGuard = seq.SamplingOptions.EnableReasoningLoopGuard;
        float origRepetitionPenalty = seq.SamplingOptions.RepetitionPenalty;
        periodicTriggeredPreSample = false;

        if (origEnableLoopGuard && isReasoningActive && !recentTokens.IsEmpty)
        {
            int loopPeriod = Sampler.DetectPeriodicReasoningLoop(
                recentTokens,
                minPeriod: seq.SamplingOptions.ReasoningLoopMinPeriod);
            if (loopPeriod > 0)
            {
                periodicTriggeredPreSample = true;
                seq.PeriodicLoopTriggers++;
                seq.LoopRecoveryTriggers++;
                seq.SamplingOptions.RepetitionPenalty = Math.Max(origRepetitionPenalty, seq.SamplingOptions.LoopRecoveryRepetitionPenalty);
                int windowLen = Math.Min(recentTokens.Length, loopPeriod * 3);
                penaltyWindow = recentTokens.Slice(recentTokens.Length - windowLen, windowLen);
            }
        }

        seq.SamplingOptions.EnableReasoningLoopGuard = false;
        ApplySleepExpansionResidual(seqLogits);
        try
        {
            if (isGreedy)
            {
                return activeSampler.Sample(seqLogits, seq.SamplingOptions, penaltyWindow);
            }

            Sampler.ComputeFilteredDistribution(seqLogits, seq.SamplingOptions, penaltyWindow, targetProbs);
            return 0;
        }
        finally
        {
            seq.SamplingOptions.EnableReasoningLoopGuard = origEnableLoopGuard;
            seq.SamplingOptions.RepetitionPenalty = origRepetitionPenalty;
        }
    }

    private static int SampleResidualDistribution(
        ReadOnlySpan<float> p,
        ReadOnlySpan<float> q,
        Span<float> residualScratch,
        Random rng)
    {
        int n = p.Length;
        float sumResidual = 0f;
        for (int v = 0; v < n; v++)
        {
            float diff = p[v] - q[v];
            if (diff > 0f)
            {
                residualScratch[v] = diff;
                sumResidual += diff;
            }
            else
            {
                residualScratch[v] = 0f;
            }
        }

        if (!(sumResidual > 1e-12f) || !float.IsFinite(sumResidual))
        {
            return Sampler.SampleFromDistribution(p, rng);
        }

        float r = (float)rng.NextDouble() * sumResidual;
        float acc = 0f;
        int lastPositive = 0;
        for (int v = 0; v < n; v++)
        {
            float w = residualScratch[v];
            if (w <= 0f) continue;
            lastPositive = v;
            acc += w;
            if (r <= acc)
            {
                return v;
            }
        }
        return lastPositive;
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
        int blockSize = _pool.BlockSize;

        if (startPos >= promptLen)
        {
            // Fully cached prefix: evaluate final token to obtain logits for first token sampling
            var hiddenCtx = _runner as IPagedHiddenStateContext;
            hiddenCtx?.BindHiddenStateHistory(seq.HiddenStateHistory);
            bool tokenOk = _runner.ForwardToken(
                seq.PromptTokens[promptLen - 1],
                promptLen - 1,
                seq.BlockTable,
                _pool,
                logitsBuffer.AsSpan(),
                computeLogits: true,
                seq.LoraAdapter);
            hiddenCtx?.BindHiddenStateHistory(null);

            if (!tokenOk)
            {
                seq.IsFinished = true;
                seq.FinishReason = "pool_oom";
                CompleteAndRecycleSequence(seq);
                _prefillingBatch.RemoveAt(0);
                return;
            }

            _prefillingBatch.RemoveAt(0);
            seq.IsPrefillDone = true;

            if (seq.CancellationToken.IsCancellationRequested)
            {
                CancelAndRecycleSequence(seq);
                return;
            }

            ProcessSampledToken(seq, logitsBuffer.AsSpan(), isFirstToken: true);
            if (seq.IsFinished)
            {
                CompleteAndRecycleSequence(seq);
            }
            else
            {
                _activeBatch.Add(seq);
            }
            return;
        }

        int chunkLen = Math.Min(promptLen - startPos, _prefillChunkSize);
        bool isFinalChunk = (startPos + chunkLen == promptLen);

        var chunkHiddenCtx = _runner as IPagedHiddenStateContext;
        chunkHiddenCtx?.BindHiddenStateHistory(seq.HiddenStateHistory);
        bool ok = _runner.ForwardPrefillChunk(
            seq.PromptTokens.AsSpan(startPos, chunkLen),
            startPos,
            seq.BlockTable,
            _pool,
            isFinalChunk ? logitsBuffer.AsSpan() : Span<float>.Empty,
            computeLogits: isFinalChunk,
            seq.LoraAdapter);
        chunkHiddenCtx?.BindHiddenStateHistory(null);

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

        // Bind any newly completed full blocks into the prefix hash cache
        int completedBlocks = seq.PrefilledTokens / blockSize;
        while (seq.BoundPrefixBlocks < completedBlocks)
        {
            int blockIdx = seq.BoundPrefixBlocks;
            ReadOnlySpan<int> blockTokens = seq.PromptTokens.AsSpan(blockIdx * blockSize, blockSize);
            seq.LastPrefixHash = PagedBlockPool.ComputePrefixHash(seq.LastPrefixHash, blockTokens);
            if (seq.LastPrefixHash == 0UL) seq.LastPrefixHash = 1UL;
            int physicalId = seq.BlockTable.GetBlockId(blockIdx);
            _pool.BindPrefixBlock(seq.LastPrefixHash, physicalId);
            seq.BoundPrefixBlocks++;
        }

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
        // 0. If forced closing tokens (\n</think>\n\n) are queued, emit next queued token without sampling
        if (seq.TryPopForcedToken(out int forcedToken))
        {
            seq.LastToken = forcedToken;
            seq.GeneratedTokens.Add(forcedToken);
            Interlocked.Increment(ref _totalTokensGenerated);
            DecodeTokenBytesIntoBuilder(seq, forcedToken);

            if (EmitOrStopOnSequences(seq))
                return;

            if (seq.GeneratedTokens.Count >= seq.MaxNewTokens || (!isFirstToken && seq.CurrentPos >= _maxSeqLen))
            {
                FlushDecoderAndPendingText(seq);
                seq.IsFinished = true;
                seq.FinishReason = "length";
            }
            return;
        }

        // 1. Sample next token using per-sequence reasoning loop guard & sleep residual
        Sampler activeSampler = seq.SequenceSampler ?? _sampler;
        int nextToken = SampleWithSequencePolicy(
            seq,
            seqLogits,
            activeSampler,
            isGreedy: true,
            targetProbs: Span<float>.Empty,
            out bool periodicTriggeredPreSample);

        if (isFirstToken)
        {
            seq.FirstTokenTime = DateTimeOffset.UtcNow;
        }

        CommitDecodedToken(seq, nextToken, isFirstToken, periodicTriggeredPreSample, out _);
    }

    private void CommitDecodedToken(
        SequenceState seq,
        int nextToken,
        bool isFirstToken,
        bool periodicTriggeredPreSample,
        out bool shouldForceCloseThink)
    {
        shouldForceCloseThink = false;

        // 2. EOS non-leakage check BEFORE appending to GeneratedTokens or emitting text
        if (IsEosToken(nextToken))
        {
            FlushDecoderAndPendingText(seq);
            seq.IsFinished = true;
            seq.FinishReason = "stop";
            return;
        }

        // 3. Record valid non-EOS token
        seq.LastToken = nextToken;
        seq.GeneratedTokens.Add(nextToken);
        Interlocked.Increment(ref _totalTokensGenerated);

        // 4. Incremental stateful UTF-8 decoding (prevents U+FFFD on split multi-byte BPE tokens)
        DecodeTokenBytesIntoBuilder(seq, nextToken);

        // 5. Update single-token or multi-token <think> reasoning tracker and detect forced close
        bool useSingleTokenThinkDelimiters = seq.SamplingOptions.ThinkTokenId >= 0 && seq.SamplingOptions.EndThinkTokenId >= 0;
        if (seq.SamplingOptions.MaxReasoningTokens > 0)
        {
            shouldForceCloseThink = seq.ObserveReasoningTokenAndShouldForceClose(nextToken, useSingleTokenThinkDelimiters);
            if (shouldForceCloseThink)
            {
                seq.BudgetClampTriggers++;
            }
            else if (seq.SamplingOptions.EnableReasoningLoopGuard &&
                     seq.IsInReasoningBlock &&
                     !seq.ForcedThinkClose &&
                     seq.CurrentReasoningTokens >= 24 &&
                     Sampler.DetectPeriodicReasoningLoop(
                         CollectionsMarshal.AsSpan(seq.GeneratedTokens),
                         minPeriod: Math.Max(2, seq.SamplingOptions.ReasoningLoopMinPeriod),
                         maxPeriod: 64,
                         minRepeats: 4) > 0)
            {
                seq.MarkForcedThinkClose();
                if (!periodicTriggeredPreSample)
                {
                    seq.PeriodicLoopTriggers++;
                    seq.LoopRecoveryTriggers++;
                }
                shouldForceCloseThink = true;
            }
        }
        else
        {
            seq.ObserveUnbudgetedReasoningToken(
                nextToken,
                seq.SamplingOptions.ThinkTokenId,
                seq.SamplingOptions.EndThinkTokenId,
                useSingleTokenThinkDelimiters);
        }

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
            return;
        }

        // 8. Queue forced \n</think>\n\n tokens if reasoning budget or loop guard triggered
        if (shouldForceCloseThink)
        {
            int[] closeTokens = ReasoningBudgetPolicy.BuildEndThinkTokens(_tokenizer, seq.SamplingOptions.EndThinkTokenId);
            if (closeTokens.Length > 0)
            {
                seq.EnqueueForcedTokens(closeTokens);
            }
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

    private void SweepCancelledSequences(List<SequenceState> batch)
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

    private void CancelAndRecycleSequence(SequenceState seq)
    {
        seq.IsFinished = true;
        if (string.IsNullOrEmpty(seq.FinishReason))
            seq.FinishReason = "cancelled";
        RecycleSequenceResources(seq);
        try
        {
            seq.BlockTable.ReleaseAll();
        }
        catch (ObjectDisposedException)
        {
        }
        seq.TokenChannel.Writer.TryComplete();
        if (seq.CancellationToken.IsCancellationRequested)
            seq.Completion.TrySetCanceled(seq.CancellationToken);
        else
            seq.Completion.TrySetCanceled();
    }

    private bool RunnerSupportsHiddenStateForSequence(LoraAdapterWeights? lora) =>
        _runner is IPagedHiddenStateContext ctx && ctx.SupportsHiddenStateCapture(lora);

    private float[] RentHiddenStateHistory()
    {
        if (_hiddenHistoryPool.TryPop(out var buffer))
            return buffer;
        return new float[checked(_maxSeqLen * _weights.EmbeddingLength)];
    }

    private MtpDraftProvider? RentOrCreateMtpProvider()
    {
        if (_mtpPool.TryPop(out var cached))
            return cached;

        if (_runner is not IHiddenStateSource trunk)
            return null;

        return CreateMtpProvider(trunk);
    }

    private MtpDraftProvider? CreateMtpProvider(IHiddenStateSource trunk)
    {
        if (_mtpSidecarPath != null)
        {
            return MtpDraftProvider.LoadFromSidecar(trunk, _weights, _maxSeqLen, _mtpSidecarPath);
        }

        if (_weights.EmbeddedMtpBlockIndex >= 0)
        {
            return MtpDraftProvider.TryCreateFromModel(trunk, _weights, _maxSeqLen);
        }

        return null;
    }

    private void RecycleSequenceResources(SequenceState seq)
    {
        if (seq.MtpDrafter != null)
        {
            seq.MtpDrafter.Reset();
            _mtpPool.Push(seq.MtpDrafter);
            seq.MtpDrafter = null;
        }
        if (seq.HiddenStateHistory != null)
        {
            _hiddenHistoryPool.Push(seq.HiddenStateHistory);
            seq.HiddenStateHistory = null;
        }
    }

    private void ApplySleepExpansionResidual(Span<float> logits)
    {
        var router = _expansionRouter;
        if (router == null || router.ActiveExpertCount <= 0)
            return;

        int dim = router.Dimension;
        int rank = router.Rank;
        if (dim <= 0 || rank <= 0 || logits.Length < dim)
            return;

        float[]? rentedDelta = null;
        float[]? rentedRank = null;
        Span<float> delta = dim <= 512
            ? stackalloc float[dim]
            : (rentedDelta = ArrayPool<float>.Shared.Rent(dim)).AsSpan(0, dim);
        Span<float> rankScratch = rank <= 128
            ? stackalloc float[rank]
            : (rentedRank = ArrayPool<float>.Shared.Rent(rank)).AsSpan(0, rank);

        try
        {
            router.ForwardDelta(logits[..dim], delta, rankScratch);
            float scale = _expansionResidualScale;
            for (int i = 0; i < dim; i++)
            {
                logits[i] += delta[i] * scale;
            }
        }
        finally
        {
            if (rentedDelta != null) ArrayPool<float>.Shared.Return(rentedDelta);
            if (rentedRank != null) ArrayPool<float>.Shared.Return(rentedRank);
        }
    }

    private void CompleteAndRecycleSequence(SequenceState seq)
    {
        RecycleSequenceResources(seq);
        try
        {
            seq.BlockTable.ReleaseAll();
        }
        catch (ObjectDisposedException)
        {
        }
        seq.TokenChannel.Writer.TryComplete();
        string completedText = seq.GeneratedTextBuilder.ToString();
        seq.Completion.TrySetResult(completedText);
        Interlocked.Increment(ref _totalRequestsServed);

        var sleepMemory = _sleepMemory;
        if (sleepMemory != null)
        {
            if (seq.FinishReason == "stop" && !string.IsNullOrWhiteSpace(completedText))
            {
                sleepMemory.RecordTurnCompletion(seq.RawUserQueryForSleep, completedText);
            }
            return;
        }

        var sleepDaemon = _sleepDaemon;
        if (sleepDaemon != null && seq.FinishReason == "stop" && !string.IsNullOrWhiteSpace(completedText))
        {
            int dim = _sleepVectorDimension;
            float[] queryVec = BuildTokenFeatureVector(seq.PromptTokens, dim);
            float[] targetVec = BuildTokenFeatureVector(seq.GeneratedTokens, dim);
            sleepDaemon.QueueDream(new DreamCandidate(
                PromptPrefix: seq.RequestId,
                GeneratedDream: completedText,
                QueryVector: queryVec,
                TargetTeacherOutput: targetVec,
                SemanticSimilarity: 1.0f,
                LevenshteinRatio: 1.0f,
                CompositeScore: 0.90f));
        }
    }

    private static float[] BuildTokenFeatureVector(IReadOnlyList<int> tokens, int dimension)
    {
        float[] vec = new float[dimension];
        if (tokens.Count == 0)
        {
            vec[0] = 1.0f;
            return vec;
        }

        for (int i = 0; i < tokens.Count; i++)
        {
            uint h = (uint)tokens[i] * 2654435761u;
            int idx = (int)(h % (uint)dimension);
            vec[idx] += ((h & 1u) == 0u) ? 1.0f : -1.0f;
        }

        float normSq = 0f;
        for (int i = 0; i < dimension; i++) normSq += vec[i] * vec[i];
        if (normSq > 0f)
        {
            float invNorm = 1.0f / MathF.Sqrt(normSq);
            for (int i = 0; i < dimension; i++) vec[i] *= invNorm;
        }
        else
        {
            vec[0] = 1.0f;
        }
        return vec;
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
            return FormatMessages(req.Messages, req.EffectiveEnableThinking);
        }
        return req.Prompt ?? string.Empty;
    }

    private string FormatMessages(IReadOnlyList<ChatMessage> messages, bool? effectiveEnableThinking)
    {
        var list = new List<(string Role, string Content)>(messages.Count);
        foreach (var m in messages)
        {
            list.Add((m.Role, m.Content));
        }
        bool hasThinkToken = _tokenizer.TryGetTokenId("<think>", out _);
        bool thinking = effectiveEnableThinking ?? hasThinkToken;
        return _tokenizer.FormatConversation(list, thinkingMode: thinking);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

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

        if (_loopTask.IsCompleted)
        {
            DisposeOwnedResources();
        }
        else
        {
            _ = _loopTask.ContinueWith(
                _ => DisposeOwnedResources(),
                CancellationToken.None,
                TaskContinuationOptions.ExecuteSynchronously,
                TaskScheduler.Default);
        }
    }

    private void DisposeOwnedResources()
    {
        _cts.Dispose();
        while (_mtpPool.TryPop(out var mtp))
        {
            mtp.Dispose();
        }
        _hiddenHistoryPool.Clear();
        _promptLookupProvider.Dispose();
        _runner.Dispose();
        _pool.Dispose();
        _gguf.Dispose();
    }
}

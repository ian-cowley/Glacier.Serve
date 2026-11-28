namespace Glacier.Serve.Daemons;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Continuum;
using Glacier.Inference.Hardware;
using Glacier.MemEx;
using Glacier.MemEx.Models;
using Glacier.Rag.Embeddings;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Serialization;
using Glacier.Tune.Expansion;
using Glacier.Tune.Sleep;

/// <summary>
/// Configuration options for <see cref="SleepMemoryService"/> including optional NPU semantic embedding support.
/// </summary>
public sealed class SleepMemoryOptions
{
    public int TokenBudget { get; init; } = 400;
    public TimeSpan? IdleThreshold { get; init; }
    public TimeSpan? PollingInterval { get; init; }
    public int MinDreamsToTriggerSleep { get; init; } = 1;

    public bool EnableNpuEmbeddings { get; init; } = true;
    public string? EmbeddingEndpoint { get; init; }
    public string? EmbeddingModelId { get; init; }
    public string? EmbeddingApiKey { get; init; }
    public int EmbeddingDimensions { get; init; }
    public HttpClient? HttpClient { get; init; }
    public IEmbeddingModel? EmbeddingModel { get; init; }
    public TimeSpan? ProbeTimeout { get; init; }
    public TimeSpan? ProbeCacheTtl { get; init; }
    public TimeSpan? EmbeddingTimeout { get; init; }
}

/// <summary>
/// Online wake-sleep episodic consolidation service backed by Pillar 16 HOPE Continuum,
/// MemEx ACT-R/HippoRAG/Knapsack retrieval, Preallocated MoE Expansion, and AutonomousSleepDaemon.
/// </summary>
public sealed partial class SleepMemoryService : IDisposable, IAsyncDisposable
{
    private const int Dim = 64;
    private const int PruneKeepLimit = 512;
    private const string DefaultEmbeddingModelId = "embeddinggemma";

    private sealed record StoredChunk(
        string Id,
        string Content,
        long TimestampUs,
        float[] KeyVec,
        float[] ValueVec,
        float BaseImportance,
        EpisodicTrace Trace,
        bool IsSemantic);

    private sealed record Segment(string Id, string Content, long OffsetOrder);

    private sealed record PreparedSegment(
        Segment Seg,
        ulong Hash,
        (ulong Hash, int Length) DedupeKey,
        float[] Fnv64,
        List<string> Entities,
        float[]? SemanticVec);

    private static readonly HashSet<string> s_stopWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "the", "and", "for", "that", "with", "this", "from", "into", "your", "you",
        "are", "was", "were", "have", "has", "not", "but", "all", "can", "will",
        "only", "when", "then", "than", "each", "must", "use", "using", "via"
    };

    [GeneratedRegex(@"<think>.*?(?:</think>|$)", RegexOptions.Singleline)]
    private static partial Regex ThinkBlockRegex();

    private readonly object _memorySync = new();
    private readonly object _bankSync = new();
    private readonly object _opDrainSync = new();
    private readonly SemaphoreSlim _sleepGate = new(1, 1);
    private readonly SemaphoreSlim _embedGate = new(1, 1);
    private readonly CancellationTokenSource _disposeCts = new();
    private readonly TaskCompletionSource _opsDrainedTcs = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly List<IDisposable> _retiredModels = new();

    private readonly SleepMemoryOptions _options;
    private readonly HopeContinuumHierarchy _hope;
    private readonly MemExEngine _memEx;
    private readonly PreallocatedMoeExpansionBank _bank;
    private readonly SleepConsolidationTrainer _trainer;
    private readonly AutonomousSleepDaemon _daemon;

    private readonly List<StoredChunk> _records = new();
    private readonly Dictionary<string, StoredChunk> _recordById = new(StringComparer.Ordinal);
    private readonly HashSet<(ulong Hash, int Length)> _ingestedChunkKeys = new();
    private readonly float[] _prevK = new float[Dim];
    private readonly float[] _centroidSum = new float[Dim];
    private long _nextTimestampSeq;
    private int _rawChunks;
    private int _prunedChunks;
    private int _lastSelectedTraces;
    private int _lastPackedTokens;
    private double _lastElapsedMs;
    private int _disposed;
    private int _activeOps;

    private IEmbeddingModel? _resolvedSemanticModel;
    private bool _ownsResolvedModel;
    private FastHashEmbeddingModel? _fallbackHashModel;
    private string? _activeEndpoint;
    private string? _activeModelId;
    private long _lastProbeFailureTimestamp;
    private int _hasProbeFailure;
    private int _activeMemExDim;

    public int TokenBudget { get; }

    public int MinDreamsToTriggerSleep { get; }

    public AutonomousSleepDaemon Daemon => _daemon;

    public PreallocatedMoeExpansionBank ExpansionBank => _bank;

    public bool IsUsingNpuEmbeddings => Volatile.Read(ref _resolvedSemanticModel) is OpenAiEmbeddingModel;

    public int ActiveEmbeddingDimensions
    {
        get
        {
            int latched = Volatile.Read(ref _activeMemExDim);
            if (latched > 0) return latched;
            IEmbeddingModel? resolved = Volatile.Read(ref _resolvedSemanticModel);
            return resolved != null ? resolved.Dimensions : Dim;
        }
    }

    public string? ActiveEmbeddingEndpoint => Volatile.Read(ref _activeEndpoint);

    public string? ActiveEmbeddingModelId => Volatile.Read(ref _activeModelId) ?? "fnv1a-64";

    public SleepMemoryService(
        int tokenBudget = 400,
        TimeSpan? idleThreshold = null,
        TimeSpan? pollingInterval = null,
        int minDreamsToTriggerSleep = 1)
        : this(new SleepMemoryOptions
        {
            TokenBudget = tokenBudget,
            IdleThreshold = idleThreshold,
            PollingInterval = pollingInterval,
            MinDreamsToTriggerSleep = minDreamsToTriggerSleep
        })
    {
    }

    public SleepMemoryService(SleepMemoryOptions? options = null)
    {
        options ??= new SleepMemoryOptions();
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(options.TokenBudget, 0);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(options.MinDreamsToTriggerSleep, 0);
        ArgumentOutOfRangeException.ThrowIfNegative(options.EmbeddingDimensions);

        if (options.ProbeTimeout is { } probeTimeout &&
            probeTimeout <= TimeSpan.Zero &&
            probeTimeout != Timeout.InfiniteTimeSpan)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "ProbeTimeout must be positive.");
        }

        if (options.EmbeddingTimeout is { } embeddingTimeout &&
            embeddingTimeout <= TimeSpan.Zero &&
            embeddingTimeout != Timeout.InfiniteTimeSpan)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "EmbeddingTimeout must be positive.");
        }

        if (options.EmbeddingModel is { Dimensions: <= 0 })
        {
            throw new ArgumentOutOfRangeException(nameof(options), "EmbeddingModel.Dimensions must be positive.");
        }

        _options = options;
        TokenBudget = options.TokenBudget;
        MinDreamsToTriggerSleep = options.MinDreamsToTriggerSleep;

        _hope = HopeContinuumHierarchy.CreateDefault4Tier(dimension: Dim, surpriseThreshold: 0.45f);
        _memEx = new MemExEngine(new MemExOptions
        {
            MaxPromptTokens = options.TokenBudget,
            DecayParameterD = 0.5,
            TimeScaleTauSeconds = 1.0,
            WeightDecay = 0.20f,
            WeightVector = 1.85f,
            WeightGraph = 1.75f,
            WeightImportance = 1.15f,
            CoarseCandidateLimit = 512,
            EnableNpuEmbeddings = false,
            AutoEmbedMissingVectors = false,
            FallbackHashDimensions = Dim,
        });
        _bank = new PreallocatedMoeExpansionBank(dimension: Dim, rank: 16, maxExperts: 4);
        _trainer = new SleepConsolidationTrainer(_bank, synapticPruningCallback: OnSynapticPrune);
        var pipeline = new CapacityAwareSleepPipeline(_bank, _trainer, _bankSync);
        _daemon = new AutonomousSleepDaemon(pipeline, new SleepDaemonOptions(
            IdleThreshold: options.IdleThreshold ?? TimeSpan.FromSeconds(15),
            PollingInterval: options.PollingInterval ?? TimeSpan.FromMilliseconds(200),
            DistillationSteps: 60,
            LearningRate: 0.20f,
            QualityThreshold: 0.50f,
            MinDreamsToTriggerSleep: options.MinDreamsToTriggerSleep));
        _daemon.Start();
    }

    public async Task<bool> InitializeEmbeddingsAsync(CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ct.ThrowIfCancellationRequested();
        if (!TryEnterOperation())
        {
            ObjectDisposedException.ThrowIf(true, this);
        }

        try
        {
            IEmbeddingModel? model = await ResolveSemanticModelAsync(ct, forceReprobe: true).ConfigureAwait(false);
            return model is OpenAiEmbeddingModel;
        }
        finally
        {
            ExitOperation();
        }
    }

    public SleepMemoryStatusResponse GetStatus()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!TryEnterOperation())
        {
            ObjectDisposedException.ThrowIf(true, this);
        }

        try
        {
            lock (_memorySync)
            {
                int activeDims = _activeMemExDim > 0
                    ? _activeMemExDim
                    : Volatile.Read(ref _resolvedSemanticModel)?.Dimensions ?? Dim;

                return new SleepMemoryStatusResponse
                {
                    Enabled = true,
                    TokenBudget = TokenBudget,
                    State = _daemon.CurrentState.ToString(),
                    RawChunks = _rawChunks,
                    RetainedTraces = _memEx.Count,
                    PrunedChunks = _prunedChunks,
                    SleepCycles = _daemon.CompletedSleepCycles,
                    DreamsConsolidated = _daemon.TotalDreamsConsolidated,
                    PendingDreams = _daemon.PendingDreamCount,
                    LastSelectedTraces = _lastSelectedTraces,
                    LastPackedTokens = _lastPackedTokens,
                    LastElapsedMs = _lastElapsedMs,
                    NpuEmbeddingsActive = IsUsingNpuEmbeddings,
                    EmbeddingDimensions = activeDims,
                    EmbeddingEndpoint = ActiveEmbeddingEndpoint,
                    EmbeddingModel = ActiveEmbeddingModelId
                };
            }
        }
        finally
        {
            ExitOperation();
        }
    }

    /// <summary>
    /// Consolidates chat messages into a packed context + trailing question when the input exceeds
    /// <c>3x</c> the token budget on the final prompt or <c>2x</c> across non-system history.
    /// </summary>
    public async Task<List<(string Role, string Content)>> ConsolidateMessagesAsync(
        List<(string Role, string Content)> messages,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(messages);
        ct.ThrowIfCancellationRequested();
        if (!TryEnterOperation())
        {
            ObjectDisposedException.ThrowIf(true, this);
        }

        try
        {
            return await ConsolidateMessagesCoreAsync(messages, ct).ConfigureAwait(false);
        }
        finally
        {
            ExitOperation();
        }
    }

    private async Task<List<(string Role, string Content)>> ConsolidateMessagesCoreAsync(
        List<(string Role, string Content)> messages,
        CancellationToken ct)
    {
        _daemon.RecordActivity();
        var sw = Stopwatch.StartNew();

        var systemMessages = new List<(string Role, string Content)>();
        var nonSystemMessages = new List<(string Role, string Content)>();
        foreach (var msg in messages)
        {
            if (string.Equals(msg.Role, "system", StringComparison.OrdinalIgnoreCase))
            {
                systemMessages.Add(msg);
            }
            else
            {
                nonSystemMessages.Add(msg);
            }
        }

        if (nonSystemMessages.Count == 0)
        {
            return new List<(string Role, string Content)>(messages);
        }

        // Locate trailing user message (fallback to final non-system message).
        int lastUserIdx = -1;
        for (int i = nonSystemMessages.Count - 1; i >= 0; i--)
        {
            if (string.Equals(nonSystemMessages[i].Role, "user", StringComparison.OrdinalIgnoreCase))
            {
                lastUserIdx = i;
                break;
            }
        }
        if (lastUserIdx < 0)
        {
            lastUserIdx = nonSystemMessages.Count - 1;
        }

        int totalNonSystemTokens = 0;
        for (int i = 0; i < nonSystemMessages.Count; i++)
        {
            totalNonSystemTokens += EstimateTokens(nonSystemMessages[i].Content);
        }

        string lastUserContent = nonSystemMessages[lastUserIdx].Content ?? "";
        int lastUserTokens = EstimateTokens(lastUserContent);
        bool shouldConsolidate = lastUserTokens > TokenBudget * 3 || totalNonSystemTokens > TokenBudget * 2;

        // Always ingest prior non-system turns (deduped) so multi-turn memory persists.
        int newlyIngested = 0;
        for (int i = 0; i < lastUserIdx; i++)
        {
            string cleanHistory = StripThinkBlocks(nonSystemMessages[i].Content ?? "");
            if (!string.IsNullOrWhiteSpace(cleanHistory))
            {
                newlyIngested += await IngestTextAsync(cleanHistory, nonSystemMessages[i].Role ?? "user", ct).ConfigureAwait(false);
            }
        }

        if (!shouldConsolidate)
        {
            return new List<(string Role, string Content)>(messages);
        }

        var (haystackBody, questionBlock) = SplitHaystackAndQuestion(lastUserContent);
        if (!string.IsNullOrWhiteSpace(haystackBody))
        {
            newlyIngested += await IngestTextAsync(haystackBody, "ctx", ct).ConfigureAwait(false);
        }

        if (newlyIngested >= 8)
        {
            await TriggerSleepCoreAsync(ct).ConfigureAwait(false);
        }

        string queryText = string.IsNullOrWhiteSpace(questionBlock) ? lastUserContent : questionBlock;
        var (selectedContent, selectedCount, packedTokens) = await QueryMemoryAsync(queryText, ct).ConfigureAwait(false);

        sw.Stop();
        lock (_memorySync)
        {
            _lastSelectedTraces = selectedCount;
            _lastPackedTokens = packedTokens;
            _lastElapsedMs = Math.Round(sw.Elapsed.TotalMilliseconds, 2);
        }

        if (selectedCount == 0 || string.IsNullOrWhiteSpace(selectedContent))
        {
            return new List<(string Role, string Content)>(messages);
        }

        var sb = new StringBuilder();
        sb.AppendLine("Consolidated Context:");
        sb.Append(selectedContent);
        sb.AppendLine();
        sb.AppendLine();
        sb.AppendLine("Question:");
        sb.Append(queryText.Trim());

        var consolidated = new List<(string Role, string Content)>(systemMessages.Count + 1);
        consolidated.AddRange(systemMessages);
        consolidated.Add(("user", sb.ToString()));
        _daemon.RecordActivity();
        return consolidated;
    }

    /// <summary>
    /// Consolidates OpenAI/Ollama <see cref="ChatMessage"/> lists into a packed context + trailing question.
    /// </summary>
    public async Task<List<ChatMessage>> ConsolidateMessagesAsync(
        IReadOnlyList<ChatMessage> messages,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(messages);

        var tuples = new List<(string Role, string Content)>(messages.Count);
        for (int i = 0; i < messages.Count; i++)
        {
            var m = messages[i];
            tuples.Add((m.Role ?? "user", m.Content ?? string.Empty));
        }

        var consolidated = await ConsolidateMessagesAsync(tuples, ct).ConfigureAwait(false);
        var result = new List<ChatMessage>(consolidated.Count);
        for (int i = 0; i < consolidated.Count; i++)
        {
            result.Add(new ChatMessage
            {
                Role = consolidated[i].Role,
                Content = consolidated[i].Content
            });
        }
        return result;
    }

    /// <summary>
    /// Consolidates a single raw prompt (e.g. <c>/api/generate</c>) when it exceeds <c>3x</c> the token budget.
    /// </summary>
    public async Task<string> ConsolidatePromptAsync(string prompt, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ct.ThrowIfCancellationRequested();
        if (!TryEnterOperation())
        {
            ObjectDisposedException.ThrowIf(true, this);
        }

        try
        {
            if (string.IsNullOrEmpty(prompt) || EstimateTokens(prompt) <= TokenBudget * 3)
            {
                _daemon.RecordActivity();
                return prompt;
            }

            var list = await ConsolidateMessagesCoreAsync([("user", prompt)], ct).ConfigureAwait(false);
            return list.Count > 0 ? list[^1].Content : prompt;
        }
        finally
        {
            ExitOperation();
        }
    }

    /// <summary>
    /// Records a completed inference turn into episodic memory after stripping <c>&lt;think&gt;</c> traces,
    /// queueing high-surprise segments for offline dream consolidation.
    /// </summary>
    public void RecordTurnCompletion(string? userQuery, string? assistantResponse)
    {
        if (!TryEnterOperation()) return;

        try
        {
            string cleanUser = (userQuery ?? "").Trim();
            if (!string.IsNullOrEmpty(cleanUser) && EstimateTokens(cleanUser) <= TokenBudget * 3)
            {
                IngestText(cleanUser, "user");
            }

            string cleanAssistant = StripThinkBlocks(assistantResponse ?? "").Trim();
            if (!string.IsNullOrEmpty(cleanAssistant))
            {
                IngestText(cleanAssistant, "assistant");
            }

            if (Volatile.Read(ref _disposed) == 0)
            {
                _daemon.RecordActivity();
            }
        }
        catch (OperationCanceledException) when (Volatile.Read(ref _disposed) != 0 || IsDisposeCancellationRequested())
        {
        }
        catch (ObjectDisposedException) when (Volatile.Read(ref _disposed) != 0)
        {
        }
        finally
        {
            ExitOperation();
        }
    }

    /// <summary>
    /// Asynchronously records a completed inference turn into episodic memory, using NPU semantic embeddings when available.
    /// </summary>
    public async Task RecordTurnCompletionAsync(string? userQuery, string? assistantResponse, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ct.ThrowIfCancellationRequested();
        if (!TryEnterOperation())
        {
            ObjectDisposedException.ThrowIf(true, this);
        }

        try
        {
            string cleanUser = (userQuery ?? "").Trim();
            if (!string.IsNullOrEmpty(cleanUser) && EstimateTokens(cleanUser) <= TokenBudget * 3)
            {
                await IngestTextAsync(cleanUser, "user", ct).ConfigureAwait(false);
            }

            string cleanAssistant = StripThinkBlocks(assistantResponse ?? "").Trim();
            if (!string.IsNullOrEmpty(cleanAssistant))
            {
                await IngestTextAsync(cleanAssistant, "assistant", ct).ConfigureAwait(false);
            }

            if (Volatile.Read(ref _disposed) == 0)
            {
                _daemon.RecordActivity();
            }
        }
        finally
        {
            ExitOperation();
        }
    }

    /// <summary>
    /// Enqueues top candidate traces and runs an immediate offline sleep consolidation cycle.
    /// </summary>
    public async Task<ConsolidationResult?> TriggerSleepAsync(CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ct.ThrowIfCancellationRequested();
        if (!TryEnterOperation())
        {
            ObjectDisposedException.ThrowIf(true, this);
        }

        try
        {
            return await TriggerSleepCoreAsync(ct).ConfigureAwait(false);
        }
        finally
        {
            ExitOperation();
        }
    }

    private async Task<ConsolidationResult?> TriggerSleepCoreAsync(CancellationToken ct)
    {
        using var linkedCts = CreateLinkedDisposeCts(ct);
        await _sleepGate.WaitAsync(linkedCts.Token).ConfigureAwait(false);
        try
        {
            if (Volatile.Read(ref _disposed) != 0)
            {
                return null;
            }

            List<(string Id, string Content)>? upgradeSnapshot = null;
            lock (_memorySync)
            {
                if (Volatile.Read(ref _disposed) != 0)
                {
                    return null;
                }

                if (_options.EnableNpuEmbeddings &&
                    (_activeMemExDim == 0 || _activeMemExDim == Dim) &&
                    _records.Any(r => !r.IsSemantic))
                {
                    upgradeSnapshot = new List<(string Id, string Content)>(_records.Count);
                    for (int i = 0; i < _records.Count; i++)
                    {
                        upgradeSnapshot.Add((_records[i].Id, _records[i].Content));
                    }
                }
            }

            if (upgradeSnapshot is { Count: > 0 })
            {
                await TryUpgradeOfflineTracesAsync(upgradeSnapshot, linkedCts.Token).ConfigureAwait(false);
            }

            float[] centroidCopy = new float[Dim];
            int queued = 0;

            lock (_memorySync)
            {
                if (Volatile.Read(ref _disposed) != 0)
                {
                    return null;
                }

                Array.Copy(_centroidSum, centroidCopy, Dim);
                NormalizeInPlace(centroidCopy);

                long nowUs = GetCurrentWallOrSeqUs();
                var exported = _memEx.ExportConsolidationCandidates(maxCount: 32, minAccessCount: 0, currentTimestampUs: nowUs);
                foreach (var trace in exported)
                {
                    if (!_recordById.TryGetValue(trace.Id, out var rec)) continue;
                    string preview = rec.Content.Length <= 220 ? rec.Content : rec.Content[..220];
                    _daemon.QueueDream(new DreamCandidate(
                        PromptPrefix: preview,
                        GeneratedDream: preview,
                        QueryVector: rec.KeyVec,
                        TargetTeacherOutput: rec.ValueVec,
                        SemanticSimilarity: 0f,
                        LevenshteinRatio: 0f,
                        CompositeScore: 0f));
                    queued++;
                }
            }

            if (Volatile.Read(ref _disposed) != 0)
            {
                return null;
            }

            if (queued == 0 && _daemon.PendingDreamCount == 0)
            {
                return null;
            }

            try
            {
                return await _daemon.TriggerImmediateSleepAsync(centroidCopy, linkedCts.Token).ConfigureAwait(false);
            }
            catch (ObjectDisposedException) when (Volatile.Read(ref _disposed) != 0)
            {
                return null;
            }
        }
        finally
        {
            try
            {
                _sleepGate.Release();
            }
            catch (ObjectDisposedException ex)
            {
                Glacier.Serve.Diagnostics.GlacierDiagnostics.LogDebug($"[SleepMemoryService] Sleep gate already disposed: {ex.Message}");
            }
        }
    }

    private void OnSynapticPrune()
    {
        lock (_memorySync)
        {
            if (Volatile.Read(ref _disposed) != 0)
            {
                return;
            }

            _hope.GetTier(0).ResetVolatileState();
            if (_memEx.Count > PruneKeepLimit)
            {
                var dropList = _records
                    .OrderByDescending(r => r.BaseImportance)
                    .ThenByDescending(r => r.TimestampUs)
                    .Skip(PruneKeepLimit)
                    .ToList();

                if (dropList.Count > 0)
                {
                    string[] dropIds = dropList.Select(r => r.Id).ToArray();
                    _prunedChunks += _memEx.PruneTraces(dropIds);
                    foreach (var drop in dropList)
                    {
                        _recordById.Remove(drop.Id);
                    }
                    var dropSet = new HashSet<string>(dropIds, StringComparer.Ordinal);
                    _records.RemoveAll(r => dropSet.Contains(r.Id));
                }
            }
        }
    }

    private int IngestText(string text, string prefix)
    {
        if (!_options.EnableNpuEmbeddings && _options.EmbeddingModel is null or FastHashEmbeddingModel)
        {
            return IngestTextOfflineSync(text, prefix);
        }

        IEmbeddingModel? resolved = Volatile.Read(ref _resolvedSemanticModel);
        if (resolved != null && resolved is not OpenAiEmbeddingModel)
        {
            return IngestTextSyncWithModel(text, prefix, resolved);
        }

        return BridgeSync(() => IngestTextAsync(text, prefix, CancellationToken.None));
    }

    private int IngestTextOfflineSync(string text, string prefix)
    {
        var candidates = CollectNewSegments(text, prefix);
        if (candidates.Count == 0) return 0;

        var prepared = new List<PreparedSegment>(candidates.Count);
        foreach (var (seg, hash, dedupeKey) in candidates)
        {
            var (fnv64, entities) = EncodeTextAndEntities(seg.Content);
            prepared.Add(new PreparedSegment(seg, hash, dedupeKey, fnv64, entities, null));
        }

        return CommitPreparedSegments(prepared);
    }

    private int IngestTextSyncWithModel(string text, string prefix, IEmbeddingModel model)
    {
        var candidates = CollectNewSegments(text, prefix);
        if (candidates.Count == 0) return 0;

        var prepared = new List<PreparedSegment>(candidates.Count);
        foreach (var (seg, hash, dedupeKey) in candidates)
        {
            var (fnv64, entities) = EncodeTextAndEntities(seg.Content);
            float[]? semVec = TryGenerateSyncModelVector(model, seg.Content);
            prepared.Add(new PreparedSegment(seg, hash, dedupeKey, fnv64, entities, semVec));
        }

        return CommitPreparedSegments(prepared);
    }

    private async Task<int> IngestTextAsync(string text, string prefix, CancellationToken ct)
    {
        var candidates = CollectNewSegments(text, prefix);
        if (candidates.Count == 0)
        {
            return 0;
        }

        var prepared = new List<PreparedSegment>(candidates.Count);
        foreach (var (seg, hash, dedupeKey) in candidates)
        {
            ct.ThrowIfCancellationRequested();
            var (fnv64, entities) = EncodeTextAndEntities(seg.Content);
            float[]? semVec = await TryComputeSemanticEmbeddingAsync(seg.Content, ct).ConfigureAwait(false);
            prepared.Add(new PreparedSegment(seg, hash, dedupeKey, fnv64, entities, semVec));
        }

        return CommitPreparedSegments(prepared);
    }

    private List<(Segment Seg, ulong Hash, (ulong Hash, int Length) DedupeKey)> CollectNewSegments(string text, string prefix)
    {
        var segments = SegmentText(text, prefix);
        if (segments.Count == 0)
        {
            return [];
        }

        var candidates = new List<(Segment Seg, ulong Hash, (ulong Hash, int Length) DedupeKey)>(segments.Count);
        var batchSeen = new HashSet<(ulong Hash, int Length)>();

        lock (_memorySync)
        {
            foreach (var seg in segments)
            {
                ulong hash = ComputeFnv1a64(seg.Content);
                var dedupeKey = (hash, seg.Content.Length);
                if (!_ingestedChunkKeys.Contains(dedupeKey) && batchSeen.Add(dedupeKey))
                {
                    candidates.Add((seg, hash, dedupeKey));
                }
            }
        }

        return candidates;
    }

    private int CommitPreparedSegments(List<PreparedSegment> prepared)
    {
        int added = 0;
        lock (_memorySync)
        {
            if (Volatile.Read(ref _disposed) != 0)
            {
                return 0;
            }

            long baseNowUs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1000L;
            foreach (var item in prepared)
            {
                if (!_ingestedChunkKeys.Add(item.DedupeKey))
                {
                    continue;
                }

                float[] traceVec;
                bool isSemantic;

                if (_activeMemExDim == 0)
                {
                    if (item.SemanticVec != null)
                    {
                        Volatile.Write(ref _activeMemExDim, item.SemanticVec.Length);
                        traceVec = item.SemanticVec;
                        isSemantic = true;
                    }
                    else
                    {
                        Volatile.Write(ref _activeMemExDim, Dim);
                        traceVec = item.Fnv64;
                        isSemantic = false;
                    }
                }
                else if (item.SemanticVec != null && item.SemanticVec.Length == _activeMemExDim)
                {
                    traceVec = item.SemanticVec;
                    isSemantic = true;
                }
                else if (item.SemanticVec != null && _activeMemExDim == Dim)
                {
                    float[] folded = new float[Dim];
                    FoldToHopeDim(item.SemanticVec, folded);
                    traceVec = folded;
                    isSemantic = item.SemanticVec.Length == Dim;
                }
                else
                {
                    traceVec = ComputeMatchedFallbackVector(item.Seg.Content, item.Fnv64, _activeMemExDim);
                    isSemantic = false;
                }

                float[] k64;
                if (isSemantic)
                {
                    k64 = new float[Dim];
                    FoldToHopeDim(traceVec, k64);
                }
                else
                {
                    k64 = new float[Dim];
                    Array.Copy(item.Fnv64, k64, Dim);
                }

                float[] vVec = BuildValueVector(k64, _prevK, _rawChunks > 0);
                Array.Copy(k64, _prevK, Dim);

                float surprise = _hope.Step(k64, vVec);
                for (int d = 0; d < Dim; d++)
                {
                    _centroidSum[d] += k64[d];
                }

                float structuredBonus = MathF.Min(1.2f, item.Entities.Count * 0.14f);
                float importance = Math.Clamp(0.4f + 3.0f * surprise + structuredBonus, 0.2f, 5.0f);
                long seq = ++_nextTimestampSeq;
                long tsUs = baseNowUs + seq;
                int estTokens = Math.Max(8, (item.Seg.Content.Length + 3) / 4);
                string traceId = $"{item.Seg.Id}_{seq}_{item.Hash:x8}";

                var trace = new EpisodicTrace
                {
                    Id = traceId,
                    Content = item.Seg.Content,
                    TimestampUs = tsUs,
                    Breadcrumbs = [],
                    Embedding = traceVec,
                    Entities = item.Entities,
                    TokenCount = estTokens,
                    BaseImportance = importance,
                };

                _memEx.StoreTrace(trace);
                var stored = new StoredChunk(traceId, item.Seg.Content, tsUs, k64, vVec, importance, trace, isSemantic);
                _records.Add(stored);
                _recordById[traceId] = stored;
                _rawChunks++;
                added++;

                if (surprise >= 0.45f)
                {
                    string preview = item.Seg.Content.Length <= 220 ? item.Seg.Content : item.Seg.Content[..220];
                    _daemon.QueueDream(new DreamCandidate(
                        PromptPrefix: preview,
                        GeneratedDream: preview,
                        QueryVector: k64,
                        TargetTeacherOutput: vVec,
                        SemanticSimilarity: 0f,
                        LevenshteinRatio: 0f,
                        CompositeScore: 0f));
                }
            }
        }

        return added;
    }

    private async Task<(string FormattedContent, int SelectedCount, int PackedTokens)> QueryMemoryAsync(string query, CancellationToken ct)
    {
        var (qFnv64, qEntities) = EncodeTextAndEntities(query);
        float[]? qSem = await TryComputeSemanticEmbeddingAsync(query, ct).ConfigureAwait(false);

        float[] q64;
        if (qSem != null)
        {
            q64 = new float[Dim];
            FoldToHopeDim(qSem, q64);
        }
        else
        {
            q64 = qFnv64;
        }

        float[] hopeOut = new float[Dim];
        float[] moeDelta = new float[Dim];
        float[] rankScratch = new float[_bank.Rank];

        lock (_bankSync)
        {
            if (Volatile.Read(ref _disposed) != 0) return ("", 0, 0);
            _bank.ForwardDelta(q64, moeDelta, rankScratch);
        }

        MemoryQueryResult queryResult;
        lock (_memorySync)
        {
            if (Volatile.Read(ref _disposed) != 0) return ("", 0, 0);
            _hope.Forward(q64, hopeOut);

            int targetDim = _activeMemExDim > 0 ? _activeMemExDim : Dim;
            float[] qBase;
            if (qSem != null && qSem.Length == targetDim)
            {
                qBase = qSem;
            }
            else if (qSem != null && targetDim == Dim)
            {
                qBase = q64;
            }
            else
            {
                qBase = ComputeMatchedFallbackVector(query, qFnv64, targetDim);
            }

            float[] hopeMoe64 = new float[Dim];
            for (int d = 0; d < Dim; d++)
            {
                hopeMoe64[d] = (0.16f * hopeOut[d]) + (0.10f * moeDelta[d]);
            }

            float[] qRecall = new float[targetDim];
            LiftFromHopeDim(hopeMoe64, qRecall);
            for (int d = 0; d < targetDim; d++)
            {
                qRecall[d] = (0.74f * qBase[d]) + qRecall[d];
            }
            NormalizeInPlace(qRecall);

            long queryNowUs = GetCurrentWallOrSeqUs() + 10_000_000L;
            queryResult = _memEx.Query(
                query,
                qRecall,
                queryEntities: qEntities,
                currentTimestampUs: queryNowUs,
                maxTokens: TokenBudget);
        }

        var chronological = queryResult.SelectedTraces
            .OrderBy(t => t.TimestampUs)
            .ToList();

        if (chronological.Count == 0)
        {
            return ("", 0, 0);
        }

        var sb = new StringBuilder();
        for (int i = 0; i < chronological.Count; i++)
        {
            if (i > 0) sb.AppendLine();
            sb.Append("- ").Append(chronological[i].Content);
        }

        return (sb.ToString(), chronological.Count, queryResult.TotalTokens);
    }

    private async Task TryUpgradeOfflineTracesAsync(List<(string Id, string Content)> snapshot, CancellationToken ct)
    {
        IEmbeddingModel? model = await ResolveSemanticModelAsync(ct, forceReprobe: false).ConfigureAwait(false);
        if (model == null)
        {
            return;
        }

        var upgraded = new Dictionary<string, float[]>(snapshot.Count, StringComparer.Ordinal);
        int uniformDim = 0;

        for (int i = 0; i < snapshot.Count; i++)
        {
            var (id, content) = snapshot[i];
            float[]? vec = await TryComputeVectorWithModelAsync(model, content, ct).ConfigureAwait(false);
            if (vec == null || vec.Length == 0 || !AreAllFinite(vec))
            {
                return;
            }

            if (uniformDim == 0)
            {
                uniformDim = vec.Length;
            }
            else if (vec.Length != uniformDim)
            {
                return;
            }

            upgraded[id] = vec;
        }

        if (uniformDim <= 0)
        {
            return;
        }

        lock (_memorySync)
        {
            if (Volatile.Read(ref _disposed) != 0 ||
                (_activeMemExDim != 0 && _activeMemExDim != Dim))
            {
                return;
            }

            for (int i = 0; i < _records.Count; i++)
            {
                if (!upgraded.ContainsKey(_records[i].Id))
                {
                    return;
                }
            }

            string[] oldIds = new string[_records.Count];
            for (int i = 0; i < _records.Count; i++)
            {
                oldIds[i] = _records[i].Id;
            }

            _memEx.PruneTraces(oldIds);
            Array.Clear(_centroidSum, 0, Dim);
            Array.Clear(_prevK, 0, Dim);
            Volatile.Write(ref _activeMemExDim, uniformDim);

            for (int i = 0; i < _records.Count; i++)
            {
                StoredChunk oldRec = _records[i];
                float[] semVec = upgraded[oldRec.Id];
                float[] k64 = new float[Dim];
                FoldToHopeDim(semVec, k64);
                float[] v64 = BuildValueVector(k64, _prevK, i > 0);
                Array.Copy(k64, _prevK, Dim);

                for (int d = 0; d < Dim; d++)
                {
                    _centroidSum[d] += k64[d];
                }

                EpisodicTrace oldTrace = oldRec.Trace;
                long[] oldAccess = oldTrace.GetAccessTimestampsSnapshot();
                var newTrace = new EpisodicTrace
                {
                    Id = oldTrace.Id,
                    Content = oldTrace.Content,
                    TimestampUs = oldTrace.TimestampUs,
                    Breadcrumbs = [.. oldTrace.Breadcrumbs],
                    Embedding = semVec,
                    Entities = [.. oldTrace.Entities],
                    TokenCount = oldTrace.TokenCount,
                    BaseImportance = oldTrace.BaseImportance,
                };
                if (oldAccess.Length > 0)
                {
                    newTrace.RecordAccess(oldAccess);
                }

                _memEx.StoreTrace(newTrace);
                var updatedChunk = oldRec with
                {
                    KeyVec = k64,
                    ValueVec = v64,
                    Trace = newTrace,
                    IsSemantic = true
                };
                _records[i] = updatedChunk;
                _recordById[updatedChunk.Id] = updatedChunk;
            }
        }
    }

    private async Task<float[]?> TryComputeSemanticEmbeddingAsync(string text, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return null;
        }

        IEmbeddingModel? model = await ResolveSemanticModelAsync(ct, forceReprobe: false).ConfigureAwait(false);
        if (model == null)
        {
            return null;
        }

        return await TryComputeVectorWithModelAsync(model, text, ct).ConfigureAwait(false);
    }

    private async Task<float[]?> TryComputeVectorWithModelAsync(IEmbeddingModel model, string text, CancellationToken ct)
    {
        if (model is OpenAiEmbeddingModel openAi)
        {
            using var linkedCts = CreateLinkedTimeoutCts(_options.EmbeddingTimeout ?? TimeSpan.FromSeconds(5), ct);
            CancellationToken effectiveToken = linkedCts.Token;
            try
            {
                float[] vec = await openAi.GenerateEmbeddingAsync(text, effectiveToken).ConfigureAwait(false);
                if (vec.Length != openAi.Dimensions || !AreAllFinite(vec))
                {
                    return null;
                }
                return vec;
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested || IsDisposeCancellationRequested())
            {
                throw;
            }
            catch (ObjectDisposedException) when (Volatile.Read(ref _disposed) != 0)
            {
                throw new OperationCanceledException(ct.IsCancellationRequested ? ct : GetSafeDisposeToken());
            }
            catch (Exception ex) when (IsRecoverableEmbeddingException(ex))
            {
                return null;
            }
        }

        return TryGenerateSyncModelVector(model, text);
    }

    private static float[]? TryGenerateSyncModelVector(IEmbeddingModel model, string text)
    {
        try
        {
            float[] vec = new float[model.Dimensions];
            model.GenerateEmbedding(text.AsSpan(), vec);
            if (!AreAllFinite(vec))
            {
                return null;
            }
            NormalizeInPlace(vec);
            return vec;
        }
        catch (Exception ex) when (IsRecoverableEmbeddingException(ex))
        {
            return null;
        }
    }

    private async Task<IEmbeddingModel?> ResolveSemanticModelAsync(CancellationToken ct, bool forceReprobe)
    {
        IEmbeddingModel? existing = Volatile.Read(ref _resolvedSemanticModel);
        if (existing != null)
        {
            return existing;
        }

        if (_options.EmbeddingModel == null && !_options.EnableNpuEmbeddings)
        {
            return null;
        }

        TimeSpan failureCooldown = _options.ProbeCacheTtl ?? NpuSidecarDiscovery.DefaultCacheTtl;
        if (!forceReprobe &&
            Volatile.Read(ref _hasProbeFailure) != 0 &&
            failureCooldown > TimeSpan.Zero &&
            Stopwatch.GetElapsedTime(Volatile.Read(ref _lastProbeFailureTimestamp)) < failureCooldown)
        {
            return null;
        }

        try
        {
            using var linkedCts = CreateLinkedDisposeCts(ct);
            await _embedGate.WaitAsync(linkedCts.Token).ConfigureAwait(false);
            try
            {
                if (Volatile.Read(ref _disposed) != 0)
                {
                    return null;
                }

                existing = Volatile.Read(ref _resolvedSemanticModel);
                if (existing != null)
                {
                    return existing;
                }

                if (_options.EmbeddingModel != null)
                {
                    _ownsResolvedModel = false;
                    Volatile.Write(ref _activeEndpoint, (_options.EmbeddingModel as OpenAiEmbeddingModel)?.BaseUrl);
                    Volatile.Write(ref _activeModelId, (_options.EmbeddingModel as OpenAiEmbeddingModel)?.ModelId ?? _options.EmbeddingModel.GetType().Name);
                    Volatile.Write(ref _hasProbeFailure, 0);
                    Volatile.Write(ref _resolvedSemanticModel, _options.EmbeddingModel);
                    return _options.EmbeddingModel;
                }

                if (!_options.EnableNpuEmbeddings)
                {
                    return null;
                }

                if (!forceReprobe &&
                    Volatile.Read(ref _hasProbeFailure) != 0 &&
                    failureCooldown > TimeSpan.Zero &&
                    Stopwatch.GetElapsedTime(Volatile.Read(ref _lastProbeFailureTimestamp)) < failureCooldown)
                {
                    return null;
                }

                string? rawEndpoint = _options.EmbeddingEndpoint?.Trim();
                bool isAutoEndpoint = string.IsNullOrWhiteSpace(rawEndpoint) ||
                    string.Equals(rawEndpoint, "auto-npu", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(rawEndpoint, "npu", StringComparison.OrdinalIgnoreCase);

                OpenAiEmbeddingModel? openAi = isAutoEndpoint
                    ? await TryDiscoverOpenAiModelAsync(linkedCts.Token, forceReprobe).ConfigureAwait(false)
                    : await TryCreateExplicitOpenAiModelAsync(rawEndpoint!, linkedCts.Token).ConfigureAwait(false);

                if (openAi != null)
                {
                    if (Volatile.Read(ref _disposed) != 0)
                    {
                        openAi.Dispose();
                        return null;
                    }

                    _ownsResolvedModel = true;
                    Volatile.Write(ref _activeEndpoint, openAi.BaseUrl);
                    Volatile.Write(ref _activeModelId, openAi.ModelId);
                    Volatile.Write(ref _hasProbeFailure, 0);
                    Volatile.Write(ref _resolvedSemanticModel, openAi);
                    return openAi;
                }

                Volatile.Write(ref _lastProbeFailureTimestamp, Stopwatch.GetTimestamp());
                Volatile.Write(ref _hasProbeFailure, 1);
                return null;
            }
            finally
            {
                try
                {
                    _embedGate.Release();
                }
                catch (ObjectDisposedException ex)
                {
                    Glacier.Serve.Diagnostics.GlacierDiagnostics.LogDebug($"[SleepMemoryService] Embed gate already disposed: {ex.Message}");
                }
            }
        }
        catch (ObjectDisposedException) when (Volatile.Read(ref _disposed) != 0)
        {
            return null;
        }
    }

    private async Task<OpenAiEmbeddingModel?> TryCreateExplicitOpenAiModelAsync(string rawEndpoint, CancellationToken ct)
    {
        if (!IsValidHttpEndpoint(rawEndpoint, _options.HttpClient))
        {
            return null;
        }

        try
        {
            string? modelId = string.IsNullOrWhiteSpace(_options.EmbeddingModelId)
                ? null
                : _options.EmbeddingModelId.Trim();

            if (modelId == null)
            {
                NpuSidecarEndpoint? sidecar = await NpuSidecarDiscovery.ProbeAsync(
                    httpClient: _options.HttpClient,
                    candidates: [(rawEndpoint, NpuSidecarRuntime.CustomOpenAi)],
                    perEndpointTimeout: _options.ProbeTimeout,
                    cancellationToken: ct).ConfigureAwait(false);

                modelId = SelectEmbeddingModelId(sidecar?.Models) ?? DefaultEmbeddingModelId;
            }

            TimeSpan probeTimeout = _options.ProbeTimeout ?? _options.EmbeddingTimeout ?? TimeSpan.FromSeconds(2);
            int? knownDims = _options.EmbeddingDimensions > 0
                ? _options.EmbeddingDimensions
                : await OpenAiEmbeddingModel.ProbeDimensionsAsync(
                    rawEndpoint,
                    modelId,
                    _options.EmbeddingApiKey,
                    _options.HttpClient,
                    probeTimeout,
                    ct).ConfigureAwait(false);

            if (knownDims is <= 0)
            {
                return null;
            }

            return await OpenAiEmbeddingModel.CreateAutoAsync(
                rawEndpoint,
                modelId,
                _options.EmbeddingApiKey,
                _options.HttpClient,
                _options.EmbeddingTimeout ?? TimeSpan.FromSeconds(5),
                knownDimensions: knownDims,
                cancellationToken: ct).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            throw;
        }
        catch (ObjectDisposedException) when (Volatile.Read(ref _disposed) != 0)
        {
            throw new OperationCanceledException(ct);
        }
        catch (Exception ex) when (IsRecoverableEmbeddingException(ex))
        {
            return null;
        }
    }

    private async Task<OpenAiEmbeddingModel?> TryDiscoverOpenAiModelAsync(CancellationToken ct, bool forceReprobe)
    {
        try
        {
            NpuSidecarEndpoint? sidecar = forceReprobe
                ? await NpuSidecarDiscovery.ProbeAsync(
                    httpClient: _options.HttpClient,
                    perEndpointTimeout: _options.ProbeTimeout,
                    cancellationToken: ct).ConfigureAwait(false)
                : await NpuSidecarDiscovery.ProbeCachedAsync(
                    cacheTtl: _options.ProbeCacheTtl,
                    httpClient: _options.HttpClient,
                    perEndpointTimeout: _options.ProbeTimeout,
                    cancellationToken: ct).ConfigureAwait(false);

            if (sidecar == null)
            {
                return null;
            }

            string modelId = !string.IsNullOrWhiteSpace(_options.EmbeddingModelId)
                ? _options.EmbeddingModelId.Trim()
                : SelectEmbeddingModelId(sidecar.Models) ?? DefaultEmbeddingModelId;

            TimeSpan probeTimeout = _options.ProbeTimeout ?? _options.EmbeddingTimeout ?? TimeSpan.FromSeconds(2);
            int? knownDims = _options.EmbeddingDimensions > 0
                ? _options.EmbeddingDimensions
                : await OpenAiEmbeddingModel.ProbeDimensionsAsync(
                    sidecar.BaseUrl,
                    modelId,
                    _options.EmbeddingApiKey,
                    _options.HttpClient,
                    probeTimeout,
                    ct).ConfigureAwait(false);

            if (knownDims is <= 0)
            {
                return null;
            }

            return await OpenAiEmbeddingModel.CreateAutoAsync(
                sidecar.BaseUrl,
                modelId,
                _options.EmbeddingApiKey,
                _options.HttpClient,
                _options.EmbeddingTimeout ?? TimeSpan.FromSeconds(5),
                knownDimensions: knownDims,
                cancellationToken: ct).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            throw;
        }
        catch (ObjectDisposedException) when (Volatile.Read(ref _disposed) != 0)
        {
            throw new OperationCanceledException(ct);
        }
        catch (Exception ex) when (IsRecoverableEmbeddingException(ex))
        {
            return null;
        }
    }

    private float[] ComputeMatchedFallbackVector(string text, float[] fnv64, int targetDim)
    {
        if (targetDim == Dim)
        {
            return fnv64;
        }

        if (targetDim % 8 == 0)
        {
            FastHashEmbeddingModel hashModel = GetOrCreateFallbackHashModel(targetDim);
            float[] vec = new float[targetDim];
            hashModel.GenerateEmbedding(text.AsSpan(), vec);
            return vec;
        }

        float[] lifted = new float[targetDim];
        LiftFromHopeDim(fnv64, lifted);
        NormalizeInPlace(lifted);
        return lifted;
    }

    private FastHashEmbeddingModel GetOrCreateFallbackHashModel(int dimensions)
    {
        lock (_retiredModels)
        {
            if (_fallbackHashModel != null && _fallbackHashModel.Dimensions == dimensions)
            {
                return _fallbackHashModel;
            }

            _fallbackHashModel?.Dispose();
            _fallbackHashModel = new FastHashEmbeddingModel(dimensions);
            return _fallbackHashModel;
        }
    }

    internal static void FoldToHopeDim(ReadOnlySpan<float> source, Span<float> destination64)
    {
        destination64.Clear();
        if (source.IsEmpty)
        {
            return;
        }

        if (source.Length == Dim)
        {
            source.CopyTo(destination64);
            NormalizeInPlace(destination64);
            return;
        }

        for (int i = 0; i < source.Length; i++)
        {
            int block = i / Dim;
            int slot = i & (Dim - 1);
            float sign = (block == 0 || ((block + slot) & 1) == 0) ? 1.0f : -1.0f;
            destination64[slot] += sign * source[i];
        }

        NormalizeInPlace(destination64);
    }

    internal static void LiftFromHopeDim(ReadOnlySpan<float> source64, Span<float> destination)
    {
        destination.Clear();
        if (destination.IsEmpty)
        {
            return;
        }

        if (destination.Length == Dim)
        {
            source64[..Dim].CopyTo(destination);
            return;
        }

        float scale = destination.Length <= Dim ? 1.0f : MathF.Sqrt((float)Dim / destination.Length);
        for (int i = 0; i < destination.Length; i++)
        {
            int block = i / Dim;
            int slot = i & (Dim - 1);
            float sign = (block == 0 || ((block + slot) & 1) == 0) ? 1.0f : -1.0f;
            destination[i] = source64[slot] * sign * scale;
        }
    }

    private static bool IsValidHttpEndpoint(string endpoint, HttpClient? httpClient)
    {
        if (Uri.TryCreate(endpoint, UriKind.Absolute, out Uri? uri))
        {
            return string.Equals(uri.Scheme, Uri.UriSchemeHttp, StringComparison.OrdinalIgnoreCase) ||
                   string.Equals(uri.Scheme, Uri.UriSchemeHttps, StringComparison.OrdinalIgnoreCase);
        }

        return httpClient?.BaseAddress != null && Uri.TryCreate(endpoint, UriKind.Relative, out _);
    }

    private static string? SelectEmbeddingModelId(IReadOnlyList<string>? models)
    {
        if (models == null || models.Count == 0)
            return null;

        for (int i = 0; i < models.Count; i++)
        {
            string? m = models[i];
            if (!string.IsNullOrWhiteSpace(m) &&
                m.Contains("embed", StringComparison.OrdinalIgnoreCase))
            {
                return m.Trim();
            }
        }

        for (int i = 0; i < models.Count; i++)
        {
            string? m = models[i];
            if (!string.IsNullOrWhiteSpace(m))
                return m.Trim();
        }

        return null;
    }

    private static bool IsRecoverableEmbeddingException(Exception ex) =>
        ex is HttpRequestException
            or TaskCanceledException
            or OperationCanceledException
            or JsonException
            or InvalidOperationException
            or UriFormatException
            or NotSupportedException
            or ArgumentException;

    private bool TryEnterOperation()
    {
        lock (_opDrainSync)
        {
            if (_disposed != 0)
            {
                return false;
            }

            _activeOps++;
            return true;
        }
    }

    private void ExitOperation()
    {
        TaskCompletionSource? drainTcs = null;
        lock (_opDrainSync)
        {
            int remaining = --_activeOps;
            if (remaining == 0 && _disposed != 0)
            {
                drainTcs = _opsDrainedTcs;
            }
        }

        drainTcs?.TrySetResult();
    }

    private bool TryBeginDispose(out Task drainTask)
    {
        bool completeImmediately = false;
        lock (_opDrainSync)
        {
            if (_disposed != 0)
            {
                drainTask = _opsDrainedTcs.Task;
                return false;
            }

            Volatile.Write(ref _disposed, 1);
            if (_activeOps == 0)
            {
                completeImmediately = true;
            }

            drainTask = _opsDrainedTcs.Task;
        }

        if (completeImmediately)
        {
            _opsDrainedTcs.TrySetResult();
        }

        return true;
    }

    private bool IsDisposeCancellationRequested()
    {
        if (Volatile.Read(ref _disposed) != 0)
        {
            return true;
        }

        try
        {
            return _disposeCts.IsCancellationRequested;
        }
        catch (ObjectDisposedException)
        {
            return true;
        }
    }

    private CancellationToken GetSafeDisposeToken()
    {
        try
        {
            return _disposeCts.Token;
        }
        catch (ObjectDisposedException)
        {
            return new CancellationToken(canceled: true);
        }
    }

    private CancellationTokenSource CreateLinkedDisposeCts(CancellationToken ct)
    {
        if (Volatile.Read(ref _disposed) != 0)
        {
            var alreadyCanceled = new CancellationTokenSource();
            alreadyCanceled.Cancel();
            return alreadyCanceled;
        }

        try
        {
            return CancellationTokenSource.CreateLinkedTokenSource(ct, _disposeCts.Token);
        }
        catch (ObjectDisposedException)
        {
            var alreadyCanceled = new CancellationTokenSource();
            alreadyCanceled.Cancel();
            return alreadyCanceled;
        }
    }

    private CancellationTokenSource CreateLinkedTimeoutCts(TimeSpan timeout, CancellationToken ct)
    {
        var cts = CreateLinkedDisposeCts(ct);
        if (timeout != Timeout.InfiniteTimeSpan && !cts.IsCancellationRequested)
        {
            cts.CancelAfter(timeout);
        }
        return cts;
    }

    private static T BridgeSync<T>(Func<Task<T>> asyncAction)
    {
        if (SynchronizationContext.Current is not null || TaskScheduler.Current != TaskScheduler.Default)
        {
            return Task.Run(asyncAction).GetAwaiter().GetResult();
        }

        return asyncAction().GetAwaiter().GetResult();
    }

    private long GetCurrentWallOrSeqUs()
    {
        long wallUs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1000L;
        return Math.Max(wallUs, _nextTimestampSeq + 1);
    }

    private static int EstimateTokens(string? text) =>
        string.IsNullOrEmpty(text) ? 0 : (text.Length + 3) / 4;

    private static string StripThinkBlocks(string text)
    {
        if (string.IsNullOrEmpty(text) || !text.Contains("<think>", StringComparison.Ordinal))
        {
            return text;
        }
        return ThinkBlockRegex().Replace(text, "").Trim();
    }

    internal static (string HaystackBody, string QuestionBlock) SplitHaystackAndQuestion(string content)
    {
        string normalized = content.Replace("\r\n", "\n").Trim();
        if (string.IsNullOrEmpty(normalized))
        {
            return ("", "");
        }

        string[] blocks = normalized.Split("\n\n", StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (blocks.Length >= 2)
        {
            string lastBlock = blocks[^1];
            string body = string.Join("\n\n", blocks, 0, blocks.Length - 1);
            return (body, lastBlock);
        }

        string[] lines = normalized.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (lines.Length > 4)
        {
            string lastLine = lines[^1];
            string body = string.Join("\n", lines, 0, lines.Length - 1);
            return (body, lastLine);
        }

        return ("", normalized);
    }

    private static List<Segment> SegmentText(string text, string prefix)
    {
        var result = new List<Segment>();
        string normalized = text.Replace("\r\n", "\n");
        if (string.IsNullOrWhiteSpace(normalized))
        {
            return result;
        }

        string[] lines = normalized.Split('\n');
        bool looksLikeLog = lines.Length > 8 && lines.Count(l => l.TrimStart().StartsWith('[')) >= 5;
        long order = 0;

        if (looksLikeLog)
        {
            foreach (string rawLine in lines)
            {
                string line = rawLine.Trim();
                if (string.IsNullOrEmpty(line)) continue;
                result.Add(new Segment($"{prefix}_L{order}", line, order++));
            }
            return result;
        }

        string[] blocks = normalized.Split("\n\n", StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        foreach (string block in blocks)
        {
            if (block.Length <= 460)
            {
                result.Add(new Segment($"{prefix}_B{order}", block, order++));
            }
            else
            {
                var cur = new StringBuilder();
                foreach (string rawLine in block.Split('\n'))
                {
                    string line = rawLine.TrimEnd();
                    if (cur.Length > 0 && cur.Length + line.Length + 1 > 400)
                    {
                        result.Add(new Segment($"{prefix}_B{order}", cur.ToString().Trim(), order++));
                        cur.Clear();
                    }
                    if (cur.Length > 0) cur.Append('\n');
                    cur.Append(line);
                }
                if (cur.Length > 0)
                {
                    result.Add(new Segment($"{prefix}_B{order}", cur.ToString().Trim(), order++));
                }
            }
        }

        return result;
    }

    private static (float[] Vec, List<string> Entities) EncodeTextAndEntities(string text)
    {
        float[] vec = new float[Dim];
        var entities = new List<string>(24);
        var seenEntities = new HashSet<string>(StringComparer.Ordinal);
        string? prevNonStop = null;

        int n = text.Length;
        int i = 0;
        while (i < n)
        {
            while (i < n && !IsTokenChar(text[i])) i++;
            int start = i;
            bool hasUpper = false, hasUnderscore = false, hasAlpha = false, hasDigit = false;
            while (i < n && IsTokenChar(text[i]))
            {
                char c = text[i];
                if (c == '_') hasUnderscore = true;
                else if (c >= 'A' && c <= 'Z') { hasUpper = true; hasAlpha = true; }
                else if (c >= 'a' && c <= 'z') hasAlpha = true;
                else if (c >= '0' && c <= '9') hasDigit = true;
                i++;
            }

            int len = i - start;
            if (len < 2) continue;

            string tok = text.Substring(start, len);
            string low = tok.ToLowerInvariant();
            bool isPureNumber = !hasAlpha && hasDigit;
            bool isStructured = !isPureNumber && (hasUnderscore || hasUpper || hasDigit);

            float weight = isPureNumber
                ? 0.40f
                : isStructured
                    ? 2.40f
                    : s_stopWords.Contains(low)
                        ? 0.25f
                        : 1.00f;

            AddHashedFeature(vec, low, weight);

            if (isStructured)
            {
                string upper = tok.ToUpperInvariant();
                if (entities.Count < 24 && seenEntities.Add(upper))
                {
                    entities.Add(upper);
                }

                if (hasUnderscore || hasUpper)
                {
                    foreach (string sub in SplitCompoundSubtokens(tok))
                    {
                        if (sub.Length < 2 || string.Equals(sub, low, StringComparison.OrdinalIgnoreCase)) continue;
                        AddHashedFeature(vec, sub.ToLowerInvariant(), 1.25f);
                        string subUpper = sub.ToUpperInvariant();
                        if (entities.Count < 24 && seenEntities.Add(subUpper))
                        {
                            entities.Add(subUpper);
                        }
                    }
                }
            }

            if (!isPureNumber && !s_stopWords.Contains(low))
            {
                if (prevNonStop != null)
                {
                    AddHashedFeature(vec, prevNonStop + "#" + low, 0.85f);
                }
                prevNonStop = low;
            }
        }

        NormalizeInPlace(vec);
        return (vec, entities);
    }

    private static IEnumerable<string> SplitCompoundSubtokens(string tok)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < tok.Length; i++)
        {
            char c = tok[i];
            if (c == '_')
            {
                if (sb.Length > 0) { yield return sb.ToString(); sb.Clear(); }
                continue;
            }
            if (i > 0 && char.IsUpper(c) && (char.IsLower(tok[i - 1]) || char.IsDigit(tok[i - 1])))
            {
                if (sb.Length > 0) { yield return sb.ToString(); sb.Clear(); }
            }
            sb.Append(c);
        }
        if (sb.Length > 0) yield return sb.ToString();
    }

    private static bool IsTokenChar(char c) =>
        (c >= 'A' && c <= 'Z') ||
        (c >= 'a' && c <= 'z') ||
        (c >= '0' && c <= '9') ||
        c == '_';

    private static ulong ComputeFnv1a64(string text)
    {
        ulong h = 14695981039346656037UL;
        for (int i = 0; i < text.Length; i++)
        {
            h ^= text[i];
            h *= 1099511628211UL;
        }
        return h;
    }

    private static void AddHashedFeature(float[] vec, string token, float weight)
    {
        ulong h = ComputeFnv1a64(token);
        int bucket = (int)(h & (Dim - 1));
        float sign = ((h >> 6) & 1UL) == 0UL ? 1.0f : -1.0f;
        vec[bucket] += sign * weight;
    }

    private static float[] BuildValueVector(float[] kVec, float[] prevK, bool hasPrev)
    {
        float[] v = new float[Dim];
        if (hasPrev)
        {
            for (int i = 0; i < Dim; i++)
            {
                v[i] = 0.78f * kVec[i] + 0.22f * prevK[i];
            }
        }
        else
        {
            for (int i = 0; i < Dim; i++)
            {
                v[i] = 0.80f * kVec[i] + 0.20f * kVec[(i + 1) & (Dim - 1)];
            }
        }
        NormalizeInPlace(v);
        return v;
    }

    private static bool AreAllFinite(ReadOnlySpan<float> vec)
    {
        for (int i = 0; i < vec.Length; i++)
        {
            if (!float.IsFinite(vec[i])) return false;
        }
        return true;
    }

    private static void NormalizeInPlace(Span<float> vec)
    {
        double sumSq = 0.0;
        for (int i = 0; i < vec.Length; i++) sumSq += (double)vec[i] * vec[i];
        float norm = MathF.Sqrt((float)sumSq);
        if (norm > 1e-8f)
        {
            float inv = 1.0f / norm;
            for (int i = 0; i < vec.Length; i++) vec[i] *= inv;
        }
    }

    private void DisposeEmbeddingResources()
    {
        IEmbeddingModel? resolved = Interlocked.Exchange(ref _resolvedSemanticModel, null);
        if (_ownsResolvedModel)
        {
            resolved?.Dispose();
        }

        lock (_retiredModels)
        {
            for (int i = 0; i < _retiredModels.Count; i++)
            {
                _retiredModels[i].Dispose();
            }
            _retiredModels.Clear();

            _fallbackHashModel?.Dispose();
            _fallbackHashModel = null;
        }

        _sleepGate.Dispose();
        _embedGate.Dispose();
        _disposeCts.Dispose();
    }

    public void Dispose()
    {
        if (!TryBeginDispose(out Task drainTask)) return;
        try { _disposeCts.Cancel(); }
        catch (ObjectDisposedException ex)
        {
            Glacier.Serve.Diagnostics.GlacierDiagnostics.LogDebug($"[SleepMemoryService] Dispose CTS already disposed: {ex.Message}");
        }
        drainTask.GetAwaiter().GetResult();
        _daemon.Dispose();
        lock (_bankSync)
        {
            _bank.Dispose();
        }
        lock (_memorySync)
        {
            _hope.Dispose();
            _memEx.Dispose();
        }
        DisposeEmbeddingResources();
    }

    public async ValueTask DisposeAsync()
    {
        if (!TryBeginDispose(out Task drainTask)) return;
        try { _disposeCts.Cancel(); }
        catch (ObjectDisposedException ex)
        {
            Glacier.Serve.Diagnostics.GlacierDiagnostics.LogDebug($"[SleepMemoryService] DisposeAsync CTS already disposed: {ex.Message}");
        }
        await drainTask.ConfigureAwait(false);
        await _daemon.DisposeAsync().ConfigureAwait(false);
        lock (_bankSync)
        {
            _bank.Dispose();
        }
        lock (_memorySync)
        {
            _hope.Dispose();
            _memEx.Dispose();
        }
        DisposeEmbeddingResources();
    }

    /// <summary>
    /// Wraps <see cref="SleepConsolidationTrainer"/> so sleep cycles beyond
    /// <see cref="PreallocatedMoeExpansionBank.MaxExperts"/> recycle and refine the closest
    /// existing MoE expert slot instead of throwing on slot exhaustion.
    /// </summary>
    private sealed unsafe class CapacityAwareSleepPipeline : ISleepConsolidationPipeline
    {
        private readonly PreallocatedMoeExpansionBank _bank;
        private readonly SleepConsolidationTrainer _inner;
        private readonly object _bankSync;
        private int _recycleCursor;

        public CapacityAwareSleepPipeline(
            PreallocatedMoeExpansionBank bank,
            SleepConsolidationTrainer inner,
            object bankSync)
        {
            _bank = bank;
            _inner = inner;
            _bankSync = bankSync;
        }

        public int ParameterExpansion(ReadOnlySpan<float> initialCentroid)
        {
            lock (_bankSync)
            {
                if (_bank.ActiveExpertCount < _bank.MaxExperts)
                {
                    return _inner.ParameterExpansion(initialCentroid);
                }

                int dim = _bank.Dimension;
                if (initialCentroid.Length < dim)
                {
                    throw new ArgumentException(
                        $"Initial centroid length ({initialCentroid.Length}) must be at least Dimension ({dim}).",
                        nameof(initialCentroid));
                }

                double initNormSq = 0.0;
                for (int d = 0; d < dim; d++)
                {
                    initNormSq += (double)initialCentroid[d] * initialCentroid[d];
                }
                bool hasSignal = initNormSq > 1e-12;

                int selectedSlot;
                if (hasSignal)
                {
                    float invNorm = 1.0f / MathF.Sqrt((float)initNormSq);
                    selectedSlot = 0;
                    float bestAffinity = float.NegativeInfinity;
                    for (int e = 0; e < _bank.MaxExperts; e++)
                    {
                        float* cPtr = _bank.GetCentroidPtr(e);
                        float dot = 0.0f;
                        for (int d = 0; d < dim; d++)
                        {
                            dot += cPtr[d] * (initialCentroid[d] * invNorm);
                        }
                        float affinity = MathF.Abs(dot);
                        if (affinity > bestAffinity)
                        {
                            bestAffinity = affinity;
                            selectedSlot = e;
                        }
                    }

                    // Smooth EMA centroid adaptation so the recycled slot tracks recent episodes.
                    float* chosenCentroid = _bank.GetCentroidPtr(selectedSlot);
                    double blendedNormSq = 0.0;
                    for (int d = 0; d < dim; d++)
                    {
                        float blended = 0.75f * chosenCentroid[d] + 0.25f * (initialCentroid[d] * invNorm);
                        chosenCentroid[d] = blended;
                        blendedNormSq += (double)blended * blended;
                    }
                    if (blendedNormSq > 1e-12)
                    {
                        float invBlended = 1.0f / MathF.Sqrt((float)blendedNormSq);
                        for (int d = 0; d < dim; d++)
                        {
                            chosenCentroid[d] *= invBlended;
                        }
                    }
                }
                else
                {
                    selectedSlot = _recycleCursor;
                    _recycleCursor = (_recycleCursor + 1) % _bank.MaxExperts;
                }

                return selectedSlot;
            }
        }

        public Task<ConsolidationResult> ConsolidateKnowledgeAsync(
            int expertSlotIndex,
            DreamCandidate[] dreamBatch,
            int distillationSteps = 100,
            float learningRate = 0.20f,
            float qualityThreshold = 0.75f,
            CancellationToken cancellationToken = default)
        {
            lock (_bankSync)
            {
                return _inner.ConsolidateKnowledgeAsync(
                    expertSlotIndex,
                    dreamBatch,
                    distillationSteps,
                    learningRate,
                    qualityThreshold,
                    cancellationToken);
            }
        }

        public void SynapticPruning()
        {
            _inner.SynapticPruning();
        }

        public void Dispose()
        {
            _inner.Dispose();
        }
    }
}

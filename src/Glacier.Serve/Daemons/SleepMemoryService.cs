namespace Glacier.Serve.Daemons;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Continuum;
using Glacier.MemEx;
using Glacier.MemEx.Models;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Serialization;
using Glacier.Tune.Expansion;
using Glacier.Tune.Sleep;

/// <summary>
/// Online wake-sleep episodic consolidation service backed by Pillar 16 HOPE Continuum,
/// MemEx ACT-R/HippoRAG/Knapsack retrieval, Preallocated MoE Expansion, and AutonomousSleepDaemon.
/// </summary>
public sealed partial class SleepMemoryService : IDisposable, IAsyncDisposable
{
    private const int Dim = 64;
    private const int PruneKeepLimit = 512;

    private sealed record StoredChunk(
        string Id,
        string Content,
        long TimestampUs,
        float[] KeyVec,
        float[] ValueVec,
        float BaseImportance);

    private sealed record Segment(string Id, string Content, long OffsetOrder);

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
    private readonly SemaphoreSlim _sleepGate = new(1, 1);

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

    public int TokenBudget { get; }

    public int MinDreamsToTriggerSleep { get; }

    public AutonomousSleepDaemon Daemon => _daemon;

    public PreallocatedMoeExpansionBank ExpansionBank => _bank;

    public SleepMemoryService(
        int tokenBudget = 400,
        TimeSpan? idleThreshold = null,
        TimeSpan? pollingInterval = null,
        int minDreamsToTriggerSleep = 1)
    {
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(tokenBudget, 0);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(minDreamsToTriggerSleep, 0);
        TokenBudget = tokenBudget;
        MinDreamsToTriggerSleep = minDreamsToTriggerSleep;

        _hope = HopeContinuumHierarchy.CreateDefault4Tier(dimension: Dim, surpriseThreshold: 0.45f);
        _memEx = new MemExEngine(new MemExOptions
        {
            MaxPromptTokens = tokenBudget,
            DecayParameterD = 0.5,
            TimeScaleTauSeconds = 1.0,
            WeightDecay = 0.20f,
            WeightVector = 1.85f,
            WeightGraph = 1.75f,
            WeightImportance = 1.15f,
            CoarseCandidateLimit = 512,
        });
        _bank = new PreallocatedMoeExpansionBank(dimension: Dim, rank: 16, maxExperts: 4);
        _trainer = new SleepConsolidationTrainer(_bank, synapticPruningCallback: OnSynapticPrune);
        var pipeline = new CapacityAwareSleepPipeline(_bank, _trainer, _bankSync);
        _daemon = new AutonomousSleepDaemon(pipeline, new SleepDaemonOptions(
            IdleThreshold: idleThreshold ?? TimeSpan.FromSeconds(15),
            PollingInterval: pollingInterval ?? TimeSpan.FromMilliseconds(200),
            DistillationSteps: 60,
            LearningRate: 0.20f,
            QualityThreshold: 0.50f,
            MinDreamsToTriggerSleep: minDreamsToTriggerSleep));
        _daemon.Start();
    }

    public SleepMemoryStatusResponse GetStatus()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        lock (_memorySync)
        {
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
                LastElapsedMs = _lastElapsedMs
            };
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
                newlyIngested += IngestText(cleanHistory, nonSystemMessages[i].Role ?? "user");
            }
        }

        if (!shouldConsolidate)
        {
            return new List<(string Role, string Content)>(messages);
        }

        var (haystackBody, questionBlock) = SplitHaystackAndQuestion(lastUserContent);
        if (!string.IsNullOrWhiteSpace(haystackBody))
        {
            newlyIngested += IngestText(haystackBody, "ctx");
        }

        if (newlyIngested >= 8)
        {
            await TriggerSleepAsync(ct).ConfigureAwait(false);
        }

        string queryText = string.IsNullOrWhiteSpace(questionBlock) ? lastUserContent : questionBlock;
        var (selectedContent, selectedCount, packedTokens) = QueryMemory(queryText);

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
        if (string.IsNullOrEmpty(prompt) || EstimateTokens(prompt) <= TokenBudget * 3)
        {
            _daemon.RecordActivity();
            return prompt;
        }

        var list = await ConsolidateMessagesAsync([("user", prompt)], ct).ConfigureAwait(false);
        return list.Count > 0 ? list[^1].Content : prompt;
    }

    /// <summary>
    /// Records a completed inference turn into episodic memory after stripping <c>&lt;think&gt;</c> traces,
    /// queueing high-surprise segments for offline dream consolidation.
    /// </summary>
    public void RecordTurnCompletion(string? userQuery, string? assistantResponse)
    {
        if (Volatile.Read(ref _disposed) != 0) return;

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

        _daemon.RecordActivity();
    }

    /// <summary>
    /// Enqueues top candidate traces and runs an immediate offline sleep consolidation cycle.
    /// </summary>
    public async Task<ConsolidationResult?> TriggerSleepAsync(CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        await _sleepGate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            float[] centroidCopy = new float[Dim];
            int queued = 0;

            lock (_memorySync)
            {
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

            if (queued == 0 && _daemon.PendingDreamCount == 0)
            {
                return null;
            }

            return await _daemon.TriggerImmediateSleepAsync(centroidCopy, ct).ConfigureAwait(false);
        }
        finally
        {
            _sleepGate.Release();
        }
    }

    private void OnSynapticPrune()
    {
        lock (_memorySync)
        {
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
        var segments = SegmentText(text, prefix);
        if (segments.Count == 0)
        {
            return 0;
        }

        int added = 0;
        lock (_memorySync)
        {
            long baseNowUs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1000L;
            foreach (var seg in segments)
            {
                ulong hash = ComputeFnv1a64(seg.Content);
                var dedupeKey = (hash, seg.Content.Length);
                if (!_ingestedChunkKeys.Add(dedupeKey))
                {
                    continue;
                }

                var (kVec, entities) = EncodeTextAndEntities(seg.Content);
                float[] vVec = BuildValueVector(kVec, _prevK, _rawChunks > 0);
                Array.Copy(kVec, _prevK, Dim);

                float surprise = _hope.Step(kVec, vVec);
                for (int d = 0; d < Dim; d++)
                {
                    _centroidSum[d] += kVec[d];
                }

                float structuredBonus = MathF.Min(1.2f, entities.Count * 0.14f);
                float importance = Math.Clamp(0.4f + 3.0f * surprise + structuredBonus, 0.2f, 5.0f);
                long seq = ++_nextTimestampSeq;
                long tsUs = baseNowUs + seq;
                int estTokens = Math.Max(8, (seg.Content.Length + 3) / 4);
                string traceId = $"{seg.Id}_{seq}_{hash:x8}";

                var trace = new EpisodicTrace
                {
                    Id = traceId,
                    Content = seg.Content,
                    TimestampUs = tsUs,
                    Breadcrumbs = [],
                    Embedding = kVec,
                    Entities = entities,
                    TokenCount = estTokens,
                    BaseImportance = importance,
                };

                _memEx.StoreTrace(trace);
                var stored = new StoredChunk(traceId, seg.Content, tsUs, kVec, vVec, importance);
                _records.Add(stored);
                _recordById[traceId] = stored;
                _rawChunks++;
                added++;

                if (surprise >= 0.45f)
                {
                    string preview = seg.Content.Length <= 220 ? seg.Content : seg.Content[..220];
                    _daemon.QueueDream(new DreamCandidate(
                        PromptPrefix: preview,
                        GeneratedDream: preview,
                        QueryVector: kVec,
                        TargetTeacherOutput: vVec,
                        SemanticSimilarity: 0f,
                        LevenshteinRatio: 0f,
                        CompositeScore: 0f));
                }
            }
        }

        return added;
    }

    private (string FormattedContent, int SelectedCount, int PackedTokens) QueryMemory(string query)
    {
        var (qVec, qEntities) = EncodeTextAndEntities(query);
        float[] hopeOut = new float[Dim];
        float[] moeDelta = new float[Dim];
        float[] rankScratch = new float[_bank.Rank];

        lock (_bankSync)
        {
            _bank.ForwardDelta(qVec, moeDelta, rankScratch);
        }

        MemoryQueryResult queryResult;
        lock (_memorySync)
        {
            _hope.Forward(qVec, hopeOut);

            float[] qRecall = new float[Dim];
            for (int d = 0; d < Dim; d++)
            {
                qRecall[d] = (0.74f * qVec[d]) + (0.16f * hopeOut[d]) + (0.10f * moeDelta[d]);
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

    private static void NormalizeInPlace(float[] vec)
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

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _daemon.Dispose();
        lock (_bankSync)
        {
            _bank.Dispose();
        }
        lock (_memorySync)
        {
            _hope.Dispose();
        }
        _sleepGate.Dispose();
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        await _daemon.DisposeAsync().ConfigureAwait(false);
        lock (_bankSync)
        {
            _bank.Dispose();
        }
        lock (_memorySync)
        {
            _hope.Dispose();
        }
        _sleepGate.Dispose();
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

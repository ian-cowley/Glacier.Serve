namespace Glacier.Serve.Daemons;

using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Tune.Sleep;

/// <summary>
/// Operational state of the biomimetic Sleep/Wake lifecycle.
/// </summary>
public enum SleepLifecycleState : byte
{
    /// <summary>
    /// Model is awake, serving live incoming requests and accumulating fast-memory context.
    /// </summary>
    Awake = 0,

    /// <summary>
    /// Server is idle; daemon is preparing offline parameters and synthetic rollouts.
    /// </summary>
    EnteringSleep = 1,

    /// <summary>
    /// Offline sleep consolidation in progress: Parameter Expansion, Dreaming &amp; GKD distillation.
    /// </summary>
    Sleeping = 2,

    /// <summary>
    /// Finalizing synaptic pruning and restoring online serving routing.
    /// </summary>
    Waking = 3
}

/// <summary>
/// Configuration for <see cref="AutonomousSleepDaemon"/>.
/// </summary>
public sealed record SleepDaemonOptions(
    TimeSpan IdleThreshold,
    TimeSpan PollingInterval,
    int DistillationSteps = 100,
    float LearningRate = 0.20f,
    float QualityThreshold = 0.75f,
    int MinDreamsToTriggerSleep = 1
);

/// <summary>
/// Autonomous background service that detects server request quiescence (idle periods),
/// transitioning Glacier from the online Wake state into the biological Sleep consolidation phase.
/// Coordinates parameter expansion, synthetic dream filtering, Generalized Knowledge Distillation (GKD),
/// and synaptic pruning without human intervention.
/// </summary>
public sealed class AutonomousSleepDaemon : IDisposable, IAsyncDisposable
{
    private readonly ISleepConsolidationPipeline _pipeline;
    private readonly SleepDaemonOptions _options;
    private readonly ConcurrentQueue<DreamCandidate> _pendingDreams = new();
    private readonly object _stateLock = new();

    private SleepLifecycleState _state = SleepLifecycleState.Awake;
    private long _lastActivityTimestamp;
    private int _completedSleepCycles;
    private int _totalDreamsConsolidated;
    private CancellationTokenSource? _cts;
    private Task? _daemonLoopTask;
    private bool _disposed;

    public SleepLifecycleState CurrentState
    {
        get
        {
            lock (_stateLock) return _state;
        }
    }

    public int CompletedSleepCycles => Volatile.Read(ref _completedSleepCycles);
    public int TotalDreamsConsolidated => Volatile.Read(ref _totalDreamsConsolidated);
    public int PendingDreamCount => _pendingDreams.Count;

    public event Action<SleepLifecycleState>? StateChanged;
    public event Action<ConsolidationResult>? ConsolidationCompleted;

    public AutonomousSleepDaemon(
        ISleepConsolidationPipeline pipeline,
        SleepDaemonOptions? options = null)
    {
        _pipeline = pipeline ?? throw new ArgumentNullException(nameof(pipeline));
        _options = options ?? new SleepDaemonOptions(
            IdleThreshold: TimeSpan.FromSeconds(5),
            PollingInterval: TimeSpan.FromMilliseconds(200));

        _lastActivityTimestamp = Stopwatch.GetTimestamp();
    }

    /// <summary>
    /// Signals the daemon that serving activity occurred, resetting the idle countdown.
    /// </summary>
    public void RecordActivity()
    {
        Volatile.Write(ref _lastActivityTimestamp, Stopwatch.GetTimestamp());
    }

    /// <summary>
    /// Queues an on-policy synthetic dream candidate for offline consolidation.
    /// </summary>
    public void QueueDream(DreamCandidate candidate)
    {
        ArgumentNullException.ThrowIfNull(candidate);
        _pendingDreams.Enqueue(candidate);
    }

    /// <summary>
    /// Starts the background monitoring daemon loop.
    /// </summary>
    public void Start()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_daemonLoopTask != null) return;

        _cts = new CancellationTokenSource();
        _daemonLoopTask = Task.Run(() => MonitorLoopAsync(_cts.Token));
    }

    /// <summary>
    /// Triggers an immediate offline sleep consolidation cycle regardless of idle elapsed duration.
    /// </summary>
    public async Task<ConsolidationResult?> TriggerImmediateSleepAsync(
        ReadOnlyMemory<float> initialCentroid,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_stateLock)
        {
            if (_state == SleepLifecycleState.Sleeping || _state == SleepLifecycleState.EnteringSleep)
            {
                return null;
            }
            SetState(SleepLifecycleState.EnteringSleep);
        }

        try
        {
            SetState(SleepLifecycleState.Sleeping);

            // Dequeue accumulated dream candidates
            var batch = new DreamCandidate[_pendingDreams.Count];
            int count = 0;
            while (_pendingDreams.TryDequeue(out var dream) && count < batch.Length)
            {
                batch[count++] = dream;
            }

            if (count == 0)
            {
                SetState(SleepLifecycleState.Awake);
                return null;
            }

            if (count < batch.Length)
            {
                Array.Resize(ref batch, count);
            }

            // Phase 1: Parameter Expansion
            int expertSlot = _pipeline.ParameterExpansion(initialCentroid.Span);

            // Phase 2: Knowledge Seeding & GKD Distillation + Phase 3: Synaptic Pruning
            ConsolidationResult result = await _pipeline.ConsolidateKnowledgeAsync(
                expertSlot,
                batch,
                _options.DistillationSteps,
                _options.LearningRate,
                _options.QualityThreshold,
                cancellationToken);

            SetState(SleepLifecycleState.Waking);
            Interlocked.Increment(ref _completedSleepCycles);
            Interlocked.Add(ref _totalDreamsConsolidated, result.TotalDreamsAccepted);

            ConsolidationCompleted?.Invoke(result);
            return result;
        }
        finally
        {
            SetState(SleepLifecycleState.Awake);
            RecordActivity();
        }
    }

    private async Task MonitorLoopAsync(CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                await Task.Delay(_options.PollingInterval, cancellationToken);

                long lastTimestamp = Volatile.Read(ref _lastActivityTimestamp);
                TimeSpan elapsedIdle = Stopwatch.GetElapsedTime(lastTimestamp);

                if (elapsedIdle >= _options.IdleThreshold && _pendingDreams.Count >= _options.MinDreamsToTriggerSleep)
                {
                    if (_pendingDreams.TryPeek(out var firstDream) && firstDream.QueryVector != null)
                    {
                        await TriggerImmediateSleepAsync(firstDream.QueryVector, cancellationToken);
                    }
                }
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                break;
            }
            catch (Exception)
            {
                SetState(SleepLifecycleState.Awake);
            }
        }
    }

    private void SetState(SleepLifecycleState newState)
    {
        lock (_stateLock)
        {
            if (_state != newState)
            {
                _state = newState;
                StateChanged?.Invoke(newState);
            }
        }
    }

    public void Stop()
    {
        if (_cts != null)
        {
            _cts.Cancel();
            try { _daemonLoopTask?.Wait(TimeSpan.FromSeconds(2)); } catch { }
            _cts.Dispose();
            _cts = null;
            _daemonLoopTask = null;
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            Stop();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    public async ValueTask DisposeAsync()
    {
        if (!_disposed)
        {
            if (_cts != null)
            {
                _cts.Cancel();
                if (_daemonLoopTask != null)
                {
                    try { await _daemonLoopTask; } catch { }
                }
                _cts.Dispose();
                _cts = null;
                _daemonLoopTask = null;
            }
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}

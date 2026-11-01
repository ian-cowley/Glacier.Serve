using System;
using System.Collections.Generic;

namespace Glacier.Serve.Inference.PagedAttention;

/// <summary>
/// Hardware-agnostic contract for PagedAttention Transformer execution runners
/// (VRAM-resident CUDA PTX runner or SIMD CPU fallback runner).
/// </summary>
public interface IPagedModelRunner : IDisposable
{
    bool IsGpuAccelerated { get; }
    string BackendName { get; }
    int VocabSize { get; }
    int MaxSeqLen { get; }
    int MaxBatchSize { get; }

    /// <summary>
    /// Evaluates a single token step at <paramref name="pos"/> for one sequence.
    /// Returns false if <paramref name="pool"/> runs out of physical blocks.
    /// </summary>
    bool ForwardToken(
        int token,
        int pos,
        BlockTable blockTable,
        PagedBlockPool pool,
        Span<float> logits,
        bool computeLogits = true);

    /// <summary>
    /// Evaluates a causal contiguous prompt chunk starting at <paramref name="startPos"/> for a single sequence.
    /// Writes only the final token's logits into <paramref name="logits"/> when <paramref name="computeLogits"/> is true.
    /// Returns false if <paramref name="pool"/> runs out of physical blocks.
    /// </summary>
    bool ForwardPrefillChunk(
        ReadOnlySpan<int> tokens,
        int startPos,
        BlockTable blockTable,
        PagedBlockPool pool,
        Span<float> logits,
        bool computeLogits = true);

    /// <summary>
    /// Evaluates one decode token per active sequence across a batch of concurrent sequences.
    /// </summary>
    bool ForwardBatch(
        ReadOnlySpan<int> tokens,
        ReadOnlySpan<int> positions,
        IReadOnlyList<BlockTable> blockTables,
        PagedBlockPool pool,
        Span<float> logitsBatch,
        Span<bool> successFlags,
        bool computeLogits = true);
}

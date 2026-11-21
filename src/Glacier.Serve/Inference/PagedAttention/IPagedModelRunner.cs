using System;
using System.Collections.Generic;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;

namespace Glacier.Serve.Inference.PagedAttention;

internal interface IPagedHiddenStateContext
{
    bool SupportsHiddenStateCapture(LoraAdapterWeights? lora);
    void BindHiddenStateHistory(float[]? history);
    void BindBatchHiddenStateHistories(IReadOnlyList<float[]?>? histories);
}

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
    KvCachePrecision KvPrecision => KvCachePrecision.Auto;

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
    /// Evaluates a single token step with an optional runtime LoRA adapter.
    /// </summary>
    bool ForwardToken(
        int token,
        int pos,
        BlockTable blockTable,
        PagedBlockPool pool,
        Span<float> logits,
        bool computeLogits,
        LoraAdapterWeights? lora) =>
        lora == null
            ? ForwardToken(token, pos, blockTable, pool, logits, computeLogits)
            : throw new NotSupportedException($"Runner '{BackendName}' does not support runtime LoRA adapters.");

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
    /// Evaluates a causal contiguous prompt chunk with an optional runtime LoRA adapter.
    /// </summary>
    bool ForwardPrefillChunk(
        ReadOnlySpan<int> tokens,
        int startPos,
        BlockTable blockTable,
        PagedBlockPool pool,
        Span<float> logits,
        bool computeLogits,
        LoraAdapterWeights? lora) =>
        lora == null
            ? ForwardPrefillChunk(tokens, startPos, blockTable, pool, logits, computeLogits)
            : throw new NotSupportedException($"Runner '{BackendName}' does not support runtime LoRA adapters.");

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

    /// <summary>
    /// Evaluates one decode token per active sequence across a heterogeneous batch with optional per-sequence LoRA adapters.
    /// </summary>
    bool ForwardBatch(
        ReadOnlySpan<int> tokens,
        ReadOnlySpan<int> positions,
        IReadOnlyList<BlockTable> blockTables,
        PagedBlockPool pool,
        Span<float> logitsBatch,
        Span<bool> successFlags,
        bool computeLogits,
        IReadOnlyList<LoraAdapterWeights?>? loras)
    {
        if (loras != null)
        {
            for (int i = 0; i < loras.Count; i++)
            {
                if (loras[i] != null)
                    throw new NotSupportedException($"Runner '{BackendName}' does not support runtime LoRA adapters.");
            }
        }
        return ForwardBatch(tokens, positions, blockTables, pool, logitsBatch, successFlags, computeLogits);
    }

    /// <summary>
    /// Verifies a speculative draft chunk starting at <paramref name="startPos"/> and writes logits
    /// for all <paramref name="tokens"/> positions into <paramref name="logitsOut"/>.
    /// Rolls back <paramref name="blockTable"/> to its initial token count if pool allocation fails.
    /// </summary>
    bool VerifyDraftChunk(
        ReadOnlySpan<int> tokens,
        int startPos,
        BlockTable blockTable,
        PagedBlockPool pool,
        Span<float> logitsOut) =>
        VerifyDraftChunk(tokens, startPos, blockTable, pool, logitsOut, lora: null);

    /// <summary>
    /// Verifies a speculative draft chunk with an optional runtime LoRA adapter and writes logits
    /// for all <paramref name="tokens"/> positions into <paramref name="logitsOut"/>.
    /// Rolls back <paramref name="blockTable"/> to its initial token count if pool allocation fails.
    /// </summary>
    bool VerifyDraftChunk(
        ReadOnlySpan<int> tokens,
        int startPos,
        BlockTable blockTable,
        PagedBlockPool pool,
        Span<float> logitsOut,
        LoraAdapterWeights? lora)
    {
        if (tokens.IsEmpty)
            return true;

        int vocab = VocabSize;
        if (logitsOut.Length < tokens.Length * vocab)
            throw new ArgumentException($"logitsOut too small ({logitsOut.Length} < {tokens.Length * vocab}).", nameof(logitsOut));

        int initialTokens = blockTable.TokenCount;
        for (int t = 0; t < tokens.Length; t++)
        {
            if (!ForwardToken(tokens[t], startPos + t, blockTable, pool, logitsOut.Slice(t * vocab, vocab), computeLogits: true, lora))
            {
                blockTable.TruncateTo(initialTokens);
                return false;
            }
        }
        return true;
    }
}

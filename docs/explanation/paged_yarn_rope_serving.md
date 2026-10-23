# PagedAttention and YaRN RoPE Serving Mechanics Across 1M Context

**Document Type:** Diátaxis Explanation  
**Subsystem:** `Glacier.Serve`  
**Related Components:** `PagedAttentionKernel.cs`, `PagedTransformerRunner.cs`, `BlockTable.cs`, `PagedBlockPool.cs`  
**Status:** Production Normative Specification  

---

## 1. Introduction: The Memory and Positional Challenge of 1M Context

Serving LLMs at ultra-long context windows (128,000 to 1,000,000 tokens) presents two severe engineering challenges:

1. **Non-Contiguous Memory Fragmentation:** Standard attention engines require key and value tensors to reside in contiguous physical memory buffers $[B, L, H, D]$. Because request lengths are unpredictable, servers must pre-allocate contiguous memory for the maximum possible sequence length. For 1M tokens, this requires tens of gigabytes per request upfront, causing massive internal memory fragmentation (allocated memory that is never used) and external memory fragmentation (disjoint holes preventing new requests from starting).
2. **Rotational Position Discontinuity:** Rotary Position Embedding (RoPE) requires knowing the exact logical sequence index $t \in [0, 10^6)$ of every token to compute its rotational phase angle $\theta_i \cdot t$. When key-value pairs are stored in arbitrary, non-contiguous memory pages allocated over time, how does the attention kernel reconstruct continuous positional awareness without expensive tensor re-assembly or memory copies?

`Glacier.Serve` resolves both problems by coupling a **virtual memory paging architecture (PagedAttention)** with **piecewise YaRN RoPE precomputation and position indexing**.

---

## 2. PagedAttention Memory Architecture

Inspired by operating system virtual memory paging, PagedAttention partitions the physical KV cache into fixed-size contiguous blocks of tokens.

```
Logical Sequence View (Token Index Space):
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│ Tok 0 │ Tok 1 │  ...  │Tok 15 │Tok 16 │Tok 17 │  ...  │Tok 31 │Tok 32 │  ...  │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
  ▲                               ▲                               ▲
  │ Block 0 (Tokens 0..15)        │ Block 1 (Tokens 16..31)       │ Block 2 (Tokens 32..47)
  │                               │                               │
  ▼                               ▼                               ▼
Virtual Block Table (Per Sequence):
┌───────────────────────────────┬───────────────────────────────┬───────────────────────────────┐
│ Logical Block Index 0         │ Logical Block Index 1         │ Logical Block Index 2         │
│ Physical Block ID: #104       │ Physical Block ID: #12        │ Physical Block ID: #879       │
└───────────────────────────────┴───────────────────────────────┴───────────────────────────────┘
  │                               │                               │
  ▼                               ▼                               ▼
Physical PagedBlockPool (Scattered Physical Memory Pages):
┌──────────────┐                ┌──────────────┐                ┌──────────────┐
│ Physical #12 │                │ Physical #104│                │ Physical #879│
│ [Tok 16..31] │                │ [Tok 0..15]  │                │ [Tok 32..47] │
└──────────────┘                └──────────────┘                └──────────────┘
```

### 2.1 The Physical Pool: `PagedBlockPool`
`PagedBlockPool` manages an unmanaged arena in device or host memory. It allocates blocks of fixed size $B_s = 16$ tokens:
- Each block stores Key and Value tensors for all layers:
  $$\text{BlockSizeInBytes} = N_{\text{layers}} \times 2 \times B_s \times N_{\text{headsKV}} \times d_{\text{head}} \times \text{sizeof}(\text{float})$$
- Blocks are allocated dynamically on-demand from a lock-free free-list.
- When a sequence completes, its allocated blocks are instantly returned to the free list in $O(K)$ time (where $K$ is the number of blocks).

### 2.2 Virtual Address Translation: `BlockTable`
Every active sequence maintains a `BlockTable` containing an array of physical block identifiers `_physicalBlockIds`:
- **Logical Token to Physical Location:** For any logical sequence index $t$:
  $$\text{blockIndex} = \lfloor t / B_s \rfloor$$
  $$\text{offsetInBlock} = t \pmod{B_s}$$
  $$\text{physicalBlockId} = \text{BlockTable}[\text{blockIndex}]$$
- In C# (`BlockTable.cs`, lines 51–59):
  ```csharp
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public void GetTokenLocation(int tokenIndex, out int physicalBlockId, out int offsetInBlock)
  {
      int blockIndex = tokenIndex / BlockSize;
      physicalBlockId = _physicalBlockIds[blockIndex];
      offsetInBlock = tokenIndex % BlockSize;
  }
  ```
- **Zero Copy Slicing:** A sequence can append tokens one by one without reallocating buffers or copying previous KV pairs. Physical memory is allocated strictly as needed, achieving $>96\%$ memory utilization.

---

## 3. Position Indexing Across Non-Contiguous Pages

A common misconception is that storing keys in scattered memory blocks disrupts RoPE phase alignment. In reality, RoPE is applied **at the moment the token is generated**, before it is stored in physical memory.

### 3.1 Pre-Storage RoPE Application

When token $t$ is processed in layer $l$:
1. The transformer computes linear projections $Q_t = x_t W_Q$, $K_t = x_t W_K$, $V_t = x_t W_V$.
2. The logical token position $pos = t$ is passed to `QuantKernels.RoPE`:
   ```csharp
   QuantKernels.RoPE(
       _qBatch + b * qDim, 
       _kBatch + b * kvDim, 
       _nHeads, _nHeadsKv, _headDim, 
       positions[b],            // Logical position t ∈ [0, 1M)
       _weights.RopeFreqBase, 
       invFreqTable: _yarnInvFreq, 
       mscale: _yarnMscale);
   ```
3. `QuantKernels.RoPE` rotates $Q_t$ and $K_t$ using the exact YaRN frequency for logical position $t$.
4. **Physical Write:** The rotated key $K_t$ and unrotated value $V_t$ are written to physical memory at the address determined by `BlockTable`:
   ```csharp
   pool.Store(pBlockIds[b], l, offsetsInBlock[b], _kBatch + b * kvDim, _vBatch + b * kvDim);
   ```

Because $K_t$ already carries its correct rotational embedding for logical position $t$, the physical location where it is stored does **not** affect its relative phase relationship to future queries!

---

## 4. Paged Attention Reduction Over Scattered Blocks

During the attention phase, query head $q_h$ at current position $t$ must attend to all previous keys $\{K_0, K_1, \dots, K_t\}$ scattered across non-contiguous physical blocks.

### 4.1 The Paged Attention Kernel Execution

In `PagedAttentionKernel.ComputeAttention`:
1. The kernel iterates through all logical tokens $j \in [0, t]$.
2. For each token $j$, it fetches its physical location from `BlockTable`:
   ```csharp
   blockTable.GetTokenLocation(j, out int blockId, out int offset);
   float* kPtr = pool.GetKeyPointer(blockId, stageLayer, offset, hKv);
   ```
3. It computes the dot product $q_h \cdot k_j$ and scales by the invariant attention scale $\alpha = 1.0 / \sqrt{d_{\text{head}}}$:
   ```csharp
   float score = QuantKernels.VecDotF32(qHead, kPtr, headDim) * attnScale;
   ```
4. Softmax is evaluated across all $t+1$ scores.
5. The attention output vector is accumulated by gathering values directly from the physical blocks:
   ```csharp
   float* vPtr = pool.GetValuePointer(blockId, stageLayer, offset, hKv);
   QuantKernels.VecAddWeighted(vPtr, outHead, prob, vHeadDim);
   ```

### 4.2 Why YaRN $(mscale)^2$ is Maintained in Paged Serving

Because each stored key $K_j$ was pre-scaled by $mscale$ during its generation step, and the current query $Q_t$ is scaled by $mscale$ during the current step:

$$Q_t \cdot K_j = (mscale \cdot \tilde{Q}_t) \cdot (mscale \cdot \tilde{K}_j) = (mscale)^2 \cdot (\tilde{Q}_t \cdot \tilde{K}_j)$$

The paged attention kernel multiplies this dot product by $\alpha = 1 / \sqrt{d_{\text{head}}}$:

$$\text{Logit}_j = \frac{1}{\sqrt{d_{\text{head}}}} (Q_t \cdot K_j) = \frac{(mscale)^2}{\sqrt{d_{\text{head}}}} (\tilde{Q}_t \cdot \tilde{K}_j)$$

The temperature sharpening factor $(mscale)^2$ is identically preserved across all tokens $j \in [0, t]$, whether they are stored in block #0 or block #100,000!

---

## 5. Scaling to 1M Context: Memory Economics

Consider serving a 1,000,000 token context request on `Glacier.Serve`:

```
Parameters:
- Model: 7B Parameters, 28 Layers, 8 KV Heads, HeadDim = 128
- Block Size: 16 tokens
- Number of Blocks for 1M tokens: 1,000,000 / 16 = 62,500 blocks
- Memory per block (28 layers, K+V, FP16):
  28 layers * 2 * 16 tokens * 8 heads * 128 dims * 2 bytes = 1,835,008 bytes (~1.75 MB)
- Total Memory for 1M Context:
  62,500 * 1.75 MB = ~109.3 GB
```

In a traditional serving system with static pre-allocation:
- Allocating 109 GB up-front would crash if the user only asked a 20-token question.
- Multiple concurrent requests would quickly cause out-of-memory errors.

In `Glacier.Serve` with PagedAttention:
- A sequence starting with a 1,000 token prompt consumes only **63 blocks ($\approx 110\text{ MB}$)**.
- Blocks are allocated incrementally every 16 tokens during generation.
- If the sequence stops at token 2,500, only **157 blocks ($\approx 275\text{ MB}$)** are ever allocated.
- When generation finishes, all 157 blocks are instantly recycled in microseconds.

---

## 6. Summary

The combination of PagedAttention and YaRN RoPE in `Glacier.Serve` provides:
1. **Zero Memory Waste:** Virtual paging eliminates internal and external memory fragmentation.
2. **Pristine Phase Coherence:** YaRN frequencies maintain sharp, unambiguous token discrimination across 1,000,000 tokens.
3. **Exact Invariant Preservation:** The attention scaling invariant $\alpha = 1.0 / \sqrt{d_{\text{head}}}$ maintains numerical stability and prevents degenerative repetition loops across arbitrary page configurations.

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Quant;
using Glacier.Serve.Inference.PagedAttention;

namespace Glacier.Serve.Tests;

internal static unsafe class M4bTestHelpers
{
    public const int EosId = 0;
    public const int BosId = 1;
    public const int ThinkId = 2;
    public const int EndThinkId = 3;

    public static string[] CreateDefaultVocab(int vocabSize = 96)
    {
        if (vocabSize < 64)
            throw new ArgumentOutOfRangeException(nameof(vocabSize));

        var list = new List<string>(vocabSize)
        {
            "<|im_end|>",   // 0 (EOS)
            "<|im_start|>", // 1 (BOS)
            "<think>",      // 2
            "</think>",     // 3
            "\u010A",       // 4 ('\n' in GPT-2 byte-level BPE)
            "\u0120",       // 5 (' ' in GPT-2 byte-level BPE)
            "!",            // 6
            ":",            // 7
            ",",            // 8
            "."             // 9
        };

        for (char c = 'a'; c <= 'z' && list.Count < vocabSize; c++)
            list.Add(c.ToString());
        for (char c = '0'; c <= '9' && list.Count < vocabSize; c++)
            list.Add(c.ToString());
        for (char c = 'A'; c <= 'Z' && list.Count < vocabSize; c++)
            list.Add(c.ToString());
        while (list.Count < vocabSize)
            list.Add($"<tok{list.Count}>");

        return list.ToArray();
    }

    public static string CreateSyntheticDenseGguf(
        string directory,
        string fileName = "m4b_test_model.gguf",
        int dim = 64,
        int ffnDim = 128,
        int nHeads = 4,
        int nHeadsKv = 2,
        int headDim = 16,
        int layers = 2,
        int vocabSize = 96,
        int maxSeqLen = 256,
        bool includeEmbeddedMtp = false,
        int seed = 42,
        bool useQ8Weights = false,
        int dominantTokenId = -1)
    {
        Directory.CreateDirectory(directory);
        string path = Path.Combine(directory, fileName);

        var rng = new Random(seed);
        string[] vocab = CreateDefaultVocab(vocabSize);
        int qDim = nHeads * headDim;
        int kvDim = nHeadsKv * headDim;
        GgufType matType = useQ8Weights ? GgufType.Q8_0 : GgufType.F32;

        int totalBlocksInFile = includeEmbeddedMtp ? layers + 1 : layers;

        var metadata = new List<(string Key, GgufValueType Type, object Value)>
        {
            ("general.architecture", GgufValueType.String, "qwen2"),
            ("general.alignment", GgufValueType.Uint32, 32u),
            ("qwen2.block_count", GgufValueType.Uint32, (uint)totalBlocksInFile),
            ("qwen2.context_length", GgufValueType.Uint32, (uint)maxSeqLen),
            ("qwen2.embedding_length", GgufValueType.Uint32, (uint)dim),
            ("qwen2.feed_forward_length", GgufValueType.Uint32, (uint)ffnDim),
            ("qwen2.attention.head_count", GgufValueType.Uint32, (uint)nHeads),
            ("qwen2.attention.head_count_kv", GgufValueType.Uint32, (uint)nHeadsKv),
            ("qwen2.attention.key_length", GgufValueType.Uint32, (uint)headDim),
            ("qwen2.attention.value_length", GgufValueType.Uint32, (uint)headDim),
            ("qwen2.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-5f),
            ("qwen2.rope.freq_base", GgufValueType.Float32, 10000.0f),
            ("qwen2.rope.dimension_count", GgufValueType.Uint32, (uint)headDim),
            ("tokenizer.ggml.model", GgufValueType.String, "gpt2"),
            ("tokenizer.ggml.tokens", GgufValueType.Array, vocab),
            ("tokenizer.ggml.bos_token_id", GgufValueType.Uint32, (uint)BosId),
            ("tokenizer.ggml.eos_token_id", GgufValueType.Uint32, (uint)EosId),
            ("tokenizer.ggml.padding_token_id", GgufValueType.Uint32, (uint)EosId),
            ("tokenizer.ggml.add_bos_token", GgufValueType.Bool, false),
            ("tokenizer.ggml.add_eos_token", GgufValueType.Bool, false)
        };

        var tensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Bytes)>();

        float[] embdData = dominantTokenId >= 0
            ? CreatePositiveEmbeddings(vocabSize, dim)
            : CreateRandomMatrix(vocabSize, dim, rng, 0.25f);
        float[] outData = dominantTokenId >= 0
            ? CreateDominantOutputMatrix(vocabSize, dim, dominantTokenId)
            : CreateOutputMatrix(vocabSize, dim, rng);
        float layerScale = dominantTokenId >= 0 ? 0.002f : 0.12f;

        tensors.Add(CreateTensor("token_embd.weight", [dim, vocabSize], matType, embdData));
        tensors.Add(CreateTensor("output_norm.weight", [dim], GgufType.F32, CreateOnes(dim)));
        tensors.Add(CreateTensor("output.weight", [dim, vocabSize], matType, outData));

        for (int l = 0; l < layers; l++)
        {
            AddDenseLayerTensors(tensors, l, dim, ffnDim, qDim, kvDim, matType, rng, layerScale);
        }

        if (includeEmbeddedMtp)
        {
            int mtpIdx = layers;
            tensors.Add(CreateTensor($"blk.{mtpIdx}.nextn.enorm.weight", [dim], GgufType.F32, CreateOnes(dim)));
            tensors.Add(CreateTensor($"blk.{mtpIdx}.nextn.hnorm.weight", [dim], GgufType.F32, CreateOnes(dim)));
            tensors.Add(CreateTensor($"blk.{mtpIdx}.nextn.eh_proj.weight", [2 * dim, dim], matType, CreateRandomMatrix(dim, 2 * dim, rng, 0.1f)));
            tensors.Add(CreateTensor($"blk.{mtpIdx}.nextn.shared_head_norm.weight", [dim], GgufType.F32, CreateOnes(dim)));
            tensors.Add(CreateTensor($"blk.{mtpIdx}.nextn.shared_head_head.weight", [dim, vocabSize], matType, CreateOutputMatrix(vocabSize, dim, rng)));
            AddDenseLayerTensors(tensors, mtpIdx, dim, ffnDim, qDim, kvDim, matType, rng);
        }

        WriteGgufFile(path, metadata, tensors);
        return path;
    }

    public static string CreateSyntheticMtpSidecarGguf(
        string directory,
        string fileName = "m4b_mtp_sidecar.gguf",
        int dim = 64,
        int ffnDim = 128,
        int nHeads = 4,
        int nHeadsKv = 2,
        int headDim = 16,
        int vocabSize = 96,
        int mtpBlockIdx = 2,
        int seed = 99)
    {
        Directory.CreateDirectory(directory);
        string path = Path.Combine(directory, fileName);

        var rng = new Random(seed);
        int qDim = nHeads * headDim;
        int kvDim = nHeadsKv * headDim;

        var metadata = new List<(string Key, GgufValueType Type, object Value)>
        {
            ("general.architecture", GgufValueType.String, "qwen2"),
            ("general.alignment", GgufValueType.Uint32, 32u),
            ("qwen2.block_count", GgufValueType.Uint32, (uint)(mtpBlockIdx + 1)),
            ("qwen2.embedding_length", GgufValueType.Uint32, (uint)dim),
            ("qwen2.feed_forward_length", GgufValueType.Uint32, (uint)ffnDim),
            ("qwen2.attention.head_count", GgufValueType.Uint32, (uint)nHeads),
            ("qwen2.attention.head_count_kv", GgufValueType.Uint32, (uint)nHeadsKv),
            ("qwen2.attention.key_length", GgufValueType.Uint32, (uint)headDim),
            ("qwen2.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-5f),
            ("qwen2.rope.freq_base", GgufValueType.Float32, 10000.0f),
            ("qwen2.rope.dimension_count", GgufValueType.Uint32, (uint)headDim)
        };

        var tensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Bytes)>
        {
            CreateTensor($"blk.{mtpBlockIdx}.nextn.enorm.weight", [dim], GgufType.F32, CreateOnes(dim)),
            CreateTensor($"blk.{mtpBlockIdx}.nextn.hnorm.weight", [dim], GgufType.F32, CreateOnes(dim)),
            CreateTensor($"blk.{mtpBlockIdx}.nextn.eh_proj.weight", [2 * dim, dim], GgufType.F32, CreateRandomMatrix(dim, 2 * dim, rng, 0.1f)),
            CreateTensor($"blk.{mtpBlockIdx}.nextn.shared_head_norm.weight", [dim], GgufType.F32, CreateOnes(dim)),
            CreateTensor($"blk.{mtpBlockIdx}.nextn.shared_head_head.weight", [dim, vocabSize], GgufType.F32, CreateOutputMatrix(vocabSize, dim, rng))
        };
        AddDenseLayerTensors(tensors, mtpBlockIdx, dim, ffnDim, qDim, kvDim, GgufType.F32, rng);

        WriteGgufFile(path, metadata, tensors);
        return path;
    }

    public static string CreateSyntheticLoraFile(
        string directory,
        string fileName = "m4b_lora.bin",
        int layerCount = 2,
        int rank = 4,
        float alpha = 8.0f,
        int dim = 64,
        int qDim = 64,
        int seed = 123)
    {
        Directory.CreateDirectory(directory);
        string path = Path.Combine(directory, fileName);
        var rng = new Random(seed);

        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs);

        bw.Write(0x41524F4Cu);
        bw.Write(1u);
        bw.Write(layerCount);
        bw.Write(rank);
        bw.Write(alpha);
        bw.Write(2); // blk.0.attn_q.lora_a + lora_b

        bw.Write("blk.0.attn_q.lora_a.weight");
        bw.Write(dim);
        bw.Write(rank);
        var aData = CreateRandomMatrix(dim, rank, rng, 0.15f);
        bw.Write(MemoryMarshal.AsBytes(aData.AsSpan()));

        bw.Write("blk.0.attn_q.lora_b.weight");
        bw.Write(rank);
        bw.Write(qDim);
        var bData = CreateRandomMatrix(rank, qDim, rng, 0.15f);
        bw.Write(MemoryMarshal.AsBytes(bData.AsSpan()));

        return path;
    }

    public static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double dot = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            double va = a[i];
            double vb = b[i];
            dot += va * vb;
            normA += va * va;
            normB += vb * vb;
        }
        double denom = Math.Sqrt(normA) * Math.Sqrt(normB);
        return denom > 1e-12 ? (float)(dot / denom) : 0f;
    }

    public static float MaxAbsDiff(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float maxDiff = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            float d = MathF.Abs(a[i] - b[i]);
            if (d > maxDiff) maxDiff = d;
        }
        return maxDiff;
    }

    private static void AddDenseLayerTensors(
        List<(string Name, ulong[] Dims, GgufType Type, byte[] Bytes)> tensors,
        int layer,
        int dim,
        int ffnDim,
        int qDim,
        int kvDim,
        GgufType matType,
        Random rng,
        float scale = 0.12f)
    {
        tensors.Add(CreateTensor($"blk.{layer}.attn_norm.weight", [dim], GgufType.F32, CreateOnes(dim)));
        tensors.Add(CreateTensor($"blk.{layer}.attn_q.weight", [dim, qDim], matType, CreateRandomMatrix(qDim, dim, rng, scale)));
        tensors.Add(CreateTensor($"blk.{layer}.attn_k.weight", [dim, kvDim], matType, CreateRandomMatrix(kvDim, dim, rng, scale)));
        tensors.Add(CreateTensor($"blk.{layer}.attn_v.weight", [dim, kvDim], matType, CreateRandomMatrix(kvDim, dim, rng, scale)));
        tensors.Add(CreateTensor($"blk.{layer}.attn_output.weight", [qDim, dim], matType, CreateRandomMatrix(dim, qDim, rng, scale)));
        tensors.Add(CreateTensor($"blk.{layer}.ffn_norm.weight", [dim], GgufType.F32, CreateOnes(dim)));
        tensors.Add(CreateTensor($"blk.{layer}.ffn_gate.weight", [dim, ffnDim], matType, CreateRandomMatrix(ffnDim, dim, rng, scale)));
        tensors.Add(CreateTensor($"blk.{layer}.ffn_up.weight", [dim, ffnDim], matType, CreateRandomMatrix(ffnDim, dim, rng, scale)));
        tensors.Add(CreateTensor($"blk.{layer}.ffn_down.weight", [ffnDim, dim], matType, CreateRandomMatrix(dim, ffnDim, rng, scale)));
    }

    private static float[] CreatePositiveEmbeddings(int vocabSize, int dim)
    {
        var arr = new float[vocabSize * dim];
        Array.Fill(arr, 1.0f);
        return arr;
    }

    private static float[] CreateDominantOutputMatrix(int vocabSize, int dim, int dominantTokenId)
    {
        var arr = new float[vocabSize * dim];
        for (int c = 0; c < dim; c++)
        {
            arr[dominantTokenId * dim + c] = 2.0f;
        }
        return arr;
    }

    private static float[] CreateOnes(int count)
    {
        var arr = new float[count];
        Array.Fill(arr, 1.0f);
        return arr;
    }

    private static float[] CreateRandomMatrix(int rows, int cols, Random rng, float scale)
    {
        var arr = new float[rows * cols];
        for (int i = 0; i < arr.Length; i++)
        {
            arr[i] = ((float)rng.NextDouble() * 2.0f - 1.0f) * scale;
        }
        return arr;
    }

    private static float[] CreateOutputMatrix(int vocabSize, int dim, Random rng)
    {
        var arr = new float[vocabSize * dim];
        for (int r = 0; r < vocabSize; r++)
        {
            for (int c = 0; c < dim; c++)
            {
                arr[r * dim + c] = r == EosId
                    ? 0.0f
                    : ((float)rng.NextDouble() * 2.0f - 1.0f) * 0.2f;
            }
        }

        // Ensure a non-EOS pair (10, 11) are exact opposites so one always has logit > 0 (EOS = 0 never wins)
        for (int c = 0; c < dim; c++)
        {
            float v = arr[10 * dim + c];
            if (MathF.Abs(v) < 0.05f) v = 0.15f;
            arr[10 * dim + c] = v;
            arr[11 * dim + c] = -v;
        }
        return arr;
    }

    private static (string Name, ulong[] Dims, GgufType Type, byte[] Bytes) CreateTensor(
        string name,
        int[] dims,
        GgufType type,
        float[] floats)
    {
        ulong[] uDims = new ulong[dims.Length];
        for (int i = 0; i < dims.Length; i++) uDims[i] = (ulong)dims[i];

        byte[] bytes;
        if (type == GgufType.F32)
        {
            bytes = MemoryMarshal.AsBytes(floats.AsSpan()).ToArray();
        }
        else if (type == GgufType.Q8_0)
        {
            int cols = dims[0];
            int rows = dims.Length > 1 ? dims[1] : 1;
            int rowBlocks = cols / 32;
            if (cols % 32 != 0)
                throw new ArgumentException($"Q8_0 synthetic matrix requires cols ({cols}) divisible by 32.");
            int blockBytes = sizeof(BlockQ8_0);
            bytes = new byte[rows * rowBlocks * blockBytes];
            fixed (float* pSrc = floats)
            fixed (byte* pDst = bytes)
            {
                for (int r = 0; r < rows; r++)
                {
                    QuantKernels.QuantizeQ8_0(pSrc + r * cols, (BlockQ8_0*)(pDst + r * rowBlocks * blockBytes), cols);
                }
            }
        }
        else
        {
            throw new NotSupportedException($"Unsupported synthetic tensor type: {type}");
        }

        return (name, uDims, type, bytes);
    }

    private static void WriteGgufFile(
        string path,
        List<(string Key, GgufValueType Type, object Value)> metadata,
        List<(string Name, ulong[] Dims, GgufType Type, byte[] Bytes)> tensors)
    {
        const uint Alignment = 32;
        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: true);

        bw.Write(0x46554747u); // 'GGUF'
        bw.Write(3u);          // version 3
        bw.Write((ulong)tensors.Count);
        bw.Write((ulong)metadata.Count);

        foreach (var (key, type, val) in metadata)
        {
            WriteGgufString(bw, key);
            bw.Write((uint)type);
            WriteGgufValue(bw, type, val);
        }

        ulong currentDataOffset = 0;
        foreach (var (name, dims, type, bytes) in tensors)
        {
            WriteGgufString(bw, name);
            bw.Write((uint)dims.Length);
            for (int d = 0; d < dims.Length; d++) bw.Write(dims[d]);
            bw.Write((uint)type);
            bw.Write(currentDataOffset);
            ulong padded = ((ulong)bytes.Length + Alignment - 1) / Alignment * Alignment;
            currentDataOffset += padded;
        }

        long pos = fs.Position;
        long pad = (Alignment - (pos % Alignment)) % Alignment;
        for (long i = 0; i < pad; i++) bw.Write((byte)0);

        foreach (var (_, _, _, bytes) in tensors)
        {
            bw.Write(bytes);
            long tPad = (Alignment - (bytes.Length % Alignment)) % Alignment;
            for (long i = 0; i < tPad; i++) bw.Write((byte)0);
        }
    }

    private static void WriteGgufString(BinaryWriter bw, string text)
    {
        byte[] utf8 = Encoding.UTF8.GetBytes(text);
        bw.Write((ulong)utf8.Length);
        bw.Write(utf8);
    }

    private static void WriteGgufValue(BinaryWriter bw, GgufValueType type, object val)
    {
        switch (type)
        {
            case GgufValueType.Uint32:
                bw.Write((uint)val);
                break;
            case GgufValueType.Float32:
                bw.Write((float)val);
                break;
            case GgufValueType.Bool:
                bw.Write((byte)((bool)val ? 1 : 0));
                break;
            case GgufValueType.String:
                WriteGgufString(bw, (string)val);
                break;
            case GgufValueType.Array:
                string[] arr = (string[])val;
                bw.Write((uint)GgufValueType.String);
                bw.Write((ulong)arr.Length);
                for (int i = 0; i < arr.Length; i++) WriteGgufString(bw, arr[i]);
                break;
            default:
                throw new NotSupportedException($"Unsupported GGUF metadata type: {type}");
        }
    }
}

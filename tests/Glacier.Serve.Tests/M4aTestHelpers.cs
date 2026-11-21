using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;

namespace Glacier.Serve.Tests;

internal static class M4aTestHelpers
{
    internal static int GetFreeTcpPort()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        return ((IPEndPoint)listener.LocalEndpoint).Port;
    }

    internal static string CreateTempDirectory(string prefix)
    {
        string dir = Path.Combine(Path.GetTempPath(), $"{prefix}_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        return dir;
    }

    /// <summary>
    /// Writes a valid GGUF v3 F32 model file with deterministic embedding/output routing
    /// or random/custom layer weights for continuous batching and LoRA tests.
    /// </summary>
    internal static string WriteSyntheticGguf(
        string filePath,
        string[] vocabTokens,
        int eosTokenId,
        int bosTokenId,
        int dim = 16,
        int ffnDim = 32,
        int nHeads = 2,
        int nHeadsKv = 2,
        int layerCount = 1,
        string architecture = "qwen2",
        float[]? tokenEmbeddings = null,
        float[]? outputWeights = null,
        bool nonZeroLayerWeights = false,
        string[]? merges = null)
    {
        int headDim = dim / nHeads;
        int kvDim = nHeadsKv * headDim;
        int vocabSize = vocabTokens.Length;

        var metadata = new List<(string Key, uint Type, object Value)>
        {
            ("general.architecture", 8u, architecture),
            ("general.alignment", 4u, 32u),
            ($"{architecture}.block_count", 4u, (uint)layerCount),
            ($"{architecture}.context_length", 4u, 512u),
            ($"{architecture}.embedding_length", 4u, (uint)dim),
            ($"{architecture}.feed_forward_length", 4u, (uint)ffnDim),
            ($"{architecture}.attention.head_count", 4u, (uint)nHeads),
            ($"{architecture}.attention.head_count_kv", 4u, (uint)nHeadsKv),
            ($"{architecture}.attention.key_length", 4u, (uint)headDim),
            ($"{architecture}.attention.value_length", 4u, (uint)headDim),
            ($"{architecture}.rope.freq_base", 6u, 10000.0f),
            ($"{architecture}.attention.layer_norm_rms_epsilon", 6u, 1e-5f),
            ("tokenizer.ggml.eos_token_id", 4u, (uint)eosTokenId),
            ("tokenizer.ggml.bos_token_id", 4u, (uint)bosTokenId),
            ("tokenizer.ggml.padding_token_id", 4u, (uint)bosTokenId),
            ("tokenizer.ggml.add_bos_token", 7u, false),
            ("tokenizer.ggml.add_eos_token", 7u, false),
            ("tokenizer.ggml.tokens", 9u, vocabTokens),
            ("tokenizer.ggml.merges", 9u, merges ?? Array.Empty<string>())
        };

        var tensors = new List<(string Name, ulong[] Dims, float[] Data)>();

        float[] embd = tokenEmbeddings ?? CreateDeterministicMatrix(vocabSize * dim, 0.05f, 1);
        float[] outNorm = new float[dim];
        Array.Fill(outNorm, 1.0f);
        float[] outW = outputWeights ?? CreateDeterministicMatrix(vocabSize * dim, 0.05f, 2);

        tensors.Add(("token_embd.weight", new[] { (ulong)dim, (ulong)vocabSize }, embd));
        tensors.Add(("output_norm.weight", new[] { (ulong)dim }, outNorm));
        tensors.Add(("output.weight", new[] { (ulong)dim, (ulong)vocabSize }, outW));

        for (int l = 0; l < layerCount; l++)
        {
            float[] attnNorm = new float[dim];
            float[] ffnNorm = new float[dim];
            Array.Fill(attnNorm, 1.0f);
            Array.Fill(ffnNorm, 1.0f);

            tensors.Add(($"blk.{l}.attn_norm.weight", new[] { (ulong)dim }, attnNorm));
            tensors.Add(($"blk.{l}.attn_q.weight", new[] { (ulong)dim, (ulong)dim }, nonZeroLayerWeights ? CreateDeterministicMatrix(dim * dim, 0.03f, 10 + l) : new float[dim * dim]));
            tensors.Add(($"blk.{l}.attn_k.weight", new[] { (ulong)dim, (ulong)kvDim }, nonZeroLayerWeights ? CreateDeterministicMatrix(dim * kvDim, 0.03f, 20 + l) : new float[dim * kvDim]));
            tensors.Add(($"blk.{l}.attn_v.weight", new[] { (ulong)dim, (ulong)kvDim }, nonZeroLayerWeights ? CreateDeterministicMatrix(dim * kvDim, 0.03f, 30 + l) : new float[dim * kvDim]));
            tensors.Add(($"blk.{l}.attn_output.weight", new[] { (ulong)dim, (ulong)dim }, nonZeroLayerWeights ? CreateDeterministicMatrix(dim * dim, 0.03f, 40 + l) : new float[dim * dim]));
            tensors.Add(($"blk.{l}.ffn_norm.weight", new[] { (ulong)dim }, ffnNorm));
            tensors.Add(($"blk.{l}.ffn_gate.weight", new[] { (ulong)dim, (ulong)ffnDim }, nonZeroLayerWeights ? CreateDeterministicMatrix(dim * ffnDim, 0.03f, 50 + l) : new float[dim * ffnDim]));
            tensors.Add(($"blk.{l}.ffn_up.weight", new[] { (ulong)dim, (ulong)ffnDim }, nonZeroLayerWeights ? CreateDeterministicMatrix(dim * ffnDim, 0.03f, 60 + l) : new float[dim * ffnDim]));
            tensors.Add(($"blk.{l}.ffn_down.weight", new[] { (ulong)ffnDim, (ulong)dim }, nonZeroLayerWeights ? CreateDeterministicMatrix(ffnDim * dim, 0.03f, 70 + l) : new float[ffnDim * dim]));
        }

        using var fs = File.Create(filePath);
        using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);

        // Header
        bw.Write(0x46554747u); // 'GGUF'
        bw.Write(3u);          // version 3
        bw.Write((ulong)tensors.Count);
        bw.Write((ulong)metadata.Count);

        // Metadata
        foreach (var (key, type, val) in metadata)
        {
            WriteGgufString(bw, key);
            bw.Write(type);
            switch (type)
            {
                case 4u:
                    bw.Write((uint)val);
                    break;
                case 6u:
                    bw.Write((float)val);
                    break;
                case 7u:
                    bw.Write((byte)((bool)val ? 1 : 0));
                    break;
                case 8u:
                    WriteGgufString(bw, (string)val);
                    break;
                case 9u:
                    var arr = (string[])val;
                    bw.Write(8u); // element type = String
                    bw.Write((ulong)arr.Length);
                    foreach (string s in arr)
                        WriteGgufString(bw, s);
                    break;
            }
        }

        // Tensor directory (compute 32-byte aligned offsets)
        ulong dataOffset = 0;
        var offsets = new ulong[tensors.Count];
        for (int i = 0; i < tensors.Count; i++)
        {
            offsets[i] = dataOffset;
            ulong byteSize = (ulong)tensors[i].Data.Length * sizeof(float);
            ulong rem = byteSize % 32UL;
            dataOffset += rem == 0 ? byteSize : byteSize + (32UL - rem);
        }

        for (int i = 0; i < tensors.Count; i++)
        {
            var (name, dims, _) = tensors[i];
            WriteGgufString(bw, name);
            bw.Write((uint)dims.Length);
            for (int d = 0; d < dims.Length; d++)
                bw.Write(dims[d]);
            bw.Write(0u); // GgufType.F32
            bw.Write(offsets[i]);
        }

        // Pad to 32-byte alignment before tensor data
        long headerEnd = fs.Position;
        long pad = (32 - (headerEnd % 32)) % 32;
        for (int i = 0; i < pad; i++) bw.Write((byte)0);

        // Write tensor data
        for (int i = 0; i < tensors.Count; i++)
        {
            byte[] bytes = MemoryMarshal.AsBytes(tensors[i].Data.AsSpan()).ToArray();
            bw.Write(bytes);
            long rem = bytes.Length % 32;
            if (rem != 0)
            {
                for (long p = 0; p < 32 - rem; p++)
                    bw.Write((byte)0);
            }
        }

        bw.Flush();
        return filePath;
    }

    private static void WriteGgufString(BinaryWriter bw, string s)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(s);
        bw.Write((ulong)bytes.Length);
        bw.Write(bytes);
    }

    internal static float[] CreateDeterministicMatrix(int length, float scale, int seed)
    {
        var arr = new float[length];
        for (int i = 0; i < length; i++)
        {
            arr[i] = (((i * 37 + seed * 13) % 19) - 9) * scale;
        }
        return arr;
    }

    /// <summary>
    /// Writes a <c>0x41524F4C</c> v1 LoRA adapter binary.
    /// When <paramref name="peftTransposed"/> is false, writes Glacier layout A[in, rank] and B[rank, out].
    /// When <paramref name="peftTransposed"/> is true, writes PEFT layout A[rank, in] and B[out, rank].
    /// </summary>
    internal static string WriteSyntheticLoraBinary(
        string filePath,
        int layerCount,
        int rank,
        float alpha,
        IReadOnlyList<(int Layer, string ProjName, int InDim, int OutDim, float[] A, float[] B)> entries,
        bool peftTransposed = false)
    {
        using var fs = File.Create(filePath);
        using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);

        bw.Write(0x41524F4Cu); // 'LORA'
        bw.Write(1u);          // version 1
        bw.Write(layerCount);
        bw.Write(rank);
        bw.Write(alpha);
        bw.Write(entries.Count * 2);

        foreach (var (layer, projName, inDim, outDim, a, b) in entries)
        {
            if (peftTransposed)
            {
                // A is [rank, inDim], B is [outDim, rank]
                WriteLoraTensor(bw, $"blk.{layer}.{projName}.lora_a.weight", rank, inDim, a);
                WriteLoraTensor(bw, $"blk.{layer}.{projName}.lora_b.weight", outDim, rank, b);
            }
            else
            {
                // A is [inDim, rank], B is [rank, outDim]
                WriteLoraTensor(bw, $"blk.{layer}.{projName}.lora_a.weight", inDim, rank, a);
                WriteLoraTensor(bw, $"blk.{layer}.{projName}.lora_b.weight", rank, outDim, b);
            }
        }

        bw.Flush();
        return filePath;
    }

    private static void WriteLoraTensor(BinaryWriter bw, string name, int dim0, int dim1, float[] data)
    {
        if (data.Length != dim0 * dim1)
            throw new ArgumentException($"Data length {data.Length} != {dim0} * {dim1} for {name}");
        bw.Write(name);
        bw.Write(dim0);
        bw.Write(dim1);
        bw.Write(MemoryMarshal.AsBytes(data.AsSpan()));
    }

    internal static HttpClient CreateHttpClient(int port) =>
        new() { BaseAddress = new Uri($"http://127.0.0.1:{port}"), Timeout = TimeSpan.FromSeconds(15) };
}

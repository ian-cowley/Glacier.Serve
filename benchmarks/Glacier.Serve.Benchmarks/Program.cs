namespace Glacier.Serve.Benchmarks;

using System;
using System.Diagnostics;
using BenchmarkDotNet.Running;

using Glacier.Serve.Inference.PagedAttention;

public static class Program
{
    public static void Main(string[] args)
    {
        if (args.Length > 0 && args[0].Equals("--bdn", StringComparison.OrdinalIgnoreCase))
        {
            BenchmarkRunner.Run<ParserBenchmarks>();
            return;
        }

        Console.WriteLine("================================================================================");
        Console.WriteLine("            GLACIER.SERVE HIGH-SPEED SIMD PARSER & ROUTER BENCHMARK             ");
        Console.WriteLine("================================================================================");

        var bench = new ParserBenchmarks();
        bench.Setup();

        // Warmup
        for (int i = 0; i < 100_000; i++)
        {
            bench.ParseRequestLine();
            bench.ParseHeader();
            bench.MatchRoute();
        }

        const int iters = 2_000_000;

        // 1. SIMD Request-Line Parsing
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) bench.ParseRequestLine();
        sw.Stop();
        double nsPerLine = (sw.Elapsed.TotalMilliseconds * 1_000_000.0) / iters;
        double throughputLine = iters / sw.Elapsed.TotalSeconds;
        Console.WriteLine($"  SIMD Parse Request Line:          {nsPerLine:F1} ns/op ({throughputLine / 1_000_000.0:F2} M req/s)");

        // 2. SIMD Header Parsing
        sw.Restart();
        for (int i = 0; i < iters; i++) bench.ParseHeader();
        sw.Stop();
        double nsHeader = (sw.Elapsed.TotalMilliseconds * 1_000_000.0) / iters;
        double throughputHeader = iters / sw.Elapsed.TotalSeconds;
        Console.WriteLine($"  SIMD Parse Header with OWS:       {nsHeader:F1} ns/op ({throughputHeader / 1_000_000.0:F2} M headers/s)");

        // 3. Radix Tree Route Matching
        sw.Restart();
        for (int i = 0; i < iters; i++) bench.MatchRoute();
        sw.Stop();
        double nsRoute = (sw.Elapsed.TotalMilliseconds * 1_000_000.0) / iters;
        double throughputRoute = iters / sw.Elapsed.TotalSeconds;
        Console.WriteLine($"  Radix Tree Parameterized Match:   {nsRoute:F1} ns/op ({throughputRoute / 1_000_000.0:F2} M matches/s)");

        // 4. PagedAttention Decode Step
        RunPagedAttentionBenchmark();

        Console.WriteLine("================================================================================");
    }

    private static unsafe void RunPagedAttentionBenchmark()
    {
        const int headDim = 128;
        const int blockSize = 16;
        const int totalTokens = 1024;
        const int nHeads = 32;
        const int nHeadsKv = 8;
        const int layers = 1;

        using var pool = new PagedBlockPool(totalBlocks: 128, layers: layers, headsKv: nHeadsKv, headDim: headDim, blockSize: blockSize);
        var table = new BlockTable(pool);

        for (int t = 0; t < totalTokens; t++)
        {
            table.AppendToken(out int blockId, out int offset);
            float[] k = new float[nHeadsKv * headDim];
            float[] v = new float[nHeadsKv * headDim];
            for (int i = 0; i < k.Length; i++) { k[i] = 0.01f; v[i] = 0.02f; }
            fixed (float* pK = k, pV = v)
            {
                pool.Store(blockId, layer: 0, offsetInBlock: offset, pK, pV);
            }
        }

        float[] q = new float[nHeads * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = 1.0f;
        float[] outAttn = new float[nHeads * headDim];
        float[] scores = new float[nHeads * totalTokens];

        // Warmup
        fixed (float* pQ = q, pOut = outAttn, pScores = scores)
        {
            for (int i = 0; i < 50; i++)
            {
                PagedAttentionKernel.ComputeAttention(0, 0, totalTokens - 1, pQ, pOut, table, pool, nHeads, nHeadsKv, headDim, headDim, 1.0f / MathF.Sqrt(headDim), pScores, totalTokens);
            }

            const int iters = 1000;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                PagedAttentionKernel.ComputeAttention(0, 0, totalTokens - 1, pQ, pOut, table, pool, nHeads, nHeadsKv, headDim, headDim, 1.0f / MathF.Sqrt(headDim), pScores, totalTokens);
            }
            sw.Stop();
            double usPerStep = (sw.Elapsed.TotalMilliseconds * 1000.0) / iters;
            double stepsPerSec = iters / sw.Elapsed.TotalSeconds;
            Console.WriteLine($"  PagedAttention Decode (1024 ctx): {usPerStep:F2} us/step ({stepsPerSec:N0} steps/s)");
        }
        table.ReleaseAll();
    }
}

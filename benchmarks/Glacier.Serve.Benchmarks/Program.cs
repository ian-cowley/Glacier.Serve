namespace Glacier.Serve.Benchmarks;

using System;
using System.Diagnostics;
using BenchmarkDotNet.Running;

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

        Console.WriteLine("================================================================================");
    }
}

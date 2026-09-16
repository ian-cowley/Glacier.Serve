using System;
using System.IO;
using System.Threading.Tasks;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Inference.PagedAttention;
using Xunit;

namespace Glacier.Serve.Tests;

public class ContinuousBatchingTests
{
    [Fact]
    public void SequenceState_TracksTokensAndAllocations()
    {
        using var pool = new PagedBlockPool(totalBlocks: 8, layers: 2, headsKv: 2, headDim: 16, blockSize: 8);
        var table = new BlockTable(pool);
        int[] prompt = [100, 200, 300];

        var seq = new SequenceState("test-req-1", prompt, table, maxNewTokens: 32, temperature: 0.5f);

        Assert.Equal("test-req-1", seq.RequestId);
        Assert.Equal(3, seq.PromptTokens.Length);
        Assert.Equal(300, seq.LastToken);
        Assert.Equal(32, seq.MaxNewTokens);
        Assert.Equal(0.5f, seq.SamplingOptions.Temperature);
        Assert.False(seq.IsFinished);
        Assert.Empty(seq.GeneratedTokens);
    }

    [Fact]
    public void PagedBlockPool_ThrowsOnInvalidParameters()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new PagedBlockPool(totalBlocks: 0, layers: 2, headsKv: 2, headDim: 16));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PagedBlockPool(totalBlocks: 10, layers: 0, headsKv: 2, headDim: 16));
    }
}

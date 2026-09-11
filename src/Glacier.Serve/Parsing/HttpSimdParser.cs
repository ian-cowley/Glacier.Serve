namespace Glacier.Serve.Parsing;

using System;
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using Glacier.Serve.Routing;

/// <summary>
/// Hardware-accelerated HTTP/1.1 delimiter and header parser using Vector256 and SearchValues intrinsics.
/// </summary>
public static class HttpSimdParser
{
    private static readonly SearchValues<byte> NewLineSearch = SearchValues.Create([(byte)'\r', (byte)'\n']);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe int IndexOfByteVector256(ReadOnlySpan<byte> span, byte value)
    {
        if (Vector256.IsHardwareAccelerated && span.Length >= 32)
        {
            var vTarget = Vector256.Create(value);
            fixed (byte* p = span)
            {
                int i = 0;
                for (; i <= span.Length - 32; i += 32)
                {
                    var v = Vector256.Load(p + i);
                    var cmp = Vector256.Equals(v, vTarget);
                    uint mask = Vector256.ExtractMostSignificantBits(cmp);
                    if (mask != 0)
                    {
                        return i + BitOperations.TrailingZeroCount(mask);
                    }
                }
                for (; i < span.Length; i++)
                {
                    if (p[i] == value) return i;
                }
                return -1;
            }
        }
        return span.IndexOf(value);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe int IndexOfCrlfVector256(ReadOnlySpan<byte> span)
    {
        if (Vector256.IsHardwareAccelerated && span.Length >= 32)
        {
            var vCr = Vector256.Create((byte)'\r');
            var vLf = Vector256.Create((byte)'\n');
            fixed (byte* p = span)
            {
                int i = 0;
                for (; i <= span.Length - 32; i += 32)
                {
                    var v = Vector256.Load(p + i);
                    var cmpCr = Vector256.Equals(v, vCr);
                    var cmpLf = Vector256.Equals(v, vLf);
                    var cmp = Vector256.BitwiseOr(cmpCr, cmpLf);
                    uint mask = Vector256.ExtractMostSignificantBits(cmp);
                    if (mask != 0)
                    {
                        return i + BitOperations.TrailingZeroCount(mask);
                    }
                }
                for (; i < span.Length; i++)
                {
                    if (p[i] == (byte)'\r' || p[i] == (byte)'\n') return i;
                }
                return -1;
            }
        }
        return span.IndexOfAny(NewLineSearch);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static bool TryParseRequestLine(
        ReadOnlySpan<byte> buffer,
        out ReadOnlySpan<byte> method,
        out ReadOnlySpan<byte> path,
        out int bytesConsumed)
    {
        method = default;
        path = default;
        bytesConsumed = 0;

        int eol = IndexOfCrlfVector256(buffer);
        if (eol < 0) return false;

        var line = buffer[..eol];

        // 1. Locate first space delimiter separating HTTP Method and Path
        int firstSpace = IndexOfByteVector256(line, (byte)' ');
        if (firstSpace <= 0) return false;

        method = line[..firstSpace];
        var remaining = line[(firstSpace + 1)..];

        // 2. Locate second space separating Path and HTTP Version
        int secondSpace = IndexOfByteVector256(remaining, (byte)' ');
        if (secondSpace <= 0) return false;

        path = remaining[..secondSpace];
        var version = remaining[(secondSpace + 1)..];

        if (version.Length < 8 || !version.StartsWith("HTTP/"u8)) return false;

        int lineEndLength = (buffer[eol] == (byte)'\r' && buffer.Length > eol + 1 && buffer[eol + 1] == (byte)'\n') ? 2 : 1;
        bytesConsumed = eol + lineEndLength;
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static bool TryParseHeader(
        ReadOnlySpan<byte> buffer,
        out ReadOnlySpan<byte> headerName,
        out ReadOnlySpan<byte> headerValue,
        out int bytesConsumed)
    {
        headerName = default;
        headerValue = default;
        bytesConsumed = 0;

        int colonIdx = IndexOfByteVector256(buffer, (byte)':');
        if (colonIdx <= 0) return false;

        headerName = buffer[..colonIdx];
        var valSpan = buffer[(colonIdx + 1)..];

        // RFC 7230 §3.2.4: Trim leading OWS (spaces and tabs)
        while (valSpan.Length > 0 && (valSpan[0] == (byte)' ' || valSpan[0] == (byte)'\t'))
            valSpan = valSpan[1..];

        int eol = IndexOfCrlfVector256(valSpan);
        if (eol < 0) return false;

        var rawValue = valSpan[..eol];

        // Trim trailing OWS
        while (rawValue.Length > 0 && (rawValue[^1] == (byte)' ' || rawValue[^1] == (byte)'\t'))
            rawValue = rawValue[..^1];

        headerValue = rawValue;
        int lineEndLength = (valSpan[eol] == (byte)'\r' && valSpan.Length > eol + 1 && valSpan[eol + 1] == (byte)'\n') ? 2 : 1;
        bytesConsumed = (colonIdx + 1) + (buffer.Length - (colonIdx + 1) - valSpan.Length) + eol + lineEndLength;
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static HttpMethod ParseMethod(ReadOnlySpan<byte> method)
    {
        if (method.SequenceEqual("GET"u8)) return HttpMethod.Get;
        if (method.SequenceEqual("POST"u8)) return HttpMethod.Post;
        if (method.SequenceEqual("PUT"u8)) return HttpMethod.Put;
        if (method.SequenceEqual("DELETE"u8)) return HttpMethod.Delete;
        if (method.SequenceEqual("PATCH"u8)) return HttpMethod.Patch;
        if (method.SequenceEqual("HEAD"u8)) return HttpMethod.Head;
        if (method.SequenceEqual("OPTIONS"u8)) return HttpMethod.Options;
        return HttpMethod.Unknown;
    }
}

namespace Glacier.Serve.Parsing;

using System;
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Glacier.Serve.Routing;

public enum HttpProtocolVersion : byte
{
    Unknown = 0,
    Http10 = 1,
    Http11 = 2
}

/// <summary>
/// Hardware-accelerated HTTP/1.1 delimiter and header parser using Vector512/Vector256/Vector128 and SearchValues intrinsics.
/// </summary>
public static class HttpSimdParser
{
    private static readonly SearchValues<byte> NewLineSearch = SearchValues.Create([(byte)'\r', (byte)'\n']);
    private static readonly SearchValues<byte> InvalidHeaderValueBytes = SearchValues.Create(CreateInvalidHeaderValueBytes());

    private static byte[] CreateInvalidHeaderValueBytes()
    {
        // RFC 9110 §5.5: field-value excludes NUL and all CTLs except HTAB (0x09)
        var list = new byte[32];
        int idx = 0;
        for (int b = 0; b < 0x20; b++)
        {
            if (b != 0x09)
                list[idx++] = (byte)b;
        }
        list[idx] = 0x7F;
        return list;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int IndexOfByteVector256(ReadOnlySpan<byte> span, byte value)
    {
        int i = 0;
        ref byte src = ref MemoryMarshal.GetReference(span);

        if (Vector512.IsHardwareAccelerated && span.Length >= 64)
        {
            var vTarget = Vector512.Create(value);
            for (; i <= span.Length - 64; i += 64)
            {
                var v = Vector512.LoadUnsafe(ref src, (nuint)i);
                ulong mask = Vector512.ExtractMostSignificantBits(Vector512.Equals(v, vTarget));
                if (mask != 0)
                    return i + BitOperations.TrailingZeroCount(mask);
            }
        }
        else if (Vector256.IsHardwareAccelerated && span.Length >= 32)
        {
            var vTarget = Vector256.Create(value);
            for (; i <= span.Length - 32; i += 32)
            {
                var v = Vector256.LoadUnsafe(ref src, (nuint)i);
                uint mask = Vector256.ExtractMostSignificantBits(Vector256.Equals(v, vTarget));
                if (mask != 0)
                    return i + BitOperations.TrailingZeroCount(mask);
            }
        }

        if (Vector128.IsHardwareAccelerated && span.Length - i >= 16)
        {
            var vTarget = Vector128.Create(value);
            for (; i <= span.Length - 16; i += 16)
            {
                var v = Vector128.LoadUnsafe(ref src, (nuint)i);
                uint mask = Vector128.ExtractMostSignificantBits(Vector128.Equals(v, vTarget));
                if (mask != 0)
                    return i + BitOperations.TrailingZeroCount(mask);
            }
        }

        int tailIdx = span[i..].IndexOf(value);
        return tailIdx >= 0 ? i + tailIdx : -1;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int IndexOfCrlfVector256(ReadOnlySpan<byte> span)
    {
        int i = 0;
        ref byte src = ref MemoryMarshal.GetReference(span);

        if (Vector512.IsHardwareAccelerated && span.Length >= 64)
        {
            var vCr = Vector512.Create((byte)'\r');
            var vLf = Vector512.Create((byte)'\n');
            for (; i <= span.Length - 64; i += 64)
            {
                var v = Vector512.LoadUnsafe(ref src, (nuint)i);
                var cmp = Vector512.BitwiseOr(Vector512.Equals(v, vCr), Vector512.Equals(v, vLf));
                ulong mask = Vector512.ExtractMostSignificantBits(cmp);
                if (mask != 0)
                    return i + BitOperations.TrailingZeroCount(mask);
            }
        }
        else if (Vector256.IsHardwareAccelerated && span.Length >= 32)
        {
            var vCr = Vector256.Create((byte)'\r');
            var vLf = Vector256.Create((byte)'\n');
            for (; i <= span.Length - 32; i += 32)
            {
                var v = Vector256.LoadUnsafe(ref src, (nuint)i);
                var cmp = Vector256.BitwiseOr(Vector256.Equals(v, vCr), Vector256.Equals(v, vLf));
                uint mask = Vector256.ExtractMostSignificantBits(cmp);
                if (mask != 0)
                    return i + BitOperations.TrailingZeroCount(mask);
            }
        }

        if (Vector128.IsHardwareAccelerated && span.Length - i >= 16)
        {
            var vCr = Vector128.Create((byte)'\r');
            var vLf = Vector128.Create((byte)'\n');
            for (; i <= span.Length - 16; i += 16)
            {
                var v = Vector128.LoadUnsafe(ref src, (nuint)i);
                var cmp = Vector128.BitwiseOr(Vector128.Equals(v, vCr), Vector128.Equals(v, vLf));
                uint mask = Vector128.ExtractMostSignificantBits(cmp);
                if (mask != 0)
                    return i + BitOperations.TrailingZeroCount(mask);
            }
        }

        int tailIdx = span[i..].IndexOfAny(NewLineSearch);
        return tailIdx >= 0 ? i + tailIdx : -1;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static bool TryParseRequestLine(
        ReadOnlySpan<byte> buffer,
        out ReadOnlySpan<byte> method,
        out ReadOnlySpan<byte> path,
        out int bytesConsumed)
        => TryParseRequestLine(buffer, out method, out path, out _, out bytesConsumed);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static bool TryParseRequestLine(
        ReadOnlySpan<byte> buffer,
        out ReadOnlySpan<byte> method,
        out ReadOnlySpan<byte> path,
        out HttpProtocolVersion version,
        out int bytesConsumed)
    {
        method = default;
        path = default;
        version = HttpProtocolVersion.Unknown;
        bytesConsumed = 0;

        int eol = IndexOfCrlfVector256(buffer);
        if (eol < 0) return false;
        if (buffer[eol] != (byte)'\r' || buffer.Length <= eol + 1 || buffer[eol + 1] != (byte)'\n') return false;

        var line = buffer[..eol];

        // 1. Locate first space delimiter separating HTTP Method and Path
        int firstSpace = IndexOfByteVector256(line, (byte)' ');
        if (firstSpace <= 0) return false;

        var methodSpan = line[..firstSpace];
        for (int i = 0; i < methodSpan.Length; i++)
        {
            byte b = methodSpan[i];
            if (b <= 0x20 || b >= 0x7F) return false;
        }

        var remaining = line[(firstSpace + 1)..];

        // 2. Locate second space separating Path and HTTP Version
        int secondSpace = IndexOfByteVector256(remaining, (byte)' ');
        if (secondSpace <= 0) return false;

        var pathSpan = remaining[..secondSpace];
        for (int i = 0; i < pathSpan.Length; i++)
        {
            byte b = pathSpan[i];
            if (b <= 0x20 || b == 0x7F) return false;
        }

        var versionSpan = remaining[(secondSpace + 1)..];
        if (versionSpan.Length != 8) return false;

        if (versionSpan.SequenceEqual("HTTP/1.1"u8))
            version = HttpProtocolVersion.Http11;
        else if (versionSpan.SequenceEqual("HTTP/1.0"u8))
            version = HttpProtocolVersion.Http10;
        else
            return false;

        method = methodSpan;
        path = pathSpan;
        bytesConsumed = eol + 2;
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

        int eol = IndexOfCrlfVector256(buffer);
        if (eol < 0) return false;
        if (buffer[eol] != (byte)'\r' || buffer.Length <= eol + 1 || buffer[eol + 1] != (byte)'\n') return false;

        var line = buffer[..eol];
        int colonIdx = IndexOfByteVector256(line, (byte)':');
        if (colonIdx <= 0) return false;

        var nameSpan = line[..colonIdx];
        for (int i = 0; i < nameSpan.Length; i++)
        {
            byte b = nameSpan[i];
            if (b <= 0x20 || b >= 0x7F) return false;
        }
        var valSpan = line[(colonIdx + 1)..];

        // RFC 9110 §5.5: Reject NUL and CTL characters (except HTAB) in field values
        if (valSpan.IndexOfAny(InvalidHeaderValueBytes) >= 0)
            return false;

        // RFC 9110 §5.5: Trim leading OWS
        while (valSpan.Length > 0 && (valSpan[0] == (byte)' ' || valSpan[0] == (byte)'\t'))
            valSpan = valSpan[1..];

        // Trim trailing OWS
        while (valSpan.Length > 0 && (valSpan[^1] == (byte)' ' || valSpan[^1] == (byte)'\t'))
            valSpan = valSpan[..^1];

        headerName = nameSpan;
        headerValue = valSpan;
        bytesConsumed = eol + 2;
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

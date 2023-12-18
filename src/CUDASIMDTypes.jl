module CUDASIMDTypes

using BFloat16s
using CUDA
using LLVM
using Random

const SmallInt = Union{Int8,Int16,Int32,UInt8,UInt16,UInt32}

################################################################################

export prmt
"""
    prmt(a, b, op)

Permute bytes bytes from a pair of inputs.

Call the [PTX `prmt`
instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt).
This picks four arbitrary bytes from the input values `a` and `b`.
"""
function prmt(a::UInt32, b::UInt32, op::UInt32)
    ab = (b % UInt64) << 0x20 | (a % UInt64) << 0x00
    vals = ntuple(n -> (ab >> (0x8 * (n - 1))) % UInt8, 8)
    signs = ntuple(n -> ((vals[n] % Int8) >> 0x7) % UInt8, 8)
    ops = ntuple(n -> ((op >> (0x4 * (n - 1))) & 0xf) % UInt8, 4)
    res = ntuple(n -> ifelse(ops[n] & 0x8 ≠ 0, signs[ops[n] & 0x7 + 1], vals[ops[n] & 0x7 + 1]), 4)
    res2 = ntuple(n -> (res[n] % UInt32) << (8 * (n - 1)), 4)
    return reduce(|, res2)::UInt32
end
CUDA.@device_override function prmt(a::UInt32, b::UInt32, op::UInt32)
    return LLVM.Interop.@asmcall("prmt.b32 \$0, \$1, \$2, \$3;", "=r,r,r,r", UInt32, Tuple{UInt32,UInt32,UInt32}, a, b, op)
end
prmt(a::T, b::T, op::SmallInt) where {T<:SmallInt} = prmt(a % UInt32, b % UInt32, op % UInt32)::UInt32 % T

################################################################################

export lop3
"""
    lop3(a, b, c, lut)

Arbitrary logical operation on 3 inputs.

Call the [PTX `prmt`
instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt).
This computes a bitwise logical operation on the inputs `a`, `b`, and
`c`.

See [`make_lop3_lut`](@ref) for creating the look-up table `lut`.
"""
function lop3(a::UInt32, b::UInt32, c::UInt32, lut::UInt32)
    z = UInt32(0)
    return (ifelse(lut & 0x01 ≠ 0, ~z, z) & ~a & ~b & ~c) |
           (ifelse(lut & 0x02 ≠ 0, ~z, z) & ~a & ~b & c) |
           (ifelse(lut & 0x04 ≠ 0, ~z, z) & ~a & b & ~c) |
           (ifelse(lut & 0x08 ≠ 0, ~z, z) & ~a & b & c) |
           (ifelse(lut & 0x10 ≠ 0, ~z, z) & a & ~b & ~c) |
           (ifelse(lut & 0x20 ≠ 0, ~z, z) & a & ~b & c) |
           (ifelse(lut & 0x40 ≠ 0, ~z, z) & a & b & ~c) |
           (ifelse(lut & 0x80 ≠ 0, ~z, z) & a & b & c)
end
CUDA.@device_override function lop3(a::UInt32, b::UInt32, c::UInt32, lut::UInt32)
    return LLVM.Interop.@asmcall(
        "lop3.b32 \$0, \$1, \$2, \$3, \$4;", "=r,r,r,r,i", UInt32, Tuple{UInt32,UInt32,UInt32,UInt32}, a, b, c, lut
    )
end
lop3(a::T, b::T, c::T, lut::SmallInt) where {T<:SmallInt} = lop3(a % UInt32, b % UInt32, c % UInt32, lut % UInt32) % T
lop3(a::T, b::T, c::T, ::Val{lut}) where {T<:SmallInt,lut} = lop3(a % UInt32, b % UInt32, c % UInt32, lut % UInt32) % T

export make_lop3_lut
"""
    make_lop3_lut(f)

Create a look-up table for [`lop3`](@ref).
"""
function make_lop3_lut(f)
    ta = 0xf0
    tb = 0xcc
    tc = 0xaa
    lut = f(ta, tb, tc)::UInt8
    return lut
end

# For integer bit width conversions below
const xor_and_lut = make_lop3_lut((a, b, c) -> (a ⊻ b) & c)

################################################################################

export bitifelse
"""
    bitifelse(cond, x, y)

Bitwise version of `ifelse`.

For each bit of the output, the respective bit in `cond` determines
whether the respective bit of `x` or of `y` is selected.
"""
bitifelse(cond::UInt32, x::UInt32, y::UInt32) = ((cond & x) | (~cond & y))::UInt32
const bitifelse_lut = make_lop3_lut((cond, x, y) -> (cond & x) | (~cond & y))
CUDA.@device_override bitifelse(cond::UInt32, x::UInt32, y::UInt32) = lop3(cond, x, y, bitifelse_lut)::UInt32
bitifelse(cond::SmallInt, x::T, y::T) where {T<:SmallInt} = bitifelse(cond % UInt32, x % UInt32, y % UInt32) % T

################################################################################

"""
    d = cvt_pack_s8(a::Int32, b::Int32)
    d::UInt32
    d[1] = sat(b)
    d[2] = sat(a)
    d[3] = 0
    d[4] = 0
"""
function cvt_pack_s8(a::Int32, b::Int32)
    # I2IP.S8.S32.SAT
    return LLVM.Interop.@asmcall("cvt.pack.sat.s8.s32.b32 \$0, \$1, \$2, 0;", "=r,r,r", UInt32, Tuple{Int32,Int32}, a, b)
end

"""
    d = cvt_pack_s8(a::Int32, b::Int32, c::UInt32)
    d::UInt32
    d[1] = sat(b)
    d[2] = sat(a)
    d[3] = c[1]
    d[4] = c[2]
"""
function cvt_pack_s8(a::Int32, b::Int32, c::UInt32)
    # I2IP.S8.S32.SAT
    return LLVM.Interop.@asmcall(
        "cvt.pack.sat.s8.s32.b32 \$0, \$1, \$2, \$3;", "=r,r,r,r", UInt32, Tuple{Int32,Int32,UInt32}, a, b, c
    )
end

################################################################################

"""
    d = cvt_pack_s16(a::Int32, b::Int32)
    d::UInt32
    d[1] = sat(b)
    d[2] = sat(a)
"""
function cvt_pack_s16(a::Int32, b::Int32)
    return LLVM.Interop.@asmcall("cvt.pack.sat.s16.s32 \$0, \$1, \$2;", "=r,r,r", UInt32, Tuple{Int32,Int32}, a, b)
end

################################################################################

export dp4a
"""
    d = dp4a(a::UInt32, b::UInt32, c::Int32)
    d::Int32
    d = a[1] * b[1] + a[2] * b[2] + a[3] * b[3] + a[4] * b[4] + c
"""
function dp4a(a::UInt32, b::UInt32, c::Int32)
    # IDP.4A.S8.S8
    return LLVM.Interop.@asmcall("dp4a.s32.s32 \$0, \$1, \$2, \$3;", "=r,r,r,r", Int32, Tuple{UInt32,UInt32,Int32}, a, b, c)
end

################################################################################

export Int2x4
"""
    struct Int2x4

A SIMD type holding 4 2-bit integers in a combined 8-bit value.
"""
struct Int2x4
    val::UInt8
end

export Int4x2
"""
    struct Int4x2

A SIMD type holding 2 4-bit integers in a combined 8-bit value.
"""
struct Int4x2
    val::UInt8
end

export Int2x16
"""
    struct Int2x16

A SIMD type holding 16 2-bit integers in a combined 32-bit value.
"""
struct Int2x16
    val::UInt32
end

export Int4x8
"""
    struct Int4x8

A SIMD type holding 8 4-bit integers in a combined 32-bit value.
"""
struct Int4x8
    val::UInt32
end

export Int8x4
"""
    struct Int8x4

A SIMD type holding 4 8-bit integers in a combined 32-bit value.
"""
struct Int8x4
    val::UInt32
end

export Int16x2
"""
    struct Int16x2

A SIMD type holding 2 16-bit integers in a combined 32-bit value.
"""
struct Int16x2
    val::UInt32
end

################################################################################

function Int2x4(a1::Int8, a2::Int8, a3::Int8, a4::Int8)
    return Int2x4((a1 << 0x00) & 0x03 | (a2 << 0x02) & 0x0c | (a3 << 0x04) & 0x30 | (a4 << 0x06) & 0xc0)
end
Int2x4(a1::Integer, a2::Integer, a3::Integer, a4::Integer) = Int2x4(a1 % Int8, a2 % Int8, a3 % Int8, a4 % Int8)
Int2x4(a::NTuple{4,<:Integer}) = Int2x4(a...)
Int2x4(a::Int8x4) = Int2x4(convert(NTuple{4,Int8}, a))

function Base.convert(::Type{NTuple{4,Int8}}, a::Int2x4)
    a1 = a.val ⊻ 0b10101010     # a + 2
    a2_0 = (a1 >>> 0x00) & 0b11 # extract individual number
    a2_1 = (a1 >>> 0x02) & 0b11
    a2_2 = (a1 >>> 0x04) & 0b11
    a2_3 = (a1 >>> 0x06) & 0b11
    a3_0 = a2_0 + 0b01111110    # a + 128
    a3_1 = a2_1 + 0b01111110
    a3_2 = a2_2 + 0b01111110
    a3_3 = a2_3 + 0b01111110
    a4_0 = a3_0 ⊻ 0x80          # a
    a4_1 = a3_1 ⊻ 0x80
    a4_2 = a3_2 ⊻ 0x80
    a4_3 = a3_3 ⊻ 0x80
    return (a4_0 % Int8, a4_1 % Int8, a4_2 % Int8, a4_3 % Int8)::NTuple{4,Int8}
end
function Base.convert(::Type{NTuple{4,I}}, a::Int2x4) where {I<:Integer}
    a8_0, a8_1, a8_2, a8_3 = convert(NTuple{4,Int8}, a)
    a_0 = convert(I, a8_0)
    a_1 = convert(I, a8_1)
    a_2 = convert(I, a8_2)
    a_3 = convert(I, a8_3)
    return (a_0, a_1, a_2, a_3)::NTuple{4,I}
end

Base.show(io::IO, a::Int2x4) = print(io, "Int2x4", convert(NTuple{4,Int32}, a))

Base.length(::Int2x4) = 1

Base.zero(::Type{Int2x4}) = Int2x4(UInt8(0))
Base.zero(::Int2x4) = zero(Int2x4)
Base.iszero(a::Int2x4) = a == zero(a)
Random.rand(rng::AbstractRNG, ::Random.SamplerType{Int2x4}) = Int2x4(rand(rng, UInt8))

Base.:~(a::Int2x4) = Int2x4(~a.val)
Base.:&(a::Int2x4, b::Int2x4) = Int2x4(a.val & b.val)
Base.:|(a::Int2x4, b::Int2x4) = Int2x4(a.val | b.val)
Base.xor(a::Int2x4, b::Int2x4) = Int2x4(a.val ⊻ b.val)

Base.:+(a::Int2x4) = a
Base.:-(a::Int2x4) = Int2x4(.-convert(NTuple{4,Int8}, a))
Base.:+(a::Int2x4, b::Int2x4) = Int2x4(convert(NTuple{4,Int8}, a) .+ convert(NTuple{4,Int8}, b))
Base.:-(a::Int2x4, b::Int2x4) = Int2x4(convert(NTuple{4,Int8}, a) .- convert(NTuple{4,Int8}, b))
Base.min(a::Int2x4, b::Int2x4) = Int2x4(min.(convert(NTuple{4,Int8}, a), convert(NTuple{4,Int8}, b)))
Base.max(a::Int2x4, b::Int2x4) = Int2x4(max.(convert(NTuple{4,Int8}, a), convert(NTuple{4,Int8}, b)))
Base.clamp(a::Int2x4, alo::Int2x4, ahi::Int2x4) = min(max(a, alo), ahi)

################################################################################

Int4x2(a1::Int8, a2::Int8) = Int4x2((a1 << 0x00) & 0x0f | (a2 << 0x04) & 0xf0)
Int4x2(a1::Integer, a2::Integer) = Int4x2(a1 % Int8, a2 % Int8)
Int4x2(a::NTuple{2,<:Integer}) = Int4x2(a...)
Int4x2(a::Int16x2) = Int4x2(convert(NTuple{2,Int16}, a))

function Base.convert(::Type{NTuple{2,Int8}}, a::Int4x2)
    # a1 = a.val ⊻ 0x88                  # a + 8
    # a2_lo = a1 & 0x0f                  # extract low part
    a2_lo = lop3(a.val, 0x08, 0x0f, xor_and_lut)
    a3_lo = a2_lo + 0x78               # a + 128
    a4_lo = a3_lo ⊻ 0x80               # a
    # a2_hi = (a1 >>> 0x04) & 0x0f       # extract high part
    a2_hi = lop3(a.val >>> 0x04, 0x08, 0x0f, xor_and_lut)
    a3_hi = a2_hi + 0x78               # a + 128
    a4_hi = a3_hi ⊻ 0x80               # a
    return (a4_lo % Int8, a4_hi % Int8)::NTuple{2,Int8}
end
function Base.convert(::Type{NTuple{2,I}}, a::Int4x2) where {I<:Integer}
    alo8, ahi8 = convert(NTuple{2,Int8}, a)
    alo = convert(I, alo8)
    ahi = convert(I, ahi8)
    return (alo, ahi)::NTuple{2,I}
end

Base.show(io::IO, a::Int4x2) = print(io, "Int4x2", convert(NTuple{2,Int32}, a))

Base.length(::Int4x2) = 1

Base.zero(::Type{Int4x2}) = Int4x2(UInt8(0))
Base.zero(::Int4x2) = zero(Int4x2)
Base.iszero(a::Int4x2) = a == zero(a)
Random.rand(rng::AbstractRNG, ::Random.SamplerType{Int4x2}) = Int4x2(rand(rng, UInt8))

Base.:~(a::Int4x2) = Int4x2(~a.val)
Base.:&(a::Int4x2, b::Int4x2) = Int4x2(a.val & b.val)
Base.:|(a::Int4x2, b::Int4x2) = Int4x2(a.val | b.val)
Base.xor(a::Int4x2, b::Int4x2) = Int4x2(a.val ⊻ b.val)

Base.:+(a::Int4x2) = a
function Base.:-(a::Int4x2)
    rlo = (-a.val) & 0x0f
    rhi = -(a.val & 0xf0)
    return Int4x2(rlo | rhi)
end
function Base.:+(a::Int4x2, b::Int4x2)
    rlo = (a.val + b.val) & 0x0f
    rhi = (a.val & 0xf0) + (b.val & 0xf0)
    return Int4x2(rlo | rhi)
end
function Base.:-(a::Int4x2, b::Int4x2)
    rlo = (a.val - b.val) & 0x0f
    rhi = (a.val & 0xf0) - (b.val & 0xf0)
    return Int4x2(rlo | rhi)
end
function Base.min(a::Int4x2, b::Int4x2)
    as = convert(NTuple{2,Int32}, a)
    bs = convert(NTuple{2,Int32}, b)
    rs = min.(as, bs)
    return Int4x2(rs...)
end
function Base.max(a::Int4x2, b::Int4x2)
    as = convert(NTuple{2,Int32}, a)
    bs = convert(NTuple{2,Int32}, b)
    rs = max.(as, bs)
    return Int4x2(rs...)
end
Base.clamp(a::Int4x2, alo::Int4x2, ahi::Int4x2) = min(max(a, alo), ahi)

################################################################################

function Int2x16(
    a1::Int8,
    a2::Int8,
    a3::Int8,
    a4::Int8,
    a5::Int8,
    a6::Int8,
    a7::Int8,
    a8::Int8,
    a9::Int8,
    a10::Int8,
    a11::Int8,
    a12::Int8,
    a13::Int8,
    a14::Int8,
    a15::Int8,
    a16::Int8,
)
    return Int2x16((Int4x8(a1, a3, a5, a7, a9, a11, a13, a15), Int4x8(a2, a4, a6, a8, a10, a12, a14, a16)))
end
function Int2x16(
    a1::Integer,
    a2::Integer,
    a3::Integer,
    a4::Integer,
    a5::Integer,
    a6::Integer,
    a7::Integer,
    a8::Integer,
    a9::Integer,
    a10::Integer,
    a11::Integer,
    a12::Integer,
    a13::Integer,
    a14::Integer,
    a15::Integer,
    a16::Integer,
)
    return Int2x16(
        a1 % Int8,
        a2 % Int8,
        a3 % Int8,
        a4 % Int8,
        a5 % Int8,
        a6 % Int8,
        a7 % Int8,
        a8 % Int8,
        a9 % Int8,
        a10 % Int8,
        a11 % Int8,
        a12 % Int8,
        a13 % Int8,
        a14 % Int8,
        a15 % Int8,
        a16 % Int8,
    )
end
Int2x16(a::NTuple{16,<:Integer}) = Int2x16(a...)
Int2x16(a::NTuple{2,Int4x8}) = Int2x16(bitifelse(0x33333333, a[1].val << 0x00, a[2].val << 0x02))
Int2x16(a::NTuple{4,Int8x4}) = Int2x16((Int4x8((a[1], a[3])), Int4x8((a[2], a[4]))))
Int2x16(a::NTuple{8,Int16x2}) = Int2x16((Int8x4((a[1], a[5])), Int8x4((a[2], a[6])), Int8x4((a[3], a[7])), Int8x4((a[4], a[8]))))

function Base.convert(::Type{NTuple{2,Int4x8}}, a::Int2x16)
    a1 = a.val ⊻ 0xaaaaaaaa            # a + 2
    a2_lo = (a1 >>> 0x00) & 0x33333333 # extract individual number
    a2_hi = (a1 >>> 0x02) & 0x33333333
    a3_lo = a2_lo + 0x66666666  # a + 8
    a3_hi = a2_hi + 0x66666666
    a4_lo = a3_lo ⊻ 0x88888888  # a
    a4_hi = a3_hi ⊻ 0x88888888
    return (Int4x8(a4_lo), Int4x8(a4_hi))::NTuple{2,Int4x8}
end
function Base.convert(::Type{NTuple{4,Int8x4}}, a::Int2x16)
    alo4, ahi4 = convert(NTuple{2,Int4x8}, a)
    alo = convert(NTuple{2,Int8x4}, alo4)
    ahi = convert(NTuple{2,Int8x4}, ahi4)
    return (alo[1], ahi[1], alo[2], ahi[2])::NTuple{4,Int8x4}
end
function Base.convert(::Type{NTuple{8,Int16x2}}, a::Int2x16)
    alo4, ahi4 = convert(NTuple{2,Int4x8}, a)
    alo = convert(NTuple{4,Int16x2}, alo4)
    ahi = convert(NTuple{4,Int16x2}, ahi4)
    return (alo[1], ahi[1], alo[2], ahi[2], alo[3], ahi[3], alo[4], ahi[4])::NTuple{8,Int16x2}
end
function Base.convert(::Type{NTuple{16,I}}, a::Int2x16) where {I<:Integer}
    alo4, ahi4 = convert(NTuple{2,Int4x8}, a)
    alo = convert(NTuple{8,I}, alo4)
    ahi = convert(NTuple{8,I}, ahi4)
    return (
        alo[1],
        ahi[1],
        alo[2],
        ahi[2],
        alo[3],
        ahi[3],
        alo[4],
        ahi[4],
        alo[5],
        ahi[5],
        alo[6],
        ahi[6],
        alo[7],
        ahi[7],
        alo[8],
        ahi[8],
    )
end

Base.show(io::IO, a::Int2x16) = print(io, "Int2x16", convert(NTuple{16,Int32}, a))

Base.zero(::Type{Int2x16}) = Int2x16(UInt32(0))
Base.zero(::Int2x16) = zero(Int2x16)
Base.iszero(a::Int2x16) = a == zero(a)
Random.rand(rng::AbstractRNG, ::Random.SamplerType{Int2x16}) = Int2x16(rand(rng, UInt32))

Base.:~(a::Int2x16) = Int2x16(~a.val)
Base.:&(a::Int2x16, b::Int2x16) = Int2x16(a.val & b.val)
Base.:|(a::Int2x16, b::Int2x16) = Int2x16(a.val | b.val)
Base.xor(a::Int2x16, b::Int2x16) = Int2x16(a.val ⊻ b.val)

Base.:+(a::Int2x16) = a
function Base.:-(a::Int2x16)
    alo = a.val & 0x33333333
    rlo = 0x88888888 - alo
    ahi = a.val & 0xcccccccc
    rhi = 0x22222222 - ahi
    return Int2x16(bitifelse(0x33333333, rlo, rhi))
end
function Base.:+(a::Int2x16, b::Int2x16)
    # TODO: Combine these via LOP3
    a1 = a.val ⊻ 0xaaaaaaaa
    b1 = b.val ⊻ 0xaaaaaaaa
    alo = a1 & 0x33333333
    blo = b1 & 0x33333333
    rlo = alo + blo
    ahi = a1 & 0xcccccccc
    bhi = b1 & 0xcccccccc
    rhi = ahi + bhi
    return Int2x16(bitifelse(0x33333333, rlo, rhi))
end
function Base.:-(a::Int2x16, b::Int2x16)
    # TODO: Combine these via LOP3
    a1 = a.val ⊻ 0xaaaaaaaa
    b1 = b.val ⊻ 0xaaaaaaaa
    alo = (a1 & 0x33333333) ⊻ 0x44444444
    blo = b1 & 0x33333333
    rlo = alo - blo
    ahi = (a1 & 0xcccccccc) ⊻ 0x11111111
    bhi = b1 & 0xcccccccc
    rhi = ahi - bhi
    return Int2x16(bitifelse(0x33333333, rlo, rhi))
end
function Base.min(a::Int2x16, b::Int2x16)
    as = convert(NTuple{16,Int32}, a)
    bs = convert(NTuple{16,Int32}, b)
    rs = min.(as, bs)
    return Int2x16(rs...)
end
function Base.max(a::Int2x16, b::Int2x16)
    as = convert(NTuple{16,Int32}, a)
    bs = convert(NTuple{16,Int32}, b)
    rs = max.(as, bs)
    return Int2x16(rs...)
end
Base.clamp(a::Int2x16, alo::Int2x16, ahi::Int2x16) = min(max(a, alo), ahi)

################################################################################

function Int4x8(a1::Int8, a2::Int8, a3::Int8, a4::Int8, a5::Int8, a6::Int8, a7::Int8, a8::Int8)
    return Int4x8((Int8x4(a1, a3, a5, a7), Int8x4(a2, a4, a6, a8)))
end
function Int4x8(a1::Integer, a2::Integer, a3::Integer, a4::Integer, a5::Integer, a6::Integer, a7::Integer, a8::Integer)
    return Int4x8(a1 % Int8, a2 % Int8, a3 % Int8, a4 % Int8, a5 % Int8, a6 % Int8, a7 % Int8, a8 % Int8)
end
Int4x8(a::NTuple{8,<:Integer}) = Int4x8(a...)
Int4x8(a::NTuple{2,Int8x4}) = Int4x8(bitifelse(0x0f0f0f0f, a[1].val << 0x00, a[2].val << 0x04))
Int4x8(a::NTuple{4,Int16x2}) = Int4x8((Int8x4((a[1], a[3])), Int8x4((a[2], a[4]))))

function Base.convert(::Type{NTuple{2,Int8x4}}, a::Int4x8)
    # a1 = a.val ⊻ 0x88888888            # a + 8
    # a2_lo = a1 & 0x0f0f0f0f            # extract low part
    a2_lo = lop3(a.val, 0x08080808, 0x0f0f0f0f, xor_and_lut)
    a3_lo = a2_lo + 0x78787878         # a + 128
    a4_lo = a3_lo ⊻ 0x80808080         # a
    # a2_hi = (a1 >>> 0x04) & 0x0f0f0f0f # extract high part
    a2_hi = lop3(a.val >>> 0x04, 0x08080808, 0x0f0f0f0f, xor_and_lut)
    a3_hi = a2_hi + 0x78787878         # a + 128
    a4_hi = a3_hi ⊻ 0x80808080         # a
    return (Int8x4(a4_lo), Int8x4(a4_hi))::NTuple{2,Int8x4}
end
function Base.convert(::Type{NTuple{4,Int16x2}}, a::Int4x8)
    alo, ahi = convert(NTuple{2,Int8x4}, a)
    alolo, alohi = convert(NTuple{2,Int16x2}, alo)
    ahilo, ahihi = convert(NTuple{2,Int16x2}, ahi)
    return (alolo, ahilo, alohi, ahihi)::NTuple{4,Int16x2}
end
function Base.convert(::Type{NTuple{8,I}}, a::Int4x8) where {I<:Integer}
    alo8, ahi8 = convert(NTuple{2,Int8x4}, a)
    alo = convert(NTuple{4,I}, alo8)
    ahi = convert(NTuple{4,I}, ahi8)
    return (alo[1], ahi[1], alo[2], ahi[2], alo[3], ahi[3], alo[4], ahi[4])
end

Base.show(io::IO, a::Int4x8) = print(io, "Int4x8", convert(NTuple{8,Int32}, a))

Base.zero(::Type{Int4x8}) = Int4x8(UInt32(0))
Base.zero(::Int4x8) = zero(Int4x8)
Base.iszero(a::Int4x8) = a == zero(a)
Random.rand(rng::AbstractRNG, ::Random.SamplerType{Int4x8}) = Int4x8(rand(rng, UInt32))

Base.:~(a::Int4x8) = Int4x8(~a.val)
Base.:&(a::Int4x8, b::Int4x8) = Int4x8(a.val & b.val)
Base.:|(a::Int4x8, b::Int4x8) = Int4x8(a.val | b.val)
Base.xor(a::Int4x8, b::Int4x8) = Int4x8(a.val ⊻ b.val)

Base.:+(a::Int4x8) = a
function Base.:-(a::Int4x8)
    alo = a.val & 0x0f0f0f0f
    rlo = 0x80808080 - alo
    ahi = a.val & 0xf0f0f0f0
    rhi = 0x08080800 - ahi
    return Int4x8(bitifelse(0x0f0f0f0f, rlo, rhi))
end
function Base.:+(a::Int4x8, b::Int4x8)
    # TODO: Combine these via LOP3
    a1 = a.val ⊻ 0x88888888
    b1 = b.val ⊻ 0x88888888
    alo = a1 & 0x0f0f0f0f
    blo = b1 & 0x0f0f0f0f
    rlo = alo + blo
    ahi = a1 & 0xf0f0f0f0
    bhi = b1 & 0xf0f0f0f0
    rhi = ahi + bhi
    return Int4x8(bitifelse(0x0f0f0f0f, rlo, rhi))
end
function Base.:-(a::Int4x8, b::Int4x8)
    # TODO: Combine these via LOP3
    a1 = a.val ⊻ 0x88888888
    b1 = b.val ⊻ 0x88888888
    alo = (a1 & 0x0f0f0f0f) ⊻ 0x10101010
    blo = b1 & 0x0f0f0f0f
    rlo = alo - blo
    ahi = (a1 & 0xf0f0f0f0) ⊻ 0x01010100
    bhi = b1 & 0xf0f0f0f0
    rhi = ahi - bhi
    return Int4x8(bitifelse(0x0f0f0f0f, rlo, rhi))
end
function Base.min(a::Int4x8, b::Int4x8)
    as = convert(NTuple{8,Int32}, a)
    bs = convert(NTuple{8,Int32}, b)
    rs = min.(as, bs)
    return Int4x8(rs...)
end
function Base.max(a::Int4x8, b::Int4x8)
    as = convert(NTuple{8,Int32}, a)
    bs = convert(NTuple{8,Int32}, b)
    rs = max.(as, bs)
    return Int4x8(rs...)
end
Base.clamp(a::Int4x8, alo::Int4x8, ahi::Int4x8) = min(max(a, alo), ahi)

################################################################################

function Int8x4(a1::Int8, a2::Int8, a3::Int8, a4::Int8)
    return Int8x4(
        (a4 % UInt8 % UInt32) << 0x18 |
        (a3 % UInt8 % UInt32) << 0x10 |
        (a2 % UInt8 % UInt32) << 0x08 |
        (a1 % UInt8 % UInt32) << 0x00,
    )
end
CUDA.@device_override Int8x4(a1::Int8, a2::Int8, a3::Int8, a4::Int8) = Int8x4(a1 % Int32, a2 % Int32, a3 % Int32, a4 % Int32)
Int8x4(a1::Int32, a2::Int32, a3::Int32, a4::Int32) = Int8x4(a1 % Int8, a2 % Int8, a3 % Int8, a4 % Int8)
CUDA.@device_override Int8x4(a1::Int32, a2::Int32, a3::Int32, a4::Int32) = Int8x4(cvt_pack_s8(a2, a1, cvt_pack_s8(a4, a3)))
Int8x4(a1::Integer, a2::Integer, a3::Integer, a4::Integer) = Int8x4(a1 % Int32, a2 % Int32, a3 % Int32, a4 % Int32)
Int8x4(a::NTuple{4,<:Integer}) = Int8x4(a...)
Int8x4(a::NTuple{2,Int16x2}) = Int8x4(a[1].val >>> 0x00, a[2].val >>> 0x00, a[1].val >>> 0x10, a[2].val >>> 0x10)
CUDA.@device_override Int8x4(a::NTuple{2,Int16x2}) = Int8x4(prmt(a[1].val, a[2].val, 0x6240) % UInt32)
Int8x4(a::Int2x4) = Int8x4(convert(NTuple{4,Int8}, a))

function Base.convert(::Type{NTuple{2,Int16x2}}, a::Int8x4)
    a1 = ((a.val >>> 0x00) & 0xff) % Int8 % Int32
    a2 = ((a.val >>> 0x08) & 0xff) % Int8 % Int32
    a3 = ((a.val >>> 0x10) & 0xff) % Int8 % Int32
    a4 = ((a.val >>> 0x18) & 0xff) % Int8 % Int32
    return (Int16x2(a1, a3), Int16x2(a2, a4))::NTuple{2,Int16x2}
end
CUDA.@device_override function Base.convert(::Type{NTuple{2,Int16x2}}, a::Int8x4)
    return (Int16x2(prmt(a.val, UInt32(0), 0xa280)), Int16x2(prmt(a.val, UInt32(0), 0xb391)))::NTuple{2,Int16x2}
end
function Base.convert(::Type{NTuple{4,Int32}}, a::Int8x4)
    return (
        (a.val >>> 0x00) % Int8 % Int32,
        (a.val >>> 0x08) % Int8 % Int32,
        (a.val >>> 0x10) % Int8 % Int32,
        (a.val >>> 0x18) % Int8 % Int32,
    )::NTuple{4,Int32}
end
CUDA.@device_override function Base.convert(::Type{NTuple{4,Int32}}, a::Int8x4)
    return (
        prmt(a.val, UInt32(0), 0x8880) % Int32,
        prmt(a.val, UInt32(0), 0x9991) % Int32,
        prmt(a.val, UInt32(0), 0xaaa2) % Int32,
        prmt(a.val, UInt32(0), 0xbbb3) % Int32,
    )::NTuple{4,Int32}
end
Base.convert(::Type{NTuple{4,I}}, a::Int8x4) where {I<:Integer} = convert(NTuple{4,I}, convert(NTuple{4,Int32}, a))

Base.show(io::IO, a::Int8x4) = print(io, "Int8x4", convert(NTuple{4,Int32}, a))

Base.zero(::Type{Int8x4}) = Int8x4(UInt32(0))
Base.zero(::Int8x4) = zero(Int8x4)
Base.iszero(a::Int8x4) = a == zero(a)
Random.rand(rng::AbstractRNG, ::Random.SamplerType{Int8x4}) = Int8x4(rand(rng, UInt32))

Base.:~(a::Int8x4) = Int8x4(~a.val)
Base.:&(a::Int8x4, b::Int8x4) = Int8x4(a.val & b.val)
Base.:|(a::Int8x4, b::Int8x4) = Int8x4(a.val | b.val)
Base.xor(a::Int8x4, b::Int8x4) = Int8x4(a.val ⊻ b.val)

Base.:+(a::Int8x4) = a
function Base.:-(a::Int8x4)
    alo = a.val & 0x00ff00ff
    rlo = 0x80008000 - alo
    ahi = a.val & 0xff00ff00
    rhi = 0x00800000 - ahi
    return Int8x4(bitifelse(0x00ff00ff, rlo, rhi))
end
function Base.:+(a::Int8x4, b::Int8x4)
    # TODO: Combine these via LOP3
    a1 = a.val ⊻ 0x80808080
    b1 = b.val ⊻ 0x80808080
    alo = a1 & 0x00ff00ff
    blo = b1 & 0x00ff00ff
    rlo = alo + blo
    ahi = a1 & 0xff00ff00
    bhi = b1 & 0xff00ff00
    rhi = ahi + bhi
    return Int8x4(bitifelse(0x00ff00ff, rlo, rhi))
end
function Base.:-(a::Int8x4, b::Int8x4)
    # TODO: Combine these via LOP3
    a1 = a.val ⊻ 0x80808080
    b1 = b.val ⊻ 0x80808080
    alo = (a1 & 0x00ff00ff) ⊻ 0x01000100
    blo = b1 & 0x00ff00ff
    rlo = alo - blo
    ahi = (a1 & 0xff00ff00) ⊻ 0x00010000
    bhi = b1 & 0xff00ff00
    rhi = ahi - bhi
    return Int8x4(bitifelse(0x00ff00ff, rlo, rhi))
end
function Base.min(a::Int8x4, b::Int8x4)
    as = convert(NTuple{4,Int32}, a)
    bs = convert(NTuple{4,Int32}, b)
    rs = min.(as, bs)
    return Int8x4(rs...)
end
function Base.max(a::Int8x4, b::Int8x4)
    as = convert(NTuple{4,Int32}, a)
    bs = convert(NTuple{4,Int32}, b)
    rs = max.(as, bs)
    return Int8x4(rs...)
end
Base.clamp(a::Int8x4, alo::Int8x4, ahi::Int8x4) = min(max(a, alo), ahi)

################################################################################

Int16x2(a1::Int16, a2::Int16) = Int16x2((a1 % UInt16 % UInt32) << 0x00 | (a2 % UInt16 % UInt32) << 0x10)
CUDA.@device_override Int16x2(a1::Int16, a2::Int16) = Int16x2(prmt(a1 % UInt16 % UInt32, a2 % UInt16 % UInt32, 0x5410))
Int16x2(a1::Int32, a2::Int32) = Int16x2(a1 % Int16, a2 % Int16)
CUDA.@device_override Int16x2(a1::Int32, a2::Int32) = Int16x2(cvt_pack_s16(a2, a1))
Int16x2(a1::Integer, a2::Integer) = Int16x2(a1 % Int16, a2 % Int16)
Int16x2(a::NTuple{2,<:Integer}) = Int16x2(a...)
Int16x2(a::Int4x2) = Int16x2(convert(NTuple{2,Int8}, a))

Base.convert(::Type{NTuple{2,Int16}}, a::Int16x2) = ((a.val >>> 0x00) % Int16, (a.val >>> 0x10) % Int16)::NTuple{2,Int16}
function Base.convert(::Type{NTuple{2,Int32}}, a::Int16x2)
    return ((a.val >>> 0x00) % Int16 % Int32, (a.val >>> 0x10) % Int16 % Int32)::NTuple{2,Int32}
end
Base.convert(::Type{NTuple{2,I}}, a::Int16x2) where {I<:Integer} = I.(convert(NTuple{2,Int16}, a))

Base.show(io::IO, a::Int16x2) = print(io, "Int16x2", convert(NTuple{2,Int32}, a))

Base.zero(::Type{Int16x2}) = Int16x2(UInt32(0))
Base.zero(::Int16x2) = zero(Int16x2)
Base.iszero(a::Int16x2) = a == zero(a)
Random.rand(rng::AbstractRNG, ::Random.SamplerType{Int16x2}) = Int16x2(rand(rng, UInt32))

Base.:~(a::Int16x2) = Int16x2(~a.val)
Base.:&(a::Int16x2, b::Int16x2) = Int16x2(a.val & b.val)
Base.:|(a::Int16x2, b::Int16x2) = Int16x2(a.val | b.val)
Base.xor(a::Int16x2, b::Int16x2) = Int16x2(a.val ⊻ b.val)

Base.:+(a::Int16x2) = a
function Base.:-(a::Int16x2)
    alo = a.val
    rlo = -alo
    ahi = a.val & 0xffff0000
    rhi = -ahi
    return Int16x2(bitifelse(0x0000ffff, rlo, rhi))
end
function Base.:+(a::Int16x2, b::Int16x2)
    a1 = a.val ⊻ 0x80008000
    b1 = b.val ⊻ 0x80008000
    alo = a1
    blo = b1
    rlo = alo + blo
    ahi = a1 & 0xffff0000
    bhi = b1 & 0xffff0000
    rhi = ahi + bhi
    return Int16x2(bitifelse(0x0000ffff, rlo, rhi))
end
function Base.:-(a::Int16x2, b::Int16x2)
    a1 = a.val ⊻ 0x80008000
    b1 = b.val ⊻ 0x80008000
    alo = a1
    blo = b1
    rlo = alo - blo
    ahi = a1 & 0xffff0000
    bhi = b1 & 0xffff0000
    rhi = ahi - bhi
    return Int16x2(bitifelse(0x0000ffff, rlo, rhi))
end
function Base.min(a::Int16x2, b::Int16x2)
    as = convert(NTuple{2,Int32}, a)
    bs = convert(NTuple{2,Int32}, b)
    rs = min.(as, bs)
    return Int16x2(rs...)
end
function Base.max(a::Int16x2, b::Int16x2)
    as = convert(NTuple{2,Int32}, a)
    bs = convert(NTuple{2,Int32}, b)
    rs = max.(as, bs)
    return Int16x2(rs...)
end
Base.clamp(a::Int16x2, alo::Int16x2, ahi::Int16x2) = min(max(a, alo), ahi)

################################################################################

export Float16x2
"""
    struct Float16x2

A SIMD type holding 2 Float16 in a combined 32-bit value.
"""
struct Float16x2
    val::UInt32
end

export BFloat16x2
"""
    struct BFloat16x2

A SIMD type holding 2 BFloat16 in a combined 32-bit value.
"""
struct BFloat16x2
    val::UInt32
end

################################################################################

function Float16x2(a1::Float16, a2::Float16)
    return Float16x2((reinterpret(UInt16, a1) % UInt32) << 0x00 | (reinterpret(UInt16, a2) % UInt32) << 0x10)
end
CUDA.@device_override function Float16x2(a1::Float16, a2::Float16)
    return Float16x2(
        LLVM.Interop.@asmcall(
            "mov.b32 \$0, {\$1, \$2};", "=r,h,h", UInt32, Tuple{UInt16,UInt16}, reinterpret(UInt16, a1), reinterpret(UInt16, a2),
        )
    )
end
function Float16x2(a1::Float32, a2::Float32)
    return Float16x2((reinterpret(UInt16, Float16(a1)) % UInt32) << 0x00 | (reinterpret(UInt16, Float16(a2)) % UInt32) << 0x10)
end
CUDA.@device_override function Float16x2(a1::Float32, a2::Float32)
    return Float16x2(LLVM.Interop.@asmcall("cvt.rn.f16x2.f32 \$0, \$2, \$1;", "=r,r,r", UInt32, Tuple{Float32,Float32}, a1, a2))
end
Float16x2(a1::Real, a2::Real) = Float16x2(Float16(a1), Float16(a2))
Float16x2(a::NTuple{2,<:Real}) = Float16x2(a...)

function Base.convert(::Type{NTuple{2,Float16}}, a::Float16x2)
    return (reinterpret(Float16, (a.val >> 0x00) % UInt16), reinterpret(Float16, (a.val >> 0x10) % UInt16))
end
function Base.convert(::Type{NTuple{2,Float32}}, a::Float16x2)
    return (Float32(reinterpret(Float16, (a.val >> 0x00) % UInt16)), Float32(reinterpret(Float16, (a.val >> 0x10) % UInt16)))
end
CUDA.@device_override function Base.convert(::Type{NTuple{2,Float32}}, a::Float16x2)
    return (
        LLVM.Interop.@asmcall("cvt.f32.f16 \$0, \$1;", "=r,r", Float32, Tuple{UInt32}, a.val >> 0x00),
        LLVM.Interop.@asmcall("cvt.f32.f16 \$0, \$1;", "=r,r", Float32, Tuple{UInt32}, a.val >> 0x10)
    )::NTuple{2,Float32}
end

Base.show(io::IO, a::Float16x2) = print(io, "Float16x2", convert(NTuple{2,Float32}, a))

Base.reverse(a::Float16x2) = Float16x2(reverse(convert(NTuple{2,Float16}, a)))
CUDA.@device_override function Base.reverse(a::Float16x2)
    return Float16x2(LLVM.Interop.@asmcall(
        """{
               .reg .f16 %lo, %hi;
               mov.b32 {%lo, %hi}, \$1;
               mov.b32 \$0, {%hi, %lo};
           }
           """,
        "=r,r",
        UInt32,
        Tuple{UInt32},
        a.val,
    ))
end

# Base.zero(::Type{Float16x2}) = Float16x2(0.0f0, 0.0f0)
Base.zero(::Type{Float16x2}) = Float16x2(UInt32(0))
Base.zero(::Float16x2) = zero(Float16x2)
Base.iszero(a::Float16x2) = a == zero(a)
Random.rand(rng::AbstractRNG, ::Random.SamplerType{Float16x2}) = Float16x2(rand(rng, Float32), rand(rng, Float32))

Base.:+(a::Float16x2) = a
Base.:-(a::Float16x2) = Float16x2(a.val ⊻ 0x80008000)
CUDA.@device_override function Base.:-(a::Float16x2)
    return Float16x2(LLVM.Interop.@asmcall("neg.f16x2 \$0, \$1;", "=r,r", UInt32, Tuple{UInt32}, a.val))
end

export negate1
"""Negate a[1]"""
# negate1(a::Float16x2) = Float16x2(a.val ⊻ 0x00008000)
negate1(a::Float16x2) = a * Float16x2(-1.0f0, 1.0f0)

export negate2
"""Negate a[2]"""
# negate2(a::Float16x2) = Float16x2(a.val ⊻ 0x80000000)
negate2(a::Float16x2) = a * Float16x2(1.0f0, -1.0f0)

Base.abs(a::Float16x2) = Float16x2(a.val & 0x7fff7fff)
CUDA.@device_override function Base.abs(a::Float16x2)
    return Float16x2(LLVM.Interop.@asmcall("abs.f16x2 \$0, \$1;", "=r,r", UInt32, Tuple{UInt32}, a.val))
end

function Base.:+(a::Float16x2, b::Float16x2)
    alo, ahi = convert(NTuple{2,Float16}, a)
    blo, bhi = convert(NTuple{2,Float16}, b)
    return Float16x2(alo + blo, ahi + bhi)
end
CUDA.@device_override function Base.:+(a::Float16x2, b::Float16x2)
    return Float16x2(LLVM.Interop.@asmcall("add.rn.f16x2 \$0, \$1, \$2;", "=r,r,r", UInt32, Tuple{UInt32,UInt32}, a.val, b.val))
end
function Base.:-(a::Float16x2, b::Float16x2)
    alo, ahi = convert(NTuple{2,Float16}, a)
    blo, bhi = convert(NTuple{2,Float16}, b)
    return Float16x2(alo - blo, ahi - bhi)
end
CUDA.@device_override function Base.:-(a::Float16x2, b::Float16x2)
    return Float16x2(LLVM.Interop.@asmcall("sub.rn.f16x2 \$0, \$1, \$2;", "=r,r,r", UInt32, Tuple{UInt32,UInt32}, a.val, b.val))
end
function Base.:*(a::Float16x2, b::Float16x2)
    alo, ahi = convert(NTuple{2,Float16}, a)
    blo, bhi = convert(NTuple{2,Float16}, b)
    return Float16x2(alo * blo, ahi * bhi)
end
CUDA.@device_override function Base.:*(a::Float16x2, b::Float16x2)
    return Float16x2(LLVM.Interop.@asmcall("mul.rn.f16x2 \$0, \$1, \$2;", "=r,r,r", UInt32, Tuple{UInt32,UInt32}, a.val, b.val))
end
function Base.muladd(a::Float16x2, b::Float16x2, c::Float16x2)
    alo, ahi = convert(NTuple{2,Float16}, a)
    blo, bhi = convert(NTuple{2,Float16}, b)
    clo, chi = convert(NTuple{2,Float16}, c)
    return Float16x2(muladd(alo, blo, clo), muladd(ahi, bhi, chi))
end
CUDA.@device_override function Base.muladd(a::Float16x2, b::Float16x2, c::Float16x2)
    return Float16x2(
        LLVM.Interop.@asmcall(
            "fma.rn.f16x2 \$0, \$1, \$2, \$3;", "=r,r,r,r", UInt32, Tuple{UInt32,UInt32,UInt32}, a.val, b.val, c.val
        )
    )
end
export muladd_sat
function muladd_sat(a::Float16x2, b::Float16x2, c::Float16x2)
    return Float16x2(
        LLVM.Interop.@asmcall(
            "fma.rn.sat.f16x2 \$0, \$1, \$2, \$3;", "=r,r,r,r", UInt32, Tuple{UInt32,UInt32,UInt32}, a.val, b.val, c.val
        )
    )
end
function Base.max(a::Float16x2, b::Float16x2)
    alo, ahi = convert(NTuple{2,Float16}, a)
    blo, bhi = convert(NTuple{2,Float16}, b)
    return Float16x2(max(alo, blo), max(ahi, bhi))
end
CUDA.@device_override function Base.max(a::Float16x2, b::Float16x2)
    return Float16x2(LLVM.Interop.@asmcall("max.f16x2 \$0, \$1, \$2;", "=r,r,r", UInt32, Tuple{UInt32,UInt32}, a.val, b.val))
end
function Base.min(a::Float16x2, b::Float16x2)
    alo, ahi = convert(NTuple{2,Float16}, a)
    blo, bhi = convert(NTuple{2,Float16}, b)
    return Float16x2(min(alo, blo), min(ahi, bhi))
end
CUDA.@device_override function Base.min(a::Float16x2, b::Float16x2)
    return Float16x2(LLVM.Interop.@asmcall("min.f16x2 \$0, \$1, \$2;", "=r,r,r", UInt32, Tuple{UInt32,UInt32}, a.val, b.val))
end
Base.clamp(a::Float16x2, alo::Float16x2, ahi::Float16x2) = min(max(a, alo), ahi)

# CUDA SDK 11.6.2, file "cuda/targets/x86_64-linux/include/cuda_fp16.hpp", lines 2419 and following:
# __CUDA_FP16_DECL__ __half2 __hcmadd(const __half2 a, const __half2 b, const __half2 c)
# {
#     // fast version of complex multiply-accumulate
#     // (a.re, a.im) * (b.re, b.im) + (c.re, c.im)
#     // acc.re = (c.re + a.re*b.re) - a.im*b.im
#     // acc.im = (c.im + a.re*b.im) + a.im*b.re
#     __half real_tmp =  __hfma(a.x, b.x, c.x);
#     __half img_tmp  =  __hfma(a.x, b.y, c.y);
#     real_tmp = __hfma(__hneg(a.y), b.y, real_tmp);
#     img_tmp  = __hfma(a.y,         b.x, img_tmp);
#     return make_half2(real_tmp, img_tmp);
# }

export complex_mul
function complex_mul(a::Float16x2, b::Float16x2)
    alo, ahi = convert(NTuple{2,Float16}, a)
    blo, bhi = convert(NTuple{2,Float16}, b)
    rlo = muladd(-ahi, bhi, alo * blo)
    rhi = muladd(ahi, blo, alo * bhi)
    return Float16x2(rlo, rhi)
end
CUDA.@device_override function complex_mul(a::Float16x2, b::Float16x2)
    # See `__hcmadd`
    return Float16x2(
        LLVM.Interop.@asmcall(
            """{
                   .reg .f16 %r1re, %r1im, %r2re, %r2im, %r1imneg, %retmp, %r0re, %imtmp, %r0im;
                   mov.b32 {%r1re, %r1im}, \$1;
                   mov.b32 {%r2re, %r2im}, \$2;
                   mul.f16 %retmp, %r1re, %r2re;
                   mul.f16 %imtmp, %r1re, %r2im;
                   neg.f16 %r1imneg, %r1im;
                   fma.rn.f16 %r0re, %r1imneg, %r2im, %retmp;
                   fma.rn.f16 %r0im, %r1im, %r2re, %imtmp;
                   mov.b32 \$0, {%r0re, %r0im};
               }
               """,
            "=r,r,r",
            UInt32,
            Tuple{UInt32,UInt32},
            a.val,
            b.val,
        )
    )
end
export swapped_complex_mul
swapped_complex_mul(a::Float16x2, b::Float16x2) = reverse(complex_mul(reverse(a), reverse(b)))
CUDA.@device_override function swapped_complex_mul(a::Float16x2, b::Float16x2)
    # See `__hcmadd`
    return Float16x2(
        LLVM.Interop.@asmcall(
            """{
                   .reg .f16 %r1re, %r1im, %r2re, %r2im, %r1imneg, %retmp, %r0re, %imtmp, %r0im;
                   mov.b32 {%r1im, %r1re}, \$1;
                   mov.b32 {%r2im, %r2re}, \$2;
                   mul.f16 %retmp, %r1re, %r2re;
                   mul.f16 %imtmp, %r1re, %r2im;
                   neg.f16 %r1imneg, %r1im;
                   fma.rn.f16 %r0re, %r1imneg, %r2im, %retmp;
                   fma.rn.f16 %r0im, %r1im, %r2re, %imtmp;
                   mov.b32 \$0, {%r0im, %r0re};
               }
               """,
            "=r,r,r",
            UInt32,
            Tuple{UInt32,UInt32},
            a.val,
            b.val,
        )
    )
end
export complex_muladd
function complex_muladd(a::Float16x2, b::Float16x2, c::Float16x2)
    alo, ahi = convert(NTuple{2,Float16}, a)
    blo, bhi = convert(NTuple{2,Float16}, b)
    clo, chi = convert(NTuple{2,Float16}, c)
    rlo = muladd(-ahi, bhi, muladd(alo, blo, clo))
    rhi = muladd(ahi, blo, muladd(alo, bhi, chi))
    return Float16x2(rlo, rhi)
end
CUDA.@device_override function complex_muladd(a::Float16x2, b::Float16x2, c::Float16x2)
    return Float16x2(
        LLVM.Interop.@asmcall(
            """{
                   .reg .f16 %r1re, %r1im, %r2re, %r2im, %r3re, %r3im, %r1imneg, %retmp, %r0re, %imtmp, %r0im;
                   mov.b32 {%r1re, %r1im}, \$1;
                   mov.b32 {%r2re, %r2im}, \$2;
                   mov.b32 {%r3re, %r3im}, \$3;
                   fma.rn.f16 %retmp, %r1re, %r2re, %r3re;
                   fma.rn.f16 %imtmp, %r1re, %r2im, %r3im;
                   neg.f16 %r1imneg, %r1im;
                   fma.rn.f16 %r0re, %r1imneg, %r2im, %retmp;
                   fma.rn.f16 %r0im, %r1im, %r2re, %imtmp;
                   mov.b32 \$0, {%r0re, %r0im};
               }
               """,
            "=r,r,r,r",
            UInt32,
            Tuple{UInt32,UInt32,UInt32},
            a.val,
            b.val,
            c.val,
        )
    )
end
export swapped_complex_muladd
swapped_complex_muladd(a::Float16x2, b::Float16x2, c::Float16x2) = reverse(complex_muladd(reverse(a), reverse(b), reverse(c)))
CUDA.@device_override function swapped_complex_muladd(a::Float16x2, b::Float16x2, c::Float16x2)
    return Float16x2(
        LLVM.Interop.@asmcall(
            """{
                   .reg .f16 %r1re, %r1im, %r2re, %r2im, %r3re, %r3im, %r1imneg, %retmp, %r0re, %imtmp, %r0im;
                   mov.b32 {%r1im, %r1re}, \$1;
                   mov.b32 {%r2im, %r2re}, \$2;
                   mov.b32 {%r3im, %r3re}, \$3;
                   fma.rn.f16 %retmp, %r1re, %r2re, %r3re;
                   fma.rn.f16 %imtmp, %r1re, %r2im, %r3im;
                   neg.f16 %r1imneg, %r1im;
                   fma.rn.f16 %r0re, %r1imneg, %r2im, %retmp;
                   fma.rn.f16 %r0im, %r1im, %r2re, %imtmp;
                   mov.b32 \$0, {%r0im, %r0re};
               }
               """,
            "=r,r,r,r",
            UInt32,
            Tuple{UInt32,UInt32,UInt32},
            a.val,
            b.val,
            c.val,
        )
    )
end

################################################################################

function BFloat16x2(a1::BFloat16, a2::BFloat16)
    return BFloat16x2((reinterpret(UInt16, a1) % UInt32) << 0x00 | (reinterpret(UInt16, a2) % UInt32) << 0x10)
end
CUDA.@device_override function BFloat16x2(a1::BFloat16, a2::BFloat16)
    return BFloat16x2(prmt(reinterpret(UInt16, a1) % UInt32, reinterpret(UInt16, a2) % UInt32, 0x5410))
end
function BFloat16x2(a1::Float32, a2::Float32)
    return BFloat16x2((reinterpret(UInt16, BFloat16(a1)) % UInt32) << 0x00 | (reinterpret(UInt16, BFloat16(a2)) % UInt32) << 0x10)
end
CUDA.@device_override function BFloat16x2(a1::Float32, a2::Float32)
    return BFloat16x2(LLVM.Interop.@asmcall("cvt.rn.bf16x2.f32 \$0, \$2, \$1;", "=r,r,r", UInt32, Tuple{Float32,Float32}, a1, a2))
end
BFloat16x2(a1::Real, a2::Real) = BFloat16x2(BFloat16(a1), BFloat16(a2))
BFloat16x2(a::NTuple{2,<:Real}) = BFloat16x2(a...)

function Base.convert(::Type{NTuple{2,BFloat16}}, a::BFloat16x2)
    return (reinterpret(BFloat16, (a.val >> 0x00) % UInt16), reinterpret(BFloat16, (a.val >> 0x10) % UInt16))
end
function Base.convert(::Type{NTuple{2,Float32}}, a::BFloat16x2)
    return (Float32(reinterpret(BFloat16, (a.val >> 0x00) % UInt16)), Float32(reinterpret(BFloat16, (a.val >> 0x10) % UInt16)))
end
CUDA.@device_override function Base.convert(::Type{NTuple{2,Float32}}, a::BFloat16x2)
    # Converting from `bf16` requires a 16-bit register as input.
    # (Converting from `f16` is more lenient, it can have a 32-bit
    # register as input.)
    return (
        LLVM.Interop.@asmcall(
            """{
                   .reg .b16 %rlo, %rhi;
                   mov.b32 {%rlo, %rhi}, \$1;
                   cvt.f32.bf16 \$0, %rlo;
               }
               """,
            "=r,r",
            Float32,
            Tuple{UInt32},
            a.val,
        ),
        LLVM.Interop.@asmcall(
            """{
                   .reg .b16 %rlo, %rhi;
                   mov.b32 {%rlo, %rhi}, \$1;
                   cvt.f32.bf16 \$0, %rhi;
               }
               """,
            "=r,r",
            Float32,
            Tuple{UInt32},
            a.val,
        ),
    )::NTuple{2,Float32}
end

Base.show(io::IO, a::BFloat16x2) = print(io, "BFloat16x2", convert(NTuple{2,Float32}, a))

Base.reverse(a::BFloat16x2) = BFloat16x2(reverse(convert(NTuple{2,BFloat16}, a)))
CUDA.@device_override function Base.reverse(a::BFloat16x2)
    return BFloat16x2(LLVM.Interop.@asmcall(
        """{
               .reg .b16 %lo, %hi;
               mov.b32 {%lo, %hi}, \$1;
               mov.b32 \$0, {%hi, %lo};
           }
           """,
        "=r,r",
        UInt32,
        Tuple{UInt32},
        a.val,
    ))
end

# Base.zero(::Type{BFloat16x2}) = BFloat16x2(0.0f0, 0.0f0)
Base.zero(::Type{BFloat16x2}) = BFloat16x2(UInt32(0))
Base.zero(::BFloat16x2) = zero(BFloat16x2)
Base.iszero(a::BFloat16x2) = a == zero(a)
Random.rand(rng::AbstractRNG, ::Random.SamplerType{BFloat16x2}) = BFloat16x2(rand(rng, Float32), rand(rng, Float32))

Base.:+(a::BFloat16x2) = a
Base.:-(a::BFloat16x2) = BFloat16x2(a.val ⊻ 0x80008000)
CUDA.@device_override function Base.:-(a::BFloat16x2)
    return BFloat16x2(LLVM.Interop.@asmcall("neg.bf16x2 \$0, \$1;", "=r,r", UInt32, Tuple{UInt32}, a.val))
end

export negate1
"""Negate a[1]"""
# negate1(a::BFloat16x2) = BFloat16x2(a.val ⊻ 0x00008000)
negate1(a::BFloat16x2) = a * BFloat16x2(-1.0f0, 1.0f0)

export negate2
"""Negate a[2]"""
# negate2(a::BFloat16x2) = BFloat16x2(a.val ⊻ 0x80000000)
negate2(a::BFloat16x2) = a * BFloat16x2(1.0f0, -1.0f0)

Base.abs(a::BFloat16x2) = BFloat16x2(a.val & 0x7fff7fff)
CUDA.@device_override function Base.abs(a::BFloat16x2)
    return BFloat16x2(LLVM.Interop.@asmcall("abs.bf16x2 \$0, \$1;", "=r,r", UInt32, Tuple{UInt32}, a.val))
end

function Base.:+(a::BFloat16x2, b::BFloat16x2)
    alo, ahi = convert(NTuple{2,BFloat16}, a)
    blo, bhi = convert(NTuple{2,BFloat16}, b)
    return BFloat16x2(alo + blo, ahi + bhi)
end
CUDA.@device_override function Base.:+(a::BFloat16x2, b::BFloat16x2)
    # This requires sm_90
    # return BFloat16x2(LLVM.Interop.@asmcall("add.rn.bf16x2 \$0, \$1, \$2;", "=r,r,r", UInt32, Tuple{UInt32,UInt32}, a.val, b.val))
    return muladd(BFloat16x2(1.0f0, 1.0f0), a, b)
end
function Base.:-(a::BFloat16x2, b::BFloat16x2)
    alo, ahi = convert(NTuple{2,BFloat16}, a)
    blo, bhi = convert(NTuple{2,BFloat16}, b)
    return BFloat16x2(alo - blo, ahi - bhi)
end
CUDA.@device_override function Base.:-(a::BFloat16x2, b::BFloat16x2)
    # This requires sm_90
    # return BFloat16x2(LLVM.Interop.@asmcall("sub.rn.bf16x2 \$0, \$1, \$2;", "=r,r,r", UInt32, Tuple{UInt32,UInt32}, a.val, b.val))
    return muladd(BFloat16x2(-1.0f0, -1.0f0), b, a)
end
function Base.:*(a::BFloat16x2, b::BFloat16x2)
    alo, ahi = convert(NTuple{2,BFloat16}, a)
    blo, bhi = convert(NTuple{2,BFloat16}, b)
    return BFloat16x2(alo * blo, ahi * bhi)
end
CUDA.@device_override function Base.:*(a::BFloat16x2, b::BFloat16x2)
    # This requires sm_90
    # return BFloat16x2(LLVM.Interop.@asmcall("mul.rn.bf16x2 \$0, \$1, \$2;", "=r,r,r", UInt32, Tuple{UInt32,UInt32}, a.val, b.val))
    return muladd(a, b, BFloat16x2(-0.0f0, -0.0f0))
end
function Base.muladd(a::BFloat16x2, b::BFloat16x2, c::BFloat16x2)
    alo, ahi = convert(NTuple{2,BFloat16}, a)
    blo, bhi = convert(NTuple{2,BFloat16}, b)
    clo, chi = convert(NTuple{2,BFloat16}, c)
    return BFloat16x2(muladd(alo, blo, clo), muladd(ahi, bhi, chi))
end
CUDA.@device_override function Base.muladd(a::BFloat16x2, b::BFloat16x2, c::BFloat16x2)
    return BFloat16x2(
        LLVM.Interop.@asmcall(
            "fma.rn.bf16x2 \$0, \$1, \$2, \$3;", "=r,r,r,r", UInt32, Tuple{UInt32,UInt32,UInt32}, a.val, b.val, c.val
        )
    )
end
export muladd_sat
function muladd_sat(a::BFloat16x2, b::BFloat16x2, c::BFloat16x2)
    return BFloat16x2(
        LLVM.Interop.@asmcall(
            "fma.rn.sat.bf16x2 \$0, \$1, \$2, \$3;", "=r,r,r,r", UInt32, Tuple{UInt32,UInt32,UInt32}, a.val, b.val, c.val
        )
    )
end
function Base.max(a::BFloat16x2, b::BFloat16x2)
    alo, ahi = convert(NTuple{2,BFloat16}, a)
    blo, bhi = convert(NTuple{2,BFloat16}, b)
    return BFloat16x2(max(alo, blo), max(ahi, bhi))
end
CUDA.@device_override function Base.max(a::BFloat16x2, b::BFloat16x2)
    return BFloat16x2(LLVM.Interop.@asmcall("max.bf16x2 \$0, \$1, \$2;", "=r,r,r", UInt32, Tuple{UInt32,UInt32}, a.val, b.val))
end
function Base.min(a::BFloat16x2, b::BFloat16x2)
    alo, ahi = convert(NTuple{2,BFloat16}, a)
    blo, bhi = convert(NTuple{2,BFloat16}, b)
    return BFloat16x2(min(alo, blo), min(ahi, bhi))
end
CUDA.@device_override function Base.min(a::BFloat16x2, b::BFloat16x2)
    return BFloat16x2(LLVM.Interop.@asmcall("min.bf16x2 \$0, \$1, \$2;", "=r,r,r", UInt32, Tuple{UInt32,UInt32}, a.val, b.val))
end
Base.clamp(a::BFloat16x2, alo::BFloat16x2, ahi::BFloat16x2) = min(max(a, alo), ahi)

export complex_mul
function complex_mul(a::BFloat16x2, b::BFloat16x2)
    alo, ahi = convert(NTuple{2,BFloat16}, a)
    blo, bhi = convert(NTuple{2,BFloat16}, b)
    rlo = muladd(-ahi, bhi, alo * blo)
    rhi = muladd(ahi, blo, alo * bhi)
    return BFloat16x2(rlo, rhi)
end
CUDA.@device_override complex_mul(a::BFloat16x2, b::BFloat16x2) = complex_muladd(a, b, zero(BFloat16x2))
export swapped_complex_mul
swapped_complex_mul(a::BFloat16x2, b::BFloat16x2) = reverse(complex_mul(reverse(a), reverse(b)))
export complex_muladd
function complex_muladd(a::BFloat16x2, b::BFloat16x2, c::BFloat16x2)
    alo, ahi = convert(NTuple{2,BFloat16}, a)
    blo, bhi = convert(NTuple{2,BFloat16}, b)
    clo, chi = convert(NTuple{2,BFloat16}, c)
    rlo = muladd(-ahi, bhi, muladd(alo, blo, clo))
    rhi = muladd(ahi, blo, muladd(alo, bhi, chi))
    return BFloat16x2(rlo, rhi)
end
CUDA.@device_override function complex_muladd(a::BFloat16x2, b::BFloat16x2, c::BFloat16x2)
    return BFloat16x2(
        LLVM.Interop.@asmcall(
            """{
                   .reg .b16 %r1re, %r1im, %r2re, %r2im, %r3re, %r3im, %r1imneg, %retmp, %r0re, %imtmp, %r0im;
                   mov.b32 {%r1re, %r1im}, \$1;
                   mov.b32 {%r2re, %r2im}, \$2;
                   mov.b32 {%r3re, %r3im}, \$3;
                   fma.rn.bf16 %retmp, %r1re, %r2re, %r3re;
                   neg.bf16 %r1imneg, %r1im;
                   fma.rn.bf16 %r0re, %r1imneg, %r2im, %retmp;
                   fma.rn.bf16 %imtmp, %r1re, %r2im, %r3im;
                   fma.rn.bf16 %r0im, %r1im, %r2re, %imtmp;
                   mov.b32 \$0, {%r0re, %r0im};
               }
               """,
            "=r,r,r,r",
            UInt32,
            Tuple{UInt32,UInt32,UInt32},
            a.val,
            b.val,
            c.val,
        )
    )
end
export swapped_complex_muladd
swapped_complex_muladd(a::BFloat16x2, b::BFloat16x2, c::BFloat16x2) = reverse(complex_muladd(reverse(a), reverse(b), reverse(c)))

################################################################################

const Types8bit = Int4x2
const Types32bit = Union{Int4x8,Int8x4,Int16x2,Float16x2,BFloat16x2}

Base.reinterpret(::Type{I}, a::Int4x2) where {I<:Union{Int8,UInt8}} = reinterpret(I, a.val)
Base.reinterpret(::Type{Int4x2}, a::Union{Int8,UInt8}) = Int4x2(reinterpret(UInt8, a))
Base.reinterpret(::Type{T}, a::Int4x2) where {T<:Types8bit} = T(a.val)

Base.reinterpret(::Type{I}, a::Types32bit) where {I<:Union{Int32,UInt32}} = reinterpret(I, a.val)
Base.reinterpret(::Type{T}, a::Union{Int32,UInt32}) where {T<:Types32bit} = T(reinterpret(a, UInt32))
Base.reinterpret(::Type{T}, a::Types32bit) where {T<:Types32bit} = T(a.val)

################################################################################

# Note: `Float(1024 + i)` has the bit pattern for `i` in the lowermost bits. This works for 0 ≤ i < 1024.
# Note: `Float(1536 + i)` has the bit pattern for `i` in the lowermost bits. This works for -512 ≤ i < 512.

# From/to Int16

Float16x2(a::Int16x2) = Float16x2(Float16.(convert(NTuple{2,Int16}, a)))

Int16x2(a::Float16x2) = Int16x2(round.(Int16, convert(NTuple{2,Float16}, a)))
CUDA.@device_override function Int16x2(a::Float16x2)
    return Int16x2(LLVM.Interop.@asmcall(
        """{
               .reg .f16 %r1lo, %r1hi;
               .reg .s16 %r0lo, %r0hi;
               mov.b32 {%r1lo, %r1hi}, \$1;
               cvt.rni.s16.f16 %r0lo, %r1lo;
               cvt.rni.s16.f16 %r0hi, %r1hi;
               mov.b32 \$0, {%r0lo, %r0hi};
           }
           """,
        "=r,r",
        UInt32,
        Tuple{UInt32},
        a.val,
    ))
end

# From/to Int8

# const lop3_and_xor_lut = Val(make_lop3_lut((a, b, c) -> (a & b) ⊻ c))

function Base.convert(::Type{NTuple{2,Float16x2}}, a::Int8x4)
    offset = Float16x2(1536, 1536)
    alo, ahi = convert(NTuple{2,Int16x2}, a)
    blo = reinterpret(Float16x2, reinterpret(Int16x2, offset) + alo) - offset
    bhi = reinterpret(Float16x2, reinterpret(Int16x2, offset) + ahi) - offset
    return (blo, bhi)
end
CUDA.@device_override function Base.convert(::Type{NTuple{2,Float16x2}}, a::Int8x4)
    offset = Float16x2(0x400, 0x400)
    # alo = lop3(a.val, 0x00ff00ff, 0x00800080, lop3_and_xor_lut)::UInt32
    # blo = Float16x2(offset.val + alo) - (offset + Float16x2(0x80, 0x80))
    # ahi = lop3(a.val >> 0x08, 0x00ff00ff, 0x00800080, lop3_and_xor_lut)::UInt32
    # bhi = Float16x2(offset.val + ahi) - (offset + Float16x2(0x80, 0x80))
    a′ = Int8x4(a.val ⊻ 0x80808080)
    blo = Float16x2(prmt(a′.val, offset.val, 0x7250)) - (offset + Float16x2(0x80, 0x80))
    bhi = Float16x2(prmt(a′.val, offset.val, 0x7351)) - (offset + Float16x2(0x80, 0x80))
    return (blo, bhi)
end

Int8x4(a::NTuple{2,Float16x2}) = Int8x4(Int16x2.(a))
CUDA.@device_override function Int8x4(a::NTuple{2,Float16x2})
    offset = Float16x2(0x400, 0x400)
    alo, ahi = a
    blo = alo + (offset + Float16x2(0x80, 0x80))
    bhi = ahi + (offset + Float16x2(0x80, 0x80))
    b = prmt(blo.val, bhi.val, 0x6240)
    return Int8x4(b ⊻ 0x80808080)
end

# From/to Int4

Float16x2(a::Int4x2) = Float16x2(Int16x2(a))

Int4x2(a::Float16x2) = Int4x2(Int16x2(a))

@inline function Base.convert(::Type{NTuple{4,Float16x2}}, a::Int4x8)
    # Note: `Float(1536 + i)` has the bit pattern for `i` in the lowermost bits. This works for -512 ≤ i < 512.
    offset = Float16x2(1536, 1536)
    a1, a2, a3, a4 = convert(NTuple{4,Int16x2}, a)
    b1 = reinterpret(Float16x2, reinterpret(Int16x2, offset) + a1) - offset
    b2 = reinterpret(Float16x2, reinterpret(Int16x2, offset) + a2) - offset
    b3 = reinterpret(Float16x2, reinterpret(Int16x2, offset) + a3) - offset
    b4 = reinterpret(Float16x2, reinterpret(Int16x2, offset) + a4) - offset
    return (b1, b2, b3, b4)
end
CUDA.@device_override @inline function Base.convert(::Type{NTuple{4,Float16x2}}, a::Int4x8)
    offset1 = Float16x2(0x0400, 0x0400)
    offset2 = Float16x2(0x0040, 0x0040)
    a′ = Int8x4(a.val ⊻ 0x88888888)
    b1 = Float16x2(bitifelse(0x000f000f, a′.val >> 0x00, offset1.val)) - (offset1 + Float16x2(0x08, 0x08))
    b2 = Float16x2(bitifelse(0x00f000f0, a′.val >> 0x00, offset2.val)) - (offset2 + Float16x2(0x08, 0x08))
    b3 = Float16x2(bitifelse(0x000f000f, a′.val >> 0x08, offset1.val)) - (offset1 + Float16x2(0x08, 0x08))
    b4 = Float16x2(bitifelse(0x00f000f0, a′.val >> 0x08, offset2.val)) - (offset2 + Float16x2(0x08, 0x08))
    return (b1, b2, b3, b4)
end

Int4x8(a::NTuple{4,Float16x2}) = Int4x8(Int16x2.(a))
CUDA.@device_override @inline function Int4x8(a::NTuple{4,Float16x2})
    offset = Float16x2(0x0400, 0x0400)
    a1, a2, a3, a4 = a
    b1 = a1 + (offset + Float16x2(0x8, 0x8))
    b2 = a2 + (offset + Float16x2(0x8, 0x8))
    b3 = a3 + (offset + Float16x2(0x8, 0x8))
    b4 = a4 + (offset + Float16x2(0x8, 0x8))
    b13 = prmt(b1.val, b3.val, 0x6240)
    b24 = prmt(b2.val, b4.val, 0x6240)
    return Int4x8(bitifelse(0x0f0f0f0f, b13 << 0x00, b24 << 0x04) ⊻ 0x88888888)
end

################################################################################

# Note: `BFloat(128 + i)` has the bit pattern for `i` in the lowermost bits. This works for 0 ≤ i < 128.
# Note: `BFloat(192 + i)` has the bit pattern for `i` in the lowermost bits. This works for -64 ≤ i < 64.

BFloat16x2(a::Int16x2) = BFloat16x2(BFloat16.(convert(NTuple{2,Int16}, a)))

Int16x2(a::BFloat16x2) = Int16x2(round.(Int16, Float32.(convert(NTuple{2,BFloat16}, a))))

end

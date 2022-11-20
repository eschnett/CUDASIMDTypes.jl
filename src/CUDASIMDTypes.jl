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

export Int4x2
"""
    struct Int4x2

A SIMD type holding 2 4-bit integers in a combined 8-bit value.
"""
struct Int4x2
    val::UInt8
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

Int4x2(a1::Int32, a2::Int32) = Int4x2((a1 << 0x00) & 0x0f | (a2 << 0x04) & 0xf0)

const xor_and_lut = make_lop3_lut((a, b, c) -> (a ⊻ b) & c)
Base.convert(::Type{Int4x2}, a::NTuple{2,Int8}) = Int4x2(a[1], a[2])
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
function Base.convert(::Type{NTuple{2,Int32}}, a::Int4x2)
    alo8, ahi8 = convert(NTuple{2,Int8}, a)
    alo32 = convert(Int32, alo8)
    ahi32 = convert(Int32, ahi8)
    return (alo32, ahi32)
end

Base.show(io::IO, a::Int4x2) = print(io, convert(NTuple{2,Int32}, a))

Base.length(::Int4x2) = 1

Base.zero(::Type{Int4x2}) = Int4x2(Int8(0))
Random.rand(rng::AbstractRNG, ::Random.SamplerType{Int4x2}) = Int4x2(rand(UInt8))

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

################################################################################

function Int4x8(a1::Int32, a2::Int32, a3::Int32, a4::Int32, a5::Int32, a6::Int32, a7::Int32, a8::Int32)
    return Int4x8(bitifelse(0x0f0f0f0f, Int8x4(a1, a3, a5, a7).val << 0x00, Int8x4(a2, a4, a6, a8).val << 0x04))
end

Base.convert(::Type{Int4x8}, a::NTuple{2,Int8x4}) = Int4x8(bitifelse(0x0f0f0f0f, a[1].val, a[2].val << 0x04))
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
function Base.convert(::Type{NTuple{8,Int32}}, a::Int4x8)
    alo8, ahi8 = convert(NTuple{2,Int8x4}, a)
    alo32 = convert(NTuple{4,Int32}, alo8)
    ahi32 = convert(NTuple{4,Int32}, ahi8)
    return (alo32[1], ahi32[1], alo32[2], ahi32[2], alo32[3], ahi32[3], alo32[4], ahi32[4])
end

Base.show(io::IO, a::Int4x8) = print(io, convert(NTuple{8,Int32}, a))

Base.zero(::Type{Int4x8}) = Int4x8(Int32(0))
Random.rand(rng::AbstractRNG, ::Random.SamplerType{Int4x8}) = Int4x8(rand(UInt32))

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

################################################################################

function Int8x4(a1::Int32, a2::Int32, a3::Int32, a4::Int32)
    return Int8x4(
        (a4 % UInt8 % UInt32) << 0x18 |
        (a3 % UInt8 % UInt32) << 0x10 |
        (a2 % UInt8 % UInt32) << 0x08 |
        (a1 % UInt8 % UInt32) << 0x00,
    )
end
CUDA.@device_override Int8x4(a1::Int32, a2::Int32, a3::Int32, a4::Int32) = Int8x4(cvt_pack_s8(a2, a1, cvt_pack_s8(a4, a3)))

function Base.convert(::Type{Int8x4}, a::NTuple{2,Int16x2})
    return Int8x4(
        (a[1].val >>> 0x00) % Int32, (a[1].val >>> 0x10) % Int32, (a[2].val >>> 0x00) % Int32, (a[2].val >>> 0x10) % Int32
    )
end
CUDA.@device_override Base.convert(::Type{Int8x4}, a::NTuple{2,Int16x2}) = Int8x4(prmt(a[1].val, a[2].val, 0x6240) % UInt32)

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

Base.convert(::Type{Int8x4}, a::NTuple{4,Int32}) = Int8x4(a[1], a[2], a[3], a[4])
CUDA.@device_override Base.convert(::Type{Int8x4}, a::NTuple{4,Int32}) = cvt_pack_s8(a[2], a[1], cvt_pack_s8(a[4], a[3]))

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

Base.show(io::IO, a::Int8x4) = print(io, convert(NTuple{4,Int32}, a))

Base.zero(::Type{Int8x4}) = Int8x4(Int32(0))
Random.rand(rng::AbstractRNG, ::Random.SamplerType{Int8x4}) = Int8x4(rand(UInt32))

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

################################################################################

Int16x2(a1::Int32, a2::Int32) = Int16x2(((a1 % UInt32) & 0xffff) << 0x00 | ((a2 % UInt32) & 0xffff) << 0x10)
CUDA.@device_override Int16x2(a1::Int32, a2::Int32) = Int16x2(cvt_pack_s16(a2, a1))

Base.convert(::Type{Int16x2}, a::NTuple{2,Int32}) = Int16x2(a[1], a[2])
function Base.convert(::Type{NTuple{2,Int32}}, a::Int16x2)
    return ((a.val >>> 0x00) % Int16 % Int32, (a.val >>> 0x10) % Int16 % Int32)::NTuple{2,Int32}
end

Base.show(io::IO, a::Int16x2) = print(io, convert(NTuple{2,Int32}, a))

Base.zero(::Type{Int16x2}) = Int16x2(Int32(0))
Random.rand(rng::AbstractRNG, ::Random.SamplerType{Int16x2}) = Int16x2(rand(UInt32))

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

function Float16x2(a1::Float32, a2::Float32)
    return Float16x2((reinterpret(UInt16, Float16(a1)) % UInt32) << 0x00 | (reinterpret(UInt16, Float16(a2)) % UInt32) << 0x10)
end
CUDA.@device_override function Float16x2(a1::Float32, a2::Float32)
    return Float16x2(LLVM.Interop.@asmcall("cvt.rn.f16x2.f32 \$0, \$1, \$2;", "=r,r,r", UInt32, Tuple{Float32,Float32}, a2, a1))
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

function Float16x2(a1::Float16, a2::Float16)
    return Float16x2((reinterpret(UInt16, a1) % UInt32) << 0x00 | (reinterpret(UInt16, a2) % UInt32) << 0x10)
end
function Base.convert(::Type{NTuple{2,Float16}}, a::Float16x2)
    return (reinterpret(Float16, (a.val >> 0x00) % UInt16), reinterpret(Float16, (a.val >> 0x10) % UInt16))
end

Base.show(io::IO, a::Float16x2) = print(io, convert(NTuple{2,Float32}, a))

Base.zero(::Type{Float16x2}) = Float16x2(0.0f0, 0.0f0)

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

################################################################################

function BFloat16x2(a1::Float32, a2::Float32)
    return BFloat16x2((reinterpret(UInt16, BFloat16(a1)) % UInt32) << 0x00 | (reinterpret(UInt16, BFloat16(a2)) % UInt32) << 0x10)
end
CUDA.@device_override function BFloat16x2(a1::Float32, a2::Float32)
    return BFloat16x2(LLVM.Interop.@asmcall("cvt.rn.bf16x2.f32 \$0, \$1, \$2;", "=r,r,r", UInt32, Tuple{Float32,Float32}, a2, a1))
end

function Base.convert(::Type{NTuple{2,Float32}}, a::BFloat16x2)
    return (Float32(reinterpret(BFloat16, (a.val >> 0x00) % UInt16)), Float32(reinterpret(BFloat16, (a.val >> 0x10) % UInt16)))
end
CUDA.@device_override function Base.convert(::Type{NTuple{2,Float32}}, a::BFloat16x2)
    # Converting from `bf16` requires a 16-bit register as input.
    # (Converting from `f16` is more lenient, it can have a 32-bit
    # register as input.)
    return (
        LLVM.Interop.@asmcall("cvt.f32.bf16 \$0, \$1;", "=r,h", Float32, Tuple{UInt16}, (a.val >> 0x00) % UInt16),
        LLVM.Interop.@asmcall("cvt.f32.bf16 \$0, \$1;", "=r,h", Float32, Tuple{UInt16}, (a.val >> 0x10) % UInt16)
    )::NTuple{2,Float32}
end

function BFloat16x2(a1::BFloat16, a2::BFloat16)
    return BFloat16x2((reinterpret(UInt16, a1) % UInt32) << 0x00 | (reinterpret(UInt16, a2) % UInt32) << 0x10)
end
function Base.convert(::Type{NTuple{2,BFloat16}}, a::BFloat16x2)
    return (reinterpret(BFloat16, (a.val >> 0x00) % UInt16), reinterpret(BFloat16, (a.val >> 0x10) % UInt16))
end

Base.show(io::IO, a::BFloat16x2) = print(io, convert(NTuple{2,Float32}, a))

Base.zero(::Type{BFloat16x2}) = BFloat16x2(0.0f0, 0.0f0)

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

end

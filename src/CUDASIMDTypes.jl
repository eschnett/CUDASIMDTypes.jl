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

export Int4x2
"""
    struct Int4x2

A SIMD type holding 2 4-bit integers in a combined 8-bit value.
"""
struct Int4x2
    val::UInt8
end

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

end

module CUDASIMDTypes

using BFloat16s
using CUDA
using LLVM

const SmallInt = Union{Int8,Int16,Int32,UInt8,UInt16,UInt32}

################################################################################

export prmt

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

function prmt(a::T, b::T, op::SmallInt) where {T<:SmallInt}
    return prmt(a % UInt32, b % UInt32, op % UInt32)::UInt32 % T
end

################################################################################

# export lop3
# function lop3(a::UInt32, b::UInt32, c::UInt32, op::UInt32)
#     z = UInt32(0)
#     return (ifelse(op & 0x01 ≠ 0, ~z, z) & ~a & ~b & ~c) |
#            (ifelse(op & 0x02 ≠ 0, ~z, z) & ~a & ~b & c) |
#            (ifelse(op & 0x04 ≠ 0, ~z, z) & ~a & b & ~c) |
#            (ifelse(op & 0x08 ≠ 0, ~z, z) & ~a & b & c) |
#            (ifelse(op & 0x10 ≠ 0, ~z, z) & a & ~b & ~c) |
#            (ifelse(op & 0x20 ≠ 0, ~z, z) & a & ~b & c) |
#            (ifelse(op & 0x40 ≠ 0, ~z, z) & a & b & ~c) |
#            (ifelse(op & 0x80 ≠ 0, ~z, z) & a & b & c)
# end
# CUDA.@device_override function lop3(x::UInt32, y::UInt32, z::UInt32, op::UInt32)
#     LLVM.Interop.@asmcall(
#         "lop3.b32 \$0, \$1, \$2, \$3, \$4;", "=r,r,r,r,i", UInt32, Tuple{UInt32,UInt32,UInt32,UInt32}, x, y, z, op
#     )
# end
# function lop3(x::Int_UInt_8_16_32, y::Int_UInt_8_16_32, z::Int_UInt_8_16_32, op::Int_UInt_8_16_32)
#     return lop3(x % UInt32, y % UInt32, z % UInt32, op % UInt32)::UInt32
# end
# 
# export make_lop3_lut
# function make_lop3_lut(f)
#     ta = 0xf0
#     tb = 0xcc
#     tc = 0xaa
#     lut = f(ta, tb, tc)::UInt8
#     return lut
# end

end

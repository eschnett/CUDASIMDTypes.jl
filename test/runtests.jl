using BFloat16s
using CUDA
using CUDASIMDTypes
using Random
using Test

################################################################################

const CR = "\r"
const ESC = "\e"
const CSI = "$(ESC)["
const EL = "$(CSI)K"

make_int2(x::Integer) = ((x + 2) & 0x3 - 2) % Int8
make_int4(x::Integer) = ((x + 8) & 0xf - 8) % Int8

tuple2complex(ab::NTuple{2}) = Complex(ab[1], ab[2])
complex2tuple(c::Complex) = (c.re, c.im)
tuple_complex_mul(x::NTuple{2}, y::NTuple{2}) = complex2tuple(tuple2complex(x) * tuple2complex(y))
function tuple_complex_muladd(x::NTuple{2}, y::NTuple{2}, z::NTuple{2})
    return complex2tuple(muladd(tuple2complex(x), tuple2complex(y), tuple2complex(z)))
end
function tuple_swapped_complex_mul(x::NTuple{2}, y::NTuple{2})
    return reverse(complex2tuple(tuple2complex(reverse(x)) * tuple2complex(reverse(y))))
end
function tuple_swapped_complex_muladd(x::NTuple{2}, y::NTuple{2}, z::NTuple{2})
    return reverse(complex2tuple(muladd(tuple2complex(reverse(x)), tuple2complex(reverse(y)), tuple2complex(reverse(z)))))
end

clamp1(a, b, c) = clamp(a, min(b, c), max(b, c))

almost_half(x::T) where {T<:Real} = max(T(0.5) - eps(x), zero(x))
inc(x::T) where {T<:Real} = x + almost_half(x)
dec(x::T) where {T<:Real} = x - almost_half(x)
inc(x::Float16x2) = Float16x2(inc.(convert(NTuple{2,Float16}, x)))
dec(x::Float16x2) = Float16x2(dec.(convert(NTuple{2,Float16}, x)))
inc(x::BFloat16x2) = BFloat16x2(inc.(convert(NTuple{2,BFloat16}, x)))
dec(x::BFloat16x2) = BFloat16x2(dec.(convert(NTuple{2,BFloat16}, x)))

################################################################################

if CUDA.functional()
    println("Testing both CPU and CUDA implementations")
else
    println("Testing only CPU implementation (CUDA is not available)")
end

run_on_cpu(f, inputs...) = f.(inputs...)

function run_on_cuda!(f, output, inputs...)
    function kernel!(cuda_output, cuda_inputs...)
        local n1 = threadIdx().x
        local n2 = blockIdx().x
        local n = (n2 - 1) * blockDim().x + (n1 - 1) + 1
        cuda_output[n] = f(getindex.(cuda_inputs, n)...)
        return nothing
    end
    cuda_inputs = CuArray.(inputs)
    cuda_output = CuArray(output)
    nitems = length(cuda_output)
    if nitems ≥ 1024
        nthreads = 1024
        @assert nitems % nthreads == 0
        nblocks = nitems ÷ nthreads
    else
        nblocks = 1
        nthreads = nitems
    end
    @cuda threads = nthreads blocks = nblocks kernel!(cuda_output, cuda_inputs...)
    synchronize()
    output .= Array(cuda_output)
    return nothing
end

################################################################################

#TODO Random.seed!(0)
#TODO @testset "prmt T=$T" for T in [UInt32, Int32]
#TODO     iters = 1000
#TODO 
#TODO     a = rand(T, iters)
#TODO     b = rand(T, iters)
#TODO     op = rand(UInt16, iters)
#TODO 
#TODO     r1 = run_on_cpu(prmt, a, b, op)
#TODO 
#TODO     if CUDA.functional()
#TODO         r2 = similar(r1)
#TODO         run_on_cuda!(prmt, r2, a, b, op)
#TODO         @test r2 == r1
#TODO     end
#TODO end
#TODO 
#TODO Random.seed!(0)
#TODO @testset "lop3 T=$T" for T in [UInt32, Int32]
#TODO     iters = 100
#TODO     for lut in rand(UInt8, 10)
#TODO         LUT = Val(lut)
#TODO 
#TODO         a = rand(T, iters)
#TODO         b = rand(T, iters)
#TODO         c = rand(T, iters)
#TODO 
#TODO         lop3′(a, b, c) = lop3(a, b, c, LUT)
#TODO 
#TODO         r1 = run_on_cpu(lop3′, a, b, c)
#TODO 
#TODO         if CUDA.functional()
#TODO             r2 = similar(r1)
#TODO             run_on_cuda!(lop3′, r2, a, b, c)
#TODO             @test r2 == r1
#TODO         end
#TODO     end
#TODO end
#TODO 
#TODO Random.seed!(0)
#TODO @testset "bitifelse T=$T" for T in [UInt32, Int32]
#TODO     iters = 1000
#TODO 
#TODO     cond = rand(T, iters)
#TODO     x = rand(T, iters)
#TODO     y = rand(T, iters)
#TODO 
#TODO     r1 = run_on_cpu(bitifelse, cond, x, y)
#TODO 
#TODO     if CUDA.functional()
#TODO         r2 = similar(r1)
#TODO         run_on_cuda!(bitifelse, r2, cond, x, y)
#TODO         @test r2 == r1
#TODO     end
#TODO end
#TODO 
#TODO Random.seed!(0)
#TODO @testset "Int2x4" begin
#TODO     # Test exhaustively for x and y
#TODO     x1 = Int32[]
#TODO     x2 = Int32[]
#TODO     x3 = Int32[]
#TODO     x4 = Int32[]
#TODO     y1 = Int32[]
#TODO     y2 = Int32[]
#TODO     y3 = Int32[]
#TODO     y4 = Int32[]
#TODO     z1 = Int32[]
#TODO     z2 = Int32[]
#TODO     z3 = Int32[]
#TODO     z4 = Int32[]
#TODO     x = Int2x4[]
#TODO     y = Int2x4[]
#TODO     z = Int2x4[]
#TODO 
#TODO     for x01 in (-Int32(2)):(+Int32(1)),
#TODO         x02 in (-Int32(2)):(+Int32(1)),
#TODO         x03 in (-Int32(2)):(+Int32(1)),
#TODO         x04 in (-Int32(2)):(+Int32(1)),
#TODO         y01 in (-Int32(2)):(+Int32(1)),
#TODO         y02 in (-Int32(2)):(+Int32(1)),
#TODO         y03 in (-Int32(2)):(+Int32(1)),
#TODO         y04 in (-Int32(2)):(+Int32(1))
#TODO 
#TODO         z01 = rand((-Int32(2)):(+Int32(1)))
#TODO         z02 = rand((-Int32(2)):(+Int32(1)))
#TODO         z03 = rand((-Int32(2)):(+Int32(1)))
#TODO         z04 = rand((-Int32(2)):(+Int32(1)))
#TODO 
#TODO         push!(x1, x01)
#TODO         push!(x2, x02)
#TODO         push!(x3, x03)
#TODO         push!(x4, x04)
#TODO         push!(y1, y01)
#TODO         push!(y2, y02)
#TODO         push!(y3, y03)
#TODO         push!(y4, y04)
#TODO         push!(z1, z01)
#TODO         push!(z2, z02)
#TODO         push!(z3, z03)
#TODO         push!(z4, z04)
#TODO         push!(x, Int2x4(x01, x02, x03, x04))
#TODO         push!(y, Int2x4(y01, y02, y03, y04))
#TODO         push!(z, Int2x4(z01, z02, z03, z04))
#TODO     end
#TODO 
#TODO     function compare(fl, fr)
#TODO         rcpul = run_on_cpu(fl, x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z)
#TODO         rcpur = run_on_cpu(fr, x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z)
#TODO         @test rcpul == rcpur
#TODO         @assert rcpul == rcpur
#TODO         if CUDA.functional()
#TODO             rcudal = similar(rcpul)
#TODO             rcudar = similar(rcpur)
#TODO             run_on_cuda!(fl, rcudal, x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z)
#TODO             run_on_cuda!(fr, rcudar, x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z)
#TODO             @test rcudal == rcudar
#TODO             @test rcudal == rcpul
#TODO             @test rcudar == rcpur
#TODO         end
#TODO         print(".")
#TODO         flush(stdout)
#TODO         return nothing
#TODO     end
#TODO 
#TODO     # Test constructors
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> Int2x4((Int8(x1), Int8(x2), Int8(x3), Int8(x4))),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> x,
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> Int2x4((Int16(x1), Int16(x2), Int16(x3), Int16(x4))),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> x,
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> Int2x4((Int32(x1), Int32(x2), Int32(x3), Int32(x4))),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> x,
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> Int2x4((Int64(x1), Int64(x2), Int64(x3), Int64(x4))),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> x,
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int8}, x),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (x1, x2, x3, x4),
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int16}, x),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (x1, x2, x3, x4),
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, x),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (x1, x2, x3, x4),
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int64}, x),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (x1, x2, x3, x4),
#TODO     )
#TODO 
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> Int2x4(Int8(x1), Int8(x2), Int8(x3), Int8(x4)),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> x,
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> Int2x4(Int16(x1), Int16(x2), Int16(x3), Int16(x4)),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> x,
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> Int2x4(Int32(x1), Int32(x2), Int32(x3), Int32(x4)),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> x,
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> Int2x4(Int64(x1), Int64(x2), Int64(x3), Int64(x4)),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> x,
#TODO     )
#TODO 
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> Int8x4(x),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> Int8x4(x1, x2, x3, x4),
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> Int2x4(Int8x4(x1, x2, x3, x4)),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> x,
#TODO     )
#TODO 
#TODO     # Test output
#TODO     @test string.(x) == "Int2x4" .* string.(tuple.(x1, x2, x3, x4))
#TODO 
#TODO     # Test half-nibble ordering
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, Int2x4((x1 & 0x03) << 0x00)),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (x1, 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, Int2x4((x2 & 0x03) << 0x02)),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (0, x2, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, Int2x4((x3 & 0x03) << 0x04)),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (0, 0, x3, 0),
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, Int2x4((x4 & 0x03) << 0x06)),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (0, 0, 0, x4),
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, Int2x4(x.val & 0x03)),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (x1, 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, Int2x4(x.val & 0x0c)),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (0, x2, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, Int2x4(x.val & 0x30)),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (0, 0, x3, 0),
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, Int2x4(x.val & 0xc0)),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (0, 0, 0, x4),
#TODO     )
#TODO 
#TODO     # zero
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, zero(Int2x4)),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (0, 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> zero(Int2x4),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> zero(x),
#TODO     )
#TODO 
#TODO     @test iszero(zero(Int2x4)) isa Bool
#TODO     @test iszero(zero(Int2x4))
#TODO     @test rand(Int2x4) isa Int2x4
#TODO 
#TODO     # logical not
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, ~x),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (~x1, ~x2, ~x3, ~x4),
#TODO     )
#TODO 
#TODO     # arithmetic pos/negation
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, +x),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> make_int2.((+x1, +x2, +x3, +x4)),
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, -x),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> make_int2.((-x1, -x2, -x3, -x4)),
#TODO     )
#TODO 
#TODO     # logical operations
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, x & y),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (x1 & y1, x2 & y2, x3 & y3, x4 & y4),
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, x | y),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (x1 | y1, x2 | y2, x3 | y3, x4 | y4),
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, x ⊻ y),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (x1 ⊻ y1, x2 ⊻ y2, x3 ⊻ y3, x4 ⊻ y4),
#TODO     )
#TODO 
#TODO     # arithmetic operations
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, x + y),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> make_int2.((x1 + y1, x2 + y2, x3 + y3, x4 + y4)),
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, x - y),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> make_int2.((x1 - y1, x2 - y2, x3 - y3, x4 - y4)),
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, min(x, y)),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) ->
#TODO             make_int2.((min(x1, y1), min(x2, y2), min(x3, y3), min(x4, y4))),
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, max(x, y)),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) ->
#TODO             make_int2.((max(x1, y1), max(x2, y2), max(x3, y3), max(x4, y4))),
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, clamp1(x, y, z)),
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) ->
#TODO             make_int2.((clamp1(x1, y1, z1), clamp1(x2, y2, z2), clamp1(x3, y3, z3), clamp1(x4, y4, z4))),
#TODO     )
#TODO 
#TODO     # comparisons
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> x == y,
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> all((x1, x2, x3, x4) .== (y1, y2, y3, y4)),
#TODO     )
#TODO     compare(
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> x != y,
#TODO         (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> any((x1, x2, x3, x4) .!= (y1, y2, y3, y4)),
#TODO     )
#TODO 
#TODO     print("$(CR)$(EL)")
#TODO     flush(stdout)
#TODO end
#TODO 
#TODO Random.seed!(0)
#TODO @testset "Int4x2" begin
#TODO     # Test exhaustively for x and y
#TODO     xlo = Int32[]
#TODO     xhi = Int32[]
#TODO     ylo = Int32[]
#TODO     yhi = Int32[]
#TODO     zlo = Int32[]
#TODO     zhi = Int32[]
#TODO     x = Int4x2[]
#TODO     y = Int4x2[]
#TODO     z = Int4x2[]
#TODO 
#TODO     for xlo1 in (-Int32(8)):(+Int32(7)),
#TODO         xhi1 in (-Int32(8)):(+Int32(7)),
#TODO         ylo1 in (-Int32(8)):(+Int32(7)),
#TODO         yhi1 in (-Int32(8)):(+Int32(7))
#TODO 
#TODO         zlo1 = rand((-Int32(8)):(+Int32(7)))
#TODO         zhi1 = rand((-Int32(8)):(+Int32(7)))
#TODO 
#TODO         push!(xlo, xlo1)
#TODO         push!(xhi, xhi1)
#TODO         push!(ylo, ylo1)
#TODO         push!(yhi, yhi1)
#TODO         push!(zlo, zlo1)
#TODO         push!(zhi, zhi1)
#TODO         push!(x, Int4x2(xlo1, xhi1))
#TODO         push!(y, Int4x2(ylo1, yhi1))
#TODO         push!(z, Int4x2(zlo1, zhi1))
#TODO     end
#TODO 
#TODO     function compare(fl, fr)
#TODO         rcpul = run_on_cpu(fl, xlo, xhi, ylo, yhi, zlo, zhi, x, y, z)
#TODO         rcpur = run_on_cpu(fr, xlo, xhi, ylo, yhi, zlo, zhi, x, y, z)
#TODO         @test rcpul == rcpur
#TODO         if CUDA.functional()
#TODO             rcudal = similar(rcpul)
#TODO             rcudar = similar(rcpur)
#TODO             run_on_cuda!(fl, rcudal, xlo, xhi, ylo, yhi, zlo, zhi, x, y, z)
#TODO             run_on_cuda!(fr, rcudar, xlo, xhi, ylo, yhi, zlo, zhi, x, y, z)
#TODO             @test rcudal == rcudar
#TODO             @test rcudal == rcpul
#TODO             @test rcudar == rcpur
#TODO         end
#TODO         print(".")
#TODO         flush(stdout)
#TODO         return nothing
#TODO     end
#TODO 
#TODO     # Test constructors
#TODO     compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2((Int8(xlo), Int8(xhi))), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2((Int16(xlo), Int16(xhi))), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x
#TODO     )
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2((Int32(xlo), Int32(xhi))), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x
#TODO     )
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2((Int64(xlo), Int64(xhi))), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x
#TODO     )
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int8}, x), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> (xlo, xhi)
#TODO     )
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int16}, x),
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> (xlo, xhi),
#TODO     )
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, x),
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> (xlo, xhi),
#TODO     )
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int64}, x),
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> (xlo, xhi),
#TODO     )
#TODO 
#TODO     compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int8(xlo), Int8(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
#TODO     compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int16(xlo), Int8(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
#TODO     compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int32(xlo), Int8(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
#TODO     compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int64(xlo), Int8(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
#TODO     compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int8(xlo), Int16(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
#TODO     compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int16(xlo), Int16(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
#TODO     compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int32(xlo), Int16(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
#TODO     compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int64(xlo), Int16(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
#TODO     compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int8(xlo), Int32(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
#TODO     compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int16(xlo), Int32(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
#TODO     compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int32(xlo), Int32(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
#TODO     compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int64(xlo), Int32(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
#TODO     compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int8(xlo), Int64(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
#TODO     compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int16(xlo), Int64(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
#TODO     compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int32(xlo), Int64(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
#TODO     compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int64(xlo), Int64(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
#TODO 
#TODO     compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int16x2(x), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int16x2(xlo, xhi))
#TODO     compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int16x2(xlo, xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
#TODO 
#TODO     # Test output
#TODO     @test string.(x) == "Int4x2" .* string.(tuple.(xlo, xhi))
#TODO 
#TODO     # Test nibble ordering
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, Int4x2(x.val & 0x0f)),
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> (xlo, 0),
#TODO     )
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, Int4x2(x.val & 0xf0)),
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> (0, xhi),
#TODO     )
#TODO 
#TODO     # zero
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, zero(Int4x2)),
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> (0, 0),
#TODO     )
#TODO     compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> zero(Int4x2), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> zero(x))
#TODO 
#TODO     @test iszero(zero(Int4x2)) isa Bool
#TODO     @test iszero(zero(Int4x2))
#TODO     @test rand(Int4x2) isa Int4x2
#TODO 
#TODO     # logical not
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, ~x),
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> (~xlo, ~xhi),
#TODO     )
#TODO 
#TODO     # arithmetic pos/negation
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, +x),
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> make_int4.((+xlo, +xhi)),
#TODO     )
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, -x),
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> make_int4.((-xlo, -xhi)),
#TODO     )
#TODO 
#TODO     # logical operations
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, x & y),
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> (xlo & ylo, xhi & yhi),
#TODO     )
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, x | y),
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> (xlo | ylo, xhi | yhi),
#TODO     )
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, x ⊻ y),
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> (xlo ⊻ ylo, xhi ⊻ yhi),
#TODO     )
#TODO 
#TODO     # arithmetic operations
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, x + y),
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> make_int4.((xlo + ylo, xhi + yhi)),
#TODO     )
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, x - y),
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> make_int4.((xlo - ylo, xhi - yhi)),
#TODO     )
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, min(x, y)),
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> make_int4.((min(xlo, ylo), min(xhi, yhi))),
#TODO     )
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, max(x, y)),
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> make_int4.((max(xlo, ylo), max(xhi, yhi))),
#TODO     )
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, clamp1(x, y, z)),
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> make_int4.((clamp1(xlo, ylo, zlo), clamp1(xhi, yhi, zhi))),
#TODO     )
#TODO 
#TODO     # comparisons
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x == y, (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> all((xlo, xhi) .== (ylo, yhi))
#TODO     )
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x != y, (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> any((xlo, xhi) .!= (ylo, yhi))
#TODO     )
#TODO 
#TODO     print("$(CR)$(EL)")
#TODO     flush(stdout)
#TODO end
#TODO 
#TODO Random.seed!(0)
#TODO @testset "Int2x16" begin
#TODO     n = Int2x16[]
#TODO     xs = NTuple{16,Int32}[]
#TODO     ys = NTuple{16,Int32}[]
#TODO     zs = NTuple{16,Int32}[]
#TODO     x = Int2x16[]
#TODO     y = Int2x16[]
#TODO     z = Int2x16[]
#TODO 
#TODO     for iter in 1:131072
#TODO         n1 = zero(Int2x16)
#TODO         xs1 = tuple(rand((-Int32(2)):(+Int32(1)), 16)...)
#TODO         ys1 = tuple(rand((-Int32(2)):(+Int32(1)), 16)...)
#TODO         zs1 = tuple(rand((-Int32(2)):(+Int32(1)), 16)...)
#TODO         x1 = Int2x16(xs1...)
#TODO         y1 = Int2x16(ys1...)
#TODO         z1 = Int2x16(zs1...)
#TODO 
#TODO         push!(n, n1)
#TODO         push!(xs, xs1)
#TODO         push!(ys, ys1)
#TODO         push!(zs, zs1)
#TODO         push!(x, x1)
#TODO         push!(y, y1)
#TODO         push!(z, z1)
#TODO     end
#TODO 
#TODO     function compare(fl, fr)
#TODO         rcpul = run_on_cpu(fl, n, xs, ys, zs, x, y, z)
#TODO         rcpur = run_on_cpu(fr, n, xs, ys, zs, x, y, z)
#TODO         @test rcpul == rcpur
#TODO         if CUDA.functional()
#TODO             rcudal = similar(rcpul)
#TODO             rcudar = similar(rcpur)
#TODO             run_on_cuda!(fl, rcudal, n, xs, ys, zs, x, y, z)
#TODO             run_on_cuda!(fr, rcudar, n, xs, ys, zs, x, y, z)
#TODO             @test rcudal == rcudar
#TODO             @test rcudal == rcpul
#TODO             @test rcudar == rcpur
#TODO         end
#TODO         print(".")
#TODO         flush(stdout)
#TODO         return nothing
#TODO     end
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int2x16(xs), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> Int2x16((
#TODO             Int4x8(xs[1], xs[3], xs[5], xs[7], xs[9], xs[11], xs[13], xs[15]),
#TODO             Int4x8(xs[2], xs[4], xs[6], xs[8], xs[10], xs[12], xs[14], xs[16]),
#TODO         )),
#TODO         (n, xs, ys, zs, x, y, z) -> x,
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> Int2x16((
#TODO             Int8x4(xs[1], xs[5], xs[9], xs[13]),
#TODO             Int8x4(xs[2], xs[6], xs[10], xs[14]),
#TODO             Int8x4(xs[3], xs[7], xs[11], xs[15]),
#TODO             Int8x4(xs[4], xs[8], xs[12], xs[16]),
#TODO         )),
#TODO         (n, xs, ys, zs, x, y, z) -> x,
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> Int2x16((
#TODO             Int16x2(xs[1], xs[9]),
#TODO             Int16x2(xs[2], xs[10]),
#TODO             Int16x2(xs[3], xs[11]),
#TODO             Int16x2(xs[4], xs[12]),
#TODO             Int16x2(xs[5], xs[13]),
#TODO             Int16x2(xs[6], xs[14]),
#TODO             Int16x2(xs[7], xs[15]),
#TODO             Int16x2(xs[8], xs[16]),
#TODO         )),
#TODO         (n, xs, ys, zs, x, y, z) -> x,
#TODO     )
#TODO 
#TODO     # TODO: Conversions from/to Int2x4, Int4x2
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int2x16(Int8.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int2x16(Int16.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int2x16(Int32.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int2x16(Int64.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int2x16(Int8.(xs)), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int2x16(Int16.(xs)), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int2x16(Int32.(xs)), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int2x16(Int64.(xs)), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int8}, x), (n, xs, ys, zs, x, y, z) -> xs)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int16}, x), (n, xs, ys, zs, x, y, z) -> xs)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, x), (n, xs, ys, zs, x, y, z) -> xs)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int64}, x), (n, xs, ys, zs, x, y, z) -> xs)
#TODO 
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int4x8}, x),
#TODO         (n, xs, ys, zs, x, y, z) -> (
#TODO             Int4x8(xs[1], xs[3], xs[5], xs[7], xs[9], xs[11], xs[13], xs[15]),
#TODO             Int4x8(xs[2], xs[4], xs[6], xs[8], xs[10], xs[12], xs[14], xs[16]),
#TODO         ),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int8x4}, x),
#TODO         (n, xs, ys, zs, x, y, z) -> (
#TODO             Int8x4(xs[1], xs[5], xs[9], xs[13]),
#TODO             Int8x4(xs[2], xs[6], xs[10], xs[14]),
#TODO             Int8x4(xs[3], xs[7], xs[11], xs[15]),
#TODO             Int8x4(xs[4], xs[8], xs[12], xs[16]),
#TODO         ),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int16x2}, x),
#TODO         (n, xs, ys, zs, x, y, z) -> (
#TODO             Int16x2(xs[1], xs[9]),
#TODO             Int16x2(xs[2], xs[10]),
#TODO             Int16x2(xs[3], xs[11]),
#TODO             Int16x2(xs[4], xs[12]),
#TODO             Int16x2(xs[5], xs[13]),
#TODO             Int16x2(xs[6], xs[14]),
#TODO             Int16x2(xs[7], xs[15]),
#TODO             Int16x2(xs[8], xs[16]),
#TODO         ),
#TODO     )
#TODO 
#TODO     @test string.(x) == "Int2x16" .* string.(xs)
#TODO 
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x00000003)),
#TODO         (n, xs, ys, zs, x, y, z) -> (xs[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x0000000c)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, xs[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x00000030)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, xs[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x000000c0)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, 0, xs[4], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x00000300)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, xs[5], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x00000c00)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, xs[6], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x00003000)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, xs[7], 0, 0, 0, 0, 0, 0, 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x0000c000)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, xs[8], 0, 0, 0, 0, 0, 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x00030000)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0, xs[9], 0, 0, 0, 0, 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x000c0000)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0, 0, xs[10], 0, 0, 0, 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x00300000)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, xs[11], 0, 0, 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x00c00000)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, xs[12], 0, 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x03000000)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, xs[13], 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x0c000000)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, xs[14], 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x30000000)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, xs[15], 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0xc0000000)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, xs[16]),
#TODO     )
#TODO 
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, zero(Int2x16)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, zero(Int2x16)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#TODO     )
#TODO     compare((n, xs, ys, zs, x, y, z) -> zero(Int2x16), (n, xs, ys, zs, x, y, z) -> zero(x))
#TODO 
#TODO     @test iszero(zero(Int2x16)) isa Bool
#TODO     @test iszero(zero(Int2x16))
#TODO     @test rand(Int2x16) isa Int2x16
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, ~x), (n, xs, ys, zs, x, y, z) -> .~xs)
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, +x), (n, xs, ys, zs, x, y, z) -> make_int2.(.+xs))
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, -x), (n, xs, ys, zs, x, y, z) -> make_int2.(.-xs))
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, x & y), (n, xs, ys, zs, x, y, z) -> xs .& ys)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, x | y), (n, xs, ys, zs, x, y, z) -> xs .| ys)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, x ⊻ y), (n, xs, ys, zs, x, y, z) -> xs .⊻ ys)
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, x + y), (n, xs, ys, zs, x, y, z) -> make_int2.(xs .+ ys))
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, x - y), (n, xs, ys, zs, x, y, z) -> make_int2.(xs .- ys))
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, min(x, y)), (n, xs, ys, zs, x, y, z) -> make_int2.(min.(xs, ys)))
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, max(x, y)), (n, xs, ys, zs, x, y, z) -> make_int2.(max.(xs, ys)))
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, clamp1(x, y, z)),
#TODO         (n, xs, ys, zs, x, y, z) -> make_int2.(clamp1.(xs, ys, zs)),
#TODO     )
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> x == y, (n, xs, ys, zs, x, y, z) -> all(xs .== ys))
#TODO     compare((n, xs, ys, zs, x, y, z) -> x != y, (n, xs, ys, zs, x, y, z) -> any(xs .!= ys))
#TODO 
#TODO     print("$(CR)$(EL)")
#TODO     flush(stdout)
#TODO end
#TODO 
#TODO Random.seed!(0)
#TODO @testset "Int4x8" begin
#TODO     n = Int4x8[]
#TODO     xs = NTuple{8,Int32}[]
#TODO     ys = NTuple{8,Int32}[]
#TODO     zs = NTuple{8,Int32}[]
#TODO     x = Int4x8[]
#TODO     y = Int4x8[]
#TODO     z = Int4x8[]
#TODO 
#TODO     for iter in 1:131072
#TODO         n1 = zero(Int4x8)
#TODO         xs1 = tuple(rand((-Int32(8)):(+Int32(7)), 8)...)
#TODO         ys1 = tuple(rand((-Int32(8)):(+Int32(7)), 8)...)
#TODO         zs1 = tuple(rand((-Int32(8)):(+Int32(7)), 8)...)
#TODO         x1 = Int4x8(xs1...)
#TODO         y1 = Int4x8(ys1...)
#TODO         z1 = Int4x8(zs1...)
#TODO 
#TODO         push!(n, n1)
#TODO         push!(xs, xs1)
#TODO         push!(ys, ys1)
#TODO         push!(zs, zs1)
#TODO         push!(x, x1)
#TODO         push!(y, y1)
#TODO         push!(z, z1)
#TODO     end
#TODO 
#TODO     function compare(fl, fr)
#TODO         rcpul = run_on_cpu(fl, n, xs, ys, zs, x, y, z)
#TODO         rcpur = run_on_cpu(fr, n, xs, ys, zs, x, y, z)
#TODO         @test rcpul == rcpur
#TODO         if CUDA.functional()
#TODO             rcudal = similar(rcpul)
#TODO             rcudar = similar(rcpur)
#TODO             run_on_cuda!(fl, rcudal, n, xs, ys, zs, x, y, z)
#TODO             run_on_cuda!(fr, rcudar, n, xs, ys, zs, x, y, z)
#TODO             @test rcudal == rcudar
#TODO             @test rcudal == rcpul
#TODO             @test rcudar == rcpur
#TODO         end
#TODO         print(".")
#TODO         flush(stdout)
#TODO         return nothing
#TODO     end
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int4x8(xs), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> Int4x8((Int8x4(xs[1], xs[3], xs[5], xs[7]), Int8x4(xs[2], xs[4], xs[6], xs[8]))),
#TODO         (n, xs, ys, zs, x, y, z) -> x,
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) ->
#TODO             Int4x8((Int16x2(xs[1], xs[5]), Int16x2(xs[2], xs[6]), Int16x2(xs[3], xs[7]), Int16x2(xs[4], xs[8]))),
#TODO         (n, xs, ys, zs, x, y, z) -> x,
#TODO     )
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int4x8(Int8.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int4x8(Int16.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int4x8(Int32.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int4x8(Int64.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int4x8(Int8.(xs)), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int4x8(Int16.(xs)), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int4x8(Int32.(xs)), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int4x8(Int64.(xs)), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int8}, x), (n, xs, ys, zs, x, y, z) -> xs)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int16}, x), (n, xs, ys, zs, x, y, z) -> xs)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, x), (n, xs, ys, zs, x, y, z) -> xs)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int64}, x), (n, xs, ys, zs, x, y, z) -> xs)
#TODO 
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int8x4}, x),
#TODO         (n, xs, ys, zs, x, y, z) -> (Int8x4(xs[1], xs[3], xs[5], xs[7]), Int8x4(xs[2], xs[4], xs[6], xs[8])),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int16x2}, x),
#TODO         (n, xs, ys, zs, x, y, z) -> (Int16x2(xs[1], xs[5]), Int16x2(xs[2], xs[6]), Int16x2(xs[3], xs[7]), Int16x2(xs[4], xs[8])),
#TODO     )
#TODO 
#TODO     @test string.(x) == "Int4x8" .* string.(xs)
#TODO 
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x0000000f)),
#TODO         (n, xs, ys, zs, x, y, z) -> (xs[1], 0, 0, 0, 0, 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x000000f0)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, xs[2], 0, 0, 0, 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x00000f00)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, xs[3], 0, 0, 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x0000f000)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, 0, xs[4], 0, 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x000f0000)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, xs[5], 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x00f00000)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, xs[6], 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x0f000000)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, xs[7], 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0xf0000000)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, xs[8]),
#TODO     )
#TODO 
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, zero(Int4x8)), (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0)
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, zero(Int4x8)), (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0)
#TODO     )
#TODO     compare((n, xs, ys, zs, x, y, z) -> zero(Int4x8), (n, xs, ys, zs, x, y, z) -> zero(x))
#TODO 
#TODO     @test iszero(zero(Int4x8)) isa Bool
#TODO     @test iszero(zero(Int4x8))
#TODO     @test rand(Int4x8) isa Int4x8
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, ~x), (n, xs, ys, zs, x, y, z) -> .~xs)
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, +x), (n, xs, ys, zs, x, y, z) -> make_int4.(.+xs))
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, -x), (n, xs, ys, zs, x, y, z) -> make_int4.(.-xs))
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, x & y), (n, xs, ys, zs, x, y, z) -> xs .& ys)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, x | y), (n, xs, ys, zs, x, y, z) -> xs .| ys)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, x ⊻ y), (n, xs, ys, zs, x, y, z) -> xs .⊻ ys)
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, x + y), (n, xs, ys, zs, x, y, z) -> make_int4.(xs .+ ys))
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, x - y), (n, xs, ys, zs, x, y, z) -> make_int4.(xs .- ys))
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, min(x, y)), (n, xs, ys, zs, x, y, z) -> make_int4.(min.(xs, ys)))
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, max(x, y)), (n, xs, ys, zs, x, y, z) -> make_int4.(max.(xs, ys)))
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, clamp1(x, y, z)),
#TODO         (n, xs, ys, zs, x, y, z) -> make_int4.(clamp1.(xs, ys, zs)),
#TODO     )
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> x == y, (n, xs, ys, zs, x, y, z) -> all(xs .== ys))
#TODO     compare((n, xs, ys, zs, x, y, z) -> x != y, (n, xs, ys, zs, x, y, z) -> any(xs .!= ys))
#TODO 
#TODO     print("$(CR)$(EL)")
#TODO     flush(stdout)
#TODO end
#TODO 
#TODO Random.seed!(0)
#TODO @testset "Int8x4" begin
#TODO     n = Int8x4[]
#TODO     xs = NTuple{4,Int32}[]
#TODO     ys = NTuple{4,Int32}[]
#TODO     zs = NTuple{4,Int32}[]
#TODO     x = Int8x4[]
#TODO     y = Int8x4[]
#TODO     z = Int8x4[]
#TODO 
#TODO     for iter in 1:131072
#TODO         n1 = zero(Int8x4)
#TODO         xs1 = tuple(Int32.(rand(Int8, 4))...)
#TODO         ys1 = tuple(Int32.(rand(Int8, 4))...)
#TODO         zs1 = tuple(Int32.(rand(Int8, 4))...)
#TODO         x1 = Int8x4(xs1...)
#TODO         y1 = Int8x4(ys1...)
#TODO         z1 = Int8x4(zs1...)
#TODO 
#TODO         push!(n, n1)
#TODO         push!(xs, xs1)
#TODO         push!(ys, ys1)
#TODO         push!(zs, zs1)
#TODO         push!(x, x1)
#TODO         push!(y, y1)
#TODO         push!(z, z1)
#TODO     end
#TODO 
#TODO     function compare(fl, fr)
#TODO         rcpul = run_on_cpu(fl, n, xs, ys, zs, x, y, z)
#TODO         rcpur = run_on_cpu(fr, n, xs, ys, zs, x, y, z)
#TODO         @test rcpul == rcpur
#TODO         if CUDA.functional()
#TODO             rcudal = similar(rcpul)
#TODO             rcudar = similar(rcpur)
#TODO             run_on_cuda!(fl, rcudal, n, xs, ys, zs, x, y, z)
#TODO             run_on_cuda!(fr, rcudar, n, xs, ys, zs, x, y, z)
#TODO             @test rcudal == rcudar
#TODO             @test rcudal == rcpul
#TODO             @test rcudar == rcpur
#TODO         end
#TODO         print(".")
#TODO         flush(stdout)
#TODO         return nothing
#TODO     end
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int8x4(xs), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int8x4((Int16x2(xs[1], xs[3]), Int16x2(xs[2], xs[4]))), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int8x4(Int8.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int8x4(Int16.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int8x4(Int32.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int8x4(Int64.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int8x4(Int8.(xs)), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int8x4(Int16.(xs)), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int8x4(Int32.(xs)), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int8x4(Int64.(xs)), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int8}, x), (n, xs, ys, zs, x, y, z) -> xs)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int16}, x), (n, xs, ys, zs, x, y, z) -> xs)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, x), (n, xs, ys, zs, x, y, z) -> xs)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int64}, x), (n, xs, ys, zs, x, y, z) -> xs)
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int16x2}, x),
#TODO         (n, xs, ys, zs, x, y, z) -> (Int16x2(xs[1], xs[3]), Int16x2(xs[2], xs[4])),
#TODO     )
#TODO 
#TODO     @test string.(x) == "Int8x4" .* string.(xs)
#TODO 
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, Int8x4(x.val & 0x000000ff)),
#TODO         (n, xs, ys, zs, x, y, z) -> (xs[1], 0, 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, Int8x4(x.val & 0x0000ff00)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, xs[2], 0, 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, Int8x4(x.val & 0x00ff0000)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, xs[3], 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, Int8x4(x.val & 0xff000000)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, 0, 0, xs[4]),
#TODO     )
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, zero(Int8x4)), (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0))
#TODO     compare((n, xs, ys, zs, x, y, z) -> zero(Int8x4), (n, xs, ys, zs, x, y, z) -> zero(x))
#TODO 
#TODO     @test iszero(zero(Int8x4)) isa Bool
#TODO     @test iszero(zero(Int8x4))
#TODO     @test rand(Int8x4) isa Int8x4
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, ~x), (n, xs, ys, zs, x, y, z) -> .~xs)
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, +x), (n, xs, ys, zs, x, y, z) -> .+xs .% Int8)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, -x), (n, xs, ys, zs, x, y, z) -> .-xs .% Int8)
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, x & y), (n, xs, ys, zs, x, y, z) -> xs .& ys)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, x | y), (n, xs, ys, zs, x, y, z) -> xs .| ys)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, x ⊻ y), (n, xs, ys, zs, x, y, z) -> xs .⊻ ys)
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, x + y), (n, xs, ys, zs, x, y, z) -> (xs .+ ys) .% Int8)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, x - y), (n, xs, ys, zs, x, y, z) -> (xs .- ys) .% Int8)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, min(x, y)), (n, xs, ys, zs, x, y, z) -> min.(xs, ys) .% Int8)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, max(x, y)), (n, xs, ys, zs, x, y, z) -> max.(xs, ys) .% Int8)
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, clamp1(x, y, z)),
#TODO         (n, xs, ys, zs, x, y, z) -> clamp1.(xs, ys, zs) .% Int8,
#TODO     )
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> x == y, (n, xs, ys, zs, x, y, z) -> all(xs .== ys))
#TODO     compare((n, xs, ys, zs, x, y, z) -> x != y, (n, xs, ys, zs, x, y, z) -> any(xs .!= ys))
#TODO 
#TODO     print("$(CR)$(EL)")
#TODO     flush(stdout)
#TODO end
#TODO 
#TODO Random.seed!(0)
#TODO @testset "Int16x2" begin
#TODO     n = Int16x2[]
#TODO     xs = NTuple{2,Int32}[]
#TODO     ys = NTuple{2,Int32}[]
#TODO     zs = NTuple{2,Int32}[]
#TODO     x = Int16x2[]
#TODO     y = Int16x2[]
#TODO     z = Int16x2[]
#TODO 
#TODO     for iter in 1:131072
#TODO         n1 = zero(Int16x2)
#TODO         xs1 = tuple(Int32.(rand(Int16, 2))...)
#TODO         ys1 = tuple(Int32.(rand(Int16, 2))...)
#TODO         zs1 = tuple(Int32.(rand(Int16, 2))...)
#TODO         x1 = Int16x2(xs1...)
#TODO         y1 = Int16x2(ys1...)
#TODO         z1 = Int16x2(zs1...)
#TODO 
#TODO         push!(n, n1)
#TODO         push!(xs, xs1)
#TODO         push!(ys, ys1)
#TODO         push!(zs, zs1)
#TODO         push!(x, x1)
#TODO         push!(y, y1)
#TODO         push!(z, z1)
#TODO     end
#TODO 
#TODO     function compare(fl, fr)
#TODO         rcpul = run_on_cpu(fl, n, xs, ys, zs, x, y, z)
#TODO         rcpur = run_on_cpu(fr, n, xs, ys, zs, x, y, z)
#TODO         @test rcpul == rcpur
#TODO         if CUDA.functional()
#TODO             rcudal = similar(rcpul)
#TODO             rcudar = similar(rcpur)
#TODO             run_on_cuda!(fl, rcudal, n, xs, ys, zs, x, y, z)
#TODO             run_on_cuda!(fr, rcudar, n, xs, ys, zs, x, y, z)
#TODO             @test rcudal == rcudar
#TODO             @test rcudal == rcpul
#TODO             @test rcudar == rcpur
#TODO         end
#TODO         print(".")
#TODO         flush(stdout)
#TODO         return nothing
#TODO     end
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int16x2(xs), (n, xs, ys, zs, x, y, z) -> x)
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int16x2(Int16.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int16x2(Int32.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int16x2(Int64.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int16x2(Int16.(xs)), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int16x2(Int32.(xs)), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Int16x2(Int64.(xs)), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int16}, x), (n, xs, ys, zs, x, y, z) -> xs)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, x), (n, xs, ys, zs, x, y, z) -> xs)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int64}, x), (n, xs, ys, zs, x, y, z) -> xs)
#TODO 
#TODO     @test string.(x) == "Int16x2" .* string.(xs)
#TODO 
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, Int16x2(x.val & 0x0000ffff)), (n, xs, ys, zs, x, y, z) -> (xs[1], 0)
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, Int16x2(x.val & 0xffff0000)), (n, xs, ys, zs, x, y, z) -> (0, xs[2])
#TODO     )
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, zero(Int16x2)), (n, xs, ys, zs, x, y, z) -> (0, 0))
#TODO     compare((n, xs, ys, zs, x, y, z) -> zero(Int16x2), (n, xs, ys, zs, x, y, z) -> zero(x))
#TODO 
#TODO     @test iszero(zero(Int16x2)) isa Bool
#TODO     @test iszero(zero(Int16x2))
#TODO     @test rand(Int16x2) isa Int16x2
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, ~x), (n, xs, ys, zs, x, y, z) -> .~xs)
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, +x), (n, xs, ys, zs, x, y, z) -> .+xs .% Int16)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, -x), (n, xs, ys, zs, x, y, z) -> .-xs .% Int16)
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, x & y), (n, xs, ys, zs, x, y, z) -> xs .& ys)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, x | y), (n, xs, ys, zs, x, y, z) -> xs .| ys)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, x ⊻ y), (n, xs, ys, zs, x, y, z) -> xs .⊻ ys)
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, x + y), (n, xs, ys, zs, x, y, z) -> (xs .+ ys) .% Int16)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, x - y), (n, xs, ys, zs, x, y, z) -> (xs .- ys) .% Int16)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, min(x, y)), (n, xs, ys, zs, x, y, z) -> min.(xs, ys) .% Int16)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, max(x, y)), (n, xs, ys, zs, x, y, z) -> max.(xs, ys) .% Int16)
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, clamp1(x, y, z)),
#TODO         (n, xs, ys, zs, x, y, z) -> clamp1.(xs, ys, zs) .% Int16,
#TODO     )
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> x == y, (n, xs, ys, zs, x, y, z) -> all(xs .== ys))
#TODO     compare((n, xs, ys, zs, x, y, z) -> x != y, (n, xs, ys, zs, x, y, z) -> any(xs .!= ys))
#TODO 
#TODO     print("$(CR)$(EL)")
#TODO     flush(stdout)
#TODO end
#TODO 
#TODO Random.seed!(0)
#TODO @testset "Float16x2" begin
#TODO     n = Float16x2[]
#TODO     xs = NTuple{2,Float32}[]
#TODO     ys = NTuple{2,Float32}[]
#TODO     zs = NTuple{2,Float32}[]
#TODO     x = Float16x2[]
#TODO     y = Float16x2[]
#TODO     z = Float16x2[]
#TODO 
#TODO     for iter in 1:131072
#TODO         n1 = zero(Float16x2)
#TODO         xs1 = tuple(Float32.(rand(Float16, 2))...)
#TODO         ys1 = tuple(Float32.(rand(Float16, 2))...)
#TODO         zs1 = tuple(Float32.(rand(Float16, 2))...)
#TODO         x1 = Float16x2(xs1...)
#TODO         y1 = Float16x2(ys1...)
#TODO         z1 = Float16x2(zs1...)
#TODO 
#TODO         push!(n, n1)
#TODO         push!(xs, xs1)
#TODO         push!(ys, ys1)
#TODO         push!(zs, zs1)
#TODO         push!(x, x1)
#TODO         push!(y, y1)
#TODO         push!(z, z1)
#TODO     end
#TODO 
#TODO     function compare(fl, fr; atol=nothing)
#TODO         rcpul = run_on_cpu(fl, n, xs, ys, zs, x, y, z)
#TODO         rcpur = run_on_cpu(fr, n, xs, ys, zs, x, y, z)
#TODO         if atol ≡ nothing
#TODO             @test rcpul == rcpur
#TODO         else
#TODO             @test all(isapprox([rcpul...], [rcpur...]; atol=atol) for (rcpul, rcpur) in zip(rcpul, rcpur))
#TODO         end
#TODO         if CUDA.functional()
#TODO             rcudal = similar(rcpul)
#TODO             rcudar = similar(rcpur)
#TODO             run_on_cuda!(fl, rcudal, n, xs, ys, zs, x, y, z)
#TODO             run_on_cuda!(fr, rcudar, n, xs, ys, zs, x, y, z)
#TODO             if atol ≡ nothing
#TODO                 @test rcudal == rcudar
#TODO                 @test rcudal == rcpul
#TODO                 @test rcudar == rcpur
#TODO             else
#TODO                 @test all(isapprox([rcudal...], [rcudar...]; atol=atol) for (rcudal, rcudar) in zip(rcudal, rcudar))
#TODO                 @test all(isapprox([rcudal...], [rcpul...]; atol=atol) for (rcudal, rcpul) in zip(rcudal, rcpul))
#TODO                 @test all(isapprox([rcudar...], [rcpur...]; atol=atol) for (rcudar, rcpur) in zip(rcudar, rcpur))
#TODO             end
#TODO         end
#TODO         print(".")
#TODO         flush(stdout)
#TODO         return nothing
#TODO     end
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> Float16x2(Float16.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Float16x2(Float32.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Float16x2(Float16.(xs)), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> Float16x2(Float32.(xs)), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float16}, x), (n, xs, ys, zs, x, y, z) -> xs)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, x), (n, xs, ys, zs, x, y, z) -> xs)
#TODO 
#TODO     @test string.(x) == "Float16x2" .* string.(xs)
#TODO 
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, Float16x2(x.val & 0x0000ffff)),
#TODO         (n, xs, ys, zs, x, y, z) -> (xs[1], 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, Float16x2(x.val & 0xffff0000)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, xs[2]),
#TODO     )
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, reverse(x)), (n, xs, ys, zs, x, y, z) -> reverse(xs))
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, zero(Float16x2)), (n, xs, ys, zs, x, y, z) -> (0, 0))
#TODO     compare((n, xs, ys, zs, x, y, z) -> zero(Float16x2), (n, xs, ys, zs, x, y, z) -> zero(x))
#TODO 
#TODO     @test iszero(zero(Float16x2)) isa Bool
#TODO     @test iszero(zero(Float16x2))
#TODO     @test rand(Float16x2) isa Float16x2
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, abs(x)), (n, xs, ys, zs, x, y, z) -> abs.(xs))
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, +x), (n, xs, ys, zs, x, y, z) -> .+xs)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, -x), (n, xs, ys, zs, x, y, z) -> .-xs)
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, x + y), (n, xs, ys, zs, x, y, z) -> xs .+ ys; atol=eps(Float16))
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, x - y), (n, xs, ys, zs, x, y, z) -> xs .- ys; atol=eps(Float16))
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, x * y), (n, xs, ys, zs, x, y, z) -> xs .* ys; atol=eps(Float16))
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, min(x, y)), (n, xs, ys, zs, x, y, z) -> min.(xs, ys))
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, max(x, y)), (n, xs, ys, zs, x, y, z) -> max.(xs, ys))
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, clamp1(x, y, z)), (n, xs, ys, zs, x, y, z) -> clamp1.(xs, ys, zs)
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, complex_mul(x, y)),
#TODO         (n, xs, ys, zs, x, y, z) -> tuple_complex_mul(xs, ys);
#TODO         atol=2 * eps(Float16),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, swapped_complex_mul(x, y)),
#TODO         (n, xs, ys, zs, x, y, z) -> tuple_swapped_complex_mul(xs, ys);
#TODO         atol=2 * eps(Float16),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, muladd(x, y, z)),
#TODO         (n, xs, ys, zs, x, y, z) -> muladd.(xs, ys, zs);
#TODO         atol=2 * eps(Float16),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, complex_muladd(x, y, z)),
#TODO         (n, xs, ys, zs, x, y, z) -> tuple_complex_muladd(xs, ys, zs);
#TODO         atol=3 * eps(Float16),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, swapped_complex_muladd(x, y, z)),
#TODO         (n, xs, ys, zs, x, y, z) -> tuple_swapped_complex_muladd(xs, ys, zs);
#TODO         atol=3 * eps(Float16),
#TODO     )
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> x == y, (n, xs, ys, zs, x, y, z) -> all(xs .=== ys))
#TODO     compare((n, xs, ys, zs, x, y, z) -> x != y, (n, xs, ys, zs, x, y, z) -> any(xs .!== ys))
#TODO 
#TODO     print("$(CR)$(EL)")
#TODO     flush(stdout)
#TODO end
#TODO 
#TODO Random.seed!(0)
#TODO @testset "BFloat16x2" begin
#TODO     n = BFloat16x2[]
#TODO     xs = NTuple{2,Float32}[]
#TODO     ys = NTuple{2,Float32}[]
#TODO     zs = NTuple{2,Float32}[]
#TODO     x = BFloat16x2[]
#TODO     y = BFloat16x2[]
#TODO     z = BFloat16x2[]
#TODO 
#TODO     for iter in 1:131072
#TODO         n1 = zero(BFloat16x2)
#TODO         xs1 = tuple(Float32.(rand(BFloat16, 2))...)
#TODO         ys1 = tuple(Float32.(rand(BFloat16, 2))...)
#TODO         zs1 = tuple(Float32.(rand(BFloat16, 2))...)
#TODO         x1 = BFloat16x2(xs1...)
#TODO         y1 = BFloat16x2(ys1...)
#TODO         z1 = BFloat16x2(zs1...)
#TODO 
#TODO         push!(n, n1)
#TODO         push!(xs, xs1)
#TODO         push!(ys, ys1)
#TODO         push!(zs, zs1)
#TODO         push!(x, x1)
#TODO         push!(y, y1)
#TODO         push!(z, z1)
#TODO     end
#TODO 
#TODO     function compare(fl, fr; atol=nothing)
#TODO         rcpul = run_on_cpu(fl, n, xs, ys, zs, x, y, z)
#TODO         rcpur = run_on_cpu(fr, n, xs, ys, zs, x, y, z)
#TODO         if atol ≡ nothing
#TODO             @test rcpul == rcpur
#TODO         else
#TODO             @test all(isapprox([rcpul...], [rcpur...]; atol=atol) for (rcpul, rcpur) in zip(rcpul, rcpur))
#TODO         end
#TODO         if CUDA.functional()
#TODO             rcudal = similar(rcpul)
#TODO             rcudar = similar(rcpur)
#TODO             run_on_cuda!(fl, rcudal, n, xs, ys, zs, x, y, z)
#TODO             run_on_cuda!(fr, rcudar, n, xs, ys, zs, x, y, z)
#TODO             if atol ≡ nothing
#TODO                 @test rcudal == rcudar
#TODO                 @test rcudal == rcpul
#TODO                 @test rcudar == rcpur
#TODO             else
#TODO                 @test all(isapprox([rcudal...], [rcudar...]; atol=atol) for (rcudal, rcudar) in zip(rcudal, rcudar))
#TODO                 @test all(isapprox([rcudal...], [rcpul...]; atol=atol) for (rcudal, rcpul) in zip(rcudal, rcpul))
#TODO                 @test all(isapprox([rcudar...], [rcpur...]; atol=atol) for (rcudar, rcpur) in zip(rcudar, rcpur))
#TODO             end
#TODO         end
#TODO         print(".")
#TODO         flush(stdout)
#TODO         return nothing
#TODO     end
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> BFloat16x2(BFloat16.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> BFloat16x2(Float32.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> BFloat16x2(BFloat16.(xs)), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> BFloat16x2(Float32.(xs)), (n, xs, ys, zs, x, y, z) -> x)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,BFloat16}, x), (n, xs, ys, zs, x, y, z) -> xs)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, x), (n, xs, ys, zs, x, y, z) -> xs)
#TODO 
#TODO     @test string.(x) == "BFloat16x2" .* string.(xs)
#TODO 
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, BFloat16x2(x.val & 0x0000ffff)),
#TODO         (n, xs, ys, zs, x, y, z) -> (xs[1], 0),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, BFloat16x2(x.val & 0xffff0000)),
#TODO         (n, xs, ys, zs, x, y, z) -> (0, xs[2]),
#TODO     )
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, zero(BFloat16x2)), (n, xs, ys, zs, x, y, z) -> (0, 0))
#TODO     compare((n, xs, ys, zs, x, y, z) -> zero(BFloat16x2), (n, xs, ys, zs, x, y, z) -> zero(x))
#TODO 
#TODO     @test iszero(zero(BFloat16x2)) isa Bool
#TODO     @test iszero(zero(BFloat16x2))
#TODO     @test rand(BFloat16x2) isa BFloat16x2
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, abs(x)), (n, xs, ys, zs, x, y, z) -> abs.(xs))
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, +x), (n, xs, ys, zs, x, y, z) -> .+xs)
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, -x), (n, xs, ys, zs, x, y, z) -> .-xs)
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, x + y), (n, xs, ys, zs, x, y, z) -> xs .+ ys; atol=eps(BFloat16))
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, x - y), (n, xs, ys, zs, x, y, z) -> xs .- ys; atol=eps(BFloat16))
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, x * y), (n, xs, ys, zs, x, y, z) -> xs .* ys; atol=eps(BFloat16))
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, min(x, y)), (n, xs, ys, zs, x, y, z) -> min.(xs, ys))
#TODO     compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, max(x, y)), (n, xs, ys, zs, x, y, z) -> max.(xs, ys))
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, clamp1(x, y, z)), (n, xs, ys, zs, x, y, z) -> clamp1.(xs, ys, zs)
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, complex_mul(x, y)),
#TODO         (n, xs, ys, zs, x, y, z) -> tuple_complex_mul(xs, ys);
#TODO         atol=2 * eps(BFloat16),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, swapped_complex_mul(x, y)),
#TODO         (n, xs, ys, zs, x, y, z) -> tuple_swapped_complex_mul(xs, ys);
#TODO         atol=2 * eps(BFloat16),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, muladd(x, y, z)),
#TODO         (n, xs, ys, zs, x, y, z) -> muladd.(xs, ys, zs);
#TODO         atol=2 * eps(BFloat16),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, complex_muladd(x, y, z)),
#TODO         (n, xs, ys, zs, x, y, z) -> tuple_complex_muladd(xs, ys, zs);
#TODO         atol=3 * eps(BFloat16),
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, swapped_complex_muladd(x, y, z)),
#TODO         (n, xs, ys, zs, x, y, z) -> tuple_swapped_complex_muladd(xs, ys, zs);
#TODO         atol=3 * eps(BFloat16),
#TODO     )
#TODO 
#TODO     compare((n, xs, ys, zs, x, y, z) -> x == y, (n, xs, ys, zs, x, y, z) -> all(xs .=== ys))
#TODO     compare((n, xs, ys, zs, x, y, z) -> x != y, (n, xs, ys, zs, x, y, z) -> any(xs .!== ys))
#TODO 
#TODO     print("$(CR)$(EL)")
#TODO     flush(stdout)
#TODO end

@testset "Convert between Int16 and Float16" begin
    xs = NTuple{2,Int16}[]
    x = Int16x2[]

    for i1 in -2048:+2048, i2 in -2048:+2048
        push!(xs, (i1, i2))
        push!(x, Int16x2(i1, i2))
    end
    while length(xs) % 1024 ≠ 0
        push!(xs, (0, 0))
        push!(x, Int16x2(0, 0))
    end

    function compare(fl, fr; atol=nothing)
        rcpul = run_on_cpu(fl, xs, x)
        rcpur = run_on_cpu(fr, xs, x)
        if atol ≡ nothing
            @test rcpul == rcpur
        else
            @test all(isapprox([rcpul...], [rcpur...]; atol=atol) for (rcpul, rcpur) in zip(rcpul, rcpur))
        end
        if CUDA.functional()
            rcudal = similar(rcpul)
            rcudar = similar(rcpur)
            run_on_cuda!(fl, rcudal, xs, x)
            run_on_cuda!(fr, rcudar, xs, x)
            if atol ≡ nothing
                @test rcudal == rcudar
                @test rcudal == rcpul
                @test rcudar == rcpur
            else
                @test all(isapprox([rcudal...], [rcudar...]; atol=atol) for (rcudal, rcudar) in zip(rcudal, rcudar))
                @test all(isapprox([rcudal...], [rcpul...]; atol=atol) for (rcudal, rcpul) in zip(rcudal, rcpul))
                @test all(isapprox([rcudar...], [rcpur...]; atol=atol) for (rcudar, rcpur) in zip(rcudar, rcpur))
            end
        end
        print(".")
        flush(stdout)
        return nothing
    end

    compare((xs, x) -> Float16(Int16(xs[1])), (xs, x) -> xs[1])
    compare((xs, x) -> round(Int16, Float16(xs[1])), (xs, x) -> xs[1])
    compare((xs, x) -> round(Int16, inc(Float16(xs[1]))), (xs, x) -> xs[1])
    compare((xs, x) -> round(Int16, dec(Float16(xs[1]))), (xs, x) -> xs[1])

    compare((xs, x) -> Float16x2(x), (xs, x) -> Float16x2(xs[1], xs[2]))
    compare((xs, x) -> Int16x2(Float16x2(xs[1], xs[2])), (xs, x) -> x)
    compare((xs, x) -> Int16x2(inc(Float16x2(xs[1], xs[2]))), (xs, x) -> x)
    compare((xs, x) -> Int16x2(dec(Float16x2(xs[1], xs[2]))), (xs, x) -> x)

    print("$(CR)$(EL)")
    flush(stdout)
end

@testset "Convert between Int16 and BFloat16" begin
    xs = NTuple{2,Int16}[]
    x = Int16x2[]

    for i1 in -256:+256, i2 in -256:+256
        push!(xs, (i1, i2))
        push!(x, Int16x2(i1, i2))
    end
    while length(xs) % 1024 ≠ 0
        push!(xs, (0, 0))
        push!(x, Int16x2(0, 0))
    end

    function compare(fl, fr; atol=nothing)
        rcpul = run_on_cpu(fl, xs, x)
        rcpur = run_on_cpu(fr, xs, x)
        if atol ≡ nothing
            @test rcpul == rcpur
        else
            @test all(isapprox([rcpul...], [rcpur...]; atol=atol) for (rcpul, rcpur) in zip(rcpul, rcpur))
        end
        if CUDA.functional()
            rcudal = similar(rcpul)
            rcudar = similar(rcpur)
            run_on_cuda!(fl, rcudal, xs, x)
            run_on_cuda!(fr, rcudar, xs, x)
            if atol ≡ nothing
                @test rcudal == rcudar
                @test rcudal == rcpul
                @test rcudar == rcpur
            else
                @test all(isapprox([rcudal...], [rcudar...]; atol=atol) for (rcudal, rcudar) in zip(rcudal, rcudar))
                @test all(isapprox([rcudal...], [rcpul...]; atol=atol) for (rcudal, rcpul) in zip(rcudal, rcpul))
                @test all(isapprox([rcudar...], [rcpur...]; atol=atol) for (rcudar, rcpur) in zip(rcudar, rcpur))
            end
        end
        print(".")
        flush(stdout)
        return nothing
    end

    compare((xs, x) -> BFloat16(Int16(xs[1])), (xs, x) -> xs[1])
    compare((xs, x) -> round(Int16, Float32(BFloat16(xs[1]))), (xs, x) -> xs[1])
    compare((xs, x) -> round(Int16, Float32(inc(BFloat16(xs[1])))), (xs, x) -> xs[1])
    compare((xs, x) -> round(Int16, Float32(dec(BFloat16(xs[1])))), (xs, x) -> xs[1])

    compare((xs, x) -> BFloat16x2(x), (xs, x) -> BFloat16x2(xs[1], xs[2]))
    compare((xs, x) -> Int16x2(BFloat16x2(xs[1], xs[2])), (xs, x) -> x)
    compare((xs, x) -> Int16x2(inc(BFloat16x2(xs[1], xs[2]))), (xs, x) -> x)
    compare((xs, x) -> Int16x2(dec(BFloat16x2(xs[1], xs[2]))), (xs, x) -> x)

    print("$(CR)$(EL)")
    flush(stdout)
end

Random.seed!(0)
@testset "Convert between Int8 and Float16" begin
    xs = NTuple{4,Int8}[]
    x = Int8x4[]

    for i1 in -128:127, i2 in -128:+127
        for iter in 1:256
            i3 = rand(Int8)
            i4 = rand(Int8)
            push!(xs, (i1, i2, i3, i4))
            push!(x, Int8x4(i1, i2, i3, i4))
        end
    end
    while length(xs) % 1024 ≠ 0
        push!(xs, (0, 0, 0, 0))
        push!(x, Int8x4(0, 0, 0, 0))
    end

    function compare(fl, fr; atol=nothing)
        rcpul = run_on_cpu(fl, xs, x)
        rcpur = run_on_cpu(fr, xs, x)
        if atol ≡ nothing
            @test rcpul == rcpur
        else
            @test all(isapprox([rcpul...], [rcpur...]; atol=atol) for (rcpul, rcpur) in zip(rcpul, rcpur))
        end
        if CUDA.functional()
            rcudal = similar(rcpul)
            rcudar = similar(rcpur)
            run_on_cuda!(fl, rcudal, xs, x)
            run_on_cuda!(fr, rcudar, xs, x)
            if atol ≡ nothing
                @test rcudal == rcudar
                @test rcudal == rcpul
                @test rcudar == rcpur
            else
                @test all(isapprox([rcudal...], [rcudar...]; atol=atol) for (rcudal, rcudar) in zip(rcudal, rcudar))
                @test all(isapprox([rcudal...], [rcpul...]; atol=atol) for (rcudal, rcpul) in zip(rcudal, rcpul))
                @test all(isapprox([rcudar...], [rcpur...]; atol=atol) for (rcudar, rcpur) in zip(rcudar, rcpur))
            end
        end
        print(".")
        flush(stdout)
        return nothing
    end

    compare((xs, x) -> Float16(Int8(xs[1])), (xs, x) -> xs[1])
    compare((xs, x) -> round(Int8, Float16(xs[1])), (xs, x) -> xs[1])
    compare((xs, x) -> round(Int8, inc(Float16(xs[1]))), (xs, x) -> xs[1])
    compare((xs, x) -> round(Int8, dec(Float16(xs[1]))), (xs, x) -> xs[1])

    compare((xs, x) -> convert(NTuple{2,Float16x2}, x), (xs, x) -> (Float16x2(xs[1], xs[3]), Float16x2(xs[2], xs[4])))
    compare((xs, x) -> Int8x4((Float16x2(xs[1], xs[3]), Float16x2(xs[2], xs[4]))), (xs, x) -> x)
    compare((xs, x) -> Int8x4((inc(Float16x2(xs[1], xs[3])), inc(Float16x2(xs[2], xs[4])))), (xs, x) -> x)
    compare((xs, x) -> Int8x4((dec(Float16x2(xs[1], xs[3])), dec(Float16x2(xs[2], xs[4])))), (xs, x) -> x)

    print("$(CR)$(EL)")
    flush(stdout)
end

Random.seed!(0)
@testset "Convert between Int4 and Float16" begin
    xs = NTuple{8,Int8}[]
    x = Int4x8[]

    for i1 in -8:7, i2 in -8:7, i3 in -8:7, i4 in -8:7
        for iter in 1:256
            i5 = rand(-8:7)
            i6 = rand(-8:7)
            i7 = rand(-8:7)
            i8 = rand(-8:7)
            push!(xs, (i1, i2, i3, i4, i5, i6, i7, i8))
            push!(x, Int4x8(i1, i2, i3, i4, i5, i6, i7, i8))
        end
    end
    while length(xs) % 1024 ≠ 0
        push!(xs, (0, 0, 0, 0, 0, 0, 0, 0))
        push!(x, Int4x8(0, 0, 0, 0, 0, 0, 0, 0))
    end

    function compare(fl, fr; atol=nothing)
        rcpul = run_on_cpu(fl, xs, x)
        rcpur = run_on_cpu(fr, xs, x)
        if atol ≡ nothing
            @test rcpul == rcpur
        else
            @test all(isapprox([rcpul...], [rcpur...]; atol=atol) for (rcpul, rcpur) in zip(rcpul, rcpur))
        end
        if CUDA.functional()
            rcudal = similar(rcpul)
            rcudar = similar(rcpur)
            run_on_cuda!(fl, rcudal, xs, x)
            run_on_cuda!(fr, rcudar, xs, x)
            if atol ≡ nothing
                @test rcudal == rcudar
                @test rcudal == rcpul
                @test rcudar == rcpur
            else
                @test all(isapprox([rcudal...], [rcudar...]; atol=atol) for (rcudal, rcudar) in zip(rcudal, rcudar))
                @test all(isapprox([rcudal...], [rcpul...]; atol=atol) for (rcudal, rcpul) in zip(rcudal, rcpul))
                @test all(isapprox([rcudar...], [rcpur...]; atol=atol) for (rcudar, rcpur) in zip(rcudar, rcpur))
            end
        end
        print(".")
        flush(stdout)
        return nothing
    end

    compare((xs, x) -> Float16x2(Int4x2(xs[1], xs[2])), (xs, x) -> Float16x2(xs[1], xs[2]))
    compare((xs, x) -> Int4x2(Float16x2(xs[1], xs[2])), (xs, x) -> Int4x2(xs[1], xs[2]))
    compare((xs, x) -> Int4x2(inc(Float16x2(xs[1], xs[2]))), (xs, x) -> Int4x2(xs[1], xs[2]))
    compare((xs, x) -> Int4x2(dec(Float16x2(xs[1], xs[2]))), (xs, x) -> Int4x2(xs[1], xs[2]))

    compare(
        (xs, x) -> convert(NTuple{4,Float16x2}, x),
        (xs, x) -> (Float16x2(xs[1], xs[5]), Float16x2(xs[2], xs[6]), Float16x2(xs[3], xs[7]), Float16x2(xs[4], xs[8])),
    )
    compare(
        (xs, x) -> Int4x8((Float16x2(xs[1], xs[5]), Float16x2(xs[2], xs[6]), Float16x2(xs[3], xs[7]), Float16x2(xs[4], xs[8]))),
        (xs, x) -> x,
    )
    compare(
        (xs, x) -> Int4x8((
            inc(Float16x2(xs[1], xs[5])),
            inc(Float16x2(xs[2], xs[6])),
            inc(Float16x2(xs[3], xs[7])),
            inc(Float16x2(xs[4], xs[8])),
        )),
        (xs, x) -> x,
    )
    compare(
        (xs, x) -> Int4x8((
            dec(Float16x2(xs[1], xs[5])),
            dec(Float16x2(xs[2], xs[6])),
            dec(Float16x2(xs[3], xs[7])),
            dec(Float16x2(xs[4], xs[8])),
        )),
        (xs, x) -> x,
    )

    print("$(CR)$(EL)")
    flush(stdout)
end

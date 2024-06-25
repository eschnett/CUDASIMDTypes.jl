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

Random.seed!(0)
@testset "prmt T=$T" for T in [UInt32, Int32]
    iters = 1000

    a = rand(T, iters)
    b = rand(T, iters)
    op = rand(UInt16, iters)

    r1 = run_on_cpu(prmt, a, b, op)

    if CUDA.functional()
        r2 = similar(r1)
        run_on_cuda!(prmt, r2, a, b, op)
        @test r2 == r1
    end
end

Random.seed!(0)
@testset "lop3 T=$T" for T in [UInt32, Int32]
    iters = 100
    for lut in rand(UInt8, 10)
        LUT = Val(lut)

        a = rand(T, iters)
        b = rand(T, iters)
        c = rand(T, iters)

        lop3′(a, b, c) = lop3(a, b, c, LUT)

        r1 = run_on_cpu(lop3′, a, b, c)

        if CUDA.functional()
            r2 = similar(r1)
            run_on_cuda!(lop3′, r2, a, b, c)
            @test r2 == r1
        end
    end
end

Random.seed!(0)
@testset "bitifelse T=$T" for T in [UInt32, Int32]
    iters = 1000

    cond = rand(T, iters)
    x = rand(T, iters)
    y = rand(T, iters)

    r1 = run_on_cpu(bitifelse, cond, x, y)

    if CUDA.functional()
        r2 = similar(r1)
        run_on_cuda!(bitifelse, r2, cond, x, y)
        @test r2 == r1
    end
end

Random.seed!(0)
@testset "Int2x4" begin
    # Test exhaustively for x and y
    x1 = Int32[]
    x2 = Int32[]
    x3 = Int32[]
    x4 = Int32[]
    y1 = Int32[]
    y2 = Int32[]
    y3 = Int32[]
    y4 = Int32[]
    z1 = Int32[]
    z2 = Int32[]
    z3 = Int32[]
    z4 = Int32[]
    x = Int2x4[]
    y = Int2x4[]
    z = Int2x4[]

    for x01 in (-Int32(2)):(+Int32(1)),
        x02 in (-Int32(2)):(+Int32(1)),
        x03 in (-Int32(2)):(+Int32(1)),
        x04 in (-Int32(2)):(+Int32(1)),
        y01 in (-Int32(2)):(+Int32(1)),
        y02 in (-Int32(2)):(+Int32(1)),
        y03 in (-Int32(2)):(+Int32(1)),
        y04 in (-Int32(2)):(+Int32(1))

        z01 = rand((-Int32(2)):(+Int32(1)))
        z02 = rand((-Int32(2)):(+Int32(1)))
        z03 = rand((-Int32(2)):(+Int32(1)))
        z04 = rand((-Int32(2)):(+Int32(1)))

        push!(x1, x01)
        push!(x2, x02)
        push!(x3, x03)
        push!(x4, x04)
        push!(y1, y01)
        push!(y2, y02)
        push!(y3, y03)
        push!(y4, y04)
        push!(z1, z01)
        push!(z2, z02)
        push!(z3, z03)
        push!(z4, z04)
        push!(x, Int2x4(x01, x02, x03, x04))
        push!(y, Int2x4(y01, y02, y03, y04))
        push!(z, Int2x4(z01, z02, z03, z04))
    end

    function compare(fl, fr)
        rcpul = run_on_cpu(fl, x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z)
        rcpur = run_on_cpu(fr, x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z)
        @test rcpul == rcpur
        @assert rcpul == rcpur
        if CUDA.functional()
            rcudal = similar(rcpul)
            rcudar = similar(rcpur)
            run_on_cuda!(fl, rcudal, x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z)
            run_on_cuda!(fr, rcudar, x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z)
            @test rcudal == rcudar
            @test rcudal == rcpul
            @test rcudar == rcpur
        end
        print(".")
        flush(stdout)
        return nothing
    end

    # Test constructors
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> Int2x4((Int8(x1), Int8(x2), Int8(x3), Int8(x4))),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> x,
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> Int2x4((Int16(x1), Int16(x2), Int16(x3), Int16(x4))),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> x,
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> Int2x4((Int32(x1), Int32(x2), Int32(x3), Int32(x4))),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> x,
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> Int2x4((Int64(x1), Int64(x2), Int64(x3), Int64(x4))),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> x,
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int8}, x),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (x1, x2, x3, x4),
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int16}, x),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (x1, x2, x3, x4),
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, x),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (x1, x2, x3, x4),
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int64}, x),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (x1, x2, x3, x4),
    )

    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> Int2x4(Int8(x1), Int8(x2), Int8(x3), Int8(x4)),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> x,
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> Int2x4(Int16(x1), Int16(x2), Int16(x3), Int16(x4)),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> x,
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> Int2x4(Int32(x1), Int32(x2), Int32(x3), Int32(x4)),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> x,
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> Int2x4(Int64(x1), Int64(x2), Int64(x3), Int64(x4)),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> x,
    )

    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> Int8x4(x),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> Int8x4(x1, x2, x3, x4),
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> Int2x4(Int8x4(x1, x2, x3, x4)),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> x,
    )

    # Test output
    @test string.(x) == "Int2x4" .* string.(tuple.(x1, x2, x3, x4))

    # Test half-nibble ordering
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, Int2x4((x1 & 0x03) << 0x00)),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (x1, 0, 0, 0),
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, Int2x4((x2 & 0x03) << 0x02)),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (0, x2, 0, 0),
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, Int2x4((x3 & 0x03) << 0x04)),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (0, 0, x3, 0),
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, Int2x4((x4 & 0x03) << 0x06)),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (0, 0, 0, x4),
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, Int2x4(x.val & 0x03)),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (x1, 0, 0, 0),
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, Int2x4(x.val & 0x0c)),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (0, x2, 0, 0),
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, Int2x4(x.val & 0x30)),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (0, 0, x3, 0),
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, Int2x4(x.val & 0xc0)),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (0, 0, 0, x4),
    )

    # zero
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, zero(Int2x4)),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (0, 0, 0, 0),
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> zero(Int2x4),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> zero(x),
    )

    @test iszero(zero(Int2x4)) isa Bool
    @test iszero(zero(Int2x4))
    @test rand(Int2x4) isa Int2x4

    # logical not
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, ~x),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (~x1, ~x2, ~x3, ~x4),
    )

    # arithmetic pos/negation
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, +x),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> make_int2.((+x1, +x2, +x3, +x4)),
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, -x),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> make_int2.((-x1, -x2, -x3, -x4)),
    )

    # logical operations
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, x & y),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (x1 & y1, x2 & y2, x3 & y3, x4 & y4),
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, x | y),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (x1 | y1, x2 | y2, x3 | y3, x4 | y4),
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, x ⊻ y),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> (x1 ⊻ y1, x2 ⊻ y2, x3 ⊻ y3, x4 ⊻ y4),
    )

    # arithmetic operations
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, x + y),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> make_int2.((x1 + y1, x2 + y2, x3 + y3, x4 + y4)),
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, x - y),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> make_int2.((x1 - y1, x2 - y2, x3 - y3, x4 - y4)),
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, min(x, y)),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) ->
            make_int2.((min(x1, y1), min(x2, y2), min(x3, y3), min(x4, y4))),
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, max(x, y)),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) ->
            make_int2.((max(x1, y1), max(x2, y2), max(x3, y3), max(x4, y4))),
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> convert(NTuple{4,Int32}, clamp1(x, y, z)),
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) ->
            make_int2.((clamp1(x1, y1, z1), clamp1(x2, y2, z2), clamp1(x3, y3, z3), clamp1(x4, y4, z4))),
    )

    # comparisons
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> x == y,
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> all((x1, x2, x3, x4) .== (y1, y2, y3, y4)),
    )
    compare(
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> x != y,
        (x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, x, y, z) -> any((x1, x2, x3, x4) .!= (y1, y2, y3, y4)),
    )

    print("$(CR)$(EL)")
    flush(stdout)
end

Random.seed!(0)
@testset "Int4x2" begin
    # Test exhaustively for x and y
    xlo = Int32[]
    xhi = Int32[]
    ylo = Int32[]
    yhi = Int32[]
    zlo = Int32[]
    zhi = Int32[]
    x = Int4x2[]
    y = Int4x2[]
    z = Int4x2[]

    for xlo1 in (-Int32(8)):(+Int32(7)),
        xhi1 in (-Int32(8)):(+Int32(7)),
        ylo1 in (-Int32(8)):(+Int32(7)),
        yhi1 in (-Int32(8)):(+Int32(7))

        zlo1 = rand((-Int32(8)):(+Int32(7)))
        zhi1 = rand((-Int32(8)):(+Int32(7)))

        push!(xlo, xlo1)
        push!(xhi, xhi1)
        push!(ylo, ylo1)
        push!(yhi, yhi1)
        push!(zlo, zlo1)
        push!(zhi, zhi1)
        push!(x, Int4x2(xlo1, xhi1))
        push!(y, Int4x2(ylo1, yhi1))
        push!(z, Int4x2(zlo1, zhi1))
    end

    function compare(fl, fr)
        rcpul = run_on_cpu(fl, xlo, xhi, ylo, yhi, zlo, zhi, x, y, z)
        rcpur = run_on_cpu(fr, xlo, xhi, ylo, yhi, zlo, zhi, x, y, z)
        @test rcpul == rcpur
        if CUDA.functional()
            rcudal = similar(rcpul)
            rcudar = similar(rcpur)
            run_on_cuda!(fl, rcudal, xlo, xhi, ylo, yhi, zlo, zhi, x, y, z)
            run_on_cuda!(fr, rcudar, xlo, xhi, ylo, yhi, zlo, zhi, x, y, z)
            @test rcudal == rcudar
            @test rcudal == rcpul
            @test rcudar == rcpur
        end
        print(".")
        flush(stdout)
        return nothing
    end

    # Test constructors
    compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2((Int8(xlo), Int8(xhi))), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2((Int16(xlo), Int16(xhi))), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x
    )
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2((Int32(xlo), Int32(xhi))), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x
    )
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2((Int64(xlo), Int64(xhi))), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x
    )
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int8}, x), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> (xlo, xhi)
    )
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int16}, x),
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> (xlo, xhi),
    )
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, x),
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> (xlo, xhi),
    )
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int64}, x),
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> (xlo, xhi),
    )

    compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int8(xlo), Int8(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
    compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int16(xlo), Int8(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
    compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int32(xlo), Int8(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
    compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int64(xlo), Int8(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
    compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int8(xlo), Int16(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
    compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int16(xlo), Int16(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
    compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int32(xlo), Int16(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
    compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int64(xlo), Int16(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
    compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int8(xlo), Int32(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
    compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int16(xlo), Int32(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
    compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int32(xlo), Int32(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
    compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int64(xlo), Int32(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
    compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int8(xlo), Int64(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
    compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int16(xlo), Int64(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
    compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int32(xlo), Int64(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)
    compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int64(xlo), Int64(xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)

    compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int16x2(x), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int16x2(xlo, xhi))
    compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> Int4x2(Int16x2(xlo, xhi)), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x)

    # Test output
    @test string.(x) == "Int4x2" .* string.(tuple.(xlo, xhi))

    # Test nibble ordering
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, Int4x2(x.val & 0x0f)),
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> (xlo, 0),
    )
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, Int4x2(x.val & 0xf0)),
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> (0, xhi),
    )

    # zero
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, zero(Int4x2)),
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> (0, 0),
    )
    compare((xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> zero(Int4x2), (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> zero(x))

    @test iszero(zero(Int4x2)) isa Bool
    @test iszero(zero(Int4x2))
    @test rand(Int4x2) isa Int4x2

    # logical not
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, ~x),
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> (~xlo, ~xhi),
    )

    # arithmetic pos/negation
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, +x),
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> make_int4.((+xlo, +xhi)),
    )
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, -x),
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> make_int4.((-xlo, -xhi)),
    )

    # logical operations
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, x & y),
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> (xlo & ylo, xhi & yhi),
    )
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, x | y),
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> (xlo | ylo, xhi | yhi),
    )
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, x ⊻ y),
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> (xlo ⊻ ylo, xhi ⊻ yhi),
    )

    # arithmetic operations
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, x + y),
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> make_int4.((xlo + ylo, xhi + yhi)),
    )
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, x - y),
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> make_int4.((xlo - ylo, xhi - yhi)),
    )
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, min(x, y)),
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> make_int4.((min(xlo, ylo), min(xhi, yhi))),
    )
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, max(x, y)),
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> make_int4.((max(xlo, ylo), max(xhi, yhi))),
    )
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> convert(NTuple{2,Int32}, clamp1(x, y, z)),
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> make_int4.((clamp1(xlo, ylo, zlo), clamp1(xhi, yhi, zhi))),
    )

    # comparisons
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x == y, (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> all((xlo, xhi) .== (ylo, yhi))
    )
    compare(
        (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> x != y, (xlo, xhi, ylo, yhi, zlo, zhi, x, y, z) -> any((xlo, xhi) .!= (ylo, yhi))
    )

    print("$(CR)$(EL)")
    flush(stdout)
end

Random.seed!(0)
@testset "Int2x16" begin
    n = Int2x16[]
    xs = NTuple{16,Int32}[]
    ys = NTuple{16,Int32}[]
    zs = NTuple{16,Int32}[]
    x = Int2x16[]
    y = Int2x16[]
    z = Int2x16[]

    for iter in 1:131072
        n1 = zero(Int2x16)
        xs1 = tuple(rand((-Int32(2)):(+Int32(1)), 16)...)
        ys1 = tuple(rand((-Int32(2)):(+Int32(1)), 16)...)
        zs1 = tuple(rand((-Int32(2)):(+Int32(1)), 16)...)
        x1 = Int2x16(xs1...)
        y1 = Int2x16(ys1...)
        z1 = Int2x16(zs1...)

        push!(n, n1)
        push!(xs, xs1)
        push!(ys, ys1)
        push!(zs, zs1)
        push!(x, x1)
        push!(y, y1)
        push!(z, z1)
    end

    function compare(fl, fr)
        rcpul = run_on_cpu(fl, n, xs, ys, zs, x, y, z)
        rcpur = run_on_cpu(fr, n, xs, ys, zs, x, y, z)
        @test rcpul == rcpur
        if CUDA.functional()
            rcudal = similar(rcpul)
            rcudar = similar(rcpur)
            run_on_cuda!(fl, rcudal, n, xs, ys, zs, x, y, z)
            run_on_cuda!(fr, rcudar, n, xs, ys, zs, x, y, z)
            @test rcudal == rcudar
            @test rcudal == rcpul
            @test rcudar == rcpur
        end
        print(".")
        flush(stdout)
        return nothing
    end

    compare((n, xs, ys, zs, x, y, z) -> Int2x16(xs), (n, xs, ys, zs, x, y, z) -> x)
    compare(
        (n, xs, ys, zs, x, y, z) -> Int2x16((
            Int4x8(xs[1], xs[3], xs[5], xs[7], xs[9], xs[11], xs[13], xs[15]),
            Int4x8(xs[2], xs[4], xs[6], xs[8], xs[10], xs[12], xs[14], xs[16]),
        )),
        (n, xs, ys, zs, x, y, z) -> x,
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> Int2x16((
            Int8x4(xs[1], xs[5], xs[9], xs[13]),
            Int8x4(xs[2], xs[6], xs[10], xs[14]),
            Int8x4(xs[3], xs[7], xs[11], xs[15]),
            Int8x4(xs[4], xs[8], xs[12], xs[16]),
        )),
        (n, xs, ys, zs, x, y, z) -> x,
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> Int2x16((
            Int16x2(xs[1], xs[9]),
            Int16x2(xs[2], xs[10]),
            Int16x2(xs[3], xs[11]),
            Int16x2(xs[4], xs[12]),
            Int16x2(xs[5], xs[13]),
            Int16x2(xs[6], xs[14]),
            Int16x2(xs[7], xs[15]),
            Int16x2(xs[8], xs[16]),
        )),
        (n, xs, ys, zs, x, y, z) -> x,
    )

    # TODO: Conversions from/to Int2x4, Int4x2
    compare((n, xs, ys, zs, x, y, z) -> Int2x16(Int8.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int2x16(Int16.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int2x16(Int32.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int2x16(Int64.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int2x16(Int8.(xs)), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int2x16(Int16.(xs)), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int2x16(Int32.(xs)), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int2x16(Int64.(xs)), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int8}, x), (n, xs, ys, zs, x, y, z) -> xs)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int16}, x), (n, xs, ys, zs, x, y, z) -> xs)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, x), (n, xs, ys, zs, x, y, z) -> xs)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int64}, x), (n, xs, ys, zs, x, y, z) -> xs)

    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int4x8}, x),
        (n, xs, ys, zs, x, y, z) -> (
            Int4x8(xs[1], xs[3], xs[5], xs[7], xs[9], xs[11], xs[13], xs[15]),
            Int4x8(xs[2], xs[4], xs[6], xs[8], xs[10], xs[12], xs[14], xs[16]),
        ),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int8x4}, x),
        (n, xs, ys, zs, x, y, z) -> (
            Int8x4(xs[1], xs[5], xs[9], xs[13]),
            Int8x4(xs[2], xs[6], xs[10], xs[14]),
            Int8x4(xs[3], xs[7], xs[11], xs[15]),
            Int8x4(xs[4], xs[8], xs[12], xs[16]),
        ),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int16x2}, x),
        (n, xs, ys, zs, x, y, z) -> (
            Int16x2(xs[1], xs[9]),
            Int16x2(xs[2], xs[10]),
            Int16x2(xs[3], xs[11]),
            Int16x2(xs[4], xs[12]),
            Int16x2(xs[5], xs[13]),
            Int16x2(xs[6], xs[14]),
            Int16x2(xs[7], xs[15]),
            Int16x2(xs[8], xs[16]),
        ),
    )

    @test string.(x) == "Int2x16" .* string.(xs)

    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x00000003)),
        (n, xs, ys, zs, x, y, z) -> (xs[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x0000000c)),
        (n, xs, ys, zs, x, y, z) -> (0, xs[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x00000030)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, xs[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x000000c0)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, 0, xs[4], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x00000300)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, xs[5], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x00000c00)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, xs[6], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x00003000)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, xs[7], 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x0000c000)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, xs[8], 0, 0, 0, 0, 0, 0, 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x00030000)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0, xs[9], 0, 0, 0, 0, 0, 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x000c0000)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0, 0, xs[10], 0, 0, 0, 0, 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x00300000)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, xs[11], 0, 0, 0, 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x00c00000)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, xs[12], 0, 0, 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x03000000)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, xs[13], 0, 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x0c000000)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, xs[14], 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0x30000000)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, xs[15], 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, Int2x16(x.val & 0xc0000000)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, xs[16]),
    )

    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, zero(Int2x16)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, zero(Int2x16)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    compare((n, xs, ys, zs, x, y, z) -> zero(Int2x16), (n, xs, ys, zs, x, y, z) -> zero(x))

    @test iszero(zero(Int2x16)) isa Bool
    @test iszero(zero(Int2x16))
    @test rand(Int2x16) isa Int2x16

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, ~x), (n, xs, ys, zs, x, y, z) -> .~xs)

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, +x), (n, xs, ys, zs, x, y, z) -> make_int2.(.+xs))
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, -x), (n, xs, ys, zs, x, y, z) -> make_int2.(.-xs))

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, x & y), (n, xs, ys, zs, x, y, z) -> xs .& ys)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, x | y), (n, xs, ys, zs, x, y, z) -> xs .| ys)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, x ⊻ y), (n, xs, ys, zs, x, y, z) -> xs .⊻ ys)

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, x + y), (n, xs, ys, zs, x, y, z) -> make_int2.(xs .+ ys))
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, x - y), (n, xs, ys, zs, x, y, z) -> make_int2.(xs .- ys))
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, min(x, y)), (n, xs, ys, zs, x, y, z) -> make_int2.(min.(xs, ys)))
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, max(x, y)), (n, xs, ys, zs, x, y, z) -> make_int2.(max.(xs, ys)))
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{16,Int32}, clamp1(x, y, z)),
        (n, xs, ys, zs, x, y, z) -> make_int2.(clamp1.(xs, ys, zs)),
    )

    compare((n, xs, ys, zs, x, y, z) -> x == y, (n, xs, ys, zs, x, y, z) -> all(xs .== ys))
    compare((n, xs, ys, zs, x, y, z) -> x != y, (n, xs, ys, zs, x, y, z) -> any(xs .!= ys))

    print("$(CR)$(EL)")
    flush(stdout)
end

Random.seed!(0)
@testset "Int4x8" begin
    n = Int4x8[]
    xs = NTuple{8,Int32}[]
    ys = NTuple{8,Int32}[]
    zs = NTuple{8,Int32}[]
    x = Int4x8[]
    y = Int4x8[]
    z = Int4x8[]

    for iter in 1:131072
        n1 = zero(Int4x8)
        xs1 = tuple(rand((-Int32(8)):(+Int32(7)), 8)...)
        ys1 = tuple(rand((-Int32(8)):(+Int32(7)), 8)...)
        zs1 = tuple(rand((-Int32(8)):(+Int32(7)), 8)...)
        x1 = Int4x8(xs1...)
        y1 = Int4x8(ys1...)
        z1 = Int4x8(zs1...)

        push!(n, n1)
        push!(xs, xs1)
        push!(ys, ys1)
        push!(zs, zs1)
        push!(x, x1)
        push!(y, y1)
        push!(z, z1)
    end

    function compare(fl, fr)
        rcpul = run_on_cpu(fl, n, xs, ys, zs, x, y, z)
        rcpur = run_on_cpu(fr, n, xs, ys, zs, x, y, z)
        @test rcpul == rcpur
        if CUDA.functional()
            rcudal = similar(rcpul)
            rcudar = similar(rcpur)
            run_on_cuda!(fl, rcudal, n, xs, ys, zs, x, y, z)
            run_on_cuda!(fr, rcudar, n, xs, ys, zs, x, y, z)
            @test rcudal == rcudar
            @test rcudal == rcpul
            @test rcudar == rcpur
        end
        print(".")
        flush(stdout)
        return nothing
    end

    compare((n, xs, ys, zs, x, y, z) -> Int4x8(xs), (n, xs, ys, zs, x, y, z) -> x)
    compare(
        (n, xs, ys, zs, x, y, z) -> Int4x8((Int8x4(xs[1], xs[3], xs[5], xs[7]), Int8x4(xs[2], xs[4], xs[6], xs[8]))),
        (n, xs, ys, zs, x, y, z) -> x,
    )
    compare(
        (n, xs, ys, zs, x, y, z) ->
            Int4x8((Int16x2(xs[1], xs[5]), Int16x2(xs[2], xs[6]), Int16x2(xs[3], xs[7]), Int16x2(xs[4], xs[8]))),
        (n, xs, ys, zs, x, y, z) -> x,
    )

    compare((n, xs, ys, zs, x, y, z) -> Int4x8(Int8.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int4x8(Int16.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int4x8(Int32.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int4x8(Int64.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int4x8(Int8.(xs)), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int4x8(Int16.(xs)), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int4x8(Int32.(xs)), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int4x8(Int64.(xs)), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int8}, x), (n, xs, ys, zs, x, y, z) -> xs)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int16}, x), (n, xs, ys, zs, x, y, z) -> xs)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, x), (n, xs, ys, zs, x, y, z) -> xs)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int64}, x), (n, xs, ys, zs, x, y, z) -> xs)

    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int8x4}, x),
        (n, xs, ys, zs, x, y, z) -> (Int8x4(xs[1], xs[3], xs[5], xs[7]), Int8x4(xs[2], xs[4], xs[6], xs[8])),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int16x2}, x),
        (n, xs, ys, zs, x, y, z) -> (Int16x2(xs[1], xs[5]), Int16x2(xs[2], xs[6]), Int16x2(xs[3], xs[7]), Int16x2(xs[4], xs[8])),
    )

    @test string.(x) == "Int4x8" .* string.(xs)

    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x0000000f)),
        (n, xs, ys, zs, x, y, z) -> (xs[1], 0, 0, 0, 0, 0, 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x000000f0)),
        (n, xs, ys, zs, x, y, z) -> (0, xs[2], 0, 0, 0, 0, 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x00000f00)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, xs[3], 0, 0, 0, 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x0000f000)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, 0, xs[4], 0, 0, 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x000f0000)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, xs[5], 0, 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x00f00000)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, xs[6], 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x0f000000)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, xs[7], 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0xf0000000)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, xs[8]),
    )

    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, zero(Int4x8)), (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0)
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, zero(Int4x8)), (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0, 0, 0, 0, 0)
    )
    compare((n, xs, ys, zs, x, y, z) -> zero(Int4x8), (n, xs, ys, zs, x, y, z) -> zero(x))

    @test iszero(zero(Int4x8)) isa Bool
    @test iszero(zero(Int4x8))
    @test rand(Int4x8) isa Int4x8

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, ~x), (n, xs, ys, zs, x, y, z) -> .~xs)

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, +x), (n, xs, ys, zs, x, y, z) -> make_int4.(.+xs))
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, -x), (n, xs, ys, zs, x, y, z) -> make_int4.(.-xs))

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, x & y), (n, xs, ys, zs, x, y, z) -> xs .& ys)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, x | y), (n, xs, ys, zs, x, y, z) -> xs .| ys)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, x ⊻ y), (n, xs, ys, zs, x, y, z) -> xs .⊻ ys)

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, x + y), (n, xs, ys, zs, x, y, z) -> make_int4.(xs .+ ys))
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, x - y), (n, xs, ys, zs, x, y, z) -> make_int4.(xs .- ys))
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, min(x, y)), (n, xs, ys, zs, x, y, z) -> make_int4.(min.(xs, ys)))
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, max(x, y)), (n, xs, ys, zs, x, y, z) -> make_int4.(max.(xs, ys)))
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{8,Int32}, clamp1(x, y, z)),
        (n, xs, ys, zs, x, y, z) -> make_int4.(clamp1.(xs, ys, zs)),
    )

    compare((n, xs, ys, zs, x, y, z) -> x == y, (n, xs, ys, zs, x, y, z) -> all(xs .== ys))
    compare((n, xs, ys, zs, x, y, z) -> x != y, (n, xs, ys, zs, x, y, z) -> any(xs .!= ys))

    print("$(CR)$(EL)")
    flush(stdout)
end

Random.seed!(0)
@testset "Int8x4" begin
    n = Int8x4[]
    xs = NTuple{4,Int32}[]
    ys = NTuple{4,Int32}[]
    zs = NTuple{4,Int32}[]
    x = Int8x4[]
    y = Int8x4[]
    z = Int8x4[]

    for iter in 1:131072
        n1 = zero(Int8x4)
        xs1 = tuple(Int32.(rand(Int8, 4))...)
        ys1 = tuple(Int32.(rand(Int8, 4))...)
        zs1 = tuple(Int32.(rand(Int8, 4))...)
        x1 = Int8x4(xs1...)
        y1 = Int8x4(ys1...)
        z1 = Int8x4(zs1...)

        push!(n, n1)
        push!(xs, xs1)
        push!(ys, ys1)
        push!(zs, zs1)
        push!(x, x1)
        push!(y, y1)
        push!(z, z1)
    end

    function compare(fl, fr)
        rcpul = run_on_cpu(fl, n, xs, ys, zs, x, y, z)
        rcpur = run_on_cpu(fr, n, xs, ys, zs, x, y, z)
        @test rcpul == rcpur
        if CUDA.functional()
            rcudal = similar(rcpul)
            rcudar = similar(rcpur)
            run_on_cuda!(fl, rcudal, n, xs, ys, zs, x, y, z)
            run_on_cuda!(fr, rcudar, n, xs, ys, zs, x, y, z)
            @test rcudal == rcudar
            @test rcudal == rcpul
            @test rcudar == rcpur
        end
        print(".")
        flush(stdout)
        return nothing
    end

    compare((n, xs, ys, zs, x, y, z) -> Int8x4(xs), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int8x4((Int16x2(xs[1], xs[3]), Int16x2(xs[2], xs[4]))), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int8x4(Int8.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int8x4(Int16.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int8x4(Int32.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int8x4(Int64.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int8x4(Int8.(xs)), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int8x4(Int16.(xs)), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int8x4(Int32.(xs)), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int8x4(Int64.(xs)), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int8}, x), (n, xs, ys, zs, x, y, z) -> xs)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int16}, x), (n, xs, ys, zs, x, y, z) -> xs)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, x), (n, xs, ys, zs, x, y, z) -> xs)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int64}, x), (n, xs, ys, zs, x, y, z) -> xs)
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int16x2}, x),
        (n, xs, ys, zs, x, y, z) -> (Int16x2(xs[1], xs[3]), Int16x2(xs[2], xs[4])),
    )

    @test string.(x) == "Int8x4" .* string.(xs)

    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, Int8x4(x.val & 0x000000ff)),
        (n, xs, ys, zs, x, y, z) -> (xs[1], 0, 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, Int8x4(x.val & 0x0000ff00)),
        (n, xs, ys, zs, x, y, z) -> (0, xs[2], 0, 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, Int8x4(x.val & 0x00ff0000)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, xs[3], 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, Int8x4(x.val & 0xff000000)),
        (n, xs, ys, zs, x, y, z) -> (0, 0, 0, xs[4]),
    )

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, zero(Int8x4)), (n, xs, ys, zs, x, y, z) -> (0, 0, 0, 0))
    compare((n, xs, ys, zs, x, y, z) -> zero(Int8x4), (n, xs, ys, zs, x, y, z) -> zero(x))

    @test iszero(zero(Int8x4)) isa Bool
    @test iszero(zero(Int8x4))
    @test rand(Int8x4) isa Int8x4

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, ~x), (n, xs, ys, zs, x, y, z) -> .~xs)

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, +x), (n, xs, ys, zs, x, y, z) -> .+xs .% Int8)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, -x), (n, xs, ys, zs, x, y, z) -> .-xs .% Int8)

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, x & y), (n, xs, ys, zs, x, y, z) -> xs .& ys)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, x | y), (n, xs, ys, zs, x, y, z) -> xs .| ys)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, x ⊻ y), (n, xs, ys, zs, x, y, z) -> xs .⊻ ys)

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, x + y), (n, xs, ys, zs, x, y, z) -> (xs .+ ys) .% Int8)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, x - y), (n, xs, ys, zs, x, y, z) -> (xs .- ys) .% Int8)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, min(x, y)), (n, xs, ys, zs, x, y, z) -> min.(xs, ys) .% Int8)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, max(x, y)), (n, xs, ys, zs, x, y, z) -> max.(xs, ys) .% Int8)
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{4,Int32}, clamp1(x, y, z)),
        (n, xs, ys, zs, x, y, z) -> clamp1.(xs, ys, zs) .% Int8,
    )

    compare((n, xs, ys, zs, x, y, z) -> x == y, (n, xs, ys, zs, x, y, z) -> all(xs .== ys))
    compare((n, xs, ys, zs, x, y, z) -> x != y, (n, xs, ys, zs, x, y, z) -> any(xs .!= ys))

    print("$(CR)$(EL)")
    flush(stdout)
end

Random.seed!(0)
@testset "Int16x2" begin
    n = Int16x2[]
    xs = NTuple{2,Int32}[]
    ys = NTuple{2,Int32}[]
    zs = NTuple{2,Int32}[]
    x = Int16x2[]
    y = Int16x2[]
    z = Int16x2[]

    for iter in 1:131072
        n1 = zero(Int16x2)
        xs1 = tuple(Int32.(rand(Int16, 2))...)
        ys1 = tuple(Int32.(rand(Int16, 2))...)
        zs1 = tuple(Int32.(rand(Int16, 2))...)
        x1 = Int16x2(xs1...)
        y1 = Int16x2(ys1...)
        z1 = Int16x2(zs1...)

        push!(n, n1)
        push!(xs, xs1)
        push!(ys, ys1)
        push!(zs, zs1)
        push!(x, x1)
        push!(y, y1)
        push!(z, z1)
    end

    function compare(fl, fr)
        rcpul = run_on_cpu(fl, n, xs, ys, zs, x, y, z)
        rcpur = run_on_cpu(fr, n, xs, ys, zs, x, y, z)
        @test rcpul == rcpur
        if CUDA.functional()
            rcudal = similar(rcpul)
            rcudar = similar(rcpur)
            run_on_cuda!(fl, rcudal, n, xs, ys, zs, x, y, z)
            run_on_cuda!(fr, rcudar, n, xs, ys, zs, x, y, z)
            @test rcudal == rcudar
            @test rcudal == rcpul
            @test rcudar == rcpur
        end
        print(".")
        flush(stdout)
        return nothing
    end

    compare((n, xs, ys, zs, x, y, z) -> Int16x2(xs), (n, xs, ys, zs, x, y, z) -> x)

    compare((n, xs, ys, zs, x, y, z) -> Int16x2(Int16.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int16x2(Int32.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int16x2(Int64.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int16x2(Int16.(xs)), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int16x2(Int32.(xs)), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Int16x2(Int64.(xs)), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int16}, x), (n, xs, ys, zs, x, y, z) -> xs)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, x), (n, xs, ys, zs, x, y, z) -> xs)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int64}, x), (n, xs, ys, zs, x, y, z) -> xs)

    @test string.(x) == "Int16x2" .* string.(xs)

    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, Int16x2(x.val & 0x0000ffff)), (n, xs, ys, zs, x, y, z) -> (xs[1], 0)
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, Int16x2(x.val & 0xffff0000)), (n, xs, ys, zs, x, y, z) -> (0, xs[2])
    )

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, zero(Int16x2)), (n, xs, ys, zs, x, y, z) -> (0, 0))
    compare((n, xs, ys, zs, x, y, z) -> zero(Int16x2), (n, xs, ys, zs, x, y, z) -> zero(x))

    @test iszero(zero(Int16x2)) isa Bool
    @test iszero(zero(Int16x2))
    @test rand(Int16x2) isa Int16x2

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, ~x), (n, xs, ys, zs, x, y, z) -> .~xs)

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, +x), (n, xs, ys, zs, x, y, z) -> .+xs .% Int16)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, -x), (n, xs, ys, zs, x, y, z) -> .-xs .% Int16)

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, x & y), (n, xs, ys, zs, x, y, z) -> xs .& ys)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, x | y), (n, xs, ys, zs, x, y, z) -> xs .| ys)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, x ⊻ y), (n, xs, ys, zs, x, y, z) -> xs .⊻ ys)

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, x + y), (n, xs, ys, zs, x, y, z) -> (xs .+ ys) .% Int16)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, x - y), (n, xs, ys, zs, x, y, z) -> (xs .- ys) .% Int16)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, min(x, y)), (n, xs, ys, zs, x, y, z) -> min.(xs, ys) .% Int16)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, max(x, y)), (n, xs, ys, zs, x, y, z) -> max.(xs, ys) .% Int16)
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Int32}, clamp1(x, y, z)),
        (n, xs, ys, zs, x, y, z) -> clamp1.(xs, ys, zs) .% Int16,
    )

    compare((n, xs, ys, zs, x, y, z) -> x == y, (n, xs, ys, zs, x, y, z) -> all(xs .== ys))
    compare((n, xs, ys, zs, x, y, z) -> x != y, (n, xs, ys, zs, x, y, z) -> any(xs .!= ys))

    print("$(CR)$(EL)")
    flush(stdout)
end

Random.seed!(0)
@testset "Float16x2" begin
    n = Float16x2[]
    xs = NTuple{2,Float32}[]
    ys = NTuple{2,Float32}[]
    zs = NTuple{2,Float32}[]
    x = Float16x2[]
    y = Float16x2[]
    z = Float16x2[]

    for iter in 1:131072
        n1 = zero(Float16x2)
        xs1 = tuple(Float32.(rand(Float16, 2))...)
        ys1 = tuple(Float32.(rand(Float16, 2))...)
        zs1 = tuple(Float32.(rand(Float16, 2))...)
        x1 = Float16x2(xs1...)
        y1 = Float16x2(ys1...)
        z1 = Float16x2(zs1...)

        push!(n, n1)
        push!(xs, xs1)
        push!(ys, ys1)
        push!(zs, zs1)
        push!(x, x1)
        push!(y, y1)
        push!(z, z1)
    end

    function compare(fl, fr; atol=nothing)
        rcpul = run_on_cpu(fl, n, xs, ys, zs, x, y, z)
        rcpur = run_on_cpu(fr, n, xs, ys, zs, x, y, z)
        if atol ≡ nothing
            @test rcpul == rcpur
        else
            @test all(isapprox([rcpul...], [rcpur...]; atol=atol) for (rcpul, rcpur) in zip(rcpul, rcpur))
        end
        if CUDA.functional()
            rcudal = similar(rcpul)
            rcudar = similar(rcpur)
            run_on_cuda!(fl, rcudal, n, xs, ys, zs, x, y, z)
            run_on_cuda!(fr, rcudar, n, xs, ys, zs, x, y, z)
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

    compare((n, xs, ys, zs, x, y, z) -> Float16x2(Float16.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Float16x2(Float32.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Float16x2(Float16.(xs)), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> Float16x2(Float32.(xs)), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float16}, x), (n, xs, ys, zs, x, y, z) -> xs)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, x), (n, xs, ys, zs, x, y, z) -> xs)

    @test string.(x) == "Float16x2" .* string.(xs)

    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, Float16x2(x.val & 0x0000ffff)),
        (n, xs, ys, zs, x, y, z) -> (xs[1], 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, Float16x2(x.val & 0xffff0000)),
        (n, xs, ys, zs, x, y, z) -> (0, xs[2]),
    )

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, reverse(x)), (n, xs, ys, zs, x, y, z) -> reverse(xs))

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, zero(Float16x2)), (n, xs, ys, zs, x, y, z) -> (0, 0))
    compare((n, xs, ys, zs, x, y, z) -> zero(Float16x2), (n, xs, ys, zs, x, y, z) -> zero(x))

    @test iszero(zero(Float16x2)) isa Bool
    @test iszero(zero(Float16x2))
    @test rand(Float16x2) isa Float16x2

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, abs(x)), (n, xs, ys, zs, x, y, z) -> abs.(xs))
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, +x), (n, xs, ys, zs, x, y, z) -> .+xs)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, -x), (n, xs, ys, zs, x, y, z) -> .-xs)

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, x + y), (n, xs, ys, zs, x, y, z) -> xs .+ ys; atol=eps(Float16))
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, x - y), (n, xs, ys, zs, x, y, z) -> xs .- ys; atol=eps(Float16))
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, x * y), (n, xs, ys, zs, x, y, z) -> xs .* ys; atol=eps(Float16))
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, min(x, y)), (n, xs, ys, zs, x, y, z) -> min.(xs, ys))
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, max(x, y)), (n, xs, ys, zs, x, y, z) -> max.(xs, ys))
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, clamp1(x, y, z)), (n, xs, ys, zs, x, y, z) -> clamp1.(xs, ys, zs)
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, complex_mul(x, y)),
        (n, xs, ys, zs, x, y, z) -> tuple_complex_mul(xs, ys);
        atol=2 * eps(Float16),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, swapped_complex_mul(x, y)),
        (n, xs, ys, zs, x, y, z) -> tuple_swapped_complex_mul(xs, ys);
        atol=2 * eps(Float16),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, muladd(x, y, z)),
        (n, xs, ys, zs, x, y, z) -> muladd.(xs, ys, zs);
        atol=2 * eps(Float16),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, complex_muladd(x, y, z)),
        (n, xs, ys, zs, x, y, z) -> tuple_complex_muladd(xs, ys, zs);
        atol=3 * eps(Float16),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, swapped_complex_muladd(x, y, z)),
        (n, xs, ys, zs, x, y, z) -> tuple_swapped_complex_muladd(xs, ys, zs);
        atol=3 * eps(Float16),
    )

    compare((n, xs, ys, zs, x, y, z) -> x == y, (n, xs, ys, zs, x, y, z) -> all(xs .=== ys))
    compare((n, xs, ys, zs, x, y, z) -> x != y, (n, xs, ys, zs, x, y, z) -> any(xs .!== ys))

    print("$(CR)$(EL)")
    flush(stdout)
end

Random.seed!(0)
@testset "BFloat16x2" begin
    n = BFloat16x2[]
    xs = NTuple{2,Float32}[]
    ys = NTuple{2,Float32}[]
    zs = NTuple{2,Float32}[]
    x = BFloat16x2[]
    y = BFloat16x2[]
    z = BFloat16x2[]

    for iter in 1:131072
        n1 = zero(BFloat16x2)
        xs1 = tuple(Float32.(rand(BFloat16, 2))...)
        ys1 = tuple(Float32.(rand(BFloat16, 2))...)
        zs1 = tuple(Float32.(rand(BFloat16, 2))...)
        x1 = BFloat16x2(xs1...)
        y1 = BFloat16x2(ys1...)
        z1 = BFloat16x2(zs1...)

        push!(n, n1)
        push!(xs, xs1)
        push!(ys, ys1)
        push!(zs, zs1)
        push!(x, x1)
        push!(y, y1)
        push!(z, z1)
    end

    function compare(fl, fr; atol=nothing)
        rcpul = run_on_cpu(fl, n, xs, ys, zs, x, y, z)
        rcpur = run_on_cpu(fr, n, xs, ys, zs, x, y, z)
        if atol ≡ nothing
            @test rcpul == rcpur
        else
            @test all(isapprox([rcpul...], [rcpur...]; atol=atol) for (rcpul, rcpur) in zip(rcpul, rcpur))
        end
        if CUDA.functional()
            rcudal = similar(rcpul)
            rcudar = similar(rcpur)
            run_on_cuda!(fl, rcudal, n, xs, ys, zs, x, y, z)
            run_on_cuda!(fr, rcudar, n, xs, ys, zs, x, y, z)
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

    compare((n, xs, ys, zs, x, y, z) -> BFloat16x2(BFloat16.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> BFloat16x2(Float32.(xs)...), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> BFloat16x2(BFloat16.(xs)), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> BFloat16x2(Float32.(xs)), (n, xs, ys, zs, x, y, z) -> x)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,BFloat16}, x), (n, xs, ys, zs, x, y, z) -> xs)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, x), (n, xs, ys, zs, x, y, z) -> xs)

    @test string.(x) == "BFloat16x2" .* string.(xs)

    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, BFloat16x2(x.val & 0x0000ffff)),
        (n, xs, ys, zs, x, y, z) -> (xs[1], 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, BFloat16x2(x.val & 0xffff0000)),
        (n, xs, ys, zs, x, y, z) -> (0, xs[2]),
    )

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, zero(BFloat16x2)), (n, xs, ys, zs, x, y, z) -> (0, 0))
    compare((n, xs, ys, zs, x, y, z) -> zero(BFloat16x2), (n, xs, ys, zs, x, y, z) -> zero(x))

    @test iszero(zero(BFloat16x2)) isa Bool
    @test iszero(zero(BFloat16x2))
    @test rand(BFloat16x2) isa BFloat16x2

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, abs(x)), (n, xs, ys, zs, x, y, z) -> abs.(xs))
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, +x), (n, xs, ys, zs, x, y, z) -> .+xs)
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, -x), (n, xs, ys, zs, x, y, z) -> .-xs)

    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, x + y), (n, xs, ys, zs, x, y, z) -> xs .+ ys; atol=eps(BFloat16))
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, x - y), (n, xs, ys, zs, x, y, z) -> xs .- ys; atol=eps(BFloat16))
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, x * y), (n, xs, ys, zs, x, y, z) -> xs .* ys; atol=eps(BFloat16))
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, min(x, y)), (n, xs, ys, zs, x, y, z) -> min.(xs, ys))
    compare((n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, max(x, y)), (n, xs, ys, zs, x, y, z) -> max.(xs, ys))
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, clamp1(x, y, z)), (n, xs, ys, zs, x, y, z) -> clamp1.(xs, ys, zs)
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, complex_mul(x, y)),
        (n, xs, ys, zs, x, y, z) -> tuple_complex_mul(xs, ys);
        atol=2 * eps(BFloat16),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, swapped_complex_mul(x, y)),
        (n, xs, ys, zs, x, y, z) -> tuple_swapped_complex_mul(xs, ys);
        atol=2 * eps(BFloat16),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, muladd(x, y, z)),
        (n, xs, ys, zs, x, y, z) -> muladd.(xs, ys, zs);
        atol=2 * eps(BFloat16),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, complex_muladd(x, y, z)),
        (n, xs, ys, zs, x, y, z) -> tuple_complex_muladd(xs, ys, zs);
        atol=3 * eps(BFloat16),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, swapped_complex_muladd(x, y, z)),
        (n, xs, ys, zs, x, y, z) -> tuple_swapped_complex_muladd(xs, ys, zs);
        atol=3 * eps(BFloat16),
    )

    compare((n, xs, ys, zs, x, y, z) -> x == y, (n, xs, ys, zs, x, y, z) -> all(xs .=== ys))
    compare((n, xs, ys, zs, x, y, z) -> x != y, (n, xs, ys, zs, x, y, z) -> any(xs .!== ys))

    print("$(CR)$(EL)")
    flush(stdout)
end

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

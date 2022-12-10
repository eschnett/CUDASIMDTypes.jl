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

make_int4(x::Integer) = ((x + 8) & 0xf - 8) % Int8

################################################################################

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

@testset "Int4x2" begin
    # Test exhaustively
    xlo = Int32[]
    xhi = Int32[]
    ylo = Int32[]
    yhi = Int32[]
    x = Int4x2[]
    y = Int4x2[]

    for xlo1 in (-Int32(8)):(+Int32(7)),
        xhi1 in (-Int32(8)):(+Int32(7)),
        ylo1 in (-Int32(8)):(+Int32(7)),
        yhi1 in (-Int32(8)):(+Int32(7))

        push!(xlo, xlo1)
        push!(xhi, xhi1)
        push!(ylo, ylo1)
        push!(yhi, yhi1)
        push!(x, Int4x2(xlo1, xhi1))
        push!(y, Int4x2(ylo1, yhi1))
    end

    function compare(fl, fr)
        rcpul = run_on_cpu(fl, xlo, xhi, ylo, yhi, x, y)
        rcpur = run_on_cpu(fr, xlo, xhi, ylo, yhi, x, y)
        @test rcpul == rcpur
        if CUDA.functional()
            rcudal = similar(rcpul)
            rcudar = similar(rcpur)
            run_on_cuda!(fl, rcudal, xlo, xhi, ylo, yhi, x, y)
            run_on_cuda!(fr, rcudar, xlo, xhi, ylo, yhi, x, y)
            @test rcudal == rcudar
            @test rcudal == rcpul
            @test rcudar == rcpur
        end
        print(".")
        flush(stdout)
        return nothing
    end

    # Test constructors
    compare((xlo, xhi, ylo, yhi, x, y) -> Int4x2((Int8(xlo), Int8(xhi))), (xlo, xhi, ylo, yhi, x, y) -> x)
    compare((xlo, xhi, ylo, yhi, x, y) -> Int4x2((Int16(xlo), Int16(xhi))), (xlo, xhi, ylo, yhi, x, y) -> x)
    compare((xlo, xhi, ylo, yhi, x, y) -> Int4x2((Int32(xlo), Int32(xhi))), (xlo, xhi, ylo, yhi, x, y) -> x)
    compare((xlo, xhi, ylo, yhi, x, y) -> Int4x2((Int64(xlo), Int64(xhi))), (xlo, xhi, ylo, yhi, x, y) -> x)
    compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int8}, x), (xlo, xhi, ylo, yhi, x, y) -> (xlo, xhi))
    compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int16}, x), (xlo, xhi, ylo, yhi, x, y) -> (xlo, xhi))
    compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, x), (xlo, xhi, ylo, yhi, x, y) -> (xlo, xhi))
    compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int64}, x), (xlo, xhi, ylo, yhi, x, y) -> (xlo, xhi))

    compare((xlo, xhi, ylo, yhi, x, y) -> Int4x2(Int8(xlo), Int8(xhi)), (xlo, xhi, ylo, yhi, x, y) -> x)
    compare((xlo, xhi, ylo, yhi, x, y) -> Int4x2(Int16(xlo), Int8(xhi)), (xlo, xhi, ylo, yhi, x, y) -> x)
    compare((xlo, xhi, ylo, yhi, x, y) -> Int4x2(Int32(xlo), Int8(xhi)), (xlo, xhi, ylo, yhi, x, y) -> x)
    compare((xlo, xhi, ylo, yhi, x, y) -> Int4x2(Int64(xlo), Int8(xhi)), (xlo, xhi, ylo, yhi, x, y) -> x)
    compare((xlo, xhi, ylo, yhi, x, y) -> Int4x2(Int8(xlo), Int16(xhi)), (xlo, xhi, ylo, yhi, x, y) -> x)
    compare((xlo, xhi, ylo, yhi, x, y) -> Int4x2(Int16(xlo), Int16(xhi)), (xlo, xhi, ylo, yhi, x, y) -> x)
    compare((xlo, xhi, ylo, yhi, x, y) -> Int4x2(Int32(xlo), Int16(xhi)), (xlo, xhi, ylo, yhi, x, y) -> x)
    compare((xlo, xhi, ylo, yhi, x, y) -> Int4x2(Int64(xlo), Int16(xhi)), (xlo, xhi, ylo, yhi, x, y) -> x)
    compare((xlo, xhi, ylo, yhi, x, y) -> Int4x2(Int8(xlo), Int32(xhi)), (xlo, xhi, ylo, yhi, x, y) -> x)
    compare((xlo, xhi, ylo, yhi, x, y) -> Int4x2(Int16(xlo), Int32(xhi)), (xlo, xhi, ylo, yhi, x, y) -> x)
    compare((xlo, xhi, ylo, yhi, x, y) -> Int4x2(Int32(xlo), Int32(xhi)), (xlo, xhi, ylo, yhi, x, y) -> x)
    compare((xlo, xhi, ylo, yhi, x, y) -> Int4x2(Int64(xlo), Int32(xhi)), (xlo, xhi, ylo, yhi, x, y) -> x)
    compare((xlo, xhi, ylo, yhi, x, y) -> Int4x2(Int8(xlo), Int64(xhi)), (xlo, xhi, ylo, yhi, x, y) -> x)
    compare((xlo, xhi, ylo, yhi, x, y) -> Int4x2(Int16(xlo), Int64(xhi)), (xlo, xhi, ylo, yhi, x, y) -> x)
    compare((xlo, xhi, ylo, yhi, x, y) -> Int4x2(Int32(xlo), Int64(xhi)), (xlo, xhi, ylo, yhi, x, y) -> x)
    compare((xlo, xhi, ylo, yhi, x, y) -> Int4x2(Int64(xlo), Int64(xhi)), (xlo, xhi, ylo, yhi, x, y) -> x)

    compare((xlo, xhi, ylo, yhi, x, y) -> Int16x2(x), (xlo, xhi, ylo, yhi, x, y) -> Int16x2(xlo, xhi))
    compare((xlo, xhi, ylo, yhi, x, y) -> Int4x2(Int16x2(xlo, xhi)), (xlo, xhi, ylo, yhi, x, y) -> x)

    # Test output
    @test string.(x) == string.(tuple.(xlo, xhi))

    # Test nibble ordering
    compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, Int4x2(x.val & 0x0f)), (xlo, xhi, ylo, yhi, x, y) -> (xlo, 0))
    compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, Int4x2(x.val & 0xf0)), (xlo, xhi, ylo, yhi, x, y) -> (0, xhi))

    # zero
    compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, zero(Int4x2)), (xlo, xhi, ylo, yhi, x, y) -> (0, 0))
    compare((xlo, xhi, ylo, yhi, x, y) -> zero(Int4x2), (xlo, xhi, ylo, yhi, x, y) -> zero(x))

    @test iszero(zero(Int4x2)) isa Bool
    @test iszero(zero(Int4x2))
    @test rand(Int4x2) isa Int4x2

    # logical not
    compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, ~x), (xlo, xhi, ylo, yhi, x, y) -> (~xlo, ~xhi))

    # arithmetic pos/negation
    compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, +x), (xlo, xhi, ylo, yhi, x, y) -> make_int4.((+xlo, +xhi)))
    compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, -x), (xlo, xhi, ylo, yhi, x, y) -> make_int4.((-xlo, -xhi)))

    # logical operations
    compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, x & y), (xlo, xhi, ylo, yhi, x, y) -> (xlo & ylo, xhi & yhi))
    compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, x | y), (xlo, xhi, ylo, yhi, x, y) -> (xlo | ylo, xhi | yhi))
    compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, x ⊻ y), (xlo, xhi, ylo, yhi, x, y) -> (xlo ⊻ ylo, xhi ⊻ yhi))

    # arithmetic operations
    compare(
        (xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, x + y),
        (xlo, xhi, ylo, yhi, x, y) -> make_int4.((xlo + ylo, xhi + yhi)),
    )
    compare(
        (xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, x - y),
        (xlo, xhi, ylo, yhi, x, y) -> make_int4.((xlo - ylo, xhi - yhi)),
    )
    compare(
        (xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, min(x, y)),
        (xlo, xhi, ylo, yhi, x, y) -> make_int4.((min(xlo, ylo), min(xhi, yhi))),
    )
    compare(
        (xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, max(x, y)),
        (xlo, xhi, ylo, yhi, x, y) -> make_int4.((max(xlo, ylo), max(xhi, yhi))),
    )

    print("$(CR)$(EL)")
    flush(stdout)
end

Random.seed!(0)
@testset "Int4x8" begin
    n = Int4x8[]
    xs = NTuple{8,Int32}[]
    ys = NTuple{8,Int32}[]
    x = Int4x8[]
    y = Int4x8[]

    for iter in 1:131072
        n1 = zero(Int4x8)
        xs1 = tuple(rand((-Int32(8)):(+Int32(7)), 8)...)
        ys1 = tuple(rand((-Int32(8)):(+Int32(7)), 8)...)
        x1 = Int4x8(xs1...)
        y1 = Int4x8(ys1...)

        push!(n, n1)
        push!(xs, xs1)
        push!(ys, ys1)
        push!(x, x1)
        push!(y, y1)
    end

    function compare(fl, fr)
        rcpul = run_on_cpu(fl, n, xs, ys, x, y)
        rcpur = run_on_cpu(fr, n, xs, ys, x, y)
        @test rcpul == rcpur
        if CUDA.functional()
            rcudal = similar(rcpul)
            rcudar = similar(rcpur)
            run_on_cuda!(fl, rcudal, n, xs, ys, x, y)
            run_on_cuda!(fr, rcudar, n, xs, ys, x, y)
            @test rcudal == rcudar
            @test rcudal == rcpul
            @test rcudar == rcpur
        end
        print(".")
        flush(stdout)
        return nothing
    end

    compare((n, xs, ys, x, y) -> Int4x8(xs), (n, xs, ys, x, y) -> x)
    compare(
        (n, xs, ys, x, y) -> Int4x8((Int8x4(xs[1], xs[3], xs[5], xs[7]), Int8x4(xs[2], xs[4], xs[6], xs[8]))),
        (n, xs, ys, x, y) -> x,
    )
    compare(
        (n, xs, ys, x, y) -> Int4x8((Int16x2(xs[1], xs[5]), Int16x2(xs[2], xs[6]), Int16x2(xs[3], xs[7]), Int16x2(xs[4], xs[8]))),
        (n, xs, ys, x, y) -> x,
    )

    compare((n, xs, ys, x, y) -> Int4x8(Int8.(xs)...), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> Int4x8(Int16.(xs)...), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> Int4x8(Int32.(xs)...), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> Int4x8(Int64.(xs)...), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> Int4x8(Int8.(xs)), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> Int4x8(Int16.(xs)), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> Int4x8(Int32.(xs)), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> Int4x8(Int64.(xs)), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> convert(NTuple{8,Int8}, x), (n, xs, ys, x, y) -> xs)
    compare((n, xs, ys, x, y) -> convert(NTuple{8,Int16}, x), (n, xs, ys, x, y) -> xs)
    compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, x), (n, xs, ys, x, y) -> xs)
    compare((n, xs, ys, x, y) -> convert(NTuple{8,Int64}, x), (n, xs, ys, x, y) -> xs)

    compare(
        (n, xs, ys, x, y) -> convert(NTuple{2,Int8x4}, x),
        (n, xs, ys, x, y) -> (Int8x4(xs[1], xs[3], xs[5], xs[7]), Int8x4(xs[2], xs[4], xs[6], xs[8])),
    )
    compare(
        (n, xs, ys, x, y) -> convert(NTuple{4,Int16x2}, x),
        (n, xs, ys, x, y) -> (Int16x2(xs[1], xs[5]), Int16x2(xs[2], xs[6]), Int16x2(xs[3], xs[7]), Int16x2(xs[4], xs[8])),
    )

    @test string.(x) == string.(xs)

    compare(
        (n, xs, ys, x, y) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x0000000f)), (n, xs, ys, x, y) -> (xs[1], 0, 0, 0, 0, 0, 0, 0)
    )
    compare(
        (n, xs, ys, x, y) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x000000f0)), (n, xs, ys, x, y) -> (0, xs[2], 0, 0, 0, 0, 0, 0)
    )
    compare(
        (n, xs, ys, x, y) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x00000f00)), (n, xs, ys, x, y) -> (0, 0, xs[3], 0, 0, 0, 0, 0)
    )
    compare(
        (n, xs, ys, x, y) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x0000f000)), (n, xs, ys, x, y) -> (0, 0, 0, xs[4], 0, 0, 0, 0)
    )
    compare(
        (n, xs, ys, x, y) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x000f0000)), (n, xs, ys, x, y) -> (0, 0, 0, 0, xs[5], 0, 0, 0)
    )
    compare(
        (n, xs, ys, x, y) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x00f00000)), (n, xs, ys, x, y) -> (0, 0, 0, 0, 0, xs[6], 0, 0)
    )
    compare(
        (n, xs, ys, x, y) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x0f000000)), (n, xs, ys, x, y) -> (0, 0, 0, 0, 0, 0, xs[7], 0)
    )
    compare(
        (n, xs, ys, x, y) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0xf0000000)), (n, xs, ys, x, y) -> (0, 0, 0, 0, 0, 0, 0, xs[8])
    )

    compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, zero(Int4x8)), (n, xs, ys, x, y) -> (0, 0, 0, 0, 0, 0, 0, 0))
    compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, zero(Int4x8)), (n, xs, ys, x, y) -> (0, 0, 0, 0, 0, 0, 0, 0))
    compare((n, xs, ys, x, y) -> zero(Int4x8), (n, xs, ys, x, y) -> zero(x))

    @test iszero(zero(Int4x8)) isa Bool
    @test iszero(zero(Int4x8))
    @test rand(Int4x8) isa Int4x8

    compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, ~x), (n, xs, ys, x, y) -> .~xs)

    compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, +x), (n, xs, ys, x, y) -> make_int4.(.+xs))
    compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, -x), (n, xs, ys, x, y) -> make_int4.(.-xs))

    compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, x & y), (n, xs, ys, x, y) -> xs .& ys)
    compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, x | y), (n, xs, ys, x, y) -> xs .| ys)
    compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, x ⊻ y), (n, xs, ys, x, y) -> xs .⊻ ys)

    compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, x + y), (n, xs, ys, x, y) -> make_int4.(xs .+ ys))
    compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, x - y), (n, xs, ys, x, y) -> make_int4.(xs .- ys))
    compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, min(x, y)), (n, xs, ys, x, y) -> make_int4.(min.(xs, ys)))
    compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, max(x, y)), (n, xs, ys, x, y) -> make_int4.(max.(xs, ys)))

    print("$(CR)$(EL)")
    flush(stdout)
end

Random.seed!(0)
@testset "Int8x4" begin
    n = Int8x4[]
    xs = NTuple{4,Int32}[]
    ys = NTuple{4,Int32}[]
    x = Int8x4[]
    y = Int8x4[]

    for iter in 1:131072
        n1 = zero(Int8x4)
        xs1 = tuple(Int32.(rand(Int8, 4))...)
        ys1 = tuple(Int32.(rand(Int8, 4))...)
        x1 = Int8x4(xs1...)
        y1 = Int8x4(ys1...)

        push!(n, n1)
        push!(xs, xs1)
        push!(ys, ys1)
        push!(x, x1)
        push!(y, y1)
    end

    function compare(fl, fr)
        rcpul = run_on_cpu(fl, n, xs, ys, x, y)
        rcpur = run_on_cpu(fr, n, xs, ys, x, y)
        @test rcpul == rcpur
        if CUDA.functional()
            rcudal = similar(rcpul)
            rcudar = similar(rcpur)
            run_on_cuda!(fl, rcudal, n, xs, ys, x, y)
            run_on_cuda!(fr, rcudar, n, xs, ys, x, y)
            @test rcudal == rcudar
            @test rcudal == rcpul
            @test rcudar == rcpur
        end
        print(".")
        flush(stdout)
        return nothing
    end

    compare((n, xs, ys, x, y) -> Int8x4(xs), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> Int8x4((Int16x2(xs[1], xs[3]), Int16x2(xs[2], xs[4]))), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> Int8x4(Int8.(xs)...), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> Int8x4(Int16.(xs)...), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> Int8x4(Int32.(xs)...), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> Int8x4(Int64.(xs)...), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> Int8x4(Int8.(xs)), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> Int8x4(Int16.(xs)), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> Int8x4(Int32.(xs)), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> Int8x4(Int64.(xs)), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> convert(NTuple{4,Int8}, x), (n, xs, ys, x, y) -> xs)
    compare((n, xs, ys, x, y) -> convert(NTuple{4,Int16}, x), (n, xs, ys, x, y) -> xs)
    compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, x), (n, xs, ys, x, y) -> xs)
    compare((n, xs, ys, x, y) -> convert(NTuple{4,Int64}, x), (n, xs, ys, x, y) -> xs)
    compare((n, xs, ys, x, y) -> convert(NTuple{2,Int16x2}, x), (n, xs, ys, x, y) -> (Int16x2(xs[1], xs[3]), Int16x2(xs[2], xs[4])))

    @test string.(x) == string.(xs)

    compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, Int8x4(x.val & 0x000000ff)), (n, xs, ys, x, y) -> (xs[1], 0, 0, 0))
    compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, Int8x4(x.val & 0x0000ff00)), (n, xs, ys, x, y) -> (0, xs[2], 0, 0))
    compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, Int8x4(x.val & 0x00ff0000)), (n, xs, ys, x, y) -> (0, 0, xs[3], 0))
    compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, Int8x4(x.val & 0xff000000)), (n, xs, ys, x, y) -> (0, 0, 0, xs[4]))

    compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, zero(Int8x4)), (n, xs, ys, x, y) -> (0, 0, 0, 0))
    compare((n, xs, ys, x, y) -> zero(Int8x4), (n, xs, ys, x, y) -> zero(x))

    @test iszero(zero(Int8x4)) isa Bool
    @test iszero(zero(Int8x4))
    @test rand(Int8x4) isa Int8x4

    compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, ~x), (n, xs, ys, x, y) -> .~xs)

    compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, +x), (n, xs, ys, x, y) -> .+xs .% Int8)
    compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, -x), (n, xs, ys, x, y) -> .-xs .% Int8)

    compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, x & y), (n, xs, ys, x, y) -> xs .& ys)
    compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, x | y), (n, xs, ys, x, y) -> xs .| ys)
    compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, x ⊻ y), (n, xs, ys, x, y) -> xs .⊻ ys)

    compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, x + y), (n, xs, ys, x, y) -> (xs .+ ys) .% Int8)
    compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, x - y), (n, xs, ys, x, y) -> (xs .- ys) .% Int8)
    compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, min(x, y)), (n, xs, ys, x, y) -> min.(xs, ys) .% Int8)
    compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, max(x, y)), (n, xs, ys, x, y) -> max.(xs, ys) .% Int8)

    print("$(CR)$(EL)")
    flush(stdout)
end

Random.seed!(0)
@testset "Int16x2" begin
    n = Int16x2[]
    xs = NTuple{2,Int32}[]
    ys = NTuple{2,Int32}[]
    x = Int16x2[]
    y = Int16x2[]

    for iter in 1:131072
        n1 = zero(Int16x2)
        xs1 = tuple(Int32.(rand(Int16, 2))...)
        ys1 = tuple(Int32.(rand(Int16, 2))...)
        x1 = Int16x2(xs1...)
        y1 = Int16x2(ys1...)

        push!(n, n1)
        push!(xs, xs1)
        push!(ys, ys1)
        push!(x, x1)
        push!(y, y1)
    end

    function compare(fl, fr)
        rcpul = run_on_cpu(fl, n, xs, ys, x, y)
        rcpur = run_on_cpu(fr, n, xs, ys, x, y)
        @test rcpul == rcpur
        if CUDA.functional()
            rcudal = similar(rcpul)
            rcudar = similar(rcpur)
            run_on_cuda!(fl, rcudal, n, xs, ys, x, y)
            run_on_cuda!(fr, rcudar, n, xs, ys, x, y)
            @test rcudal == rcudar
            @test rcudal == rcpul
            @test rcudar == rcpur
        end
        print(".")
        flush(stdout)
        return nothing
    end

    compare((n, xs, ys, x, y) -> Int16x2(xs), (n, xs, ys, x, y) -> x)

    compare((n, xs, ys, x, y) -> Int16x2(Int16.(xs)...), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> Int16x2(Int32.(xs)...), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> Int16x2(Int64.(xs)...), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> Int16x2(Int16.(xs)), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> Int16x2(Int32.(xs)), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> Int16x2(Int64.(xs)), (n, xs, ys, x, y) -> x)
    compare((n, xs, ys, x, y) -> convert(NTuple{2,Int16}, x), (n, xs, ys, x, y) -> xs)
    compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, x), (n, xs, ys, x, y) -> xs)
    compare((n, xs, ys, x, y) -> convert(NTuple{2,Int64}, x), (n, xs, ys, x, y) -> xs)

    @test string.(x) == string.(xs)

    compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, Int16x2(x.val & 0x0000ffff)), (n, xs, ys, x, y) -> (xs[1], 0))
    compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, Int16x2(x.val & 0xffff0000)), (n, xs, ys, x, y) -> (0, xs[2]))

    compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, zero(Int16x2)), (n, xs, ys, x, y) -> (0, 0))
    compare((n, xs, ys, x, y) -> zero(Int16x2), (n, xs, ys, x, y) -> zero(x))

    @test iszero(zero(Int16x2)) isa Bool
    @test iszero(zero(Int16x2))
    @test rand(Int16x2) isa Int16x2

    compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, ~x), (n, xs, ys, x, y) -> .~xs)

    compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, +x), (n, xs, ys, x, y) -> .+xs .% Int16)
    compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, -x), (n, xs, ys, x, y) -> .-xs .% Int16)

    compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, x & y), (n, xs, ys, x, y) -> xs .& ys)
    compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, x | y), (n, xs, ys, x, y) -> xs .| ys)
    compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, x ⊻ y), (n, xs, ys, x, y) -> xs .⊻ ys)

    compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, x + y), (n, xs, ys, x, y) -> (xs .+ ys) .% Int16)
    compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, x - y), (n, xs, ys, x, y) -> (xs .- ys) .% Int16)
    compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, min(x, y)), (n, xs, ys, x, y) -> min.(xs, ys) .% Int16)
    compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, max(x, y)), (n, xs, ys, x, y) -> max.(xs, ys) .% Int16)

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

    @test string.(x) == string.(xs)

    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, Float16x2(x.val & 0x0000ffff)),
        (n, xs, ys, zs, x, y, z) -> (xs[1], 0),
    )
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, Float16x2(x.val & 0xffff0000)),
        (n, xs, ys, zs, x, y, z) -> (0, xs[2]),
    )

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
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, muladd(x, y, z)),
        (n, xs, ys, zs, x, y, z) -> muladd.(xs, ys, zs);
        atol=2 * eps(Float16),
    )

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

    @test string.(x) == string.(xs)

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
    compare(
        (n, xs, ys, zs, x, y, z) -> convert(NTuple{2,Float32}, muladd(x, y, z)),
        (n, xs, ys, zs, x, y, z) -> muladd.(xs, ys, zs);
        atol=2 * eps(BFloat16),
    )

    print("$(CR)$(EL)")
    flush(stdout)
end

@testset "Convert Int16 to Float16" begin
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

    compare((xs, x) -> Float16x2(x), (xs, x) -> Float16x2(xs[1], xs[2]))
    compare((xs, x) -> Int16x2(Float16x2(xs[1], xs[2])), (xs, x) -> x)

    print("$(CR)$(EL)")
    flush(stdout)
end

@testset "Convert Int16 to BFloat16" begin
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

    compare((xs, x) -> BFloat16x2(x), (xs, x) -> BFloat16x2(xs[1], xs[2]))
    compare((xs, x) -> Int16x2(BFloat16x2(xs[1], xs[2])), (xs, x) -> x)

    print("$(CR)$(EL)")
    flush(stdout)
end

Random.seed!(0)
@testset "Convert Int8 to Float16" begin
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

    compare((xs, x) -> convert(NTuple{2,Float16x2}, x), (xs, x) -> (Float16x2(xs[1], xs[3]), Float16x2(xs[2], xs[4])))
    compare((xs, x) -> Int8x4((Float16x2(xs[1], xs[3]), Float16x2(xs[2], xs[4]))), (xs, x) -> x)

    print("$(CR)$(EL)")
    flush(stdout)
end

Random.seed!(0)
@testset "Convert Int4 to Float16" begin
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

    compare(
        (xs, x) -> convert(NTuple{4,Float16x2}, x),
        (xs, x) -> (Float16x2(xs[1], xs[5]), Float16x2(xs[2], xs[6]), Float16x2(xs[3], xs[7]), Float16x2(xs[4], xs[8])),
    )
    compare(
        (xs, x) -> Int4x8((Float16x2(xs[1], xs[5]), Float16x2(xs[2], xs[6]), Float16x2(xs[3], xs[7]), Float16x2(xs[4], xs[8]))),
        (xs, x) -> x,
    )

    print("$(CR)$(EL)")
    flush(stdout)
end

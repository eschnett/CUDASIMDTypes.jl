using CUDA
using CUDASIMDTypes
using Random
using Test

################################################################################

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
    end

    # Test constructors
    compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, x), (xlo, xhi, ylo, yhi, x, y) -> (xlo, xhi))
    compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int8}, x), (xlo, xhi, ylo, yhi, x, y) -> (xlo, xhi))

    # Test output
    @test string.(x) == string.(tuple.(xlo, xhi))

    # Test nibble ordering
    compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, Int4x2(x.val & 0x0f)), (xlo, xhi, ylo, yhi, x, y) -> (xlo, 0))
    compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, Int4x2(x.val & 0xf0)), (xlo, xhi, ylo, yhi, x, y) -> (0, xhi))

    # zero
    compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, zero(Int4x2)), (xlo, xhi, ylo, yhi, x, y) -> (0, 0))

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
end

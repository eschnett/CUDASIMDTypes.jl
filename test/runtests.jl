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
#TODO @testset "Int4x2" begin
#TODO     # Test exhaustively
#TODO     xlo = Int32[]
#TODO     xhi = Int32[]
#TODO     ylo = Int32[]
#TODO     yhi = Int32[]
#TODO     x = Int4x2[]
#TODO     y = Int4x2[]
#TODO 
#TODO     for xlo1 in (-Int32(8)):(+Int32(7)),
#TODO         xhi1 in (-Int32(8)):(+Int32(7)),
#TODO         ylo1 in (-Int32(8)):(+Int32(7)),
#TODO         yhi1 in (-Int32(8)):(+Int32(7))
#TODO 
#TODO         push!(xlo, xlo1)
#TODO         push!(xhi, xhi1)
#TODO         push!(ylo, ylo1)
#TODO         push!(yhi, yhi1)
#TODO         push!(x, Int4x2(xlo1, xhi1))
#TODO         push!(y, Int4x2(ylo1, yhi1))
#TODO     end
#TODO 
#TODO     function compare(fl, fr)
#TODO         rcpul = run_on_cpu(fl, xlo, xhi, ylo, yhi, x, y)
#TODO         rcpur = run_on_cpu(fr, xlo, xhi, ylo, yhi, x, y)
#TODO         @test rcpul == rcpur
#TODO         if CUDA.functional()
#TODO             rcudal = similar(rcpul)
#TODO             rcudar = similar(rcpur)
#TODO             run_on_cuda!(fl, rcudal, xlo, xhi, ylo, yhi, x, y)
#TODO             run_on_cuda!(fr, rcudar, xlo, xhi, ylo, yhi, x, y)
#TODO             @test rcudal == rcudar
#TODO             @test rcudal == rcpul
#TODO             @test rcudar == rcpur
#TODO         end
#TODO     end
#TODO 
#TODO     # Test constructors
#TODO     compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, x), (xlo, xhi, ylo, yhi, x, y) -> (xlo, xhi))
#TODO     compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int8}, x), (xlo, xhi, ylo, yhi, x, y) -> (xlo, xhi))
#TODO 
#TODO     # Test output
#TODO     @test string.(x) == string.(tuple.(xlo, xhi))
#TODO 
#TODO     # Test nibble ordering
#TODO     compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, Int4x2(x.val & 0x0f)), (xlo, xhi, ylo, yhi, x, y) -> (xlo, 0))
#TODO     compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, Int4x2(x.val & 0xf0)), (xlo, xhi, ylo, yhi, x, y) -> (0, xhi))
#TODO 
#TODO     # zero
#TODO     compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, zero(Int4x2)), (xlo, xhi, ylo, yhi, x, y) -> (0, 0))
#TODO 
#TODO     # logical not
#TODO     compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, ~x), (xlo, xhi, ylo, yhi, x, y) -> (~xlo, ~xhi))
#TODO 
#TODO     # arithmetic pos/negation
#TODO     compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, +x), (xlo, xhi, ylo, yhi, x, y) -> make_int4.((+xlo, +xhi)))
#TODO     compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, -x), (xlo, xhi, ylo, yhi, x, y) -> make_int4.((-xlo, -xhi)))
#TODO 
#TODO     # logical operations
#TODO     compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, x & y), (xlo, xhi, ylo, yhi, x, y) -> (xlo & ylo, xhi & yhi))
#TODO     compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, x | y), (xlo, xhi, ylo, yhi, x, y) -> (xlo | ylo, xhi | yhi))
#TODO     compare((xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, x ⊻ y), (xlo, xhi, ylo, yhi, x, y) -> (xlo ⊻ ylo, xhi ⊻ yhi))
#TODO 
#TODO     # arithmetic operations
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, x + y),
#TODO         (xlo, xhi, ylo, yhi, x, y) -> make_int4.((xlo + ylo, xhi + yhi)),
#TODO     )
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, x - y),
#TODO         (xlo, xhi, ylo, yhi, x, y) -> make_int4.((xlo - ylo, xhi - yhi)),
#TODO     )
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, min(x, y)),
#TODO         (xlo, xhi, ylo, yhi, x, y) -> make_int4.((min(xlo, ylo), min(xhi, yhi))),
#TODO     )
#TODO     compare(
#TODO         (xlo, xhi, ylo, yhi, x, y) -> convert(NTuple{2,Int32}, max(x, y)),
#TODO         (xlo, xhi, ylo, yhi, x, y) -> make_int4.((max(xlo, ylo), max(xhi, yhi))),
#TODO     )
#TODO end
#TODO 
#TODO Random.seed!(0)
#TODO @testset "Int4x8" begin
#TODO     n = Int4x8[]
#TODO     xs = NTuple{8,Int32}[]
#TODO     ys = NTuple{8,Int32}[]
#TODO     x = Int4x8[]
#TODO     y = Int4x8[]
#TODO 
#TODO     for iter in 1:131072
#TODO         n1 = zero(Int4x8)
#TODO         xs1 = tuple(rand((-Int32(8)):(+Int32(7)), 8)...)
#TODO         ys1 = tuple(rand((-Int32(8)):(+Int32(7)), 8)...)
#TODO         x1 = Int4x8(xs1...)
#TODO         y1 = Int4x8(ys1...)
#TODO 
#TODO         push!(n, n1)
#TODO         push!(xs, xs1)
#TODO         push!(ys, ys1)
#TODO         push!(x, x1)
#TODO         push!(y, y1)
#TODO     end
#TODO 
#TODO     function compare(fl, fr)
#TODO         rcpul = run_on_cpu(fl, n, xs, ys, x, y)
#TODO         rcpur = run_on_cpu(fr, n, xs, ys, x, y)
#TODO         @test rcpul == rcpur
#TODO         if CUDA.functional()
#TODO             rcudal = similar(rcpul)
#TODO             rcudar = similar(rcpur)
#TODO             run_on_cuda!(fl, rcudal, n, xs, ys, x, y)
#TODO             run_on_cuda!(fr, rcudar, n, xs, ys, x, y)
#TODO             @test rcudal == rcudar
#TODO             @test rcudal == rcpul
#TODO             @test rcudar == rcpur
#TODO         end
#TODO     end
#TODO 
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, x), (n, xs, ys, x, y) -> xs)
#TODO     compare(
#TODO         (n, xs, ys, x, y) -> convert(NTuple{2,Int8x4}, x),
#TODO         (n, xs, ys, x, y) -> (Int8x4(xs[1], xs[3], xs[5], xs[7]), Int8x4(xs[2], xs[4], xs[6], xs[8])),
#TODO     )
#TODO 
#TODO     @test string.(x) == string.(xs)
#TODO 
#TODO     compare(
#TODO         (n, xs, ys, x, y) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x0000000f)), (n, xs, ys, x, y) -> (xs[1], 0, 0, 0, 0, 0, 0, 0)
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, x, y) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x000000f0)), (n, xs, ys, x, y) -> (0, xs[2], 0, 0, 0, 0, 0, 0)
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, x, y) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x00000f00)), (n, xs, ys, x, y) -> (0, 0, xs[3], 0, 0, 0, 0, 0)
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, x, y) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x0000f000)), (n, xs, ys, x, y) -> (0, 0, 0, xs[4], 0, 0, 0, 0)
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, x, y) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x000f0000)), (n, xs, ys, x, y) -> (0, 0, 0, 0, xs[5], 0, 0, 0)
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, x, y) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x00f00000)), (n, xs, ys, x, y) -> (0, 0, 0, 0, 0, xs[6], 0, 0)
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, x, y) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0x0f000000)), (n, xs, ys, x, y) -> (0, 0, 0, 0, 0, 0, xs[7], 0)
#TODO     )
#TODO     compare(
#TODO         (n, xs, ys, x, y) -> convert(NTuple{8,Int32}, Int4x8(x.val & 0xf0000000)), (n, xs, ys, x, y) -> (0, 0, 0, 0, 0, 0, 0, xs[8])
#TODO     )
#TODO 
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, zero(Int4x8)), (n, xs, ys, x, y) -> (0, 0, 0, 0, 0, 0, 0, 0))
#TODO 
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, ~x), (n, xs, ys, x, y) -> .~xs)
#TODO 
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, +x), (n, xs, ys, x, y) -> make_int4.(.+xs))
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, -x), (n, xs, ys, x, y) -> make_int4.(.-xs))
#TODO 
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, x & y), (n, xs, ys, x, y) -> xs .& ys)
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, x | y), (n, xs, ys, x, y) -> xs .| ys)
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, x ⊻ y), (n, xs, ys, x, y) -> xs .⊻ ys)
#TODO 
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, x + y), (n, xs, ys, x, y) -> make_int4.(xs .+ ys))
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, x - y), (n, xs, ys, x, y) -> make_int4.(xs .- ys))
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, min(x, y)), (n, xs, ys, x, y) -> make_int4.(min.(xs, ys)))
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{8,Int32}, max(x, y)), (n, xs, ys, x, y) -> make_int4.(max.(xs, ys)))
#TODO end
#TODO 
#TODO Random.seed!(0)
#TODO @testset "Int8x4" begin
#TODO     n = Int8x4[]
#TODO     xs = NTuple{4,Int32}[]
#TODO     ys = NTuple{4,Int32}[]
#TODO     x = Int8x4[]
#TODO     y = Int8x4[]
#TODO 
#TODO     for iter in 1:131072
#TODO         n1 = zero(Int8x4)
#TODO         xs1 = tuple(Int32.(rand(Int8, 4))...)
#TODO         ys1 = tuple(Int32.(rand(Int8, 4))...)
#TODO         x1 = Int8x4(xs1...)
#TODO         y1 = Int8x4(ys1...)
#TODO 
#TODO         push!(n, n1)
#TODO         push!(xs, xs1)
#TODO         push!(ys, ys1)
#TODO         push!(x, x1)
#TODO         push!(y, y1)
#TODO     end
#TODO 
#TODO     function compare(fl, fr)
#TODO         rcpul = run_on_cpu(fl, n, xs, ys, x, y)
#TODO         rcpur = run_on_cpu(fr, n, xs, ys, x, y)
#TODO         @test rcpul == rcpur
#TODO         if CUDA.functional()
#TODO             rcudal = similar(rcpul)
#TODO             rcudar = similar(rcpur)
#TODO             run_on_cuda!(fl, rcudal, n, xs, ys, x, y)
#TODO             run_on_cuda!(fr, rcudar, n, xs, ys, x, y)
#TODO             @test rcudal == rcudar
#TODO             @test rcudal == rcpul
#TODO             @test rcudar == rcpur
#TODO         end
#TODO     end
#TODO 
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, x), (n, xs, ys, x, y) -> xs)
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{2,Int16x2}, x), (n, xs, ys, x, y) -> (Int16x2(xs[1], xs[3]), Int16x2(xs[2], xs[4])))
#TODO 
#TODO     @test string.(x) == string.(xs)
#TODO 
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, Int8x4(x.val & 0x000000ff)), (n, xs, ys, x, y) -> (xs[1], 0, 0, 0))
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, Int8x4(x.val & 0x0000ff00)), (n, xs, ys, x, y) -> (0, xs[2], 0, 0))
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, Int8x4(x.val & 0x00ff0000)), (n, xs, ys, x, y) -> (0, 0, xs[3], 0))
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, Int8x4(x.val & 0xff000000)), (n, xs, ys, x, y) -> (0, 0, 0, xs[4]))
#TODO 
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, zero(Int8x4)), (n, xs, ys, x, y) -> (0, 0, 0, 0))
#TODO 
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, ~x), (n, xs, ys, x, y) -> .~xs)
#TODO 
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, +x), (n, xs, ys, x, y) -> .+xs .% Int8)
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, -x), (n, xs, ys, x, y) -> .-xs .% Int8)
#TODO 
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, x & y), (n, xs, ys, x, y) -> xs .& ys)
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, x | y), (n, xs, ys, x, y) -> xs .| ys)
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, x ⊻ y), (n, xs, ys, x, y) -> xs .⊻ ys)
#TODO 
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, x + y), (n, xs, ys, x, y) -> (xs .+ ys) .% Int8)
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, x - y), (n, xs, ys, x, y) -> (xs .- ys) .% Int8)
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, min(x, y)), (n, xs, ys, x, y) -> min.(xs, ys) .% Int8)
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{4,Int32}, max(x, y)), (n, xs, ys, x, y) -> max.(xs, ys) .% Int8)
#TODO end
#TODO 
#TODO Random.seed!(0)
#TODO @testset "Int16x2" begin
#TODO     n = Int16x2[]
#TODO     xs = NTuple{2,Int32}[]
#TODO     ys = NTuple{2,Int32}[]
#TODO     x = Int16x2[]
#TODO     y = Int16x2[]
#TODO 
#TODO     for iter in 1:131072
#TODO         n1 = zero(Int16x2)
#TODO         xs1 = tuple(Int32.(rand(Int16, 2))...)
#TODO         ys1 = tuple(Int32.(rand(Int16, 2))...)
#TODO         x1 = Int16x2(xs1...)
#TODO         y1 = Int16x2(ys1...)
#TODO 
#TODO         push!(n, n1)
#TODO         push!(xs, xs1)
#TODO         push!(ys, ys1)
#TODO         push!(x, x1)
#TODO         push!(y, y1)
#TODO     end
#TODO 
#TODO     function compare(fl, fr)
#TODO         rcpul = run_on_cpu(fl, n, xs, ys, x, y)
#TODO         rcpur = run_on_cpu(fr, n, xs, ys, x, y)
#TODO         @test rcpul == rcpur
#TODO         if CUDA.functional()
#TODO             rcudal = similar(rcpul)
#TODO             rcudar = similar(rcpur)
#TODO             run_on_cuda!(fl, rcudal, n, xs, ys, x, y)
#TODO             run_on_cuda!(fr, rcudar, n, xs, ys, x, y)
#TODO             @test rcudal == rcudar
#TODO             @test rcudal == rcpul
#TODO             @test rcudar == rcpur
#TODO         end
#TODO     end
#TODO 
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, x), (n, xs, ys, x, y) -> xs)
#TODO 
#TODO     @test string.(x) == string.(xs)
#TODO 
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, Int16x2(x.val & 0x0000ffff)), (n, xs, ys, x, y) -> (xs[1], 0))
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, Int16x2(x.val & 0xffff0000)), (n, xs, ys, x, y) -> (0, xs[2]))
#TODO 
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, zero(Int16x2)), (n, xs, ys, x, y) -> (0, 0))
#TODO 
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, ~x), (n, xs, ys, x, y) -> .~xs)
#TODO 
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, +x), (n, xs, ys, x, y) -> .+xs .% Int16)
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, -x), (n, xs, ys, x, y) -> .-xs .% Int16)
#TODO 
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, x & y), (n, xs, ys, x, y) -> xs .& ys)
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, x | y), (n, xs, ys, x, y) -> xs .| ys)
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, x ⊻ y), (n, xs, ys, x, y) -> xs .⊻ ys)
#TODO 
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, x + y), (n, xs, ys, x, y) -> (xs .+ ys) .% Int16)
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, x - y), (n, xs, ys, x, y) -> (xs .- ys) .% Int16)
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, min(x, y)), (n, xs, ys, x, y) -> min.(xs, ys) .% Int16)
#TODO     compare((n, xs, ys, x, y) -> convert(NTuple{2,Int32}, max(x, y)), (n, xs, ys, x, y) -> max.(xs, ys) .% Int16)
#TODO end

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
    end

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
end

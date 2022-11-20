using CUDA
using CUDASIMDTypes
using Random
using Test

Random.seed!(0)
@testset "prmt T=$T" for T in [UInt32, Int32]
    iters = 1000

    a = rand(T, iters)
    b = rand(T, iters)
    op = rand(UInt16, iters)

    r1 = prmt.(a, b, op)

    if CUDA.functional()
        r2 = zeros(T, iters)
        function calcr2(a, b, op, r2)
            n = threadIdx().x
            r2[n] = prmt(a[n], b[n], op[n])
            return nothing
        end
        a = CuArray(a)
        b = CuArray(b)
        op = CuArray(op)
        r2 = CuArray(r2)
        @cuda threads = (iters,) blocks = 1 calcr2(a, b, op, r2)
        synchronize()
        r2 = Array(r2)

        @test r1 == r2
    end
end

Random.seed!(0)
@testset "lop3 T=$T LUT=$LUT" for T in [UInt32, Int32], LUT in Val.(rand(UInt8, 10))
    iters = 100

    a = rand(T, iters)
    b = rand(T, iters)
    c = rand(T, iters)

    r1 = lop3.(a, b, c, LUT)

    if CUDA.functional()
        r2 = zeros(T, iters)
        function calcr2(a, b, c, r2)
            n = threadIdx().x
            r2[n] = lop3(a[n], b[n], c[n], LUT)
            return nothing
        end
        a = CuArray(a)
        b = CuArray(b)
        c = CuArray(c)
        r2 = CuArray(r2)
        @cuda threads = (iters,) blocks = 1 calcr2(a, b, c, r2)
        synchronize()
        r2 = Array(r2)

        @test r1 == r2
    end
end

Random.seed!(0)
@testset "bitifelse T=$T" for T in [UInt32, Int32]
    iters = 1000

    cond = rand(T, iters)
    x = rand(T, iters)
    y = rand(T, iters)

    r1 = bitifelse.(cond, x, y)

    if CUDA.functional()
        r2 = zeros(T, iters)
        function calcr2(cond, x, y, r2)
            n = threadIdx().x
            r2[n] = bitifelse(cond[n], x[n], y[n])
            return nothing
        end
        cond = CuArray(cond)
        x = CuArray(x)
        y = CuArray(y)
        r2 = CuArray(r2)
        @cuda threads = (iters,) blocks = 1 calcr2(cond, x, y, r2)
        synchronize()
        r2 = Array(r2)

        @test r1 == r2
    end
end

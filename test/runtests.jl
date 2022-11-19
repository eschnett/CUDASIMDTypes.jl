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
            nothing
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

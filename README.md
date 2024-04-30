# CUDASIMDTypes.jl

Explicit SIMD types that live in 32 bits, optimized for CUDA with
fallbacks for regular CPUs.

* [Documentation](https://eschnett.github.io/CUDASIMDTypes.jl/dev/)
* [![GitHub
  CI](https://github.com/eschnett/CUDASIMDTypes.jl/workflows/CI/badge.svg)](https://github.com/eschnett/CUDASIMDTypes.jl/actions)
* [![codecov](https://codecov.io/gh/eschnett/CUDASIMDTypes.jl/branch/main/graph/badge.svg?token=75FT03ULHD)](https://codecov.io/gh/eschnett/CUDASIMDTypes.jl)
* [![PkgEval](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/C/CUDASIMDTypes.svg)](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/C/CUDASIMDTypes.html)

(CI code coverage is bad because the CI tests don't test the CUDA
code; these tests need to be run manually.)

## Description

CUDA supports storing SIMD integer for floating-point in its 32-bit
registers. These types are today most prominently used in the tensor
core operations. These types are `Int4x8`, `Int8x4`, `Int16x2`,
`Float16x2`, and `BFloat16x2`. Each such type stores multiple small
integer or floating point numbers in a single 32-bit register.

Unfortunately, plain CUDA has very little support for these types.
This Julia package `CUDASIMDTypes.jl` defines respective data types,
constructors, conversion routines to tuples (to decompose the SIMD
types), and simple arithmetic operations. When executing in CUDA,
these operations are highly optimized. These operations are also
supported on CPUs, but are usually less efficient there. (This could
be remedied by interfacing this package with
[`SIMD.jl`](https://github.com/eschnett/SIMD.jl).

This package also defines and exports a few helper functions that
correspond to certain CUDA PTX instructions, such as `prmt` and
`lop3`, and defines a function `bitifelse`. These are used internally
but might also be useful in other CUDA packages.

## Examples

Create two `Int8x4` numbers, add them, and convert the result into a tuple:
```Julia
julia> using CUDASIMDTypes
[ Info: Precompiling CUDASIMDTypes [ba1ee33b-8807-41fd-9812-6d5f2ce04139]

julia> i = Int8x4(1, 2, 3, 4)
(1, 2, 3, 4)

julia> j = Int8x4(5, 6, 7, 8)
(5, 6, 7, 8)

julia> k = i + j
(6, 8, 10, 12)

julia> convert(NTuple{4,Int32}, k)
(6, 8, 10, 12)
```

Create an `Int4x8` vector, and split it into its even and odd
components, converted into 2 `Int8x4` vectors. Note that `Int4` is a
rather small type, so that our input `8` overflows to `-8`.
```Julia
julia> using CUDASIMDTypes

julia> i = Int4x8(1, 2, 3, 4, 5, 6, 7, 8)
(1, 2, 3, 4, 5, 6, 7, -8)

julia> jlo, jhi = convert(NTuple{2,Int8x4}, i)
((1, 3, 5, 7), (2, 4, 6, -8))
```

Create `Float16x2` numbers, multiply and add them, and sum the result:
```Julia
julia> x = Float16x2(1.0, 2.0)
(1.0f0, 2.0f0)

julia> y = Float16x2(3.0, 4.0)
(3.0f0, 4.0f0)

julia> z = Float16x2(5.0, 6.0)
(5.0f0, 6.0f0)

julia> r = muladd(x, y, z)
(8.0f0, 14.0f0)

julia> convert(NTuple{2,Float32}, r)
(8.0f0, 14.0f0)

julia> convert(NTuple{2,Float32}, r) |> sum
22.0f0
```

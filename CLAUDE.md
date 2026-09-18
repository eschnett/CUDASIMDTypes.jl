# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this package is

`CUDASIMDTypes.jl` (repo: `eschnett/CUDASIMDTypes.jl`) defines Julia types that
pack several small integers or floats into a single 8- or 32-bit register, the
way CUDA's tensor-core instructions expect them: `Int2x4`, `Int4x2` (8-bit);
`Int2x16`, `Int4x8`, `Int8x4`, `Int16x2`, `Float16x2`, `BFloat16x2` (32-bit).
Each is an immutable struct with a single field `val::UInt8` / `val::UInt32`.

Also exported: PTX-instruction wrappers `prmt`, `lop3`, `make_lop3_lut`,
`dp4a`, and the helper `bitifelse`.

## Layout

Everything lives in one file: [src/CUDASIMDTypes.jl](src/CUDASIMDTypes.jl)
(~1600 lines), split into sections by `####...####` banner comments, roughly one
section per type. Tests are likewise a single
[test/runtests.jl](test/runtests.jl), with one `@testset` per type mirroring the
source order.

## The central pattern: dual CPU/CUDA implementations

Nearly every operation is defined **twice**:

```julia
function Base.:+(a::Float16x2, b::Float16x2)
    alo, ahi = convert(NTuple{2,Float16}, a)
    blo, bhi = convert(NTuple{2,Float16}, b)
    return Float16x2(alo + blo, ahi + bhi)          # portable CPU fallback
end
CUDA.@device_override function Base.:+(a::Float16x2, b::Float16x2)
    return Float16x2(LLVM.Interop.@asmcall("add.rn.f16x2 \$0, \$1, \$2;", "=r,r,r",
                                           UInt32, Tuple{UInt32,UInt32}, a.val, b.val))
end
```

The plain method runs on the CPU; the `CUDA.@device_override` method replaces it
inside GPU kernels with inline PTX. When adding an operation, write both, and
keep them bit-for-bit equivalent — the test suite's whole job is checking that.

Note that `$` must be escaped as `\$` inside the `@asmcall` string, since Julia
strings interpolate.

A few functions are **CUDA-only**: `dp4a` and the internal `cvt_pack_s8` are a
bare `@asmcall` with no CPU fallback, so they fail if called on the host.

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Tests detect CUDA with `CUDA.functional()`. On a machine without a GPU only the
CPU paths run, which is also why CI coverage is low — **the CUDA paths are never
exercised in CI and must be tested manually on a GPU machine.** If you change a
`@device_override` method or any inline PTX, say explicitly that it is unverified
unless you ran it on a GPU.

Each testset follows the same shape: generate random inputs, compute a reference
with `run_on_cpu`, and if CUDA is available recompute with `run_on_cuda!` (which
wraps the function in a kernel over `CuArray`s) and `@test` the two agree. When
adding a type or operation, extend the corresponding testset in the same style
rather than inventing a new one.

## Conventions

- Formatting is enforced by [.JuliaFormatter.toml](.JuliaFormatter.toml): blue
  style, 4-space indent, **132-column margin**, `short_to_long_function_def`.
  Run `julia -e 'using JuliaFormatter; format(".")'` before committing.
- Every exported name gets a docstring; `docs/src/index.md` is a bare
  `@autodocs` block, so documentation is generated from docstrings alone and no
  doc file needs editing when adding an export.
- Bit-twiddling implementations carry a short comment explaining the trick
  (e.g. the `Float16(1024 + i)` bit-pattern notes near the conversion code).
  Match that density — terse, one line, only where the code is non-obvious.
- Renamed functions keep the old name as a `const` alias marked
  `# backward compatibility` (see `any_zero`/`any_iszero`,
  `all_finite`/`all_isfinite`), and both names are exported.
- Bump `version` in `Project.toml` in the same commit that adds a feature;
  releases are tagged by TagBot from a registrator comment.
- Minimum Julia is 1.10; CI covers 1.10/1.11/1.12 on Linux, macOS, Windows,
  x64 and arm64.

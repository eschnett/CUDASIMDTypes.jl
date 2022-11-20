# Generate documentation with this command:
# (cd docs && julia make.jl)

push!(LOAD_PATH, "..")

using Documenter
using CUDASIMDTypes

makedocs(; sitename="CUDASIMDTypes", format=Documenter.HTML(), modules=[CUDASIMDTypes])

deploydocs(; repo="github.com/eschnett/CUDASIMDTypes.jl.git", devbranch="main", push_preview=true)

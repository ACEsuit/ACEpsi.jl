

using Polynomials4ML, ACEcore, ACEpsi, ACEbase, Printf
using ACEpsi: BFwf, gradient, evaluate, laplacian 
using LinearAlgebra
using BenchmarkTools

##

const ↑, ↓, ∅ = '↑','↓','∅'
Nel = 5
polys = legendre_basis(8)
wf = BFwf(Nel, polys; ν=3, purify = true)


##

X = 2 * rand(Nel) .- 1
Σ = rand([↑, ↓], Nel)
wf(X, Σ)
gradient(wf, X, Σ)
laplacian(wf, X, Σ)

## 

@info("evaluate")
@btime wf($X, Σ)

@info("gradient")
@btime gradient($wf, $X, $Σ)

@info("laplacian")
@btime laplacian($wf, $X, $Σ)

##

# @profview let wf=wf, X=X
#    for nrun = 1:50_000 
#       laplacian(wf, X)
#    end
# end

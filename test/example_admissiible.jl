using Polynomials4ML, ACEcore, ACEpsi, ACEbase, Printf
using ACEpsi: BFwf, gradient, evaluate, laplacian, envelopefcn
using LinearAlgebra
using BenchmarkTools
using JSON
const ↑, ↓, ∅ = '↑','↓','∅'
Σ = [↑, ↑, ↑, ↓, ↓];
Nel = 5
polys = legendre_basis(10)
MaxDeg = [10, 4]
wf = BFwf(Nel, polys; ν=2, sd_admissible = bb -> (length(bb) == 0 || all([bb[i][1] < MaxDeg[i] for i = 1:length(bb)])))

X = 2 * rand(Nel) .- 1
wf(X, Σ)
g = gradient(wf, X, Σ)
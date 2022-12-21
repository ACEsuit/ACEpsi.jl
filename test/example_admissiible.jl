using Polynomials4ML, ACEcore, ACEpsi, ACEbase, Printf
using ACEpsi: BFwf, gradient, evaluate, laplacian, envelopefcn, displayspec
using LinearAlgebra
using BenchmarkTools
using JSON
const ↑, ↓, ∅ = '↑','↓','∅'
Σ = [↑, ↑, ↑, ↓, ↓];
Nel = 5
polys = legendre_basis(16)
MaxDeg = [16, 5]


test_ad = bb -> (@show bb; @show all([bb[i][1] < MaxDeg[i] for i = 1:length(bb)]); (length(bb) == 0 || all([bb[i][1] <= MaxDeg[length(bb)] for i = 1:length(bb)])))
wf = BFwf(Nel, polys; ν=2, sd_admissible = test_ad)

K = length(wf.polys)
spec1p = [ (k, σ) for σ in [1, 2, 3] for k in 1:K]
spec1p = sort(spec1p,  by = b -> b[1])
LL = displayspec(wf.spec, spec1p)
@show size(LL)
for i = 1:length(LL)
    @show LL[i]
end
@show length(wf.spec)
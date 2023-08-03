using Polynomials4ML, ACEpsi, ACEbase, Printf
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
@show length(wf.polys)
LL = displayspec(wf)
@show LL
for i = 1:length(LL)
    @show LL[i]
end

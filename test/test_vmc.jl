using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: Nuc, make_nlms_spec, evaluate
using ACEpsi: BackflowPooling, BFwf_lux, setupBFState, Jastrow
using ACEpsi.vmc: gradient, laplacian, grad_params, SumH, MHSampler, Eloc_Exp_TV_clip, grad
using ACEbase.Testing: print_tf, fdtest
using LuxCore
using Lux
using Zygote
using Optimisers # mainly for the destrcuture(ps) function
using Random
using Printf
using LinearAlgebra
using BenchmarkTools

using HyperDualNumbers: Hyper


Rnldegree = 4
Ylmdegree = 4
totdegree = 8
Nel = 5
X = randn(SVector{3, Float64}, Nel)
Σ = rand(spins(), Nel)
nuclei = [ Nuc(3 * rand(SVector{3, Float64}), 1.0) for _=1:3 ]

# wrap it as HyperDualNumbers
x2dualwrtj(x, j) = SVector{3}([Hyper(x[i], i == j, i == j, 0) for i = 1:3])
hX = [x2dualwrtj(x, 0) for x in X]
hX[1] = x2dualwrtj(X[1], 1) # test eval for grad wrt x coord of first elec

##

# Defining AtomicOrbitalsBasis
bRnl = ACEpsi.AtomicOrbitals.RnlExample(Rnldegree)
bYlm = RYlmBasis(Ylmdegree)

# setup state
wf = BFwf_chain = BFwf_lux(Nel, bRnl, bYlm, nuclei; totdeg = totdegree, ν = 2)
ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)

K(wf, X::AbstractVector, ps, st) = -0.5 * laplacian(wf, X, ps, st)
Vext(wf, X::AbstractVector, ps, st) = sum(1/norm(nuclei[i].rr - X[j]) for i = 1:length(nuclei) for j in 1:length(X))
Vee(wf, X::AbstractVector, ps, st) = sum(1/norm(X[i]-X[j]) for i = 1:length(X)-1 for j = i+1:length(X))


ham = SumH(K, Vext, Vee)
sam = MHSampler(wf, Nel)

λ₀, σ, E_clip, x_clip, x0, acc = Eloc_Exp_TV_clip(wf, ps, st, sam, ham)
g = grad(wf, x_clip, ps, st, E_clip)

# Optimization
st_opt = Optimisers.setup(Optimisers.Adam(0.0001), ps)
st_opt, ps = Optimisers.update(st_opt, ps, g)
using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: Nuc, make_nlms_spec, evaluate
using ACEpsi: BackflowPooling, BFwf_lux, setupBFState, Jastrow, displayspec
using ACEpsi.vmc: gradient, laplacian, grad_params, SumH, MHSampler, VMC, gd_GradientByVMC
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



n1 = Rnldegree = 6
Ylmdegree = 1
totdegree = 10
Nel = 3
X = randn(SVector{3, Float64}, Nel)
Σ = [↑,↑,↓]
nuclei = [ Nuc(zeros(SVector{3, Float64}), 1.0)]
##

# Defining AtomicOrbitalsBasis
n2 = 1
Pn = Polynomials4ML.legendre_basis(n1+1)
spec = [(n1 = n1, n2 = n2, l = l) for n1 = 1:n1 for n2 = 1:n2 for l = 0:n1-1] 
ζ = rand(length(spec))
Dn = GaussianBasis(ζ)
bRnl = AtomicOrbitalsRadials(Pn, Dn, spec) 
bYlm = RYlmBasis(Ylmdegree)

# setup state
wf, spec, spec1p = BFwf_chain, spec, spec1p  = BFwf_lux(Nel, bRnl, bYlm, nuclei; totdeg = totdegree, ν = 2)
ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)

wf(X, ps, st)

K(wf, X::AbstractVector, ps, st) = -0.5 * laplacian(wf, X, ps, st)

Vext(wf, X::AbstractVector, ps, st) = 1/2 * sum([X[i][j]^2 for i = 1:length(X) for j = 1:3]) #sum(1/norm(nuclei[i].rr - X[j]) for i = 1:length(nuclei) for j in 1:length(X))

Vee(wf, X::AbstractVector, ps, st) = 0 # sum(1/norm(X[i]-X[j]) for i = 1:length(X)-1 for j = i+1:length(X))

ham = SumH(K, Vext, Vee)
sam = MHSampler(wf, Nel, nuclei, Δt = 1.0, burnin = 1000, nchains = 2000)

opt_vmc = VMC(100, 0.05, ACEpsi.vmc.adamW(); lr_dc = 30.0)
wf, err_opt, ps = gd_GradientByVMC(opt_vmc, sam, ham, wf, ps, st)

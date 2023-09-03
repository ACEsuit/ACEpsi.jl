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

n1 = Rnldegree = 1
Ylmdegree = 0
totdegree = 4
Nel = 1
X = randn(SVector{3, Float64}, Nel)
Σ = [↑]
nuclei = [Nuc(SVector(0.0,0.0,0.0), 1.0)]
##

# Defining AtomicOrbitalsBasis
n2 = 1
Pn = Polynomials4ML.legendre_basis(n1+1)
spec = [(n1 = 1, n2 = 1, l = 0) ] 

ζ = [0.3552322122E+02, 0.6513143725E+01, 0.1822142904E+01,0.6259552659E+00, 0.2430767471E+00, 0.1001124280E+00]
ζ = reshape(ζ, 1, length(ζ))
D = [0.9163596281E-02, 0.4936149294E-01,0.1685383049E+00,0.3705627997E+00,  0.4164915298E+00, 0.1303340841E+00]
D = reshape(D, 1, length(D))

Dn = STO_NG((ζ, D))
bRnl = AtomicOrbitalsRadials(Pn, Dn, spec) 
bYlm = RYlmBasis(Ylmdegree)

# setup state
ord = 1
wf, spec, spec1p = BFwf_chain, spec, spec1p  = BFwf_lux(Nel, bRnl, bYlm, nuclei; totdeg = totdegree, ν = ord)

ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)
p, = destructure(ps)
length(p)

K(wf, X::AbstractVector, ps, st) = -0.5 * laplacian(wf, X, ps, st)
Vext(wf, X::AbstractVector, ps, st) = -sum(nuclei[i].charge/norm(nuclei[i].rr - X[j]) for i = 1:length(nuclei) for j in 1:length(X))
function Vee(wf, X::AbstractVector, ps, st) 
    nX = length(X)
    if nX <=1 
        return 0
    else 
        return sum(1/norm(X[i]-X[j]) for i = 1:length(X)-1 for j = i+1:length(X))
    end
end

ham = SumH(K, Vext, Vee)
sam = MHSampler(wf, Nel, nuclei, Δt = 1.0, burnin = 1000, nchains = 2000)

opt_vmc = VMC(3000, 0.1, ACEpsi.vmc.adamW(); lr_dc = 100.0)
wf, err_opt, ps = gd_GradientByVMC(opt_vmc, sam, ham, wf, ps, st)

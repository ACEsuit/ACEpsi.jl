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
totdegree = 20
Nel = 10
X = randn(SVector{3, Float64}, Nel)
Σ = [↑,↑,↑,↑,↑,↓,↓,↓,↓,↓]
nuclei = [Nuc(SVector(0.0,0.0,-4.5), 1.0),Nuc(SVector(0.0,0.0,-3.5), 1.0),
Nuc(SVector(0.0,0.0,-2.5), 1.0),Nuc(SVector(0.0,0.0,-1.5), 1.0),
Nuc(SVector(0.0,0.0,-0.5), 1.0),Nuc(SVector(0.0,0.0,0.5), 1.0),
Nuc(SVector(0.0,0.0,1.5), 1.0),Nuc(SVector(0.0,0.0,2.5), 1.0),
Nuc(SVector(0.0,0.0,3.5), 1.0),Nuc(SVector(0.0,0.0,4.5), 1.0)]
## 23.11399

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
ord = 2
wf, spec, spec1p = BFwf_chain, spec, spec1p  = BFwf_lux(Nel, bRnl, bYlm, nuclei; totdeg = totdegree, ν = ord)

ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)
p, = destructure(ps)
length(p)

using BenchmarkTools
@btime wf(X, ps, st)
@btime gradient(wf, X, ps, st)

ham = SumH(nuclei)
sam = MHSampler(wf, Nel, nuclei, Δt = 0.5, burnin = 1, nchains = 2000)

opt_vmc = VMC(10, 0.1, ACEpsi.vmc.adamW(); lr_dc = 100.0)
wf, err_opt, ps = gd_GradientByVMC(opt_vmc, sam, ham, wf, ps, st)
@profview wf, err_opt, ps = gd_GradientByVMC(opt_vmc, sam, ham, wf, ps, st)

sam = MHSampler(wf, Nel, nuclei, Δt = 0.5, burnin = 100, nchains = 2000)

#@profview ACEpsi.vmc.sampler_restart(sam, ps, st)
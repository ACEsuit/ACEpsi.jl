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

ζ = [0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00]
D = [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]
D = [(2 * ζ[i]/pi)^(3/4) * D[i] for i = 1:length(ζ)] * sqrt(2) * 2 * sqrt(pi)

ζ = reshape(ζ, 1, length(ζ))
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

ham = SumH(nuclei)
sam = MHSampler(wf, Nel, nuclei, Δt = 0.5, burnin = 1000, nchains = 2000)

opt_vmc = VMC(1000, 0.015, ACEpsi.vmc.adamW(); lr_dc = 1000.0)
wf, err_opt, ps = gd_GradientByVMC(opt_vmc, sam, ham, wf, ps, st)
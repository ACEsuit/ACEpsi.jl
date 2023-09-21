using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: Nuc, make_nlms_spec, evaluate
using ACEpsi: BackflowPooling, BFwf_lux, setupBFState, Jastrow
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

n1 = Rnldegree = 5
Ylmdegree = 2
totdegree = 5
Nel = 4
X = randn(SVector{3, Float64}, Nel)
Σ = [↑,↑,↓,↓]
nuclei = [ Nuc(zeros(SVector{3, Float64}), Nel * 1.0)]
##

# Defining AtomicOrbitalsBasis
n2 = 1
Pn = Polynomials4ML.legendre_basis(n1+1)
spec = [(n1 = n1, n2 = n2, l = l) for n1 = 1:n1 for n2 = 1:n2 for l = 0:n1-1] 
ζ = 10 * rand(length(spec))
Dn = SlaterBasis(ζ)
bRnl = AtomicOrbitalsRadials(Pn, Dn, spec) 
bYlm = RYlmBasis(Ylmdegree)

# setup state
wf, spec, spec1p = BFwf_chain, spec, spec1p  = BFwf_lux(Nel, bRnl, bYlm, nuclei; totdeg = totdegree, ν = 2)
displayspec(spec, spec1p)

ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)
p, = destructure(ps)
length(p)

K(wf, X::AbstractVector, ps, st) = -0.5 * laplacian(wf, X, ps, st)
Vext(wf, X::AbstractVector, ps, st) = -sum(nuclei[i].charge/norm(nuclei[i].rr - X[j]) for i = 1:length(nuclei) for j in 1:length(X))
Vee(wf, X::AbstractVector, ps, st) = sum(1/norm(X[i]-X[j]) for i = 1:length(X)-1 for j = i+1:length(X))

ham = SumH(nuclei)
sam = MHSampler(wf, Nel, Δt = 0.5, burnin = 1000, nchains = 2000)

opt_vmc = VMC(3000, 0.1, ACEpsi.vmc.adamW(), lr_dc = 100)
wf, err_opt, ps = gd_GradientByVMC(opt_vmc, sam, ham, wf, ps, st)

err = err_opt
per = 0.2
err1 = zero(err)
for i = 1:length(err)
    err1[i] = mean(err[Int(ceil(i-per  * i)):i])
end
err1

Eref = -14.667


# using Plots
# plot(abs.(err1 .- Eref), w = 3, yscale=:log10)

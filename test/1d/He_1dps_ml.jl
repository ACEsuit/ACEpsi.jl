using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: Nuc1d, make_nlms_spec, evaluate
using ACEpsi: BackflowPooling1d, BFwf1dps_lux, setupBFState, Jastrow
using ACEpsi.vmc: gradient, laplacian, grad_params, SumH, MHSampler, VMC, gd_GradientByVMC, d1, adamW, sr
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

totdegree = 8
Nel = 2
X = randn(Nel)
Σ = [↑,↓]


# init a list of wf
Pn = Polynomials4ML.legendre_basis(totdegree+1)
wf1, spec1, spec1p1 = BFwf1dps_lux(Nel, Pn, totdeg = floor(Int, totdegree/2))
ps1, st1 = setupBFState(MersenneTwister(1234), wf1, Σ)
wf2, spec2, spec1p2 = BFwf1dps_lux(Nel, Pn, totdeg = totdegree)
ps2, st2 = setupBFState(MersenneTwister(1234), wf2, Σ)
nuclei = [ Nuc1d(zeros(SVector{1, Float64}), Nel * 1.0)]


wf_list = [wf1, wf2]
ps_list = [ps1, ps2]
st_list = [st1, st2]
spec_list = [spec1, spec2]
spec1p_list = [spec1p1, spec1p2]



K(wf, X::AbstractVector, ps, st) = -0.5 * laplacian(wf, X, ps, st)
Vext(wf, X::AbstractVector, ps, st) = -sum(nuclei[i].charge/norm(nuclei[i].rr[1] - X[j]) for i = 1:length(nuclei) for j in 1:length(X))
Vee(wf, X::AbstractVector, ps, st) = sum(1/norm(X[i]-X[j]) for i = 1:length(X)-1 for j = i+1:length(X))

ham = SumH(K, Vext, Vee)
sam = MHSampler(wf_list[1], Nel, Δt = 0.5, burnin = 10, nchains = 2000, d = d1()) # d = d1() means sample for 1d

opt_vmc = VMC(10, 0.1, adamW(), lr_dc = 100)






wf, err_opt, ps = ACEpsi.vmc.VMC_multilevel_1d(opt_vmc, sam, ham, wf_list, ps_list, st_list, spec_list, spec1p_list)

# wf, err_opt, ps = (opt_vmc, sam, ham, wf, ps, st)

err = err_opt
per = 0.2
err1 = zero(err)
for i = 1:length(err)
    err1[i] = mean(err[Int(ceil(i-per  * i)):i])
end
err1

Eref = -2.238


# using Plots
# plot(abs.(err1 .- Eref), w = 3, yscale=:log10)

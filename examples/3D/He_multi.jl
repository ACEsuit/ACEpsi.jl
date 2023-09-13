using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: Nuc, make_nlms_spec, evaluate
using ACEpsi.vmc: gradient, laplacian, grad_params, EmbeddingW!, _invmap, VMC_multilevel, wf_multilevel, VMC, gd_GradientByVMC, gd_GradientByVMC_multilevel, AdamW, SR, SumH, MHSampler
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


# Define He model 
Nel = 2
X = randn(SVector{3, Float64}, Nel)
Σ = [↑,↓]
nuclei = [ Nuc(zeros(SVector{3, Float64}), 2.0)]

K(wf, X::AbstractVector, ps, st) = -0.5 * laplacian(wf, X, ps, st)
Vext(wf, X::AbstractVector, ps, st) = -sum(nuclei[i].charge/norm(nuclei[i].rr - X[j]) for i = 1:length(nuclei) for j in 1:length(X))
Vee(wf, X::AbstractVector, ps, st) = sum(1/norm(X[i]-X[j]) for i = 1:length(X)-1 for j = i+1:length(X))

ham = SumH(K, Vext, Vee)

# Defining Multilevel
Rnldegree = [4, 6, 6, 7]
Ylmdegree = [2, 2, 3, 4]
totdegree = [2, 3, 3, 4]
n2 = [1, 1, 2, 2]
ν = [1, 1, 2, 2]
MaxIters = [3, 3, 3, 3]
##

# 
wf_list, spec_list, spec1p_list, specAO_list, ps_list, st_list = wf_multilevel(Nel, Σ, nuclei, Rnldegree, Ylmdegree, totdegree, n2, ν)

sam = MHSampler(wf_list[1], Nel, nuclei, Δt = 0.5, burnin = 1, nchains = 20)
opt_vmc = VMC_multilevel(MaxIters, 0.0015, SR(1e-4, 0.015), lr_dc = 50.0)

wf, err_opt, ps = gd_GradientByVMC_multilevel(opt_vmc, sam, ham, wf_list, ps_list, st_list, spec_list, spec1p_list, specAO_list)


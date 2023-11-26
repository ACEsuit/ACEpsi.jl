using Distributed

N_procs = 8

if nprocs() == 1
    addprocs(N_procs - 1, exeflags="--project=$(Base.active_project())")
end

@everywhere begin
    
using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: Nuc, make_nlms_spec, evaluate
using ACEpsi: BackflowPooling, BFwf_lux, setupBFState, Jastrow, displayspec
using ACEpsi.vmc: gradient, laplacian, grad_params, EmbeddingW!, _invmap, VMC_multilevel, wf_multilevel, VMC, gd_GradientByVMC, gd_GradientByVMC_multilevel, AdamW, SR, SumH, MHSampler
using ACEbase.Testing: print_tf, fdtest
using LuxCore
using Lux
using Zygote
using Optimisers
using Random
using Printf
using LinearAlgebra
using BenchmarkTools
using HyperDualNumbers: Hyper

end 
@everywhere begin
Nel = 10
X = randn(SVector{3, Float64}, Nel)
Σ = [↑,↑,↑,↑,↑,↓,↓,↓,↓,↓]
spacing = 1.0
nuclei = [Nuc(SVector(0.0,0.0,(i-1/2-Nel/2) * spacing), 1.0) for i = 1:Nel]
spec = [(n1 = 1, n2 = 1, l = 0), (n1 = 1, n2 = 2, l = 0), (n1 = 1, n2 = 3, l = 0), (n1 = 1, n2 = 4, l = 0), (n1 = 1, n2 = 5, l = 0),(n1 = 2, n2 = 1, l = 1)]
n1 = 2
Pn = Polynomials4ML.legendre_basis(n1+1)
Ylmdegree = 2
totdegree = 20
ζ = 10 * rand(length(spec))
Dn = GaussianBasis(ζ)
bYlm = RYlmBasis(Ylmdegree)

totdegree = 30 * ones(Int64, length(spec))
ν = ones(Int64, length(spec))
MaxIters = [100, 100, 100, 100, 100, 1000]
_spec = [spec[1:i] for i = 1:length(spec)]
_spec = length(ν)>length(spec) ? reduce(vcat, [_spec, [spec[1:end] for i = 1:length(ν) - length(spec)]]) : _spec
wf_list, spec_list, spec1p_list, specAO_list, ps_list, st_list = wf_multilevel(Nel, Σ, nuclei, Dn, Pn, bYlm, _spec, totdegree, ν)

ham = SumH(nuclei)
sam = MHSampler(wf_list[1], Nel, nuclei, Δt = 0.5, burnin = 1000, nchains = 2000)
opt_vmc = VMC_multilevel(MaxIters, 0.015, ACEpsi.vmc.adamW(); lr_dc = 100.0)
end
wf, err_opt, ps = gd_GradientByVMC_multilevel(opt_vmc, sam, ham, wf_list, ps_list, 
                    st_list, spec_list, spec1p_list, specAO_list, batch_size = 200)


## MRCI+Q: -23.5092
## UHF:    -23.2997
using Distributed

N_procs = 10

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
Nel = 2
X = randn(SVector{3, Float64}, Nel)
Σ = [↑,↓]
nuclei = [ Nuc(zeros(SVector{3, Float64}), Nel * 1.0)]
##

spec = [(n1 = 1, n2 = 1, l = 0), (n1 = 2, n2 = 1, l = 0), (n1 = 2, n2 = 1, l = 1), (n1 = 3, n2 = 1, l = 0), (n1 = 3, n2 = 1, l = 1)]
n1 = 3
Pn = Polynomials4ML.legendre_basis(n1+1)
Ylmdegree = 2
totdegree = 20
ζ = 10 * rand(length(spec))
Dn = SlaterBasis(ζ)
bYlm = RYlmBasis(Ylmdegree)

totdegree = [30,30,30]
ν = [1,1,2]
MaxIters = [100,100,500]
_spec = [spec[1:3], spec[1:4], spec]
_TD = [ACEpsi.TD.No_Decomposition(),ACEpsi.TD.No_Decomposition(),ACEpsi.TD.No_Decomposition()]
wf_list, spec_list, spec1p_list, specAO_list, ps_list, st_list = wf_multilevel(Nel, Σ, nuclei, Dn, Pn, bYlm, _spec, totdegree, ν, _TD)

ham = SumH(nuclei)
sam = MHSampler(wf_list[1], Nel, nuclei, Δt = 0.5, burnin = 1000, nchains = 2000)
opt_vmc = VMC_multilevel(MaxIters, 0.5, ACEpsi.vmc.adamW(); lr_dc = 50.0)
end
wf, err_opt, ps = gd_GradientByVMC_multilevel(opt_vmc, sam, ham, wf_list, ps_list, 
                    st_list, spec_list, spec1p_list, specAO_list, batch_size = 500)

# Eref = -2.9037247 He
# Eref = −7.47798 Li
# 3s+js, ord = 2, -2.8990571700941272, 205 parameters He
# 3p+js, ord = 2, -2.899325323029854, 440 parameters He

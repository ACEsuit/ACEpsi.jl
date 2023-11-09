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
Nel = 10
X = randn(SVector{3, Float64}, Nel)
Σ = [↑,↑,↑,↑,↑,↓,↓,↓,↓,↓]
spacing = 1.0
nuclei = [Nuc(SVector(0.0,0.0,(i-1/2-Nel/2) * spacing), 1.0) for i = 1:Nel]
spec = [(n1 = 1, n2 = 1, l = 0), (n1 = 2, n2 = 1, l = 0), (n1 = 2, n2 = 1, l = 1), (n1 = 3, n2 = 1, l = 0)]
n1 = 2
Pn = Polynomials4ML.legendre_basis(n1+1)
Ylmdegree = 2
totdegree = 20
ζ = 10 * rand(length(spec))
Dn = SlaterBasis(ζ)
bYlm = RYlmBasis(Ylmdegree)

totdegree = [30, 30, 30, 30, 30]
ν = [1, 1, 1, 1,2]
MaxIters = [50,  100, 100, 200,200]
_spec = [spec[1:i] for i = 1:length(spec)]
_spec = length(ν)>length(spec) ? reduce(vcat, [_spec, [spec[1:end] for i = 1:length(ν) - length(spec)]]) : _spec
wf_list, spec_list, spec1p_list, specAO_list, ps_list, st_list = wf_multilevel(Nel, Σ, nuclei, Dn, Pn, bYlm, _spec, totdegree, ν)

ham = SumH(nuclei)
sam = MHSampler(wf_list[1], Nel, nuclei, Δt = 0.5, burnin = 1000, nchains = 2000)
opt_vmc = VMC_multilevel(MaxIters, 0.2, ACEpsi.vmc.adamW(); lr_dc = 50.0)
end
wf, err_opt, ps = gd_GradientByVMC_multilevel(opt_vmc, sam, ham, wf_list, ps_list, 
                    st_list, spec_list, spec1p_list, specAO_list, batch_size = 500)


# a = -19.28968253968254 R-R term 


# -23.443315039682542: CBS UHF
# -23.722102539682542: CBS MRCI+Q (benchmark used in ferminet)

# -23.5694, var: 0.01857, ord = 1, 1s, 2s, 2p + js
# -23.6509, var: 0.01257, ord = 2, 1s, 2s, 2p + js

# -23.6083, var: 0.01951, ord = 1, 1s, 2s, 2p, 3s + js
# -23.6462, var: 0.01358, ord = 2, 1s, 2s, 2p, 3s + js, basis: 9090

# -23.6052,  var: 0.02280, ord = 1, 1s, 2s, 2p, 3s, 3p + js
# -23.6057, var: 0.03082, ord = 2, 1s, 2s, 2p, 3s, 3p + js, basis: 20385

# -23.6122, var: 0.02630, ord = 1, 1s, 2s, 2p, 3s, 3p, 3d + js




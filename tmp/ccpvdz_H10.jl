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
n1 = Rnldegree = 2
Ylmdegree = 1
totdegree = 20
Nel = 10
X = randn(SVector{3, Float64}, Nel)
Σ = [↑,↑,↑,↑,↑,↓,↓,↓,↓,↓]
spacing = 1.0
nuclei = [Nuc(SVector(0.0,0.0,(i-1/2-Nel/2) * spacing), 1.0) for i = 1:Nel]
# Ref: http://www.grant-hill.group.shef.ac.uk/ccrepo/hydrogen/hbasis.php
# (4s,1p) -> [2s,1p]
# H    S
#      1.301000E+01           1.968500E-02           0.000000E+00
#      1.962000E+00           1.379770E-01           0.000000E+00
#      4.446000E-01           4.781480E-01           0.000000E+00
#      1.220000E-01           5.012400E-01           1.000000E+00
# H    P
#      7.270000E-01           1.0000000

ζ = [[1.301000E+01, 1.962000E+00, 4.446000E-01, 1.220000E-01], [1.220000E-01], [7.270000E-01]]
D = [[1.968500E-02, 1.379770E-01, 4.781480E-01, 5.012400E-01], [1.0000000], [1.0000000]]
#ζ = [[0.3552322122E+02, 0.6513143725E+01, 0.1822142904E+01,0.6259552659E+00, 0.2430767471E+00, 0.1001124280E+00], [1.220000E-01], [7.270000E-01]]
#D = [[0.9163596281E-02, 0.4936149294E-01,0.1685383049E+00,0.3705627997E+00,  0.4164915298E+00, 0.1303340841E+00], [1.0000000], [1.0000000]]
D[1] = [(2 * ζ[1][i]/pi)^(3/4) * D[1][i] for i = 1:length(ζ[1])]

Pn = Polynomials4ML.legendre_basis(n1+1)
Dn = STO_NG((ζ, D))
bYlm = RYlmBasis(Ylmdegree)

totdegree = [30, 30, 30]
ν = [1, 1, 1]
MaxIters = [150, 200, 200]
_TD = [ACEpsi.TD.No_Decomposition(),ACEpsi.TD.No_Decomposition(),ACEpsi.TD.No_Decomposition()]
spec = [(n1 = 1, n2 = 1, l = 0), (n1 = 1, n2 = 2, l = 0), (n1 = 2, n2 = 1, l = 1)]
_spec = [spec[1:i] for i = 1:length(spec)]
_spec = length(ν)>length(spec) ? reduce(vcat, [_spec, [spec[1:end] for i = 1:length(ν) - length(spec)]]) : _spec
wf_list, spec_list, spec1p_list, specAO_list, ps_list, st_list = wf_multilevel(Nel, Σ, nuclei, Dn, Pn, bYlm, _spec, totdegree, ν, _TD)

ham = SumH(nuclei)
sam = MHSampler(wf_list[1], Nel, nuclei, Δt = 0.5, burnin = 2000, nchains = 2000)
opt_vmc = VMC_multilevel(MaxIters, 0.015, ACEpsi.vmc.adamW(); lr_dc = 100.0)
end
wf, err_opt, ps = gd_GradientByVMC_multilevel(opt_vmc, sam, ham, wf_list, ps_list, 
                    st_list, spec_list, spec1p_list, specAO_list, batch_size = 200)


## MRCI+Q: -23.5092
## UHF:    -23.2997

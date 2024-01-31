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
Pn = Polynomials4ML.legendre_basis(n1+1)
spec = [[(n1 = 1, n2 = 1, l = 0)] ]

# Ref: https://link.springer.com/book/10.1007/978-90-481-3862-3: P235
# STO: 0.7790 * e^(-1.24 * r)
# ϕ_1s(1, r) = \sum_(k = 1)^K d_1s,k g_1s(α_1k, r)
# g_1s(α, r) = (2α/π)^(3/4) * exp(-αr^2): α ∼ ζ, g ∼ D
# sto-6g: Ref: https://www.basissetexchange.org/
# BASIS SET: (6s) -> [1s]
# H    S
#      0.3552322122E+02       0.9163596281E-02
#      0.6513143725E+01       0.4936149294E-01
#      0.1822142904E+01       0.1685383049E+00
#      0.6259552659E+00       0.3705627997E+00
#      0.2430767471E+00       0.4164915298E+00
#      0.1001124280E+00       0.1303340841E+00

ζ = [[0.3552322122E+02, 0.6513143725E+01, 0.1822142904E+01,0.6259552659E+00, 0.2430767471E+00, 0.1001124280E+00]]
D = [[0.9163596281E-02, 0.4936149294E-01,0.1685383049E+00,0.3705627997E+00,  0.4164915298E+00, 0.1303340841E+00]]
D[1] = [(2 * ζ[1][i]/pi)^(3/4) * D[1][i] for i = 1:length(ζ[1])]

Dn = STO_NG((ζ, D))
bYlm = RYlmBasis(Ylmdegree)
Nbf = [1, 1, 1, 1]
speclist  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

totdegree = [30]
ν = [2]
MaxIters = [500]
_TD = [ACEpsi.TD.No_Decomposition()]
_spec = [spec]

#wf_list, spec_list, spec1p_list, specAO_list, ps_list, st_list, Nlm_list = wf_multilevel(Nel, Σ, nuclei, Dn, Pn, bYlm, _spec, speclist, Nbf, totdegree, ν, _TD; js = ACEpsi.Jastrow(nuclei))
wf_list, spec_list, spec1p_list, specAO_list, ps_list, st_list, Nlm_list, dist_list = wf_multilevel(Nel, Σ, nuclei, Dn, Pn, bYlm, _spec, speclist, Nbf, totdegree, ν, _TD; js = ACEpsi.Jastrow(nuclei))

ham = SumH(nuclei)
sam = MHSampler(wf_list[1], Nel, nuclei, 
                Δt = 0.08, 
                burnin  = 1000, 
                nchains = 2000)

lr_0  = 0.2
lr_dc = 1000.0
epsilon = 0.001
kappa_S = 0.95
kappa_m = 0.
opt_vmc = VMC_multilevel(MaxIters, lr_0,
                ACEpsi.vmc.SR(0.0, epsilon, kappa_S, kappa_m, 
                              ACEpsi.vmc.QGT(), 
                              ACEpsi.vmc.no_scale(),
                              ACEpsi.vmc.no_constraint()
                              ); 
                lr_dc = lr_dc)

 
wf = wf_list[1]
ps = ps_list[1]
st = st_list[1]
x2dualwrtj(x, j) = SVector{3}([Hyper(x[i], i == j, i == j, 0) for i = 1:3])
hX = [x2dualwrtj(x, 0) for x in X]
A = wf(X, ps, st)[1]
hA = wf(hX, ps, st)[1]
A = wf(X, ps, st)[1]
hA = wf(hX, ps, st)[1]
print_tf(@test hA.value ≈ A)
Zygote.gradient(x -> wf(x, ps, st)[1], X)
p = Zygote.gradient(p -> wf(X, p, st)[1], ps)
laplacian(wf, X, ps, st)

end
wf, err_opt, ps = gd_GradientByVMC_multilevel(opt_vmc, sam, ham, wf_list, ps_list, 
                    st_list, spec_list, spec1p_list, specAO_list, Nlm_list, dist_list, batch_size = 50, 
                    accMCMC = [10, [0.4,0.7]])


## FCI: -23.1140: ord = 2: -23.43910784182956
## UHF: -23.0414: ord = 1: -23.038158337661304
## Eref: -23.71808
# 0: -19.73658
# 1: -21.60029
# 2: -22.13555
# 3: 
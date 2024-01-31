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

n1 = Rnldegree = 2
totdegree = 2
Nel = 4
X = randn(SVector{3, Float64}, Nel)
Σ = [↑,↑,↓,↓]
nuclei = [Nuc(SVector(0.0,0.0,-3.015/2), 2.0), Nuc(SVector(0.0,0.0,3.015/2), 1.0),Nuc(SVector(0.0,0.0,4.015/2), 1.0)]

# Defining AtomicOrbitalsBasis
n2 = 1
Pn = Polynomials4ML.legendre_basis(n1+1)
Dn = SlaterBasis(10 * rand(5))
Ylmdegree = 2

bYlm = RRlmBasis(Ylmdegree)

spec1 = [[(n1 = 1, n2 = 1, l = 0), (n1 = 2, n2 = 1, l = 0), (n1 = 2, n2 = 1, l = 1)], [(n1 = 1, n2 = 1, l = 0), (n1 = 2, n2 = 1, l = 2)]]
spec2 = [[(n1 = 1, n2 = 1, l = 0), (n1 = 2, n2 = 1, l = 0), (n1 = 2, n2 = 1, l = 1), (n1 = 2, n2 = 1, l = 2)], [(n1 = 1, n2 = 1, l = 0), (n1 = 2, n2 = 1, l = 2)]]
spec3 = [[(n1 = 1, n2 = 1, l = 0), (n1 = 2, n2 = 1, l = 0), (n1 = 2, n2 = 1, l = 1), (n1 = 2, n2 = 1, l = 2)], [(n1 = 1, n2 = 1, l = 0), (n1 = 2, n2 = 1, l = 2), (n1 = 2, n2 = 1, l = 1)]]

_spec = [spec1, spec2, spec3]
speclist = [1,2,2]

totdegree = [30,30,30]
ν = [1,1,2]
MaxIters = [100,100,200]
Nbf = [1,2,3]

_TD = [ACEpsi.TD.Tucker(2),ACEpsi.TD.Tucker(3),ACEpsi.TD.Tucker(4)]
#_TD = [ACEpsi.TD.No_Decomposition(),ACEpsi.TD.No_Decomposition(),ACEpsi.TD.No_Decomposition()]
wf_list, spec_list, spec1p_list, specAO_list, ps_list, st_list, Nlm_list, dist_list = wf_multilevel(Nel, Σ, nuclei, Dn, Pn, bYlm, _spec, speclist, Nbf, totdegree, ν, _TD)

for i = 1:length(ν) - 1
    l = i
    ps = ps_list[l]
    spec = spec_list[l]
    spec1p = spec1p_list[l]
    specAO = specAO_list[l]
    dispec = dist_list[l]
    wf = wf_list[l]
    st = st_list[l]
    Nlm = Nlm_list[l]
    l = i + 1
    ps2 = ps_list[l]
    spec2 = spec_list[l]
    spec1p2 = spec1p_list[l]
    specAO2 = specAO_list[l]
    Nlm2 = Nlm_list[l]
    dispec2 = dist_list[l]
    ps2 = EmbeddingW!(ps, ps2, spec, spec2, spec1p, spec1p2, specAO, specAO2, Nlm, Nlm2, dispec, dispec2)
    wf2 = wf_list[l]
    st2 = st_list[l]
    wf(X, ps, st)
    wf2(X, ps2, st2)
    @assert wf(X, ps, st)[1] ≈ wf2(X, ps2, st2)[1]
end
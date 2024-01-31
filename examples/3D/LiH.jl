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
Nel = 4
X = randn(SVector{3, Float64}, Nel)
Σ = [↑, ↑, ↓, ↓]
nuclei = [ Nuc(SVector(0.0, 0.0, 0.0), 3.0), 
           Nuc(SVector(3.015, 0.0, 0.0), 1.0)
         ]

spec_Li = [(n1 = 1, n2 = 1, l = 0), 
        (n1 = 1, n2 = 2, l = 0), 
        (n1 = 1, n2 = 3, l = 0), 
        (n1 = 1, n2 = 1, l = 1), 
        (n1 = 1, n2 = 2, l = 1), 
        (n1 = 2, n2 = 1, l = 0), 
        (n1 = 2, n2 = 2, l = 0), 
        (n1 = 2, n2 = 3, l = 0), 
        (n1 = 2, n2 = 1, l = 1), 
        (n1 = 2, n2 = 2, l = 1), 
        (n1 = 3, n2 = 1, l = 0), 
        (n1 = 3, n2 = 2, l = 0), 
        (n1 = 3, n2 = 3, l = 0), 
        (n1 = 3, n2 = 1, l = 1), 
        (n1 = 3, n2 = 2, l = 1), 
        (n1 = 4, n2 = 1, l = 0),
        (n1 = 4, n2 = 1, l = 1),
        (n1 = 5, n2 = 1, l = 0),
        (n1 = 5, n2 = 1, l = 1),
        (n1 = 1, n2 = 1, l = 2),
        (n1 = 2, n2 = 1, l = 2),
        (n1 = 3, n2 = 1, l = 2)
        ]
spec_H  = [(n1 = 1, n2 = 1, l = 0), 
        (n1 = 1, n2 = 2, l = 0), 
        (n1 = 1, n2 = 3, l = 0), 
        (n1 = 1, n2 = 1, l = 1), 
        (n1 = 1, n2 = 2, l = 1), 
        (n1 = 2, n2 = 1, l = 0), 
        (n1 = 2, n2 = 2, l = 0), 
        (n1 = 2, n2 = 3, l = 0), 
        (n1 = 2, n2 = 1, l = 1), 
        (n1 = 2, n2 = 2, l = 1), 
        (n1 = 3, n2 = 1, l = 0), 
        (n1 = 3, n2 = 2, l = 0), 
        (n1 = 3, n2 = 3, l = 0), 
        (n1 = 3, n2 = 1, l = 1), 
        (n1 = 3, n2 = 2, l = 1), 
        (n1 = 4, n2 = 1, l = 0),
        (n1 = 4, n2 = 1, l = 1),
        (n1 = 5, n2 = 1, l = 0),
        (n1 = 5, n2 = 1, l = 1),
        (n1 = 1, n2 = 1, l = 2),
        (n1 = 2, n2 = 1, l = 2),
        (n1 = 3, n2 = 1, l = 2)
        ]
spec = [ spec_Li, spec_H ]

n1 = 5
Pn = Polynomials4ML.legendre_basis(n1+1)
Ylmdegree = 2
totdegree = 20
ζ = 8.0 * rand(length(spec))
Dn = SlaterBasis(ζ)
bYlm = RRlmBasis(Ylmdegree)

totdegree = [30, 30, 30, 30]
ν = [1, 1, 2, 2]
MaxIters = [400, 800, 1600, 3200]

_spec = [ [ spec[1][1:8], spec[2][1:8] ], 
          [ spec[1][1:13], spec[2][1:13] ], 
          [ spec[1][1:13], spec[2][1:13] ], 
          [ spec[1][1:15], spec[2][1:15] ]
        ]

_TD = [ACEpsi.TD.No_Decomposition(),
       ACEpsi.TD.No_Decomposition(),
       ACEpsi.TD.No_Decomposition(),
       ACEpsi.TD.No_Decomposition()
      ]

Nbf = [1, 1, 1, 1]

speclist  = [1, 2]

wf_list, spec_list, spec1p_list, specAO_list, ps_list, st_list, Nlm_list, dist_list = wf_multilevel(Nel, Σ, nuclei, Dn, Pn, bYlm, _spec, speclist, Nbf, totdegree, ν, _TD)

ham = SumH(nuclei)
sam = MHSampler(wf_list[1], Nel, nuclei, 
                Δt = 0.3, 
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

# -9.065525
# -9.061051

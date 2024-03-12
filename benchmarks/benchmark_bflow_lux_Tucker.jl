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

Nel = 4
X = randn(SVector{3, Float64}, Nel)
Σ = [↑,↑,↓,↓]
nuclei = [ Nuc(zeros(SVector{3, Float64}), Nel * 1.0)]

spec_Be = [(n1 = 1, n2 = 1, l = 0), 
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
        (n1 = 3, n2 = 2, l = 1)
        ]

spec = [ spec_Be ]

n1 = 5
Pn = Polynomials4ML.legendre_basis(n1+1)
Ylmdegree = 2
totdegree = 20
ζ = 10.0 * rand(length(spec))
Dn = SlaterBasis(ζ)
bYlm = RRlmBasis(Ylmdegree)

totdegree = [30, 30, 30, 30]
ν = [1, 1, 2, 2]

MaxIters = [1000, 2000, 4000, 10000]
_spec = [ [ spec[1][1:8]], 
          [ spec[1][1:13]], 
          [ spec[1][1:13]], 
          [ spec[1][1:15]]
        ]


_TD = [ACEpsi.TD.Tucker(4),
       ACEpsi.TD.Tucker(4),
       ACEpsi.TD.Tucker(4),
       ACEpsi.TD.Tucker(4)]

Nbf = [1, 1, 1, 1]
speclist  = [1]

wf_list, spec_list, spec1p_list, specAO_list, ps_list, st_list, Nlm_list, dist_list = wf_multilevel(Nel, Σ, nuclei, Dn, Pn, bYlm, _spec, speclist, Nbf, totdegree, ν, _TD)

BFwf_chain, ps, st = wf_list[1], ps_list[1], st_list[1]



gradient(BFwf_chain, X, ps, st)



@btime BFwf_chain($X, $ps, $st) # 24.761 μs
@btime gradient($BFwf_chain, $X, $ps, $st) # 2.549 ms
@btime laplacian($BFwf_chain, $X, $ps, $st) # 367.083 μs

@profview let  BFwf_chain = BFwf_chain, X = X, ps =  ps, st = st
   for i = 1:10_000
      BFwf_chain(X, ps, st)
   end
end

@profview let  BFwf_chain = BFwf_chain, X = X, ps =  ps, st = st
   for i = 1:10000
      gradient(BFwf_chain, X, ps, st)
   end
end

@profview let  BFwf_chain = BFwf_chain, X = X, ps =  ps, st = st
   for i = 1:10_00
      laplacian(BFwf_chain, X, ps, st)
   end
end
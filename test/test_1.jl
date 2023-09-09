using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: Nuc, make_nlms_spec, evaluate
using ACEpsi: BackflowPooling, BFwf_lux, setupBFState, Jastrow, displayspec
using ACEpsi.vmc: gradient, laplacian, grad_params, SumH, MHSampler, VMC, gd_GradientByVMC
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


n1 = Rnldegree = 2
Ylmdegree = 2
totdegree = 2
Nel = 5
X = randn(SVector{3, Float64}, Nel)
Σ = [↑,↑,↑,↑,↑]
nuclei = [Nuc(SVector(0.0,0.0,-4.5), 1.0),Nuc(SVector(0.0,0.0,-3.5), 1.0),
Nuc(SVector(0.0,0.0,-2.5), 1.0),Nuc(SVector(0.0,0.0,-1.5), 1.0),
Nuc(SVector(0.0,0.0,-0.5), 1.0)]
##

# Defining AtomicOrbitalsBasis
n2 = 1
Pn = Polynomials4ML.legendre_basis(n1+1)
spec = [(n1 = n1, n2 = n2, l = l) for n1 = 1:n1 for n2 = 1:n2 for l = 0:n1-1] 

bRnl = [AtomicOrbitalsRadials(Pn, SlaterBasis(10 * rand(length(spec))), spec) for i = 1:length(nuclei)]
bYlm = [RYlmBasis(Ylmdegree) for i = 1:length(nuclei)]

ord = 1
wf, spec, spec1p = BFwf_chain, spec, spec1p  = BFwf_lux(Nel, bRnl, bYlm, nuclei; totdeg = totdegree, ν = ord)

ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)
p, = destructure(ps)
length(p)
AAA = deepcopy(wf(X,ps,st)[1])
gradient(wf, X, ps, st)
using BenchmarkTools
@btime gradient(wf, X, ps, st)
@profview begin for i = 1:1000 gradient(wf, X, ps, st) end end





using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: Nuc, make_nlms_spec, evaluate
using ACEpsi: BackflowPooling, BFwf_lux, setupBFState, Jastrow
using ACEpsi.vmc: gradient, laplacian, grad_params
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

function grad_test2(f, df, X::AbstractVector)
   F = f(X) 
   ∇F = df(X)
   nX = length(X)
   EE = Matrix(I, (nX, nX))
   
   for h in 0.1.^(3:12)
      gh = [ (f(X + h * EE[:, i]) - F) / h for i = 1:nX ]
      @printf(" %.1e | %.2e \n", h, norm(gh - ∇F, Inf))
   end
end
x2dualwrtj(x, j) = SVector{3}([Hyper(x[i], i == j, i == j, 0) for i = 1:3])
hX = [x2dualwrtj(x, 0) for x in X]
hX[1] = x2dualwrtj(X[1], 1) # test eval for grad wrt x coord of first elec

@info("Test evaluate")
A1 = BFwf_chain(X, ps, st)
hA1 = BFwf_chain(hX, ps, st)

print_tf(@test hA1[1].value ≈ A1[1])

println()

##
F(X) = BFwf_chain(X, ps, st)[1]

# @profview let  F = F, X = X
#    for i = 1:10_000
#        F(X)
#    end
# end

# @btime F(X)




@info("Test ∇ψ w.r.t. X")
ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)
y, st = Lux.apply(BFwf_chain, X, ps, st)

F(X) = BFwf_chain(X, ps, st)[1]
dF(X) = Zygote.gradient(x -> BFwf_chain(x, ps, st)[1], X)[1]
fdtest(F, dF, X, verbose = true)
 
"""
function embed_diff_func(X, nuc, i)
    Xs = .-(X, Ref(nuc[i].rr))   
    return copy(Xs)
 end
 
totdeg = 20
spec1p = make_nlms_spec(bRnl, bYlm; 
                          totaldegree = totdeg)

   # ----------- Lux connections ---------
   # AtomicOrbitalsBasis: (X, Σ) -> (length(nuclei), nX, length(spec1))
   
   embed_layers = Tuple(collect(Lux.WrappedFunction(x -> embed_diff_func(x, nuclei, i)) for i = 1:length(nuclei)))
   prodbasis_layer = ACEpsi.AtomicOrbitals.ProductBasisLayer(spec1p, bRnl, bYlm)
   l_Pds = Tuple(collect(prodbasis_layer for _ = 1:length(nuclei)))
wf = Chain(; diff = Lux.BranchLayer(embed_layers...), Pds = prodbasis_layer)

ν = 2
using Polynomials4ML, Random 
using Polynomials4ML: OrthPolyBasis1D3T, LinearLayer, PooledSparseProduct, SparseSymmProdDAG, SparseSymmProd, release!
using Polynomials4ML.Utils: gensparse
using LinearAlgebra: qr, I, logabsdet, pinv, mul!, dot , tr, det
import ForwardDiff
using ACEpsi.AtomicOrbitals: make_nlms_spec
using ACEpsi: ↑, ↓, ∅, spins, extspins, Spin, spin2idx, idx2spin
using ACEpsi
using LuxCore: AbstractExplicitLayer
using LuxCore
using Lux
using Lux: Chain, WrappedFunction, BranchLayer
using ChainRulesCore
using ChainRulesCore: NoTangent
totdeg = 20
spec1p = make_nlms_spec(bRnl, bYlm; 
                          totaldegree = totdeg)

   # ----------- Lux connections ---------
   # AtomicOrbitalsBasis: (X, Σ) -> (length(nuclei), nX, length(spec1))
   prodbasis_layer = ACEpsi.AtomicOrbitals.ProductBasisLayer(spec1p, bRnl, bYlm)
   aobasis_layer = ACEpsi.AtomicOrbitals.AtomicOrbitalsBasisLayer(prodbasis_layer, nuclei)
   # BackFlowPooling: (length(nuclei), nX, length(spec1 from totaldegree)) -> (nX, 3, length(nuclei), length(spec1))
   pooling = BackflowPooling(aobasis_layer)
   pooling_layer = ACEpsi.lux(pooling)
   sd_admissible = bb -> prod(b.s != '∅' for b in bb) == 0
   function get_spec(nuclei, spec1p) 
    spec = []
    Nnuc = length(nuclei)
 
    spec = Array{Any}(undef, (3, Nnuc, length(spec1p)))
 
    for (k, nlm) in enumerate(spec1p)
       for I = 1:Nnuc 
          for (is, s) in enumerate(extspins())
             spec[is, I, k] = (s=s, I = I, nlm...)
          end
       end
    end
 
    return spec[:]
 end
   spec1p = get_spec(nuclei, spec1p)
   # define sparse for n-correlations
   tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
   default_admissible = bb -> (length(bb) == 0) || (sum(b.n1 - 1 for b in bb ) <= totdeg)

   specAA = gensparse(; NU = ν, tup2b = tup2b, admissible = default_admissible,
                        minvv = fill(0, ν), 
                        maxvv = fill(length(spec1p), ν), 
                        ordered = true)
   spec = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]
   
   # further restrict
   spec = [t for t in spec if sd_admissible([spec1p[t[j]] for j = 1:length(t)])]
   
   # define n-correlation
   corr1 = Polynomials4ML.SparseSymmProd(spec)

   # (nX, 3, length(nuclei), length(spec1 from totaldegree)) -> (nX, length(spec))
   corr_layer = Polynomials4ML.lux(corr1; use_cache = true)

   #js = Jastrow(nuclei)
   #jastrow_layer = ACEpsi.lux(js)

   reshape_func = x -> reshape(x, (size(x, 1), prod(size(x)[2:end])))

   _det = x -> size(x) == (1, 1) ? x[1,1] : det(Matrix(x))
   
   BFwf_chain1 = Chain(; ϕnlm = aobasis_layer, bA = pooling_layer,bAA = corr_layer, hidden1 = LinearLayer(length(corr1), Nel), 
   Mask = ACEpsi.MaskLayer(Nel), det = WrappedFunction(x -> _det(x)), logabs = WrappedFunction(x -> 2 * log(abs(x))) 
   ps, st = setupBFState(MersenneTwister(1234), BFwf_chain1, Σ)
   BBB = deepcopy(BFwf_chain1(X,ps,st)[1])



totdeg = 20
function embed_diff_func(X, nuc, i)
    Xs = .-(X, Ref(nuc[i].rr))   
    return copy(Xs)
 end

spec1p = make_nlms_spec(bRnl, bYlm; 
                          totaldegree = totdeg)

   # ----------- Lux connections ---------
   # AtomicOrbitalsBasis: (X, Σ) -> (length(nuclei), nX, length(spec1))
   
   embed_layers = Tuple(collect(Lux.WrappedFunction(x -> embed_diff_func(x, nuclei, i)) for i = 1:length(nuclei)))
   prodbasis_layer = ACEpsi.AtomicOrbitals.ProductBasisLayer(spec1p, bRnl, bYlm)
   l_Pds = Tuple(collect(prodbasis_layer for _ = 1:length(nuclei)))

   aobasis_layer = ACEpsi.AtomicOrbitals.AtomicOrbitalsBasisLayer(prodbasis_layer, nuclei)
   # BackFlowPooling: (length(nuclei), nX, length(spec1 from totaldegree)) -> (nX, 3, length(nuclei), length(spec1))
   pooling = BackflowPooling(aobasis_layer)
   pooling_layer = ACEpsi.lux(pooling)



   prodbasis_layer = ACEpsi.AtomicOrbitals.ProductBasisLayer(spec1p, bRnl, bYlm)
   l_Pds = Tuple(collect(prodbasis_layer for _ = 1:length(nuclei)))
   aobasis_layer = ACEpsi.AtomicOrbitals.AtomicOrbitalsBasisLayer(prodbasis_layer, nuclei)
   # BackFlowPooling: (length(nuclei), nX, length(spec1 from totaldegree)) -> (nX, 3, length(nuclei), length(spec1))
   pooling = BackflowPooling(aobasis_layer)
   pooling_layer = ACEpsi.lux(pooling)

   wf = Chain(; diff = Lux.BranchLayer(embed_layers...), Pds = Lux.Parallel(nothing, l_Pds...), bA = pooling_layer)
   ps,st = setupBFState(MersenneTwister(1234), wf,Σ)
   wf(X,ps,st)
   Zygote.jacobian(x -> wf(x, ps, st)[1], X)[1]

   totdeg = 20
spec1p = make_nlms_spec(bRnl, bYlm; 
                          totaldegree = totdeg)
function embed_diff_func(X, nuc, i)
    Xs = .-(X, Ref(nuc[i].rr))   
    return copy(Xs)
 end
embed_layers = Tuple(collect(Lux.WrappedFunction(x -> embed_diff_func(x, nuclei, i)) for i = 1:length(nuclei)))
   prodbasis_layer = ACEpsi.AtomicOrbitals.ProductBasisLayer(spec1p, bRnl, bYlm)
   l_Pds = Tuple(collect(prodbasis_layer for _ = 1:length(nuclei)))


BFwf_chain = Chain(; diff = Lux.BranchLayer(embed_layers), Pds = Lux.Parallel(nothing, l_Pds))


"""
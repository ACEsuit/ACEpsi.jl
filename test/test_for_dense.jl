
using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: AtomicOrbitalsBasis, Nuc, make_nlms_spec, ProductBasis, evaluate
using ACEpsi: BackflowPooling, BFwf_lux, setupBFState,Jastrow
using ACEbase.Testing: print_tf
using LuxCore
using Lux
using Zygote
using Optimisers # mainly for the destrcuture(ps) functionx
using Random
using HyperDualNumbers: Hyper
using Printf
using LinearAlgebra
using ACEbase.Testing: fdtest

Rnldegree = 2
Ylmdegree = 2
totdegree = 2
Nel = 5
X = randn(SVector{3, Float64}, Nel)

# wrap it as HyperDualNumbers
x2dualwrtj(x, j) = SVector{3}([Hyper(x[i], i == j, i == j, 0) for i = 1:3])

hX = [x2dualwrtj(x, 1) for x in X]
Σ = rand(spins(), Nel)

nuclei = [ Nuc(3 * rand(SVector{3, Float64}), 1.0) for _=1:3 ]
##

# Defining AtomicOrbitalsBasis
bRnl = ACEpsi.AtomicOrbitals.RnlExample(Rnldegree)
bYlm = RYlmBasis(Ylmdegree)


totdeg =  totdegree
spec1p = make_nlms_spec(bRnl, bYlm; 
                          totaldegree = totdeg)

aobasis = AtomicOrbitalsBasis(bRnl, bYlm; totaldegree = totdeg, nuclei = nuclei, )
pooling = BackflowPooling(aobasis)


tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
default_admissible = bb -> (length(bb) == 0) || (sum(b[1] - 1 for b in bb ) <= totdeg)
using ACEcore.Utils: gensparse
ν = 2
specAA = gensparse(; NU = ν, tup2b = tup2b, admissible = default_admissible,
                        minvv = fill(0, ν), 
                        maxvv = fill(length(spec1p), ν), 
                        ordered = true)
spec = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]
corr1 = Polynomials4ML.SparseSymmProd(spec)

aobasis_layer = ACEpsi.AtomicOrbitals.lux(aobasis)
pooling_layer = ACEpsi.lux(pooling)
corr_layer = Polynomials4ML.lux(corr1)

js = Jastrow(nuclei)
jastrow_layer = ACEpsi.lux(js)
reshape_func = x -> reshape(x, (size(x, 1), prod(size(x)[2:end])))
bchain = Chain(; ϕnlm = aobasis_layer, bA = pooling_layer, reshape = WrappedFunction(reshape_func), 
                        bAA = corr_layer, transpose_layer = WrappedFunction(transpose))
                        
ps, st = setupBFState(MersenneTwister(1234), bchain, Σ)

y = bchain(hX, ps, st)[1]

b = Chain(; b = ACEpsi.DenseLayer(length(corr1), Nel))
ps, st = setupBFState(MersenneTwister(1234), b, Σ)

Zygote.jacobian(p -> b(y, p, st)[1], ps)[1]


using ChainRulesCore
import ChainRules: rrule, NoTangent 


function ChainRulesCore.rrule(::typeof(Lux.apply), l::ACEpsi.DenseLayer, x::AbstractMatrix, ps, st)
    val = l(x, ps, st)
    function pb(A)
       return NoTangent, NoTangent, NoTangent, (W = A[1] * x',), NoTangent
    end
    return val, pb
end
 

Zygote.refresh()
Zygote.jacobian(p -> b(y, p, st)[1], ps)[1]

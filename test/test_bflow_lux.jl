using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: AtomicOrbitalsBasis, Nuc, make_nlms_spec, ProductBasis, evaluate
using ACEpsi: BackflowPooling, BFwf_lux, setupBFState,Jastrow
using ACEbase.Testing: print_tf
using LuxCore
using Lux
using Zygote
using Random

Rnldegree = 4
Ylmdegree = 4
totdegree = 8
Nel = 5
X = randn(SVector{3, Float64}, Nel)
Σ = rand(spins(), Nel)

nuclei = [ Nuc(3 * rand(SVector{3, Float64}), 1.0) for _=1:3 ]
##

# Defining AtomicOrbitalsBasis
bRnl = ACEpsi.AtomicOrbitals.RnlExample(Rnldegree)
bYlm = RYlmBasis(Ylmdegree)

BFwf_chain = BFwf_lux(Nel, bRnl, bYlm, nuclei; totdeg = totdegree, ν = 2)
ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)

@info("Test evaluate")
A1 = BFwf_chain(X, ps, st)

js = Jastrow(nuclei)
jatrow_layer = ACEpsi.lux(js)
js_chain = Chain(; jatrow_layer)
ps, st = setupBFState(MersenneTwister(1234), js_chain, Σ)

gs = Zygote.gradient(X -> js_chain(X, ps, st)[1], X)
Zygote.gradient(X -> ACEpsi.evaluate(js, X, Σ),X)

# # original implementation
# function BFwf_lux2(Nel::Integer, bRnl, bYlm, nuclei; totdeg = 15, 
#     ν = 3, T = Float64, 
#     sd_admissible = bb -> (true),
#     envelope = x -> x) # enveolpe to be replaced by SJ-factor
 
#     spec1p = make_nlms_spec(bRnl, bYlm; 
#                            totaldegree = totdeg)
 
#     # size(X) = (nX, 3); length(Σ) = nX
#     aobasis = AtomicOrbitalsBasis(bRnl, bYlm; totaldegree = totdeg, nuclei = nuclei, )
#     pooling = BackflowPooling(aobasis)
 
#     # define sparse for n-correlations
#     tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
#     default_admissible = bb -> (length(bb) == 0) || (sum(b[1] - 1 for b in bb ) <= totdeg)
 
#     specAA = gensparse(; NU = ν, tup2b = tup2b, admissible = default_admissible,
#                          minvv = fill(0, ν), 
#                          maxvv = fill(length(spec1p), ν), 
#                          ordered = true)
#     spec = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]
 
#     # further restrict
#     spec = [t for t in spec if sd_admissible([spec1p[t[j]] for j = 1:length(t)])]
 
#     # define n-correlation
#     corr1 = SparseSymmProd(spec; T = Float64)
 
#     # ----------- Lux connections ---------
#     # Should we break down the aobasis again into x -> (norm(x), x) -> (Rln, Ylm) -> ϕnlm for trainable radial basis later?
#     # AtomicOrbitalsBasis: (X, Σ) -> (length(nuclei), nX, length(spec1))
#     aobasis_layer = ACEpsi.AtomicOrbitals.lux(aobasis)
#     # BackFlowPooling: (length(nuclei), nX, length(spec1 from totaldegree)) -> (nX, 3, length(nuclei), length(spec1))
#     pooling_layer = ACEpsi.lux(pooling)
#     # (nX, 3, length(nuclei), length(spec1 from totaldegree)) -> (nX, length(spec))
#     corr_layer = ACEcore.lux(corr1)
 
#     # TODO: Add J-factor and add trainable basis later
#     js = Jastrow(nuclei)
#     jastrow_layer = ACEpsi.lux(js)
 
#     reshape_func = x -> reshape(x, (size(x, 1), prod(size(x)[2:end])))
#     # Questions to discuss:
#     # 1. it seems that we do not need trans since the bases have already taken care of it?
#     # 2. Why we need a tranpose here??? Seems that the output from corr_layer is (length(spec), nX)???
#     # 3. How do we define n-correlations if we use trainable basis?
#     BFwf_chain = Chain(; ϕnlm = aobasis_layer, bA = pooling_layer, reshape = WrappedFunction(reshape_func), 
#                          bAA = corr_layer, transpose_layer = WrappedFunction(transpose), hidden1 = Dense(length(corr1), Nel), 
#                          Mask = ACEpsi.MaskLayer(Nel), logabsdet = WrappedFunction(x -> det(x)))
#     return Chain(; branch = BranchLayer((js = jastrow_layer, bf = BFwf_chain, )))
#  end

# using ACEcore, Polynomials4ML
# using Polynomials4ML: OrthPolyBasis1D3T
# using ACEcore: PooledSparseProduct, SparseSymmProdDAG, SparseSymmProd, release!
# using ACEcore.Utils: gensparse
# using LinearAlgebra: qr, I, logabsdet, pinv, mul!, dot , tr, det
# import ForwardDiff
# using ACEpsi.AtomicOrbitals: make_nlms_spec

# using LuxCore: AbstractExplicitLayer
# using Lux: Dense, Chain, WrappedFunction, BranchLayer
# using ACEpsi: Jastrow

# js = Jastrow(nuclei)
# jatrow_layer = ACEpsi.lux(js)
# branchlayer = BFwf_lux2(Nel, bRnl, bYlm, nuclei; totdeg = totdegree, ν = 2)
# ps2, st2 = setupBFState(MersenneTwister(1234), branchlayer, Σ)

# A2 = branchlayer(X, ps2, st2)
# @test(2 * logabsdet(A2[1][1] *A2[1][2])[1] == A1[1])



using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: AtomicOrbitalsBasis, Nuc, make_nlms_spec, ProductBasis, evaluate
using ACEpsi: BackflowPooling, BFwf_lux, setupBFState, Jastrow
using ACEbase.Testing: print_tf
using LuxCore
using Lux
using Zygote
using Random
using ACEcore.Utils: gensparse
using ACEcore: SparseSymmProd
using ACEcore

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
spec1 = make_nlms_spec(bRnl, bYlm; totaldegree = totdegree) 

# define basis and pooling operations
prodbasis = ProductBasis(spec1, bRnl, bYlm)
aobasis = AtomicOrbitalsBasis(prodbasis, nuclei)
pooling = BackflowPooling(aobasis)
ϕnlm = prodbasis(X)
bϕnlm = aobasis(X, Σ)
A = pooling(bϕnlm, Σ)
ν = 3
tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
default_admissible = bb -> (length(bb) == 0) || (sum(b[1] - 1 for b in bb ) <= totdeg)

specAA = gensparse(; NU = ν, tup2b = tup2b, admissible = default_admissible,
                        minvv = fill(0, ν), 
                        maxvv = fill(length(spec1p), ν), 
                        ordered = true)
spec = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]

# further restrict
spec = [t for t in spec if sd_admissible([spec1p[t[j]] for j = 1:length(t)])]

# define n-correlation
corr1 = SparseSymmProd(spec; T = Float64)

# pooling layer
pooling_layer = ACEpsi.lux(pooling)
pChain = Chain(; reshape_layer)
ps, st = setupBFState(MersenneTwister(1234), pChain, Σ)
y, st = Lux.apply(pChain, A, ps, st)
(l, st_), pb = pullback(ϕnlm -> Lux.apply(pChain, ϕnlm, ps, st), ϕnlm)
gs = pb((l, nothing))[1]

# reshape layer
reshape_func = x -> reshape(x, (size(x, 1), prod(size(x)[2:end])))
reshape_layer = WrappedFunction(reshape_func)
rChain = Chain(; reshape_layer)
ps, st = setupBFState(MersenneTwister(1234), rChain, Σ)
y, st = Lux.apply(rChain, A, ps, st)
(l, st_), pb = pullback(A -> Lux.apply(rChain, A, ps, st), A)
gs = pb((l, nothing))[1]

A = l
# corr_layer
corr_layer = ACEcore.lux(corr1)
cChain = Chain(; corr_layer)
ps, st = setupBFState(MersenneTwister(1234), cChain, Σ)
y, st = Lux.apply(cChain, A, ps, st)
(l, st_), pb = pullback(A -> Lux.apply(cChain, A, ps, st), A)
gs = pb((l, nothing))[1]

# transpose layer
ϕ = l
transpose_layer = WrappedFunction(transpose)
tChain = Chain(; transpose_layer)
ps, st = setupBFState(MersenneTwister(1234), tChain, Σ)
y, st = Lux.apply(tChain, ϕ, ps, st)
(l, st_), pb = pullback(ϕ -> Lux.apply(tChain, ϕ, ps, st), ϕ)
gs = pb((l, nothing))[1]

# hidden layer
ϕ = l
hidden_layer = Dense(length(corr1), Nel)
hChain = Chain(; hidden_layer)
ps, st = setupBFState(MersenneTwister(1234), hChain, Σ)
y, st = Lux.apply(hChain, ϕ, ps, st)
(l, st_), pb = pullback(ϕ -> Lux.apply(hChain, ϕ, ps, st), ϕ)
gs = pb((l, nothing))[1]

# Mask layer
ϕ = l
Mask = ACEpsi.MaskLayer(Nel)
mChain = Chain(; Mask)
ps, st = setupBFState(MersenneTwister(1234), mChain, Σ)
y, st = Lux.apply(mChain, ϕ, ps, st)
(l, st_), pb = pullback(ϕ -> Lux.apply(mChain, ϕ, ps, st), ϕ)
gs = pb((l, nothing))[1]

# det layer
using LinearAlgebra:det
ϕ = l
det_layer = WrappedFunction(x -> det(x))
dChain = Chain(; det_layer)
ps, st = setupBFState(MersenneTwister(1234), dChain, Σ)
y, st = Lux.apply(dChain, ϕ, ps, st)
(l, st_), pb = pullback(ϕ -> Lux.apply(dChain, ϕ, ps, st), ϕ)
gs = pb((l, nothing))[1]
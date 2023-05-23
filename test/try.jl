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
ν = 2
totdeg = 10
Nel = 5
X = randn(SVector{3, Float64}, Nel)
Σ = rand(spins(), Nel)
sd_admissible = bb -> (true)


nuclei = [ Nuc(3 * rand(SVector{3, Float64}), 1.0) for _=1:3 ]
##

# Defining AtomicOrbitalsBasis
bRnl = ACEpsi.AtomicOrbitals.RnlExample(Rnldegree)
bYlm = RYlmBasis(Ylmdegree)

spec1p = make_nlms_spec(bRnl, bYlm; 
                          totaldegree = totdeg)

# size(X) = (nX, 3); length(Σ) = nX
aobasis = AtomicOrbitalsBasis(bRnl, bYlm; totaldegree = totdeg, nuclei = nuclei, )
pooling = BackflowPooling(aobasis)

# define sparse for n-correlations
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

reshape_func = x -> reshape(x, (size(x, 1), prod(size(x)[2:end])))


pooling_layer = ACEpsi.lux(pooling)
# (nX, 3, length(nuclei), length(spec1 from totaldegree)) -> (nX, length(spec))
corr_layer = ACEcore.lux(corr1)

pool2AAChain = Chain(; pooling = pooling_layer, reshape = WrappedFunction(reshape_func), 
bAA = corr_layer)


# dummy input
ϕnlm = aobasis(X, Σ)

ps, st = setupBFState(MersenneTwister(1234), pool2AAChain, Σ)

y, st = Lux.apply(pool2AAChain, ϕnlm, ps, st)

## Pullback API to capture change in state
(l, st_), pb = pullback(p -> Lux.apply(pool2AAChain, ϕnlm, p, st), ps)
gs = pb((one.(l), nothing))[1]

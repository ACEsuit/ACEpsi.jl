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

@info("Test Zygote API")
ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)

y, st = Lux.apply(BFwf_chain, X, ps, st)

## Pullback API to capture change in state
(l, st_), pb = pullback(p -> Lux.apply(BFwf_chain, X, p, st), ps)
gs = pb((one.(l), nothing))[1]

# Jastrow: try with gradient
# using ACEpsi: Jastrow
# using Lux
# using Zygote
# using ACEpsi:evaluate

# js = Jastrow(nuclei)
# jatrow_layer = ACEpsi.lux(js)
# js_chain = Chain(; jatrow_layer)
# ps, st = setupBFState(MersenneTwister(1234), js_chain, Σ)

# gs = Zygote.gradient(X -> js_chain(X, ps, st)[1], X)
# Zygote.gradient(X -> ACEpsi.evaluate(js, X, Σ), X)


# BackFlowPooling: Try with rrule



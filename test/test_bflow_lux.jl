using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: AtomicOrbitalsBasis, Nuc, make_nlms_spec, ProductBasis, evaluate
using ACEpsi: BackflowPooling
using ACEpsi: BFwf_lux
using ACEbase.Testing: print_tf
using LuxCore
using Random

function setupBFState(rng, BFwf_chain, Σ)
    chain_ps, chain_st = LuxCore.setup(rng, BFwf_chain)
    layers = keys(chain_st)
    nΣ = Tuple([(Σ = Σ, ) for _ = 1:length(layers)])
    chain_st = (; zip(layers, nΣ)...)
    return chain_ps, chain_st
end

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
BFwf_chain(X, ps, st)
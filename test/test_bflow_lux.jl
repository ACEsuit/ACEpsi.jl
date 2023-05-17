using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: AtomicOrbitalsBasis, Nuc, make_nlms_spec, ProductBasis, evaluate
using ACEpsi: BackflowPooling
using ACEpsi: BFwf_lux
using ACEbase.Testing: print_tf
using LuxCore
using Random


function replace_namedtuples(nt, rp_st, Σ)
    if length(nt) == 0
        return rp_st
    else
        for i in 1:length(nt)
            if length(nt[i]) == 0                
                rp_st = (; rp_st..., (; keys(nt)[i] => (Σ = Σ, ))...)
            else
                rp_st = (; rp_st..., keys(nt)[i] => replace_namedtuples(nt[i], (;), Σ))
            end
        end
        return rp_st
    end
end

function setupBFState(rng, bf, Σ)
    ps, st = LuxCore.setup(rng, bf)
    rp_st = replace_namedtuples(st, (;), Σ)
    return ps, rp_st
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
A1 = BFwf_chain(X, ps, st)

using ACEpsi: Jastrow
using Lux
using Zygote
using ACEpsi:evaluate

js = Jastrow(nuclei)
jatrow_layer = ACEpsi.lux(js)
js_chain = Chain(; jatrow_layer)
ps, st = setupBFState(MersenneTwister(1234), js_chain, Σ)

gs = Zygote.gradient(X -> js_chain(X, ps, st)[1], X)
Zygote.gradient(X -> ACEpsi.evaluate(js, X, Σ),X)


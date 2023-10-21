using Polynomials4ML, Random 
using Polynomials4ML: OrthPolyBasis1D3T, PooledSparseProduct, SparseSymmProdDAG, SparseSymmProd
using ObjectPools: release!
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
 
function get_spec(spec1p) 
    spec = []
 
    spec = Array{Any}(undef, (3, length(spec1p)))
 
    for (k, n) in enumerate(spec1p)
        for (is, s) in enumerate(extspins())
             spec[is, k] = (s=s, n)
        end
    end
 
    return spec[:]
end

function BFwf1d_lux(Nel::Integer, Pn; totdeg = 15, 
    ν = 3, T = Float64, 
    sd_admissible = bb -> prod(b.s != '∅' for b in bb) == 0,
    trans = identity) 
 
    spec1p = [(n = n) for n = 1:totdeg]

    l_trans = WrappedFunction(x -> trans.(x))
    l_Pn = Polynomials4ML.lux(Pn)
    # ----------- Lux connections ---------
    # BackFlowPooling: (length(nuclei), nX, length(spec1 from totaldegree)) -> (nX, 3, length(nuclei), length(spec1))
    pooling = BackflowPooling1d()
    pooling_layer = ACEpsi.lux(pooling)
 
    spec1p = get_spec(spec1p)
    # define sparse for n-correlations
    tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
    default_admissible = bb -> (length(bb) == 0) || (sum(b.n - 1 for b in bb ) <= totdeg)
 
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
    corr_layer = Polynomials4ML.lux(corr1; use_cache = false)
 
    reshape_func = x -> reshape(x, (size(x, 1), prod(size(x)[2:end])))
 
    _det = x -> size(x) == (1, 1) ? x[1,1] : det(Matrix(x))
    BFwf_chain = Chain(; trans = l_trans, Pn = l_Pn, bA = pooling_layer, reshape = WrappedFunction(reshape_func), 
                         bAA = corr_layer, hidden1 = LinearLayer(length(corr1), Nel), 
                         Mask = ACEpsi.MaskLayer(Nel), det = WrappedFunction(x -> _det(x)), prod = WrappedFunction(x -> prod(x)), logabs = WrappedFunction(x -> 2 * log(abs(x))))
    return BFwf_chain, spec, spec1p
end
 
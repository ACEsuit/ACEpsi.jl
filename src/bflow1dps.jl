using ACEcore, Polynomials4ML, Random 
using Polynomials4ML: OrthPolyBasis1D3T
using ACEcore: PooledSparseProduct, SparseSymmProdDAG, SparseSymmProd, release!
using ACEcore.Utils: gensparse
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
using Zygote 

function embed_diff_func(Xt, i)
    T = eltype(Xt)
    Nel = length(Xt)
    Xts = Zygote.Buffer(zeros(T, Nel))
    for j = 1:Nel
        Xts[j] = Xt[j] - Xt[i]
    end    
    Xts[i] = Xt[i]
    return copy(Xts)
end

function BFwf1dps_lux(Nel::Integer, Pn::OrthPolyBasis1D3T; totdeg = 15, 
    ν = 3, T = Float64, trans = x -> x,
    sd_admissible = bb -> prod(b.s != '∅' for b in bb) == 0) 
 
    spec1p = [(n = n) for n = 1:totdeg]


    l_trans = Lux.WrappedFunction(x -> trans.(x))
    l_Pn = Polynomials4ML.lux(Pn)
    # ----------- Lux connections ---------
    # BackFlowPooling: (length(nuclei), nX, length(spec1 from totaldegree)) -> (nX, 3, length(nuclei), length(spec1))
    pooling = ACEpsi.BackflowPooling1dps()
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

    embed_layers = Tuple(collect(Lux.WrappedFunction(x -> embed_diff_func(x, i)) for i = 1:Nel))
    l_Pns = Tuple(collect(l_Pn for _ = 1:Nel))


    _det = x -> size(x) == (1, 1) ? x[1,1] : det(Matrix(x))

    BFwf_chain = Chain(; trans = l_trans, diff = Lux.BranchLayer(embed_layers...), Pn = Lux.Parallel(nothing, l_Pns...), bA = pooling_layer, reshape = WrappedFunction(reshape_func), bAA = corr_layer, hidden1 = ACEpsi.DenseLayer(Nel, length(corr1)), 
                         Mask = ACEpsi.MaskLayer(Nel), det = WrappedFunction(x -> _det(x)), prod = WrappedFunction(x -> prod(x)), logabs = WrappedFunction(x -> 2 * log(abs(x))))
    return BFwf_chain
end

using Polynomials4ML, Random 
using Polynomials4ML: AbstractPoly4MLBasis, PooledSparseProduct, SparseSymmProdDAG, SparseSymmProd, LinearLayer
using ObjectPools: release!
using Polynomials4ML.Utils: gensparse
using LinearAlgebra: qr, I, logabsdet, pinv, mul!, dot , tr, det
import ForwardDiff
using ACEpsi: ↑, ↓, ∅, spins, extspins, Spin, spin2idx, idx2spin, JCasino1dVb, JCasinoChain
using ACEpsi
using LuxCore: AbstractExplicitLayer
using LuxCore
using Lux
using Lux: Chain, WrappedFunction, BranchLayer
using ChainRulesCore
using ChainRulesCore: NoTangent
using Zygote 
using StaticArrays: SA 

"""
Embed by displacement
"""
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

""" 
trivial embedding. This should be removed later.
"""
function embed_usual_func(Xt, i)
    T = eltype(Xt)
    Nel = length(Xt)
    Xts = Zygote.Buffer(zeros(T, Nel))
    for j = 1:Nel
        Xts[j] = Xt[j]
    end 
    return copy(Xts)
end

function get_spec_Wigner(spec1p) 
    spec = []
    spin = SA['∅','σ']
 
    spec = Array{Any}(undef, (2, length(spec1p)))
 
    for (k, n) in enumerate(spec1p)
        for (is, s) in enumerate(spin)
             spec[is, k] = (s=s, n)
        end
    end
 
    return spec[:]
end


"""
According to manuscript, the bf orbital ϕ(x1;x2,…,xN) is (partially) symmetric in x2,…,xN.
Thus admissible specs should be subset of (N×Z3) × (N×Z3)_ord^(B-1) (Note Z3={↑,↓,∅}).
This version of BF generates such specs. Previous version generates subset of (N×Z3)_ord^B.

Future: In principle, order 1 orbital/discretization can be different from orbtials defining pooled basis A
"""
function BFwfTrig_lux(Nel::Integer, Pn::AbstractPoly4MLBasis; totdeg = length(Pn),
    ν = 3, T = Float64, trans = x -> x,
    sd_admissible = bb -> prod(b.s != '∅' for b in bb) == 0) 
 
    # create as much as we can first, and then filter later
    spec1p = [(n = n) for n = 1:totdeg]


    l_trans = Lux.WrappedFunction(x -> trans.(x))
    l_Pn = Polynomials4ML.lux(Pn)
    # ----------- Lux connections ---------
    # BackFlowPooling: (length(nuclei), nX, length(spec1 from totaldegree)) -> (nX, 3, length(nuclei), length(spec1))
    pooling = ACEpsi.BackflowPooling1dps()
    pooling_layer = ACEpsi.lux(pooling)
 
    spec1p = get_spec(spec1p)
    spec = [[i] for i in eachindex(spec1p)] # spec of order 1
    spec = [t for t in spec if spec1p[t[1]].s == '∅'] # body order 1 term should be ∅

    if ν > 1
        # define sparse for (n-1)-correlations for order ≥ 2 terms
        tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
        default_admissible = bb -> (length(bb) == 0) || (sum(ceil((b.n - 1)/2) for b in bb ) <= ceil((totdeg+ν)/2)) # totdeg>1 is P4ML (unnatural) index for Rtrig

        specAA = gensparse(; NU = ν-1, tup2b = tup2b, admissible = default_admissible,
                            minvv = fill(0, ν-1), 
                            maxvv = fill(length(spec1p), ν-1), 
                            ordered = true) ## gensparse automatically order the spec (it assumes S_N symmetry)
        spec_bf = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]

        # combining with order 1 orital
        spec_bf = [cat(b, t; dims=1) for b in spec for t in spec_bf]
        spec = cat(spec, spec_bf; dims=1) # this is the old spec format, except now each b∈spec only has b[2]≤…≤b[B] ordered

        # further restrict
        spec = [t for t in spec if sd_admissible([spec1p[t[j]] for j = 1:length(t)])]
    end
    
    # define n-correlation
    corr1 = Polynomials4ML.SparseSymmProd(spec)
 
    # (nX, 3, length(nuclei), length(spec1 from totaldegree)) -> (nX, length(spec))
    corr_layer = Polynomials4ML.lux(corr1; use_cache = false)
 
    reshape_func = x -> reshape(x, (size(x, 1), prod(size(x)[2:end])))

    embed_layers = Tuple(collect(Lux.WrappedFunction(x -> embed_diff_func(x, i)) for i = 1:Nel))
    l_Pns = Tuple(collect(l_Pn for _ = 1:Nel))

    _det = x -> size(x) == (1, 1) ? x[1,1] : det(Matrix(x))

    BFwf_chain = Chain(; trans = l_trans, diff = Lux.BranchLayer(embed_layers...), Pn = Lux.Parallel(nothing, l_Pns...), bA = pooling_layer, reshape = WrappedFunction(reshape_func), bAA = corr_layer, hidden1 = LinearLayer(length(corr1), Nel), # hidden1 = ACEpsi.DenseLayer(Nel, length(corr1)), 
                         Mask = ACEpsi.MaskLayer(Nel), det = WrappedFunction(x -> _det(x)), logabs = WrappedFunction(x -> 2 * log(abs(x))))
    return BFwf_chain, spec, spec1p
end

"""
WignerLayer
Extra layer for basis sparsification according to Wigner Ansatz
(REF)
"""
struct WignerLayer <: AbstractExplicitLayer end

(l::WignerLayer)(A::AbstractMatrix{T}, ps, st) where T = begin 
    N = length(st.Σ)
    @assert all( t == '↑' for t in st.Σ[1:Int(N / 2)])
    @assert all( t == '↓' for t in st.Σ[Int(N / 2) + 1:N])
    
    # size(A,2) = spatial_basis x 3
    
    K = Int(size(A, 2) * 2 / 3)
    ATilde = Zygote.Buffer(zeros(T, (N, K)))

    # ∅ spin
    ATilde[:, 1:2:K] = A[:, 3:3:end]

    # non-empty spin (↓) for first N/2 electrons
    ATilde[1:Int(N/2), 2:2:K] = A[1:Int(N/2), 2:3:end]

    # non-empty spin (↑) for last N/2 electrons
    ATilde[Int(N/2)+1:N, 2:2:K] = A[Int(N/2)+1:N, 1:3:end]

    return copy(ATilde), st
end

"""
WIGwfTrig_lux
BFwf Lux chain according to Wigner Ansatz
(REF)
"""
function WIGwfTrig_lux(Nel::Integer, Pn::AbstractPoly4MLBasis; totdeg = length(Pn),
    ν = 3, T = Float64, trans = x -> x,
    sd_admissible = bb -> prod(b.s != '∅' for b in bb) == 0) 
 
    # create as much as we can first, and then filter later
    spec1p = [(n = n) for n = 1:totdeg]


    l_trans = Lux.WrappedFunction(x -> trans.(x))
    l_Pn = Polynomials4ML.lux(Pn)
    # ----------- Lux connections ---------
    # BackFlowPooling: (length(nuclei), nX, length(spec1 from totaldegree)) -> (nX, 3, length(nuclei), length(spec1))
    pooling = ACEpsi.BackflowPooling1dps()
    pooling_layer = ACEpsi.lux(pooling)

    # Wigner spec1p
    spec1p_winger = get_spec_Wigner(spec1p)

    # initalize spec for n-corr
    spec = [[i] for i in eachindex(spec1p_winger)] # spec of order 1
    spec = [t for t in spec if spec1p_winger[t[1]].s == '∅'] # body order 1 term should be ∅

    if ν > 1
        # define sparse for (n-1)-correlations for order ≥ 2 terms
        tup2b = vv -> [ spec1p_winger[v] for v in vv[vv .> 0]  ]
        default_admissible = bb -> (length(bb) == 0) || (sum(ceil((b.n - 1)/2) for b in bb ) <= ceil((totdeg+ν)/2)) # totdeg>1 is P4ML (unnatural) index for Rtrig

        specAA = gensparse(; NU = ν-1, tup2b = tup2b, admissible = default_admissible,
                            minvv = fill(0, ν-1), 
                            maxvv = fill(length(spec1p_winger), ν-1), 
                            ordered = true) ## gensparse automatically order the spec (it assumes S_N symmetry)
        spec_bf = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]

        # combining with order 1 orital
        spec_bf = [cat(b, t; dims=1) for b in spec for t in spec_bf]
        #spec=  cat(spec, spec_bf; dims=1) # this is the old spec format, except now each b∈spec only has b[2]≤…≤b[B] ordered

        # further restrict
        spec = [t for t in spec_bf if sd_admissible([spec1p_winger[t[j]] for j = 1:length(t)])]
    end
    
    # define n-correlation
    corr1 = Polynomials4ML.SparseSymmProd(spec)
 
    # (nX, 3, length(nuclei), length(spec1 from totaldegree)) -> (nX, length(spec))
    corr_layer = Polynomials4ML.lux(corr1; use_cache = false)
 
    reshape_func = x -> reshape(x, (size(x, 1), prod(size(x)[2:end])))

    embed_layers = Tuple(collect(Lux.WrappedFunction(x -> embed_diff_func(x, i)) for i = 1:Nel))
    l_Pns = Tuple(collect(l_Pn for _ = 1:Nel))

    _det = x -> size(x) == (1, 1) ? x[1,1] : det(Matrix(x))

    BFwf_chain = Chain(; trans = l_trans, diff = Lux.BranchLayer(embed_layers...), Pn = Lux.Parallel(nothing, l_Pns...), bA = pooling_layer, reshape = WrappedFunction(reshape_func), Wigner = WignerLayer(), bAA = corr_layer, hidden1 = LinearLayer(length(corr1), Nel), # hidden1 = ACEpsi.DenseLayer(Nel, length(corr1)), 
                         Mask = ACEpsi.MaskLayer(Nel), det = WrappedFunction(x -> _det(x)), logabs = WrappedFunction(x -> 2 * log(abs(x))))
    return BFwf_chain, spec, spec1p_winger
end

# === JS factor WIP === 
function BFJwfTrig_lux(Nel::Integer, Pn::AbstractPoly4MLBasis, J; totdeg = length(Pn),
    ν = 3, T = Float64, trans = x -> x,
    sd_admissible = bb -> prod(b.s != '∅' for b in bb) == 0,
    Jastrow_chain = JSPsiTrasnformer()
    ) 
 
    # create as much as we can first, and then filter later
    spec1p = [(n = n) for n = 1:totdeg]


    l_trans = Lux.WrappedFunction(x -> trans.(x))
    l_Pn = Polynomials4ML.lux(Pn)
    # ----------- Lux connections ---------
    # BackFlowPooling: (length(nuclei), nX, length(spec1 from totaldegree)) -> (nX, 3, length(nuclei), length(spec1))
    pooling = ACEpsi.BackflowPooling1dps()
    pooling_layer = ACEpsi.lux(pooling)
 
    spec1p = get_spec(spec1p)
    spec = [[i] for i in eachindex(spec1p)] # spec of order 1
    spec = [t for t in spec if spec1p[t[1]].s == '∅'] # body order 1 term should be ∅

    if ν > 1
        # define sparse for (n-1)-correlations for order ≥ 2 terms
        tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
        default_admissible = bb -> (length(bb) == 0) || (sum(ceil((b.n - 1)/2) for b in bb ) <= ceil((totdeg+ν)/2)) # totdeg>1 is P4ML (unnatural) index for Rtrig

        specAA = gensparse(; NU = ν-1, tup2b = tup2b, admissible = default_admissible,
                            minvv = fill(0, ν-1), 
                            maxvv = fill(length(spec1p), ν-1), 
                            ordered = true) ## gensparse automatically order the spec (it assumes S_N symmetry)
        spec_bf = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]

        # combining with order 1 orital
        spec_bf = [cat(b, t; dims=1) for b in spec for t in spec_bf]
        spec = cat(spec, spec_bf; dims=1) # this is the old spec format, except now each b∈spec only has b[2]≤…≤b[B] ordered

        # further restrict
        spec = [t for t in spec if sd_admissible([spec1p[t[j]] for j = 1:length(t)])]
    end
    
    # define n-correlation
    corr1 = Polynomials4ML.SparseSymmProd(spec)
 
    # (nX, 3, length(nuclei), length(spec1 from totaldegree)) -> (nX, length(spec))
    corr_layer = Polynomials4ML.lux(corr1; use_cache = false)
 
    reshape_func = x -> reshape(x, (size(x, 1), prod(size(x)[2:end])))

    embed_layers = Tuple(collect(Lux.WrappedFunction(x -> embed_diff_func(x, i)) for i = 1:Nel))
    l_Pns = Tuple(collect(l_Pn for _ = 1:Nel))


    _det = x -> size(x) == (1, 1) ? x[1,1] : det(Matrix(x))

    BFwf_chain = Chain(; diff = Lux.BranchLayer(embed_layers...), Pn = Lux.Parallel(nothing, l_Pns...), bA = pooling_layer, reshape = WrappedFunction(reshape_func), bAA = corr_layer, hidden1 = LinearLayer(length(corr1), Nel), # hidden1 = ACEpsi.DenseLayer(Nel, length(corr1)), 
                         Mask = ACEpsi.MaskLayer(Nel), det = WrappedFunction(x -> _det(x)), logabs = WrappedFunction(x -> 2 * log(abs(x))))

    # BFJwf = Chain(; trans = l_trans, diff = Lux.BranchLayer(embed_layers...), to_be_prod = Lux.BranchLayer(BFwf_chain, Jastrow_chain), Sum = WrappedFunction(x -> x[1] + x[2][1]))
    BFJwf = Chain(; trans = l_trans, to_be_prod = Lux.BranchLayer(BFwf_chain, Jastrow_chain), Sum = WrappedFunction(x -> x[1] + x[2]))

    return BFJwf, spec, spec1p
end
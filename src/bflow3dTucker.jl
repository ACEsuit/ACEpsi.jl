using Polynomials4ML, Random, ACEpsi
using Polynomials4ML: OrthPolyBasis1D3T, LinearLayer, PooledSparseProduct, SparseSymmProdDAG, SparseSymmProd, release!
using Polynomials4ML.Utils: gensparse
using LinearAlgebra: qr, I, logabsdet, pinv, mul!, dot , tr, det
using LuxCore: AbstractExplicitLayer
using Lux: Chain, WrappedFunction, BranchLayer
using ChainRulesCore: NoTangent
using ChainRulesCore
# ----------------------------------------
# some quick hacks that we should take care in P4ML later with careful thoughts
using StrideArrays
using ObjectPools: unwrap,ArrayPool, FlexArray,acquire!
using Lux
using ACEpsi.AtomicOrbitals: make_nlms_spec, _invmap, Nuc
using ACEpsi.TD: No_Decomposition, Tucker
using ACEpsi: ↑, ↓, ∅, spins, extspins, Spin, spin2idx, idx2spin

function BFwf_lux(Nel::Integer, Nbf::Integer, speclist::Vector{Int}, bRnl, bYlm, nuclei, TD::Tucker; totdeg = 100, cluster = Nel, 
    ν = 3, sd_admissible = bb -> prod(b.s != '∅' for b in bb) == 0, disspec = [],
    js = JPauliNet(nuclei)) 
    # ----------- Lux connections ---------
    # X -> (X-R[1], X-R[2], X-R[3])
    embed_layer = embed_diff_layer(nuclei)
    # X -> (ϕ1(X-R[1]), ϕ2(X-R[2]), ϕ3(X-R[3])
    prodbasis_layer = ACEpsi.AtomicOrbitals.ProductBasisLayer(speclist, bRnl, bYlm, totdeg)
    
    # BackFlowPooling: (length(nuclei), nX, length(spec1 from totaldegree)) -> (nX, 3, length(nuclei) * length(spec1))
    aobasis_layer = ACEpsi.AtomicOrbitals.AtomicOrbitalsBasisLayer(prodbasis_layer, nuclei)
    pooling = BackflowPooling(aobasis_layer)
    pooling_layer = ACEpsi.lux(pooling)
 
    tucker_layer = ACEpsi.TD.TuckerLayer(TD.P, sum(length.(pooling.basis.prodbasis.sparsebasis)), Nel)
 
    spec1p = get_spec(TD)
    tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
    default_admissible = bb -> (length(bb) == 0) || (sum(b.P - 1 for b in bb ) <= totdeg)
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
 
    l_hidden = Tuple(collect(Chain(; hidden1 = Lux.Parallel(nothing, (LinearLayer(length(corr1), 1) for j = 1:Nel)...), l_concat = WrappedFunction(x -> hcat(x...)), Mask = ACEpsi.MaskLayer(Nel), det = WrappedFunction(x::Matrix -> det(x))) for i = 1:Nbf))
    
    jastrow_layer = ACEpsi.lux(js)

    BFwf_chain = Chain(; diff = embed_layer, Pds = prodbasis_layer, 
                         bA = pooling_layer, TK = tucker_layer, 
                         bAA = Lux.Parallel(nothing, (deepcopy(corr_layer) for i = 1:Nel)...), hidden = BranchLayer(l_hidden...),
                        sum = WrappedFunction(sum))
    return Chain(; branch = BranchLayer(; js = jastrow_layer, bf = BFwf_chain, ), prod = WrappedFunction(x -> x[1] * x[2]), logabs = WrappedFunction(x -> 2 * log(abs(x))) ), spec, spec1p, disspec
end

_size(ps::NamedTuple) = begin
    if (:reshape in keys(ps.branch.bf)) && !(:bf_orbital in keys(ps.branch.bf))
        return size(ps.branch.bf.hidden.layer_1.hidden1.W, 2)
    else
        return size(ps.branch.bf.hidden.layer_1.hidden1.layer_1.W, 2)
    end
end

_classfy(ps::NamedTuple) = begin
    if !(:reshape in keys(ps.branch.bf)) 
        return ACEpsi.TD.Tucker(3)
    elseif :bf_orbital in keys(ps.branch.bf)
        return ACEpsi.Cluster._bf_orbital()
    else
        return ACEpsi.TD.No_Decomposition()
    end
end


function _invmapAO(a::AbstractVector)
    inva = Dict{eltype(a), Int}()
    for i = 1:length(a) 
       inva[a[i]] = i 
    end
    return inva 
end

function embed_ζ!(ps::NamedTuple, ps2::NamedTuple, specAO, specAO2, c::Number)
    if :Pds in keys(ps.branch.bf)
        if :ζ in keys(ps.branch.bf.Pds)
            for i = 1:length(specAO2)
                ps2.branch.bf.Pds.ζ[i] .= c
                _mapAO = _invmapAO(specAO2[i])  
                for (idx, t) in enumerate(specAO[i])
                    ps2.branch.bf.Pds.ζ[i][_mapAO[t]] = ps.branch.bf.Pds.ζ[i][idx]
                end
            end
        end
    end
end

function embed_W!(ps::NamedTuple, ps2::NamedTuple, readable_spec, Nbf1::Int, Nbf2::Int, _map, Nlm, Nlm2, dispec, dispec2, _tucker::Tucker)
    if :TK in keys(ps2.branch.bf)
        ps2.branch.bf.TK.W .= 0.0
        W = ps.branch.bf.TK.W
        idx = []
        for ii = 1:length(Nlm2)
            for k = 1:Nlm[ii]
                push!(idx, _ind(ii, k, Nlm2))
            end
        end
        ps2.branch.bf.TK.W[:,:,1:size(W)[3],idx] .= W
    end
    for i in keys(ps2.branch.bf.hidden)
        for j in keys(ps2.branch.bf.hidden[i].hidden1)
            ps2.branch.bf.hidden[i].hidden1[j].W .= 0.0
        end
    end
    for (ii, i) in enumerate(keys(ps.branch.bf.hidden))
        if ii <= Nbf1 - 1
            for j in keys(ps.branch.bf.hidden[i].hidden1)
                for (idx, t) in enumerate(readable_spec)
                    ps2.branch.bf.hidden[i].hidden1[j].W[:, _map[t]] = ps.branch.bf.hidden[i].hidden1[j].W[:, idx]
                end
            end
        elseif ii == Nbf1
            for j in keys(ps2.branch.bf.hidden)[ii:end]
                for z in keys(ps2.branch.bf.hidden[j].hidden1)
                    for (idx, t) in enumerate(readable_spec)
                        ps2.branch.bf.hidden[j].hidden1[z].W[:, _map[t]] = 1/(Nbf2 - Nbf1 + 1)* ps.branch.bf.hidden[ii].hidden1[z].W[:, idx]
                    end
                end
            end
        end
    end
end
    
function embed_W!(ps::NamedTuple, ps2::NamedTuple, readable_spec, Nbf1::Int, Nbf2::Int, _map, Nlm, Nlm2, dispec, dispec2, _tucker::ACEpsi.Cluster._bf_orbital)
    for i in keys(ps2.branch.bf.hidden)
        for j in keys(ps2.branch.bf.hidden[i].hidden1)
            ps2.branch.bf.hidden[i].hidden1[j].W .= 0.0
        end
    end
    for (ii, i) in enumerate(keys(ps.branch.bf.hidden))
        if ii <= Nbf1 - 1
            for j in keys(ps.branch.bf.hidden[i].hidden1)
                for (idx, t) in enumerate(readable_spec)
                    ps2.branch.bf.hidden[i].hidden1[j].W[:, _map[t]] = ps.branch.bf.hidden[i].hidden1[j].W[:, idx]
                end
            end
        elseif ii == Nbf1
            for j in keys(ps2.branch.bf.hidden)[ii:end]
                for (zz, z) in enumerate(keys(ps2.branch.bf.hidden[j].hidden1))
                    _map = _invmap(dispec2[zz])
                    for (idx, t) in enumerate(dispec[zz])
                        ps2.branch.bf.hidden[j].hidden1[z].W[:, _map[t]] = 1/(Nbf2 - Nbf1 + 1) * ps.branch.bf.hidden[ii].hidden1[z].W[:, idx]
                    end
                end
            end
        end
    end
end

function embed_W!(ps::NamedTuple, ps2::NamedTuple, readable_spec, Nbf1::Int, Nbf2::Int, _map, Nlm, Nlm2,dispec, dispec2, _tucker::No_Decomposition)
    for i in keys(ps2.branch.bf.hidden)
        ps2.branch.bf.hidden[i].hidden1.W .= 0.0
    end
    for (ii, i) in enumerate(keys(ps.branch.bf.hidden))
        if ii <= Nbf1 - 1
            for (idx, t) in enumerate(readable_spec)
                ps2.branch.bf.hidden[i].hidden1.W[:, _map[t]] = ps.branch.bf.hidden[i].hidden1.W[:, idx]
            end
        elseif ii == Nbf1
            for j in keys(ps2.branch.bf.hidden)[ii:end]
                for (idx, t) in enumerate(readable_spec)
                    ps2.branch.bf.hidden[j].hidden1.W[:, _map[t]] = 1/(Nbf2 - Nbf1 + 1)* ps.branch.bf.hidden[ii].hidden1.W[:, idx]
                end
            end
        end
    end
end

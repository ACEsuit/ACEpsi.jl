using ACEpsi.AtomicOrbitals: make_nlms_spec, _invmap, Nuc
using ACEpsi.TD: No_Decomposition, Tucker, SCP

_size(ps::NamedTuple) = begin
    if (:reshape in keys(ps.branch.bf)) && !(:bf_orbital in keys(ps.branch.bf))
        return size(ps.branch.bf.hidden.layer_1.hidden1.W, 2)
    else
        return size(ps.branch.bf.hidden.layer_1.hidden1.layer_1.W, 2)
    end
end

_classfy(ps::NamedTuple) = begin
    if !(:reshape in keys(ps.branch.bf)) 
        return ACEpsi.TD.Tucker(3) # tucker
    elseif :bf_orbital in keys(ps.branch.bf)
        return ACEpsi.Cluster._bf_orbital() # cluster
    else
        return ACEpsi.TD.No_Decomposition() # nothing
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

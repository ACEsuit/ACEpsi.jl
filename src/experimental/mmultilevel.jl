using Printf
using LinearAlgebra
using Optimisers
using Polynomials4ML
using Random
using ACEpsi: BackflowPooling, BFwf_lux, setupBFState, Jastrow, displayspec, mBFwf, mBFwf_sto
using ACEpsi.AtomicOrbitals: _invmap
using Plots


function mwf_multilevel_sto(Nel::Int, Σ::Vector{Char}, nuclei::Vector{Nuc{T}}, Nbf::Vector{Int}, 
                        Dn::STO_NG,
                        Pn::OrthPolyBasis1D3T,  
                        bYlm::Union{RYlmBasis, CYlmBasis, CRlmBasis},
                        _spec::Vector{Vector{NamedTuple{(:n1, :n2, :l), Tuple{Int64, Int64, Int64}}}}, 
                        totdegree::Vector{Int}, 
                        ν::Vector{Int}
                        ) where {T}
    level = length(ν)
    wf, spec, spec1p, ps, st = [], [], [], [], []
    for i = 1:level
        bRnl = AtomicOrbitalsRadials(Pn, Dn, _spec[i])
        _wf, _spec1, _spec1p = mBFwf_sto(Nel, bRnl, bYlm, nuclei, Nbf[i]; totdeg = totdegree[i], ν = ν[i])
        _ps, _st = setupBFState(MersenneTwister(1234), _wf, Σ)
        push!(wf, _wf)
        push!(spec, _spec1)
        push!(spec1p, _spec1p)
        push!(ps, _ps)
        push!(st, _st)
    end
    return wf, spec, spec1p, _spec, ps, st
end

function mwf_multilevel(Nel::Int, Σ::Vector{Char}, nuclei::Vector{Nuc{T}}, Nbf::Vector{Int},  
                        Dn::STO_NG,
                        Pn::OrthPolyBasis1D3T,  
                        bYlm::Union{RYlmBasis, CYlmBasis, CRlmBasis},
                        _spec::Vector{Vector{NamedTuple{(:n1, :n2, :l), Tuple{Int64, Int64, Int64}}}}, 
                        totdegree::Vector{Int}, 
                        ν::Vector{Int}) where {T}
    level = length(ν)
    wf, spec, spec1p, ps, st = [], [], [], [], []
    for i = 1:level
        bRnl = AtomicOrbitalsRadials(Pn, Dn, _spec[i])
        _wf, _spec1, _spec1p = mBFwf(Nel, bRnl, bYlm, nuclei, Nbf[i]; totdeg = totdegree[i], ν = ν[i])
        _ps, _st = setupBFState(MersenneTwister(1234), _wf, Σ)
        push!(wf, _wf)
        push!(spec, _spec1)
        push!(spec1p, _spec1p)
        push!(ps, _ps)
        push!(st, _st)
    end
    return wf, spec, spec1p, _spec, ps, st
end

# GaussianBasis
function mwf_multilevel(Nel::Int, Σ::Vector{Char}, nuclei::Vector{Nuc{T}}, Nbf::Vector{Int},  
                        Dn::GaussianBasis,
                        Pn::OrthPolyBasis1D3T,  
                        bYlm::Union{RYlmBasis, CYlmBasis, CRlmBasis},
                        _spec::Vector{Vector{NamedTuple{(:n1, :n2, :l), Tuple{Int64, Int64, Int64}}}}, 
                        totdegree::Vector{Int}, 
                        ν::Vector{Int}) where {T}
    level = length(ν)
    wf, spec, spec1p, ps, st = [], [], [], [], []
    for i = 1:level
        ζ =  ones(Float64,length(_spec[i]))
        Dn = GaussianBasis(ζ)
        bRnl = AtomicOrbitalsRadials(Pn, Dn, _spec[i])
        _wf, _spec1, _spec1p = mBFwf(Nel, bRnl, bYlm, nuclei, Nbf[i]; totdeg = totdegree[i], ν = ν[i])
        _ps, _st = setupBFState(MersenneTwister(1234), _wf, Σ)
        push!(wf, _wf)
        push!(spec, _spec1)
        push!(spec1p, _spec1p)
        push!(ps, _ps)
        push!(st, _st)
    end
    return wf, spec, spec1p, _spec, ps, st
end

# SlaterBasis
function mwf_multilevel(Nel::Int, Σ::Vector{Char}, nuclei::Vector{Nuc{T}}, Nbf::Vector{Int},  
                        Dn::SlaterBasis,
                        Pn::OrthPolyBasis1D3T,  
                        bYlm::Union{RYlmBasis, CYlmBasis, CRlmBasis},
                        _spec::Vector{Vector{NamedTuple{(:n1, :n2, :l), Tuple{Int64, Int64, Int64}}}}, 
                        totdegree::Vector{Int}, 
                        ν::Vector{Int}) where {T}
    level = length(ν)
    wf, spec, spec1p, ps, st = [], [], [], [], []
    for i = 1:level
        ζ = ones(Float64,length(_spec[i]))
        Dn = SlaterBasis(ζ)
        bRnl = AtomicOrbitalsRadials(Pn, Dn, _spec[i])
        _wf, _spec1, _spec1p = mBFwf(Nel, bRnl, bYlm, nuclei, Nbf[i]; totdeg = totdegree[i], ν = ν[i])
        _ps, _st = setupBFState(MersenneTwister(1234), _wf, Σ)
        push!(wf, _wf)
        push!(spec, _spec1)
        push!(spec1p, _spec1p)
        push!(ps, _ps)
        push!(st, _st)
    end
    return wf, spec, spec1p, _spec, ps, st
end


function mEmbeddingW!(ps, ps2, spec, spec2, spec1p, spec1p2, specAO, specAO2)
    readable_spec = displayspec(spec, spec1p)
    readable_spec2 = displayspec(spec2, spec1p2)
    @assert size(ps.branch.bf.Pds.layer_1.hidden1.W, 1) == size(ps2.branch.bf.Pds.layer_1.hidden1.W, 1)
    @assert size(ps.branch.bf.Pds.layer_1.hidden1.W, 2) ≤ size(ps2.branch.bf.Pds.layer_1.hidden1.W, 2)
    @assert all(t in readable_spec2 for t in readable_spec)
    @assert all(t in specAO2 for t in specAO)
 
    # set all parameters to zero
    for i in keys(ps.branch.bf.Pds)
        ps2.branch.bf.Pds[i].hidden1.W .= 0.0
    end
    if :hidden2 in keys(ps2.branch.bf)
        if size(ps.branch.bf.hidden2.W, 2) == size(ps2.branch.bf.hidden2.W, 2)
            ps2.branch.bf.hidden2.W .= ps.branch.bf.hidden2.W
        elseif size(ps.branch.bf.hidden2.W, 2) < size(ps2.branch.bf.hidden2.W, 2)
            ps2.branch.bf.hidden2.W .= 0.0
            ps2.branch.bf.hidden2.W[1:size(ps.branch.bf.hidden2.W, 2)] .= ps.branch.bf.hidden2.W
        end
    end
 
    # _map[spect] = index in readable_spec2
    _map  = _invmap(readable_spec2)
    _mapAO  = _invmapAO(specAO2)
    # embed
    for (idx, t) in enumerate(readable_spec)
        if :hidden1 in keys(ps2.branch.bf.Pds.layer_1)
            for i in keys(ps2.branch.bf.Pds)
                if i in keys(ps.branch.bf.Pds)
                    ps2.branch.bf.Pds[i].hidden1.W[:, _map[t]] = ps.branch.bf.Pds[i].hidden1.W[:, idx]
                else
                    @assert size(ps2.branch.bf.Pds.layer_1.hidden1.W, 2) == size(ps2.branch.bf.Pds[i].hidden1.W, 2)
                    ps2.branch.bf.Pds[i].hidden1.W .= ps.branch.bf.Pds[1].hidden1.W
                    ps2.branch.bf.Pds[i].hidden1.W[1,1] = 1.0
                    ps2.branch.bf.Pds[i].hidden1.W[1,2:end] .= 0.0
                end
            end
        end
    end
    if :ϕnlm in keys(ps.branch.bf)
        if :ζ in keys(ps.branch.bf.ϕnlm)
            ps2.branch.bf.ϕnlm.ζ .= 1.0
            for (idx, t) in enumerate(specAO)
                ps2.branch.bf.ϕnlm.ζ[_mapAO[t]] = ps.branch.bf.ϕnlm.ζ[idx]
            end
        end
    end

    if :branch in keys(ps)
        if :TK in keys(ps.branch.bf)
            ps2.branch.bf.TK.W .= 0
            ps2.branch.bf.TK.W[:,:,1:size(ps.branch.bf.TK.W)[3],:,1:size(ps.branch.bf.TK.W)[5]] .= ps.branch.bf.TK.W
        end
    end
    return ps2
end
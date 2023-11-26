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

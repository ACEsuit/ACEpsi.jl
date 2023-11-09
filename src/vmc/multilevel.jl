export EmbeddingW!, _invmap, VMC_multilevel, wf_multilevel, gd_GradientByVMC_multilevel
using Printf
using LinearAlgebra
using Optimisers
using Polynomials4ML
using Random
using ACEpsi: BackflowPooling, BFwf_lux, setupBFState, Jastrow, displayspec, mBFwf, mBFwf_sto
using ACEpsi.AtomicOrbitals: _invmap

mutable struct VMC_multilevel
    tol::Float64
    MaxIter::Vector{Int}
    lr::Float64
    lr_dc::Float64
    type::opt
end

VMC_multilevel(MaxIter::Vector{Int}, lr::Float64, type; tol = 1.0e-3, lr_dc = 50.0) = VMC_multilevel(tol, MaxIter, lr, lr_dc, type);
     
# TODO: this should be implemented to recursively embed the wavefunction

function _invmapAO(a::AbstractVector)
    inva = Dict{eltype(a), Int}()
    for i = 1:length(a) 
       inva[a[i]] = i 
    end
    return inva 
end

function EmbeddingW!(ps, ps2, spec, spec2, spec1p, spec1p2, specAO, specAO2)
    readable_spec = displayspec(spec, spec1p)
    readable_spec2 = displayspec(spec2, spec1p2)
    if :hidden1 in keys(ps.branch.bf)
        @assert size(ps.branch.bf.hidden1.W, 1) == size(ps2.branch.bf.hidden1.W, 1)
        @assert size(ps.branch.bf.hidden1.W, 2) ≤ size(ps2.branch.bf.hidden1.W, 2)
    elseif :hidden1 in keys(ps.branch.bf.Pds.layer_1)
        @assert size(ps.branch.bf.Pds.layer_1.hidden1.W, 1) == size(ps2.branch.bf.Pds.layer_1.hidden1.W, 1)
        @assert size(ps.branch.bf.Pds.layer_1.hidden1.W, 2) ≤ size(ps2.branch.bf.Pds.layer_1.hidden1.W, 2)
    end
    @assert all(t in readable_spec2 for t in readable_spec)
    @assert all(t in specAO2 for t in specAO)
 
    # set all parameters to zero
    if :hidden1 in keys(ps2.branch.bf)
        ps2.branch.bf.hidden1.W .= 0.0
    elseif :hidden1 in keys(ps2.branch.bf.Pds.layer_1)
        for i in keys(ps.branch.bf.Pds)
            ps2.branch.bf.Pds[i].hidden1.W .= 0.0
        end
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
        if :hidden1 in keys(ps2.branch.bf)
            ps2.branch.bf.hidden1.W[:, _map[t]] = ps.branch.bf.hidden1.W[:, idx]
        elseif :hidden1 in keys(ps2.branch.bf.Pds.layer_1)
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
        if :js in keys(ps.branch)
            if :b in keys(ps.branch.js)
                ps2.branch.js.b .= ps.branch.js.b
            end
        end
    end
    return ps2
end
 
function gd_GradientByVMC_multilevel(opt_vmc::VMC_multilevel, sam::MHSampler, ham::SumH, wf_list, ps_list, st_list, spec_list, spec1p_list, specAO_list; 
                                        verbose = true, 
                                        accMCMC = [10, [0.45, 0.55]], 
                                        batch_size = 1)
 
    # first level
    wf = wf_list[1]
    ps = ps_list[1]
    st = st_list[1]
    spec = spec_list[1]
    spec1p = spec1p_list[1]
    specAO = specAO_list[1]
 
    sam.Ψ = wf
    # burn in 
    res, λ₀, α = 1.0, 0., opt_vmc.lr
    err_opt = [zeros(opt_vmc.MaxIter[i]) for i = 1:length(opt_vmc.MaxIter)]

    x0, ~, acc = sampler_restart(sam, ps, st, batch_size = batch_size)
    acc_step, acc_range = accMCMC
    acc_opt = zeros(acc_step)
    
 
    verbose && @printf("Initialize MCMC: Δt = %.2f, accRate = %.4f \n", sam.Δt, acc)
    verbose && @printf("   k |  𝔼[E_L]  |  V[E_L] |   res   |   LR    |accRate|   Δt    \n")
    for l in 1:length(wf_list)
       # do embeddings
       if l > 1
          wf = wf_list[l]
          # embed
          ps = EmbeddingW!(ps, ps_list[l], spec, spec_list[l], spec1p, spec1p_list[l], specAO, specAO_list[l])
          st = st_list[l]
          spec = spec_list[l]
          specAO = specAO_list[l]
          spec1p = spec1p_list[l]
          sam.Ψ = wf
       end
       ν = maximum(length.(spec))
       if :hidden1 in keys(ps.branch.bf)
            _basis_size = size(ps.branch.bf.hidden1.W, 2)
            @info("level = $l, order = $ν, size of basis = $_basis_size")
       elseif :hidden1 in keys(ps.branch.bf.Pds.layer_1)
            _basis_size = size(ps.branch.bf.Pds.layer_1.hidden1.W, 2)
            @info("level = $l, order = $ν, size of basis = $_basis_size")
       else 
            @info("level = $l, order = $ν")
       end
       # optimization
       
       for k = 1 : opt_vmc.MaxIter[l]
          sam.x0 = x0
          
          # adjust Δt
          acc_opt[mod(k,acc_step)+1] = acc
          sam.Δt = acc_adjust(k, sam.Δt, acc_opt, acc_range, acc_step)
 
          # adjust learning rate
          α, ν = InverseLR(ν, opt_vmc.lr, opt_vmc.lr_dc)
 
          # optimization
          ps, acc, λ₀, res, σ, x0 = Optimization(opt_vmc.type, wf, ps, st, sam, ham, α, batch_size = batch_size)
 
          # err
          verbose && @printf(" %3.d | %.5f | %.5f | %.5f | %.5f | %.3f | %.3f \n", k, λ₀, σ, res, α, acc, sam.Δt)
          err_opt[l][k] = λ₀
 
          if res < opt_vmc.tol
                ps_list[l] = deepcopy(ps)
                break;
          end  
       end
       ps_list[l] = deepcopy(ps)
       opt_vmc.lr = α
    end
    
    return wf_list, err_opt, ps_list
end

# sto_ng
function wf_multilevel(Nel::Int, Σ::Vector{Char}, nuclei::Vector{Nuc{T}}, 
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
        _wf, _spec1, _spec1p = BFwf_lux(Nel, bRnl, bYlm, nuclei; totdeg = totdegree[i], ν = ν[i])
        _ps, _st = setupBFState(MersenneTwister(1234), _wf, Σ)
        push!(wf, _wf)
        push!(spec, _spec1)
        push!(spec1p, _spec1p)
        push!(ps, _ps)
        push!(st, _st)
    end
    return wf, spec, spec1p, _spec, ps, st
end

# gaussianbasis
function wf_multilevel(Nel::Int, Σ::Vector{Char}, nuclei::Vector{Nuc{T}}, 
                        Dn::GaussianBasis,
                        Pn::OrthPolyBasis1D3T,  
                        bYlm::Union{RYlmBasis, CYlmBasis, CRlmBasis},
                        _spec::Vector{Vector{NamedTuple{(:n1, :n2, :l), Tuple{Int64, Int64, Int64}}}}, 
                        totdegree::Vector{Int}, 
                        ν::Vector{Int}) where {T}
    level = length(ν)
    wf, spec, spec1p, ps, st = [], [], [], [], []
    for i = 1:level
        ζ = ones(Float64,length(_spec[i]))
        Dn = GaussianBasis(ζ)
        bRnl = AtomicOrbitalsRadials(Pn, Dn, _spec[i])
        _wf, _spec1, _spec1p = BFwf_lux(Nel, bRnl, bYlm, nuclei; totdeg = totdegree[i], ν = ν[i])
        _ps, _st = setupBFState(MersenneTwister(1234), _wf, Σ)
        push!(wf, _wf)
        push!(spec, _spec1)
        push!(spec1p, _spec1p)
        push!(ps, _ps)
        push!(st, _st)
    end
    return wf, spec, spec1p, _spec, ps, st
end

# slaterbasis
function wf_multilevel(Nel::Int, Σ::Vector{Char}, nuclei::Vector{Nuc{T}}, 
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
        _wf, _spec1, _spec1p = BFwf_lux(Nel, bRnl, bYlm, nuclei; totdeg = totdegree[i], ν = ν[i])
        _ps, _st = setupBFState(MersenneTwister(1234), _wf, Σ)
        push!(wf, _wf)
        push!(spec, _spec1)
        push!(spec1p, _spec1p)
        push!(ps, _ps)
        push!(st, _st)
    end
    return wf, spec, spec1p, _spec, ps, st
end


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

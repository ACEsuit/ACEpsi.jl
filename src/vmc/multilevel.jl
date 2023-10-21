export EmbeddingW!, _invmap, VMC_multilevel, wf_multilevel, gd_GradientByVMC_multilevel
using Printf
using LinearAlgebra
using Optimisers
using Polynomials4ML
using Random
using ACEpsi: BackflowPooling, BFwf_lux, setupBFState, Jastrow, displayspec
using ACEpsi.AtomicOrbitals: _invmap, Nuc

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
    @assert size(ps.hidden1.W, 1) == size(ps2.hidden1.W, 1)
    @assert size(ps.hidden1.W, 2) ≤ size(ps2.hidden1.W, 2)
    @assert all(t in readable_spec2 for t in readable_spec)
    @assert all(t in specAO2 for t in specAO)
 
    # set all parameters to zero
    ps2.hidden1.W .= 0.0
 
    # _map[spect] = index in readable_spec2
    _map  = _invmap(readable_spec2)
    _mapAO  = _invmapAO(specAO2)
    # embed
    for (idx, t) in enumerate(readable_spec)
       ps2.hidden1.W[:, _map[t]] = ps.hidden1.W[:, idx]
    end
    if :ϕnlm in keys(ps)
        if :ζ in keys(ps.ϕnlm)
            ps2.ϕnlm.ζ .= 0.0
            for (idx, t) in enumerate(specAO)
                ps2.ϕnlm.ζ[_mapAO[t]] = ps.ϕnlm.ζ[idx]
            end
        end
    end
    return ps2
end
 
function gd_GradientByVMC_multilevel(opt_vmc::VMC_multilevel, sam::MHSampler, ham::SumH, wf_list, ps_list, st_list, spec_list, spec1p_list, specAO_list; verbose = true, accMCMC = [10, [0.45, 0.55]])
 
    # first level
    wf = wf_list[1]
    ps = ps_list[1]
    st = st_list[1]
    spec = spec_list[1]
    spec1p = spec1p_list[1]
    specAO = specAO_list[1]
 
    # burn in 
    res, λ₀, α = 1.0, 0., opt_vmc.lr
    err_opt = [zeros(opt_vmc.MaxIter[i]) for i = 1:length(opt_vmc.MaxIter)]
 
    x0, ~, acc = sampler_restart(sam, ps, st)
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
          spec1p = spec1p_list[l]
          sam.Ψ = wf
       end
       _basis_size = size(ps.hidden1.W, 2)
       ν = maximum(length.(spec))
       # optimization
       @info("level = $l, order = $ν, size of basis = $_basis_size")
       for k = 1 : opt_vmc.MaxIter[l]
          sam.x0 = x0
          
          # adjust Δt
          acc_opt[mod(k,acc_step)+1] = acc
          sam.Δt = acc_adjust(k, sam.Δt, acc_opt, acc_range, acc_step)
 
          # adjust learning rate
          α, ν = InverseLR(ν, opt_vmc.lr, opt_vmc.lr_dc)
 
          # optimization
          ps, acc, λ₀, res, σ = Optimization(opt_vmc.type, wf, ps, st, sam, ham, α)
 
          # err
          verbose && @printf(" %3.d | %.5f | %.5f | %.5f | %.5f | %.3f | %.3f \n", k, λ₀, σ, res, α, acc, sam.Δt)
          err_opt[l][k] = λ₀
 
          if res < opt_vmc.tol
                ps_list[l] = deepcopy(ps)
                break;
          end  
       end
       ps_list[l] = deepcopy(ps)
    end
    
    return wf_list, err_opt, ps_list
end

function wf_multilevel(Nel::Int, Σ::Vector{Char}, nuclei::Vector{Nuc{T}}, Rnldegree::Vector{Int}, Ylmdegree::Vector{Int}, totdegree::Vector{Int}, n2::Vector{Int}, ν::Vector{Int}) where {T}
    level = length(Rnldegree)
    # init a list of wf
    wf = []
    specAO = []
    spec = []
    spec1p = []
    ps = []
    st = []
    for i = 1:level
        Pn = Polynomials4ML.legendre_basis(Rnldegree[i]+1)
        _spec = [(n1 = n1, n2 = _n2, l = l) for n1 = 1:Rnldegree[i] for _n2 = 1:n2[i] for l = 0:Rnldegree[i]-1] 
        push!(specAO, _spec)
        ζ = 10 * rand(length(_spec))
        Dn = SlaterBasis(ζ)
        bRnl = AtomicOrbitalsRadials(Pn, Dn, _spec)
        bYlm = RYlmBasis(Ylmdegree[i])
        _wf, _spec, _spec1p = BFwf_lux(Nel, bRnl, bYlm, nuclei; totdeg = totdegree[i], ν = ν[i])
        _ps, _st = setupBFState(MersenneTwister(1234), _wf, Σ)
        push!(wf, _wf)
        push!(spec, _spec)
        push!(spec1p, _spec1p)
        push!(ps, _ps)
        push!(st, _st)
    end
    return wf, spec, spec1p, specAO, ps, st
end


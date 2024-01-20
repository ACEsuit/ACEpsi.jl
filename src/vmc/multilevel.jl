export EmbeddingW!, EmbeddingWignerW!, _invmap, VMC_multilevel, wf_multilevel, gd_GradientByVMC_multilevel, wf_multilevel_Trig
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

# TODO: this should be implemented to recursively embed the wavefunction
function EmbeddingW!(ps, ps2, spec, spec2, spec1p, spec1p2)
   @assert length(spec2[1]) - length(spec[1]) <= 1
   if length(spec[1]) < length(spec2[1])
      if length(spec[1]) == 1
         spec1 = [[i[1], 3] for i in spec] # 3 <-> (1, ∅) 
      else
         spec1 = [[i[1], 3, i[2:end]...] for i in spec]
      end # so that spec1 obeys how basis labels are constructed in BFwfTrig_lux
   else
      spec1 = spec
   end
  #  @show spec1;
   readable_spec1 = ACEpsi.displayspec(spec1, spec1p2)
   readable_spec2 = ACEpsi.displayspec(spec2, spec1p2)
   @assert size(ps.hidden1.W, 1) == size(ps2.hidden1.W, 1)
   @assert size(ps.hidden1.W, 2) ≤ size(ps2.hidden1.W, 2)
   @assert all(t in readable_spec2 for t in readable_spec1)

   # set all parameters to zero
   ps2.hidden1.W .= 0.0

   # _map[spect] = index in readable_spec2
   _map  = _invmap(readable_spec2)

   # embed
   for (idx, t) in enumerate(readable_spec1)
      ps2.hidden1.W[:, _map[t]] = ps.hidden1.W[:, idx]
   end

   return ps2
end

function EmbeddingW_J!(ps, ps2, spec, spec2, spec1p, spec1p2)
    @assert length(spec2[1]) - length(spec[1]) <= 1
    if length(spec[1]) < length(spec2[1])
       if length(spec[1]) == 1
          spec1 = [[i[1], 3] for i in spec] # 3 <-> (1, ∅) 
       else
          spec1 = [[i[1], 3, i[2:end]...] for i in spec]
       end # so that spec1 obeys how basis labels are constructed in BFwfTrig_lux
    else
       spec1 = spec
    end
   #  @show spec1;
    readable_spec1 = ACEpsi.displayspec(spec1, spec1p2)
    readable_spec2 = ACEpsi.displayspec(spec2, spec1p2)
    @assert size(ps.to_be_prod.layer_1.hidden1.W, 1) == size(ps2.to_be_prod.layer_1.hidden1.W, 1)
    @assert size(ps.to_be_prod.layer_1.hidden1.W, 2) ≤ size(ps2.to_be_prod.layer_1.hidden1.W, 2)
    @assert all(t in readable_spec2 for t in readable_spec1)
 
    # set all parameters to zero
    ps2.to_be_prod.layer_1.hidden1.W .= 0.0
 
    # _map[spect] = index in readable_spec2
    _map  = _invmap(readable_spec2)
 
    # embed
    for (idx, t) in enumerate(readable_spec1)
       ps2.to_be_prod.layer_1.hidden1.W[:, _map[t]] = ps.to_be_prod.layer_1.hidden1.W[:, idx]
    end
 
    return ps2
 end

 function EmbeddingWignerW!(ps, ps2, spec, spec2, spec1p, spec1p2)
   @assert length(spec2[1]) - length(spec[1]) <= 1
   if length(spec[1]) < length(spec2[1])
      if length(spec[1]) == 1
         spec1 = [[i[1], 1] for i in spec] # 1 <-> (1, ∅) 
      else
         spec1 = [[i[1], 1, i[2:end]...] for i in spec]
      end # so that spec1 obeys how basis labels are constructed in BFwfTrig_lux
   else
      spec1 = spec
   end
  #  @show spec1;
   readable_spec1 = ACEpsi.displayspec(spec1, spec1p2)
   readable_spec2 = ACEpsi.displayspec(spec2, spec1p2)
   @assert size(ps.hidden1.W, 1) == size(ps2.hidden1.W, 1)
   @assert size(ps.hidden1.W, 2) ≤ size(ps2.hidden1.W, 2)
   @assert all(t in readable_spec2 for t in readable_spec1)

   # set all parameters to zero
   ps2.hidden1.W .= 0.0

   # _map[spect] = index in readable_spec2
   _map  = _invmap(readable_spec2)

   # embed
   for (idx, t) in enumerate(readable_spec1)
      ps2.hidden1.W[:, _map[t]] = ps.hidden1.W[:, idx]
   end

   return ps2
end

function wf_multilevel_Trig(Nel::Int, Σ::Vector{Char}, trans, 
    trigdegree::Vector{Int}, 
    totdegrees::Vector{Int},
    ν::Vector{Int},
    sd_admissible_func) where {T}
    level = length(trigdegree)
    # init a list of wf
    wf = []
    spec = []
    spec1p = []
    ps = []
    st = []
    for i = 1:level
        Pn = Polynomials4ML.RTrigBasis(totdegrees[i][1]) 
        _sd_admissible = sd_admissible_func(ν[i], totdegrees[i]) # assume B=1 always give the largest degree
        _wf, _spec, _spec1p = BFwfTrig_lux(Nel, Pn; ν = ν[i], trans = trans, totdeg = totdegree[1], sd_admissible = _sd_admissible)
        _ps, _st = setupBFState(MersenneTwister(1234), _wf, Σ)
        push!(wf, _wf)
        push!(spec, _spec)
        push!(spec1p, _spec1p)
        push!(ps, _ps)
        push!(st, _st)
    end
    return wf, spec, spec1p, ps, st
end

# function VMC_multilevel_1d(opt_vmc::VMC, sam::MHSampler, ham::SumH, wf_list, ps_list, st_list, spec_list, spec1p_list; ITERS = [100 for _ in wf_list], verbose = true, accMCMC = [10, [0.45, 0.55]])

#     # first level
#     wf = wf_list[1]
#     ps = ps_list[1]
#     st = st_list[1]
#     spec = spec_list[1]
#     spec1p = spec1p_list[1]
 
#     # burn in 
#     res, λ₀, α = 1.0, 0., opt_vmc.lr
#     err_opt = zeros(opt_vmc.MaxIter)
 
#     x0, ~, acc = sampler_restart(sam, ps, st)
#     acc_step, acc_range = accMCMC
#     acc_opt = zeros(acc_step)
    
 
#     verbose && @printf("Initialize MCMC: Δt = %.2f, accRate = %.4f \n", sam.Δt, acc)
 
#     verbose && @printf("   k |  𝔼[E_L]  |  V[E_L] |   res   |   LR    |accRate|   Δt    \n")
#     for l in 1:length(wf_list)
#        # do embeddings
#        if l > 1
#           wf = wf_list[l]
#           # embed
#           ps = EmbeddingW!(ps, ps_list[l], spec, spec_list[l], spec1p, spec1p_list[l])
#           st = st_list[l]
#           spec = spec_list[l]
#           spec1p = spec1p_list[l]
#           sam.Ψ = wf
#        end
 
#        ν = maximum(length.(spec))
#        SB = size(ps.hidden1.W, 1)
#        # optimization
#        @info("level = $l, order = $ν, size of basis = $SB")
#        opt_vmc.MaxIter = ITERS[l]
#        for k = 1 : opt_vmc.MaxIter
#           sam.x0 = x0
          
#           # adjust Δt
#           acc_opt[mod(k,acc_step)+1] = acc
#           sam.Δt = acc_adjust(k, sam.Δt, acc_opt, acc_range, acc_step)
 
#           # adjust learning rate
#           α, ν = InverseLR(ν, opt_vmc.lr, opt_vmc.lr_dc)
 
#          # optimization
#          ps, acc, λ₀, res, σ, x0 = Optimization(opt_vmc.type, wf, ps, st, sam, ham, α)
 
#           # err
#           verbose && @printf(" %3.d | %.5f | %.5f | %.5f | %.5f | %.5f | %.3f | %.3f \n", k, λ₀, λ₀/N, σ, res, α, acc, sam.Δt)
#           err_opt[k] = λ₀
 
#           if res < opt_vmc.tol
#                 break;
#           end  
#        end
 
#     end
    
#     return wf, err_opt, ps
#  end
 
#  function gd_GradientByVMC_multilevel_Trig(opt_vmc::VMC_multilevel, sam::MHSampler, ham::SumH, 
#     wf_list, ps_list, st_list, spec_list, spec1p_list; 
#     verbose = true, accMCMC = [10, [0.45, 0.55]], batch_size = 1)
 
#     # first level
#     wf = wf_list[1]
#     ps = ps_list[1]
#     st = st_list[1]
#     spec = spec_list[1]
#     spec1p = spec1p_list[1]
 
#     # burn in 
#     res, λ₀, α = 1.0, 0., opt_vmc.lr
#     err_opt = [zeros(opt_vmc.MaxIter[l]) for l = 1:length(opt_vmc.MaxIter)]
#     σ_opt = [zeros(opt_vmc.MaxIter[l]) for l = 1:length(opt_vmc.MaxIter)]
 
#     x0, ~, acc = sampler_restart(sam, ps, st)
#     acc_step, acc_range = accMCMC
#     acc_opt = zeros(acc_step)
    
 
#     verbose && @printf("Initialize MCMC: Δt = %.2f, accRate = %.4f \n", sam.Δt, acc)
 
#     verbose && @printf("   k |  𝔼[E_L]   |  𝔼[E_L]/N  |  V[E_L] |   res   |   LR    |accRate|   Δt    \n")
#     for l in 1:length(wf_list)
#        # do embeddings
#        if l > 1
#           wf = wf_list[l]
#           # embed
#           ps = EmbeddingW!(ps, ps_list[l], spec, spec_list[l], spec1p, spec1p_list[l])
#           st = st_list[l]
#           spec = spec_list[l]
#           spec1p = spec1p_list[l]
#           sam.Ψ = wf
#        end
#        _basis_size = size(ps.hidden1.W, 2)
#        ν = maximum(length.(spec))
#        # optimization
#        @info("level = $l, order = $ν, size of basis = $_basis_size")
#        for k = 1 : opt_vmc.MaxIter[l]
#         sam.x0 = x0
        
#         # adjust Δt
#         acc_opt[mod(k,acc_step)+1] = acc
#         sam.Δt = acc_adjust(k, sam.Δt, acc_opt, acc_range, acc_step)
 
#         # adjust learning rate
#         α, ν = InverseLR(ν, opt_vmc.lr, opt_vmc.lr_dc)
 
#         # optimization
#         ps, acc, λ₀, res, σ, x0 = Optimization(opt_vmc.type, wf, ps, st, sam, ham, α; batch_size = batch_size)
 
#         # err
#         verbose && @printf(" %3.d | %.5f | %.5f| %.5f | %.5f | %.5f | %.3f | %.3f \n", k, λ₀,  λ₀/N, σ, res, α, acc, sam.Δt)
#         err_opt[l][k] = λ₀
#         σ_opt[l][k] = σ
 
#        if mod(k, 10) == 0 # save intermediate results
#           json_E = JSON3.write(err_opt)
#           json_σ = JSON3.write(σ_opt)
#           json_W = JSON3.write(ps.hidden1.W)
#           json_Dic = """{"E": $(json_E), "σ": $(json_σ), "W": $(json_W)}"""
#           open("/zfs/users/berniehsu/berniehsu/OneD/ACEpsi.jl/test/1d/tmp_wf_data/Level$(l)Data$k.json", "w") do io
#              JSON3.write(io, JSON3.read(json_Dic))
#           end
#        end
 
#         if res < opt_vmc.tol
#             ps_list[l] = deepcopy(ps)
#             break;
#         end  
#     end 
#        ps_list[l] = deepcopy(ps)
#     end
    
#     return wf_list, err_opt, σ_list, ps_list
#  end
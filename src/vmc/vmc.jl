export VMC, gd_GradientByVMC, EmbeddingW!, _invmap, VMC_multilevel_1d
using Printf
using LinearAlgebra
using Optimisers
using ACEpsi
using Distributed
using ParallelDataTransfer: @getfrom
using SharedArrays

mutable struct VMC
   tol::Number
   MaxIter::Int
   lr::Number
   lr_dc::Number
   type::opt
end

VMC(MaxIter::Int, lr::Number, type; tol = 1.0e-3, lr_dc = 50.0) = VMC(tol, MaxIter, lr, lr_dc, type);
        
function gd_GradientByVMC(opt_vmc::VMC, sam::MHSampler, ham::SumH, 
               wf, ps, st, 
               ν = 1, verbose = true, accMCMC = [10, [0.45, 0.55]])

   res, λ₀, α = 1.0, 0., opt_vmc.lr
   err_opt = zeros(opt_vmc.MaxIter)
   N = length(st.trans.Σ)

   x0, ~, acc = sampler_restart(sam, ps, st)
   acc_step, acc_range = accMCMC
   acc_opt = zeros(acc_step)

   verbose && @printf("Initialize MCMC: Δt = %.2f, accRate = %.4f \n", sam.Δt, acc)
   verbose && @printf("   k |  𝔼[E_L]  |  𝔼[E_L]/N  |  V[E_L] |   res   |   LR    |accRate|   Δt    \n")
   for k = 1 : opt_vmc.MaxIter
      sam.x0 = x0
      
      # adjust Δt
      acc_opt[mod(k,acc_step)+1] = acc
      sam.Δt = acc_adjust(k, sam.Δt, acc_opt, acc_range, acc_step)

      # adjust learning rate
      α, ν = InverseLR(ν, opt_vmc.lr, opt_vmc.lr_dc)

      # optimization
      ps, acc, λ₀, res, σ = Optimization(opt_vmc.type, wf, ps, st, sam, ham, α)

      # err
      verbose && @printf(" %3.d | %.5f | %.5f | %.5f | %.5f | %.5f | %.3f | %.3f \n", k, λ₀, λ₀/N, σ, res, α, acc, sam.Δt)
      err_opt[k] = λ₀

      if res < opt_vmc.tol
         break;
      end  
   end
   return wf, err_opt, ps
end


function _invmap(a::AbstractVector)
   inva = Dict{eltype(a), Int}()
   for i = 1:length(a) 
      inva[a[i]] = i 
   end
   return inva 
end

# TODO: this should be implemented to recursively embed the wavefunction
function EmbeddingW!(ps, ps2, spec, spec2, spec1p, spec1p2)
   readable_spec = ACEpsi.displayspec(spec, spec1p)
   readable_spec2 = ACEpsi.displayspec(spec2, spec1p2)
   @assert size(ps.hidden1.W, 2) == size(ps2.hidden1.W, 2)
   @assert size(ps.hidden1.W, 1) ≤ size(ps2.hidden1.W, 1)
   @assert all(t in readable_spec2 for t in readable_spec)

   # set all parameters to zero
   ps2.hidden1.W .= 0.0

   # _map[spect] = index in readable_spec2
   _map  = _invmap(readable_spec2)

   # embed
   for (idx, t) in enumerate(readable_spec)
      ps2.hidden1.W[_map[t], :] = ps.hidden1.W[idx, :]
   end

   return ps2
end

function VMC_multilevel_1d(opt_vmc::VMC, sam::MHSampler, ham::SumH, wf_list, ps_list, st_list, spec_list, spec1p_list; ITERS = [100 for _ in wf_list], verbose = true, accMCMC = [10, [0.45, 0.55]])

   # first level
   wf = wf_list[1]
   ps = ps_list[1]
   st = st_list[1]
   spec = spec_list[1]
   spec1p = spec1p_list[1]

   # burn in 
   res, λ₀, α = 1.0, 0., opt_vmc.lr
   err_opt = zeros(opt_vmc.MaxIter)

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
         ps = EmbeddingW!(ps, ps_list[l], spec, spec_list[l], spec1p, spec1p_list[l])
         st = st_list[l]
         spec = spec_list[l]
         spec1p = spec1p_list[l]
         sam.Ψ = wf
      end

      ν = maximum(length.(spec))
      SB = size(ps.hidden1.W, 1)
      # optimization
      @info("level = $l, order = $ν, size of basis = $SB")
      opt_vmc.MaxIter = ITERS[l]
      for k = 1 : opt_vmc.MaxIter
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
         err_opt[k] = λ₀

         if res < opt_vmc.tol
               break;
         end  
      end

   end
   
   return wf, err_opt, ps
end

# === paralell code
function gd_GradientByVMC_parallel(opt_vmc::VMC, sam::MHSampler, ham::SumH, 
               wf, ps, st, 
               ν = 1, verbose = true, accMCMC = [10, [0.45, 0.55]])

   res, λ₀, α = 1.0, 0., opt_vmc.lr
   err_opt = zeros(opt_vmc.MaxIter)
   N = length(st.trans.Σ)

   @everywhere (x0, ~, acc) = ACEpsi.vmc.sampler_restart(sam, ps, st)

   acc_list = SharedArray{Float64}(nprocs())
   acc_list = [@getfrom i acc for i in 1:nprocs()]
   acc_all = mean(acc_list)

   @everywhere accMCMC = $accMCMC
   @everywhere acc_opt = zeros(accMCMC[1]) # define acc_opt on each processor for updating time step respectively


   verbose && @printf("Initialize MCMC: Δt = %.2f, accRate = %.4f \n", sam.Δt, acc_all)
   verbose && @printf("   k |  𝔼[E_L]  |  𝔼[E_L]/N  |  V[E_L] |   res   |   LR    |accRate|   Δt    \n")
   for k = 1 : opt_vmc.MaxIter
      
      # synchronize parameters
      @everywhere sam.x0 = x0
      
      @everywhere begin
         @everywhere (ps, st) = ($ps, $st)
         # sam.Ψ = wf;
         k = $k;
         # adjust the MCMC step-size every `acc_step` step
         acc_list = $acc_list
         acc_opt[mod(k, accMCMC[1])+1] = acc_list[myid()]
      end

      _acc_opt = reduce(vcat, [@getfrom i acc_opt for i in 1:nprocs()])
      macc_opt = mean(_acc_opt)
      @everywhere macc_opt = $macc_opt
      
      # adjust Δt
      if mod(k, accMCMC[1]) == 0
         if macc_opt < accMCMC[2][1]
             @everywhere sam.Δt = sam.Δt * exp(1/10 * (macc_opt - accMCMC[2][1])/accMCMC[2][1])
         elseif macc_opt > accMCMC[2][2]
             @everywhere sam.Δt = sam.Δt * exp(1/10 * (macc_opt - accMCMC[2][2])/accMCMC[2][2])
         end
      end

      # adjust learning rate
      α, ν = InverseLR(ν, opt_vmc.lr, opt_vmc.lr_dc)

      # optimization
      ps, acc, λ₀, res, σ = Optimization_parallel(opt_vmc.type, wf, ps, st, sam, ham, α)
      
      # err
      verbose && @printf(" %3.d | %.5f | %.5f | %.5f | %.5f | %.5f | %.3f | %.3f \n", k, λ₀, λ₀/N, σ, res, α, acc, sam.Δt)
      err_opt[k] = λ₀

      if res < opt_vmc.tol
         break;
      end  
   end
   return wf, err_opt, ps
end
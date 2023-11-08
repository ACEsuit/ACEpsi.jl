export VMC, gd_GradientByVMC, EmbeddingW!, _invmap, VMC_multilevel_1d

using Printf
using LinearAlgebra
using Optimisers
using ACEpsi
using Distributed
using SharedArrays
using`` JSON3

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
               ν = 1, verbose = true, accMCMC = [10, [0.45, 0.55]]; batch_size = 1)

   res, λ₀, α = 1.0, 0., opt_vmc.lr
   err_opt = zeros(opt_vmc.MaxIter)
   σ_opt = zeros(opt_vmc.MaxIter)
   N = length(st.trans.Σ)

   
   x0, ~, acc = sampler_restart(sam, ps, st; batch_size = batch_size)
   acc_step, acc_range = accMCMC
   acc_opt = zeros(acc_step)

   verbose && @printf("Initialize MCMC: Δt = %.2f, accRate = %.4f \n", sam.Δt, acc)
   verbose && @printf("   k |  𝔼[E_L]   |  𝔼[E_L]/N   |  V[E_L] |   res   |   LR    |accRate|   Δt    \n")
   for k = 1 : opt_vmc.MaxIter
      sam.x0 = x0
       
      # adjust Δt
      acc_opt[mod(k,acc_step)+1] = acc
      sam.Δt = acc_adjust(k, sam.Δt, acc_opt, acc_range, acc_step)

      # adjust learning rate
      α, ν = InverseLR(ν, opt_vmc.lr, opt_vmc.lr_dc)

      # optimization
      ps, acc, λ₀, res, σ, x0 = Optimization(opt_vmc.type, wf, ps, st, sam, ham, α; batch_size = batch_size)

      # err
      verbose && @printf(" %3.d | %.5f | %.5f | %.5f | %.5f | %.5f | %.3f | %.3f \n", k, λ₀, λ₀/N, σ, res, α, acc, sam.Δt)
      err_opt[k] = λ₀
      σ_opt[k] = σ

      if mod(k, 10) == 0 # save intermediate results
         json_E = JSON3.write(err_opt)
         json_σ = JSON3.write(σ_opt)
         json_W = JSON3.write(ps.hidden1.W)
         json_Dic = """{"E": $(json_E), "σ": $(json_σ), "W": $(json_W)}"""
         open("/zfs/users/berniehsu/berniehsu/OneD/ACEpsi.jl/test/1d/tmp_wf_data/Data$k.json", "w") do io
            JSON3.write(io, JSON3.read(json_Dic))
         end
         # save("/zfs/users/berniehsu/berniehsu/OneD/ACEpsi.jl/test/1d/tmp_wf_data/Data_$k.jld", "params", ps.hidden1.W, "err_opt", err_opt) # retiring JLD
      end

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



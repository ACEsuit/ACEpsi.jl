export VMC
using Printf
using LinearAlgebra
using Optimisers

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

    x0, ~, acc = sampler_restart(sam, ps, st)
    acc_step, acc_range = accMCMC
    acc_opt = zeros(acc_step)

    verbose && @printf("Initialize MCMC: Δt = %.2f, accRate = %.4f \n", sam.Δt, acc)
    verbose && @printf("   k |  𝔼[E_L]  |  V[E_L] |   res   |   LR    |accRate|   Δt    \n")
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
    return wf, err_opt, ps
end



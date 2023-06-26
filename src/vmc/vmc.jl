export VMC
using Printf
using LinearAlgebra
using Optimisers
#using Plots
"""
Minimizing the Rayleigh quotient by VMC
tol: tolrance
MaxIter: maximum iteration number
lr: learning rates
lr_dc_pow: rate to decrease 'lr' -- lr / (1 + k / lr_dc)
"""
mutable struct VMC
    tol::Float64
    MaxIter::Int
    lr::Float64
    lr_dc::Float64
end

function InverseLR(ν, lr, lr_dc)
    return lr / (1 + ν / lr_dc), ν+1
end

VMC(MaxIter::Int, lr::Float64; tol = 1.0e-3, lr_dc = 50.0) = VMC(tol, MaxIter, lr, lr_dc);
        
function gd_GradientByVMC(opt::VMC,
                sam::MHSampler, 
                ham::SumH, 
                wf, ps, st, ν = 1, verbose = true, accMCMC = [10, [0.45, 0.55]])
    res = 1.0;
    λ₀ = 0.;
    α = opt.lr;
    err_opt = zeros(opt.MaxIter)
    x0, ~, acc = sampler_restart(sam, ps, st);
    #x = reduce(vcat,reduce(vcat,x0))
    #display(histogram(x,xlim = (-40,40)))
    verbose && @printf("Initialize MCMC: Δt = %.2f, accRate = %.4f \n", sam.Δt, acc)
    acc_step = accMCMC[1];
    acc_opt = zeros(acc_step)
    acc_range = accMCMC[2];
    verbose && @printf("   k |  𝔼[E_L]  |  V[E_L] |   res   |   LR    |accRate|   Δt    \n")
    for k = 1 : opt.MaxIter
        sam.x0 = x0;
        sam.Ψ = wf;
        acc_opt[mod(k,acc_step)+1] = acc
        if mod(k,acc_step) == 0
            if mean(acc_opt) < acc_range[1]
                sam.Δt = sam.Δt * exp(1/10 * (mean(acc_opt) - acc_range[1])/acc_range[1])
            elseif mean(acc_opt) > acc_range[2]
                sam.Δt = sam.Δt * exp(1/10 * (mean(acc_opt) - acc_range[2])/acc_range[2])
            end
        end
        α, ν = InverseLR(ν, opt.lr, opt.lr_dc)
        # optimize
        λ₀, σ, E, x0, acc = Eloc_Exp_TV_clip(wf, ps, st, sam, ham)
        g = grad(wf, x0, ps, st, E)
        #if k % 10 == 0 
        #    x = reduce(vcat,reduce(vcat,x0))
        #    display(histogram(x,xlim = (-40,40)))
        #end
        # Optimization
        st_opt = Optimisers.setup(Optimisers.AdamW(α), ps)
        st_opt, ps = Optimisers.update(st_opt, ps, g)

        res = norm(destructure(g)[1]);
        verbose && @printf(" %3.d | %.5f | %.5f | %.5f | %.5f | %.3f | %.3f \n", 
                k, λ₀, σ, res, α, acc, sam.Δt);
        err_opt[k] = λ₀
        if res < opt.tol
            break;
        end  
    end
    return wf, err_opt, ps
end


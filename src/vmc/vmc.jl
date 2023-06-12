export VMC
using Printf
using LinearAlgebra
using Optimisers
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

VMC(MaxIter::Int, lr::Float64; tol = 1.0e-3, lr_dc = 50.0) = VMC(tol, MaxIter, lr, lr_dc);
        
function gd_GradientByVMC(opt::VMC,
                sam::MHSampler, 
                ham::SumH, 
                wf, ps, st)
    res = 1.0;
    λ₀ = 0.;
    α = opt.lr;
    err_opt = zeros(opt.MaxIter)
    x0, ~, acc = sampler_restart(sam, ps, st);
    verbose = true
    verbose && @printf("   k |  𝔼[E_L]  |  V[E_L] |   res   |   LR    |accRate|   Δt    \n")
    for k = 1 : opt.MaxIter
        sam.x0 = x0;
        sam.Ψ = wf;
        
        # optimize
        λ₀, σ, E, x0, acc = Eloc_Exp_TV_clip(wf, ps, st, sam, ham)
        g = grad(wf, x0, ps, st, E)

        # Optimization
        st_opt = Optimisers.setup(Optimisers.Adam(α), ps)
        st_opt, ps = Optimisers.update(st_opt, ps, g)

        res = norm(destructure(g)[1]);
        verbose && @printf(" %3.d | %.5f | %.5f | %.5f | %.5f | %.3f | %.3f \n", 
                k, λ₀, σ, res, α, acc, sam.Δt);
        err_opt[k] = λ₀
        if res < opt.tol
            break;
        end  
    end
    return wf, λ₀, σ, err_opt, ps
end


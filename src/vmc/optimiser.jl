export VMC

"""
Minimizing the Rayleigh quotient by VMC
tol: tolrance
MaxIter: maximum iteration number
lr: learning rates
lr_dc_pow: rate to decrease 'lr' -- lr / (1 + k / lr_dc)
"""
mutable struct VMC <: AbstractOptimizer
    tol::Float64
    MaxIter::Int
    lr::Float64
    lr_dc::Float64
end

vmc(MaxIter::Int, lr::Float64; tol = 1.0e-3, lr_dc = 50.0) = 
        VMC(tol, MaxIter, lr, lr_dc);

"""
decreasing the learning rate
"""
function InverseLR(ν, lr, lr_dc)
    return lr / (1 + ν / lr_dc), ν+1
end

function gd_GradientByVMC(
                sam::AbstractSampler, 
                ham::SumH, 
                wf)
    res = 1.0;
    λ₀ = 0.;
    g0 = 0;
    α = opt.lr;
    err_opt = zeros(opt.MaxIter)
    x0, ~, acc = sampler_restart(sam);
    verbose && @printf("   k |  𝔼[E_L]  |  V[E_L] |   res   |   LR    |accRate|   Δt   | regular\n")
    νk = ν;
    for k = 1 : opt.MaxIter
        sam.x0 = x0;
        sam.Ψ = wf;
        
        # optimize
        g0, mₜ, vₜ, c, λ₀, σ, x0, acc = adam(mₜ, vₜ, c, ν, U, sam, ham, clip, ηₜ = α; Γ = Γ);
        res = norm(g0);
        set_params!(U, c)
        verbose && @printf(" %3.d | %.5f | %.5f | %.5f | %.5f | %.3f | %.3f  | %.4f \n", 
                k, λ₀, σ, res, α, acc, sam.Δt, norm(Γ));
        err_opt[k] = λ₀
        if res < opt.tol
            break;
        end  
    end
    set_params!(U, c)
    return U, λ₀, σ, err_opt, err_avg, mₜ, vₜ
end


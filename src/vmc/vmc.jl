export VMC
using Printf
using LinearAlgebra
using Optimisers
using Plots

mutable struct VMC
    tol::Float64
    MaxIter::Int
    lr::Float64
    lr_dc::Float64
    type::opt
end

VMC(MaxIter::Int, lr::Float64, type; tol = 1.0e-3, lr_dc = 50.0) = VMC(tol, MaxIter, lr, lr_dc, type);
        
function gd_GradientByVMC(opt_vmc::VMC, sam::MHSampler, ham::SumH, 
                wf, ps, st; 
                ν = 1, verbose = true, density = false, 
                accMCMC = [10, [0.45, 0.55]], batch_size = 1)

    mₜ, vₜ = initp(opt_vmc.type, ps)
    res, λ₀, α = 1.0, 0., opt_vmc.lr
    err_opt = zeros(opt_vmc.MaxIter)

    x0, ~, acc = sampler_restart(sam, ps, st, batch_size = batch_size) 
    density && begin 
        x = reduce(vcat,reduce(vcat,x0))
        display(histogram(x, xlim = (-10,10), ylim = (0,1), normalize=:pdf))
    end
    
    acc_step, acc_range = accMCMC
    acc_opt = zeros(acc_step)

    verbose && @printf("Initialize MCMC: Δt = %.2f, accRate = %.4f \n", sam.Δt, acc)
    verbose && @printf("   k |  𝔼[E_L]   |  V[E_L] |   res   |   LR    |accRate|   Δt    \n")
    _basis_size = size(ps.branch.bf.hidden.layer_1.hidden.W, 2)
    _Nbf = length(keys(ps.branch.bf.hidden))
    @info("size of basis = $_basis_size, number of bfs = $_Nbf")
    for k = 1 : opt_vmc.MaxIter
        sam.x0 = x0
        
        # adjust Δt
        acc_opt[mod(k,acc_step)+1] = acc
        sam.Δt = acc_adjust(k, sam.Δt, acc_opt, acc_range, acc_step)

        # adjust learning rate
        α, ν = InverseLR(ν, opt_vmc.lr, opt_vmc.lr_dc)

        # optimization
        ps, acc, λ₀, res, σ, x0, mₜ, vₜ = Optimization(opt_vmc.type, wf, ps, st, sam, ham, α, mₜ, vₜ, ν, batch_size = batch_size)
        density && begin 
            if k % 10 == 0
                x = reduce(vcat,reduce(vcat,x0))
                display(histogram!(x, xlim = (-10,10), ylim = (0,1), normalize=:pdf))
            end
        end
        
        # err
        verbose && @printf(" %3.d | %.5f | %.5f | %.5f | %.5f | %.3f | %.3f \n", k, λ₀, σ, res, α, acc, sam.Δt)
        err_opt[k] = λ₀

        if res < opt_vmc.tol
            break;
        end  
    end
    return wf, err_opt, ps
end

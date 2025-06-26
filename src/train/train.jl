using ProgressMeter, Distributed
using UnPack, Printf, Statistics, ParallelDataTransfer
export train

function train(mol, model_list::Vector, ps_list::Vector, st_list::Vector, spec_list::Vector, spec1p_list::Vector, totdeg_list, ν_list, optimizer)
    @unpack iterations, clip, acc_step, acc_range, Δt, damping, nchains, burnin, lag, lr, lr_dc, η, norm_constrain, m = optimizer
    level = length(iterations)
    val_list = [zeros(iterations[i]) for i = 1:level]
    var_list = [zeros(iterations[i]) for i = 1:level]
    rank_list = [zeros(iterations[i]) for i = 1:level]
    @sync for k in 1:nprocs()
        @async begin
            sendto(k, mol = mol)
            sendto(k, nchains = nchains)
            sendto(k, burnin = burnin)
            sendto(k, lag = lag)
            sendto(k, clip = clip)
            sendto(k, Δt = Δt)
        end
    end

    @everywhere ham = SumH(mol.nuclei)
    acc_opt = zeros(acc_step)
    _iter = 0
    OptParams = init_optim(10, nchains, optimizer.sr_method)
    io = mkrespath(optimizer.res_path)
    
    for i = 1:level
        model, ps, st, spec, spec1p = model_list[i], ps_list[i], st_list[i], spec_list[i], spec1p_list[i]
        iteration = iterations[i]
        @sync for k in 1:nprocs()
            @async begin
                sendto(k, model = model)
                sendto(k, ps = ps)
                sendto(k, st = st)
            end
        end
        dim_ps = length(destructure(ps)[1])
        @everywhere dim_ps = length(destructure(ps)[1])
        @everywhere _force = zeros(Float64, dim_ps)

        progress_iter = Progress(iteration; barglyphs = BarGlyphs("[=> ]"), barlen = 50, desc = "VMC steps: ", color=:yellow)
        @info("level = $i, totdeg = $(totdeg_list[i]), order = $(ν_list[i]), number of parameters = $dim_ps")
        println(io, "level = $i, totdeg = $(totdeg_list[i]), order = $(ν_list[i]), number of parameters = $dim_ps")
        println(io, "   k         |   𝔼[E_L]       |   V[E_L]      |   accRate     |   Δt          |   free_memory |   res       |   lr     |   rank  \n")  
        @info("   k         |   𝔼[E_L]       |   V[E_L]      |   accRate     |   Δt          |   free_memory |   res       |   lr     |   rank  \n")
        
        if i == 1
            OptParams = init_optim(dim_ps, nchains * nprocs(), optimizer.sr_method)
            @everywhere begin 
                _x, _theta, _acc = init_walkers(mol, model, ps, st, burnin, nchains, Δt)
                acc = mean(_acc)
            end
            ācc = mean([@getfrom k acc for k in procs()])
            @printf("Initialize MCMC: Δt = %.2f, accRate = %.4f \n", Δt, ācc)
        else
            ps = transfer_weights!(ps_list[i-1], ps, spec_list[i-1], spec, spec1p_list[i-1], spec1p)
            idx = transfer_weights_idx!(ps_list[i-1], ps, spec_list[i-1], spec, spec1p_list[i-1], spec1p)
            OptParams = Embedding(damping, nchains * nprocs(), dim_ps, idx, OptParams, optimizer)
        end

        for iter = 1:iteration
            p, re = destructure(ps)
            @everywhere p, re = destructure(ps)
            @sync for k = 1:nprocs()
                @async sendto(k, p = p)
            end
            @everywhere ps = re(p)
            damping = max(InverseLR(i, optimizer.damping, optimizer.damping_decay), optimizer.damping_min)
            _iter += 1
            @everywhere begin 
                _x, _theta, _acc, _elocs, _o_tot = compute_Eloc_dp(ham, model, ps, st, _x, _theta, _acc, Δt, lag)
                acc = mean(_acc) 
            end
            ācc = mean([@getfrom k acc for k in 1:nprocs()])
            Eloc = vcat([@getfrom k _elocs for k in 1:nprocs()]...)
            v̄al = median(Eloc) 
            val_list[i][iter], var_list[i][iter] = mean(Eloc), sqrt(var(Eloc)/length(Eloc))
            acc_opt[mod1(iter, acc_step)] = ācc
            Δt = acc_adjust(iter, Δt, acc_opt, acc_range, acc_step)
            @sync for k = 1:nprocs()
                @async begin 
                    sendto(k, Δt = Δt)
                    sendto(k, v̄al = v̄al)
                    sendto(k, val_mean = val_list[i][iter])
                end
            end
            @everywhere ΔE = _elocs .- v̄al
            @everywhere _a = clip * Statistics.mean( abs.(ΔE) )
            ā = mean([@getfrom k _a for k in 1:nprocs()])
            @sync for k = 1:nprocs()
                @async sendto(k, ā = ā)
            end
            @everywhere begin 
                ΔE = clipE.(ΔE, Ref(ā)) 
                _elocs = v̄al .+ ΔE .- val_mean # e - ē
                _o_mean = mean(_o_tot, dims = 2)
            end
            o_mean = mean([@getfrom k _o_mean for k in 1:nprocs()])
            ldiv!(nprocs(), o_mean)
            @sync for k = 1:nprocs()
                @async sendto(k, o_mean = o_mean)
            end
            @everywhere begin 
                _o_tot .-= _o_mean
                mul!(_force, _o_tot, _elocs)
            end
            f = sum([@getfrom k _force for k in 1:nprocs()])
            OptParams.f = 2 * f/(nchains * nprocs())
            o = hcat([@getfrom k _o_tot for k = 1:nprocs()]...)
            Eloc = vcat([@getfrom k _elocs for k in 1:nprocs()]...)
            γ = InverseLR(_iter, lr, lr_dc)
            OptParams.dw_tot, r, res = opts!(iter, OptParams, optimizer.sr_method, Eloc, o, nchains * nprocs(), damping, dim_ps, η, norm_constrain, γ, m)
            rank_list[i][iter] = r
            ProgressMeter.next!(progress_iter; showvalues = [(:iter, iter), (:E, val_list[i][iter]), (:var, var_list[i][iter]), (:acc, ācc), (:lr, γ), (:res, res), (:t, Δt), (:rank, r)])
            ps = update_netparams!(ps, OptParams.dw_tot)
            _mem = mean([@getfrom k Sys.free_memory() for k in nprocs()])
            println(io, @sprintf("   %5.d     |   %.5f     |   %.5f     |   %.5f     |   %.5f     |   %.5f     |  %.5f     |   %.5f     |   %5.d ", iter, val_list[i][iter], var_list[i][iter], ācc, Δt, _mem / 2^30, res, γ, r))
            println(@sprintf("   %5.d     |   %.5f     |   %.5f     |   %.5f     |   %.5f     |   %.5f     |  %.5f     |   %.5f     |   %5.d ", iter, val_list[i][iter], var_list[i][iter], ācc, Δt, _mem / 2^30, res, γ, r))
            if iter % 10 == 0
                flush(io)
            end
        end
        per = 0.2
        _err = zero(val_list[i])
        for j = 1:length(val_list[i])
            _err[j] = mean(val_list[i][Int(ceil(j - per  * j)):j])
        end
        println(io, "average energy: $(_err[end])")
        @info("average energy: $(_err[end])")
        ps_list[i] = deepcopy(ps)
        flush(io)
        close(io)
    end
    record_energy(val_list, optimizer.res_path)
    record_rank(rank_list, optimizer.res_path)
    record_var(var_list, optimizer.res_path)
    record_ps(ps_list, optimizer.res_path)
    return model_list, ps_list, st_list, val_list, var_list, rank_list
end

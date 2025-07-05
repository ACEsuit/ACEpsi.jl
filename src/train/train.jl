using ProgressMeter, Distributed
using UnPack, Printf, Statistics, ParallelDataTransfer
export train

function train(x0, mol, model_list::Vector, ps_list::Vector, st_list::Vector, spec_list::Vector, spec1p_list::Vector, totdeg_list, ν_list, optimizer)    
    idx_i, idx_l, latest_file = ACEpsi.get_resume_index(optimizer.checkpoints, optimizer.res_path) 
    if idx_i == 1 && idx_l == 1
        level = length(optimizer.iterations) 
        val_list = [zeros(optimizer.iterations[i]) for i = 1:level]
        var_list = [zeros(optimizer.iterations[i]) for i = 1:level]
        rank_list = [zeros(optimizer.iterations[i]) for i = 1:level]
        i = 1
        model, ps, st, spec, spec1p = model_list[i], ps_list[i], st_list[i], spec_list[i], spec1p_list[i]
        dim_ps = length(destructure(ps)[1])
        io = ACEpsi.mkrespath(optimizer.res_path)
        OptParams = ACEpsi.init_optim(dim_ps, optimizer.nchains * nprocs(), optimizer.sr_method)
    else
        @assert latest_file !== nothing "Checkpoint file not found, but trying to resume."

        @info "Resuming from checkpoint file: $latest_file"
        @load latest_file x0 mol model_list ps_list st_list spec_list spec1p_list totdeg_list ν_list optimizer OptParams io val_list var_list rank_list
    end

    @unpack iterations, clip, acc_step, acc_range, damping, nchains, burnin, lag, lr, lr_dc, η, norm_constrain, m, checkpoints = optimizer
    level = length(optimizer.iterations) 
       
    for i = idx_i:level
        model, ps, st, spec, spec1p = model_list[i], ps_list[i], st_list[i], spec_list[i], spec1p_list[i]
        checkpoint = checkpoints[i]
        dim_ps = length(destructure(ps)[1])
        
        @info("level = $i, totdeg = $(totdeg_list[i]), order = $(ν_list[i]), number of parameters = $dim_ps")
        println(io, "level = $i, totdeg = $(totdeg_list[i]), order = $(ν_list[i]), number of parameters = $dim_ps")
        println(io, "   k         |   𝔼[E_L]       |   V[E_L]      |   accRate     |   Δt          |   free_memory |   res       |   lr     |   rank  \n")  
        @info("   k         |   𝔼[E_L]       |   V[E_L]      |   accRate     |   Δt          |   free_memory |   res       |   lr     |   rank  \n")
        
        if i > 1
            ps = transfer_weights!(ps_list[i-1], ps, spec_list[i-1], spec, spec1p_list[i-1], spec1p)
            idx = transfer_weights_idx!(ps_list[i-1], ps, spec_list[i-1], spec, spec1p_list[i-1], spec1p)
            OptParams = Embedding(damping, nchains * nprocs(), dim_ps, idx, OptParams, optimizer)
        end
        for (idx, _iters) in enumerate(checkpoint)
            if i == idx_i 
                if idx == idx_l
                    x0, mol, model, ps, st, optimizer, OptParams, io, val_list, var_list, rank_list = train(i, _iters, io, val_list, var_list, rank_list, OptParams, x0, mol, model, ps, st, optimizer)
                    save_path = joinpath(optimizer.res_path, "$(i)_$(_iters).jld2")
                    ps_list[i] = deepcopy(ps)
                    @save save_path x0 mol model_list ps_list st_list spec_list spec1p_list totdeg_list ν_list optimizer OptParams io val_list var_list rank_list
                end
            else
                x0, mol, model, ps, st, optimizer, OptParams, io, val_list, var_list, rank_list = train(i, _iters, io, val_list, var_list, rank_list, OptParams, x0, mol, model, ps, st, optimizer)
                save_path = joinpath(optimizer.res_path, "$(i)_$(_iters).jld2")
                ps_list[i] = deepcopy(ps)
                @save save_path x0 mol model_list ps_list st_list spec_list spec1p_list totdeg_list ν_list optimizer OptParams io val_list var_list rank_list
            end
        end
    end
    record_energy(val_list, optimizer.res_path)
    record_rank(rank_list, optimizer.res_path)
    record_var(var_list, optimizer.res_path)
    record_ps(ps_list, optimizer.res_path)
    close(io)
    return model_list, ps_list, st_list, val_list, var_list, rank_list
end

function train(i::Int, _iters, io, val_list, var_list, rank_list, OptParams, x0, mol, model::Chain, ps, st, optimizer)
    @unpack clip, acc_step, acc_range, damping, nchains, lag, lr, lr_dc, η, norm_constrain, m = optimizer
    progress_iter = Progress(_iters[2] - _iters[1] + 1; barglyphs = BarGlyphs("[=> ]"), barlen = 50, desc = "VMC steps: ", color=:yellow)
    ham = SumH(mol.nuclei)
    dim_ps = length(destructure(ps)[1])
    Δt = optimizer.Δt
    for k in 1:nprocs()
        sendto(k, mol = mol)
        sendto(k, ham = ham)
        sendto(k, nchains = nchains)
        sendto(k, lag = lag)
        sendto(k, clip = clip)
        sendto(k, Δt = Δt)
        sendto(k, model = model)
        sendto(k, ps = ps)
        sendto(k, st = st)
        sendto(k, dim_ps = dim_ps)
    end
    @everywhere _force = zeros(Float64, dim_ps)
    _x, _theta, _acc = [], [], []
    if x0 == nothing
        @everywhere begin 
            _x, _theta, _acc = init_walkers(mol, model, ps, st, burnin, nchains, optimizer.Δt)
        end
    else
        for k in 1:nprocs()
            sendto(k, _x = x0[myid()])
        end
        @everywhere _theta = evalx.(Ref(model), _x, Ref(ps), Ref(st))
        @everywhere _acc = ones(length(_theta))
    end
    for iter = _iters[1]:_iters[2]
        p, re = destructure(ps)
        @everywhere p, re = destructure(ps)
        for k = 1:nprocs()
            sendto(k, p = p)
        end
        @everywhere ps = re(p)
        damping = max(ACEpsi.InverseLR(iter, optimizer.damping, optimizer.damping_decay), optimizer.damping_min)
        @everywhere begin 
            _x, _theta, _acc, _elocs, _o_tot = compute_Eloc_dp(ham, model, ps, st, _x, _theta, _acc, Δt, lag)
            acc = mean(_acc) 
        end
        ācc = mean([@getfrom k acc for k in 1:nprocs()])
        Eloc = vcat([@getfrom k _elocs for k in 1:nprocs()]...)
        v̄al = median(Eloc) 
        val_list[i][iter], var_list[i][iter] = mean(Eloc), sqrt(var(Eloc)/length(Eloc))
        optimizer.acc_opt[mod1(iter, acc_step)] = ācc
        optimizer.Δt = ACEpsi.acc_adjust(iter, optimizer.Δt, optimizer.acc_opt, acc_range, acc_step)
        Δt = optimizer.Δt
        for k = 1:nprocs()
            sendto(k, Δt = Δt)
            sendto(k, v̄al = v̄al)
            sendto(k, val_mean = mean(Eloc))
        end
        @everywhere ΔE = _elocs .- v̄al # E - medial(E)
        @everywhere _a = clip * Statistics.mean( abs.(ΔE) )
        ā = mean([@getfrom k _a for k in 1:nprocs()])
        for k = 1:nprocs()
            sendto(k, ā = ā)
        end
        @everywhere begin 
            ΔE = ACEpsi.clipE.(ΔE, Ref(ā)) 
            _elocs = v̄al .+ ΔE .- val_mean # E - medial(E) + medial(E) - val_mean
            _o_mean = mean(_o_tot, dims = 2) 
        end
        o_mean = sum([@getfrom k _o_mean for k in 1:nprocs()])
        ldiv!(nprocs(), o_mean)
        for k = 1:nprocs()
            sendto(k, o_mean = o_mean)
        end
        @everywhere begin 
            _o_tot .-= _o_mean # o - o_mean
            mul!(_force, _o_tot, _elocs) # (o - o_mean) * (E - E_mean)
        end
        f = sum([@getfrom k _force for k in 1:nprocs()])
        OptParams.f = 2 * f/(nchains * nprocs())
        o = hcat([@getfrom k _o_tot for k = 1:nprocs()]...) # o - o_mean
        Eloc = vcat([@getfrom k _elocs for k in 1:nprocs()]...) # E - E_mean
        ldiv!(sqrt(nprocs() * nchains), o)
        ldiv!(sqrt(nprocs() * nchains), Eloc)

        γ = ACEpsi.InverseLR(iter, lr, lr_dc)
        OptParams.dw_tot, r, res = opts!(iter, OptParams, optimizer.sr_method, Eloc, o, nchains * nprocs(), damping, dim_ps, η, norm_constrain, γ, m)
        rank_list[i][iter] = r
        ProgressMeter.next!(progress_iter; showvalues = [(:iter, iter), (:E, val_list[i][iter]), (:var, var_list[i][iter]), (:acc, ācc), (:lr, γ), (:res, res), (:t, Δt), (:rank, r)])
        ps = update_netparams!(ps, OptParams.dw_tot)
        _mem = mean([@getfrom k Sys.free_memory() for k in nprocs()])
        println(io, @sprintf("   %5.d     |   %.5f     |   %.5f     |   %.5f     |   %.5f     |   %.5f     |  %.5f     |   %.5f     |   %5.d ", iter, val_list[i][iter], var_list[i][iter], ācc, Δt, _mem / 2^30, res, γ, r))
        println(@sprintf("   %5.d     |   %.5f     |   %.5f     |   %.5f     |   %.5f     |   %.5f     |  %.5f     |   %.5f     |   %5.d ", iter, val_list[i][iter], var_list[i][iter], ācc, Δt, _mem / 2^30, res, γ, r))
    end
    per = 0.2
    _err = zero(val_list[i])
    for j = 1:length(val_list[i])
        _err[j] = mean(val_list[i][Int(ceil(j - per  * j)):j])
    end
    println(io, "average energy: $(_err[end])")
    @info("average energy: $(_err[end])")
    flush(io)
    x0 = [@getfrom k _x for k in 1:nprocs()]
    return x0, mol, model, ps, st, optimizer, OptParams, io, val_list, var_list, rank_list
end
using ProgressMeter
using UnPack, Printf
using Statistics, Random
using HyperDualNumbers: Hyper
export train

function train(last_iter, iteration, io, st_opt, ham, ps_H, st_H, x0, mol, model, ps, st, optimizer;
               burnin::Integer, 
               device::Symbol = :auto, xprop_buf = nothing, logu_buf  = nothing)

    @unpack clip, acc_step, acc_range, damping, nchains, lag, Δt, lr, lr_dc,
            η, norm_constrain, m, damping_decay, damping_min, sr_method,
            acc_opt = optimizer

    prog = Progress(iteration; barglyphs = BarGlyphs("[=> ]"),
                    barlen = 50, desc = "VMC steps: ", color=:yellow)

    p0, s = destructure(ps)
    dim_ps = length(p0)

    val_list = Vector{Float64}(undef, iteration)
    var_list = similar(val_list)
    rank_list = similar(val_list)
    theta = similar(p0, nchains)

    if x0 === nothing
        x0 = initialize_around_nuclei(mol, nchains; Δt=Δt, device=device);
        x0, theta, st_eval, acc_hist = burnin!(x0, model, ps, st, burnin; Δt=Δt);
    else
        x0, theta, st_eval, acc_hist = burnin!(x0, model, ps, st, burnin; Δt=Δt);
    end
    (val, st_eval), pb = Zygote.pullback(model, x0, ps, st_eval);
    pb((val, st_eval));
    Tx = eltype(x0)
    xh = Hyper(rand(Tx), rand(Tx), rand(Tx), rand(Tx)) .* x0
    y, st_lap = model(xh, ps, st);

    Xprop = xprop_buf === nothing ? similar(x0) : xprop_buf
    logu = logu_buf === nothing ? similar(theta) : logu_buf
    fill!(Xprop, zero(Tx))
    fill!(logu, zero(Tx))

    ācc = mean(acc_hist)
    @printf("acc = %.4f\n", ācc)
    force = similar(x0, Tx, dim_ps)
    sr_st = nothing
    acc_hist = similar(acc_hist, lag)
    for iter = 1:iteration
        x0, theta, st_eval, acc_hist = distributed_sampling!(x0, model, ps, st_eval, theta, acc_hist, Δt, lag; xprop_buf = Xprop, logu_buf = logu);
        Eloc, o, st_eval, st_lap = compute_Eloc_dp(ham, ps_H, st_H, model, ps, st_eval, st_lap, x0);

        ācc = mean(acc_hist)
        v̄al = median(Eloc)
        val_mean = mean(Eloc)
        val_list[iter] = val_mean
        var_list[iter] = sqrt(var(Eloc) / length(Eloc))

        acc_opt[mod1(iter, acc_step)] = ācc
        
        Δt = acc_adjust(Δt, iter, acc_opt; acc_step=acc_step, acc_range=acc_range, η=0.10)
        optimizer.Δt = Δt

        ΔE = Eloc .- v̄al
        ā = Tx(clip * mean(abs.(ΔE)))
        ΔE = clipE(ΔE, ā)
        Eloc_adj = v̄al .+ ΔE .- val_mean

        o .-= mean(o, dims=2)

        # o :: (dim_ps, nchains), Eloc_adj :: (nchains,)
        fill!(force, zero(Tx))
        nchains_x = Tx(nchains)
        mul!(force, o, Eloc_adj)
        force    *= (2 / nchains_x)
        o        /= sqrt(nchains_x)
        Eloc_adj /= sqrt(nchains_x)

        damp_now = Tx(max(InverseLR(iter, damping, damping_decay), damping_min))
        γ = Tx(InverseLR(last_iter+iter, lr, lr_dc))


        finite_guard!(o, "input-opts")
        st_opt, dw_tot, r, res, sr_st =
            opts!(iter, st_opt, sr_method, force, Eloc_adj, o,
                  nchains, damp_now, dim_ps, Tx(η), Tx(norm_constrain), γ, Tx(m), sr_st);
        rank_list[iter] = r

        ProgressMeter.next!(prog; showvalues = [
            (:iter, iter), (:E, val_list[iter]), (:var, var_list[iter]),
            (:acc, ācc), (:lr, γ), (:res, res), (:t, Δt), (:rank, r)
        ])

        ps = update_netparams!(ps, Tx.(dw_tot))

        println(io, @sprintf(" %5d | E=%.6f | σ/√n=%.6f | acc=%.4f | Δt=%.5f | res=%.4e | lr=%.4e | rank=%5d",
                              iter, val_list[iter], var_list[iter], ācc, Δt, res, γ, r))
    end
    per = 0.2
    _err = similar(val_list)
    @inbounds for j = 1:length(val_list)
        lo = max(1, ceil(Int, j - per*j))
        _err[j] = mean(@view val_list[lo:j])
    end
    println(io, "average energy: $(_err[end])")
    @info("average energy: $(_err[end])")
    flush(io)
    return x0, mol, model, ps, st, optimizer, st_opt, io, val_list, var_list, rank_list
end

function train(mol, 
               model_list, ps_list, st_list, spec_list, spec1p_list, totdeg_list, ν_list, 
               optimizer; x0 = nothing, device::Symbol = :auto)
    levels = length(optimizer.iterations)
    val_list  = Vector{Vector{Float64}}()
    var_list  = Vector{Vector{Float64}}()
    rank_list = Vector{Vector{Float64}}()

    ham = SumH(mol)
    rng = Random.default_rng()
    ps_H, st_H = Lux.setup(rng, ham)

    use_gpu = device == :gpu || (device == :auto && CUDA.has_cuda())

    io = mkrespath(optimizer.res_path)
    ps    = ps_list[1]
    p_flat, sp = destructure(ps)
    dim_ps = length(p_flat)
    OptParams = init_optim(dim_ps, optimizer.nchains, optimizer.sr_method)
    ps_opt, st_opt = Lux.setup(rng, OptParams)
    _, ts = destructure(st_opt)
    last_iter = 0
    x_test = initialize_around_nuclei(mol, 10; Δt=optimizer.Δt, σ=1.0, device=:cpu);
    y_old, ps_old, spec_old, spec1p_old = nothing, nothing, nothing, nothing    
    for i in 1:levels
        model, ps, st, spec, spec1p, totdeg, ν = model_list[i], ps_list[i], st_list[i], spec_list[i], spec1p_list[i], totdeg_list[i], ν_list[i];
        if i >= 2
            ps  = ACEpsi.transfer_weights!(ps_old, ps,
                                    spec_old, spec,
                                    spec1p_old, spec1p)
            y_test = model(x_test, ps, st)[1];
            y_diff = mean(abs.(y_test .- y_old))
            @info("Level $i weight transfer test diff: $y_diff")    
            y_old = y_test
            ps_old = deepcopy(ps)
            spec_old = deepcopy(spec)
            spec1p_old = deepcopy(spec1p)
        else
            y_old = model(x_test, ps, st)[1];
            ps_old = deepcopy(ps)
            spec_old = deepcopy(spec)
            spec1p_old = deepcopy(spec1p)
        end
    end
    for i in 1:levels
        model, ps, st, spec, spec1p, totdeg, ν = model_list[i], ps_list[i], st_list[i], spec_list[i], spec1p_list[i], totdeg_list[i], ν_list[i];
        p_flat, sp = destructure(ps)
        dim_ps = length(p_flat)

        @info("level=$i/$(length(model_list)), totdeg=$totdeg, order=$ν, n_params=$dim_ps")
        println(io, "level=$i/$(length(model_list)), totdeg=$totdeg, order=$ν, n_params=$dim_ps")
        header = "   k     |   𝔼[E_L]     |   σ/√n       |   accRate   |   Δt        |   res        |    lr       |  rank"
        println(io, header)
        @info(header)

        if i > 1
            ps  = transfer_weights!(ps_list[i-1], ps,
                                    spec_list[i-1], spec,
                                    spec1p_list[i-1], spec1p)
            idx = transfer_weights_idx!(ps_list[i-1], ps,
                                        spec_list[i-1], spec,
                                        spec1p_list[i-1], spec1p)
            st_opt = embedding(optimizer.damping, optimizer.nchains, dim_ps, idx, st_opt, optimizer.sr_method)
            _, ts = destructure(st_opt)
            y_new = model(x_test, ps, st)[1]
            println(y_new[1])
            println(Array(y_old)[1])
        end

        if use_gpu
            dev = CUDADevice()
            ps_H, st_H, ps, st, ps_opt, st_opt = dev(ps_H), dev(st_H), dev(ps), dev(st), dev(ps_opt), dev(st_opt)
            if x0 isa AbstractArray
                x0 = dev(x0)
            end
            x_test_dev = dev(x_test)
            
        else
            if x0 isa AbstractGPUArray
                x0 = Array(x0)
            end
        end
        Xprop = x0 === nothing ? nothing : similar(x0)
        logu = x0 === nothing ? nothing : similar(x0, optimizer.nchains)

        x0, mol, model, ps, st, optimizer, st_opt, io,
        val_i, var_i, rank_i =
            ACEpsi.train(last_iter, optimizer.iterations[i], io, st_opt,
                  ham, ps_H, st_H,
                  x0, mol, model, ps, st, optimizer;
                  burnin = optimizer.burnin, 
                  device = use_gpu ? :gpu : :cpu, xprop_buf = Xprop, logu_buf = logu)

        push!(val_list,  val_i)
        push!(var_list,  var_i)
        push!(rank_list, rank_i)

        ptrain, _ = destructure(ps)
        ps = sp(ptrain)
        ps_list[i] = deepcopy(ps)

        st_opttrain, _ = destructure(st_opt)
        st_opt = ts(Array(st_opttrain))
        last_iter += optimizer.iterations[i]
        y_old = model(x_test_dev, ps, st)[1];
    end

    record_energy(val_list, optimizer.res_path)
    record_rank(rank_list,   optimizer.res_path)
    record_var(var_list,     optimizer.res_path)
    record_ps(ps_list,       optimizer.res_path)

    if io !== nothing
        close(io)
    end

    return x0, model_list, ps_list, st_list, val_list, var_list, rank_list
end

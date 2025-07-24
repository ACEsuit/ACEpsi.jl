using JSON
using StaticArrays
using ForwardDiff
using Optimisers: destructure
using LinearAlgebra

export clipE, test_wavefunction, setup, load_setup

function acc_adjust(k::Int, Δt::Number, acc_opt::AbstractVector, acc_range::AbstractVector, acc_step::Int)
    if mod(k, acc_step) == 0
        if mean(acc_opt) < acc_range[1]
            Δt *= exp(1/10 * (mean(acc_opt) - acc_range[1])/acc_range[1])
        elseif mean(acc_opt) > acc_range[2]
            Δt *= exp(1/10 * (mean(acc_opt) - acc_range[2])/acc_range[2])
        end
        if Δt > 1.0
            Δt = 1.0
        end
    end
    return Δt
end

function clipE(x::TX, ā::TA) where {TA <: Float64, TX <: Float64}
    if abs(x) <= ā
        return x
    else
        return ā * sign(x) * (1 + log((1 + (abs(x)/ā)^2)/2))
    end
end

function InverseLR(k::TK, lr::TL, lr_dc::TD) where {TK <: Int64, TL <: Float64, TD <: Int64}
    return lr / (1 + k / lr_dc)
end

function update_netparams!(ps, dw_tot)
    p, s = destructure(ps)
    ps = s(p + dw_tot)
    return ps
end

function mkrespath(res_path)
    @info("Storing results at $res_path")
    mkpath(res_path)
    io = open(res_path * "output.txt", "w+") 
    return io
end

function record_energy(E::Vector{T}, res_path) where {T}
    open(res_path * "energy.json","w") do f JSON.print(f, E) end
end

function record_var(Err::Vector{T}, res_path) where {T}
    open(res_path * "var.json","w") do f JSON.print(f, Err) end
end

function record_rank(ER::Vector{T}, res_path) where {T}
    open(res_path * "rank.json","w") do f JSON.print(f, ER) end
end

function record_ps(P::T, res_path) where {T}
    record_p = destructure(P)[1]
    open(res_path * "parameters.json","w") do f JSON.print(f, record_p) end
end


function ∇clip!(dw_tot, f::AbstractArray{TT}, norm_constrain::T, γ::TG) where {TT <: Float64, T, TG}
    res = γ * dot(dw_tot, f)
    a = min(1, sqrt(norm_constrain)/res)
    if norm_constrain > 0 && res > norm_constrain
        lmul!(a, dw_tot)
    end
end

function ∇clip!(dw_tot, norm_constrain::T, res::TG) where {T, TG}
    a = min(1, sqrt(norm_constrain)/res)
    if norm_constrain > 0 && res > norm_constrain
        lmul!(a, dw_tot)
    end
end

function test_wavefunction(model_list, ps_list, st_list, spec_list, spec1p_list, mol)
    X = [SVector{3}(rand(3)) for i = 1:mol.Nel]
    for i = 1:length(model_list)
        println("i = $i")
        model, ps, st, spec, spec1p = model_list[i], ps_list[i], st_list[i], spec_list[i], spec1p_list[i]
        p, s = destructure(X)
        d = ForwardDiff.gradient(x -> model(s(x), ps, st)[1], p)  # Compute ∇_x log|ψ| using ForwardDiff
        @assert norm(s(d) - gradx(model, X, ps, st)) < 1e-5
        p, s = destructure(ps)
        d = ForwardDiff.gradient(p -> model(X, s(p), st)[1], p)   # Compute ∇_θ log|ψ| using ForwardDiff
        @assert norm(d - gradp(model, X, ps, st)) < 1e-3
        p, s = destructure(X)
        d = ForwardDiff.hessian(x -> model(s(x), ps, st)[1], p)   # Compute full Hessian w.r.t. positions
        @assert norm(sum([d[i,i] for i=1:size(d, 1)]) - laplacian(model, X, ps, st)) < 1e-5  # Compare trace of Hessian with custom Laplacian
    end
    for i = 1:length(model_list) - 1
        model1, ps1, st1, spec1, spec1p1 = model_list[i], ps_list[i], st_list[i], spec_list[i], spec1p_list[i]
        model2, ps2, st2, spec2, spec1p2 = model_list[i+1], ps_list[i+1], st_list[i+1], spec_list[i+1], spec1p_list[i+1]
        ps2 = transfer_weights!(ps1, ps2, spec1, spec2, spec1p1, spec1p2)
        y1 = model1(X, ps1, st1)[1]
        y2 = model2(X, ps2, st2)[1]
        @assert norm(y1 - y2) < 1e-5
    end 
end

using Distributed, Printf, JLD2

function setup(mol, mol_name, method, TD, worldsize; ν = 2, basis_set = "cc-pvtz")
    atoms = [nuc.name for nuc in mol.nuclei]
    basis = Vector([load_basis_from_json("basis.json", atom, basis_set) for atom in atoms])  # Load basis from JSON
    A = []
    for (i, atom) in enumerate(atoms)
        spec = displayspec1p(basis[i].spec)
        push!(A, strip(last(split(spec, ','))))
    end

    if TD isa No_Decomposition
        totdeg = map(x -> fill(x, ν), A)
    else
        totdeg = ones(Int64, ν) * TD.P
    end

    model_list, ps_list, st_list, spec_list, spec1p_list, totdeg_list, ν_list =
        model_generator(mol, basis_set, totdeg, ν; TD = TD, ratio = 0.5);

    new_model_list = []
    new_ps_list = []
    new_st_list = []
    new_spec_list = []
    new_spec1p_list = []
    new_totdeg_list = []
    new_ν_list = []

    for i in eachindex(model_list)
        try
            test_wavefunction(
            [model_list[i]],
            [ps_list[i]],
            [st_list[i]],
            [spec_list[i]],
            [spec1p_list[i]],
            mol
        )
        push!(new_model_list, model_list[i])
        push!(new_ps_list, ps_list[i])
        push!(new_st_list, st_list[i])
        push!(new_spec_list, spec_list[i])
        push!(new_spec1p_list, spec1p_list[i])
        push!(new_totdeg_list, totdeg_list[i])
        push!(new_ν_list, ν_list[i])
        catch e
        @warn "test_wavefunction failed at layer $i, skipping it.\nError: $e"
        end
    end

    model_list   = [i for i in new_model_list[2:end]]
    ps_list      = [i for i in new_ps_list[2:end]]
    st_list      = [i for i in new_st_list[2:end]]
    spec_list    = [i for i in new_spec_list[2:end]]
    spec1p_list  = [i for i in new_spec1p_list[2:end]]
    totdeg_list  = [i for i in new_totdeg_list[2:end]]
    ν_list       = [i for i in new_ν_list[2:end]]

    ACEpsi.test_wavefunction(model_list, ps_list, st_list, spec_list, spec1p_list, mol)

    solver = (SPRINGSolver(), SketchSolver(800, 50, 50, 1.4), SVDSolver(800, 50, 50, 1.4))
    iterations = 500 * ones(Int, length(spec_list))
    iterations[end] = 40000
    
    checkpoints = []
    for i = 1:length(iterations)- 1
        push!(checkpoints, [[1, iterations[i]]])
    end
    interval = 3000
    A = []
    append!(A, collect(interval : interval : iterations[end]))
    if A[end] != iterations[end]
        push!(A, iterations[end])
    end
    checkpoint = []
    for i = 1:length(A)
        if i == 1
            a = 1
        else
            a = A[i-1]+1
        end
        push!(checkpoint, [a, A[i]])
    end
    push!(checkpoints, [i for i in checkpoint])
    checkpoints = [i for i in checkpoints]

    string = method == 1 ? "SPRING" :
             method == 2 ? "SKETCH" :
             method == 3 ? "WSSR"   : error("Invalid method")

    res_path = "$mol_name/$string/"
    optimizer = OPTSETTING(solver[method],
        iterations = iterations,
        burnin = 1000,
        lag = 10,
        nchains = 2^8,
        Δt = 0.08,
        acc_step = 10,
        acc_range = [0.45, 0.99],
        acc_opt = zeros(10), 
        clip = 5.0,
        lr = 0.015,
        lr_dc = 3000,
        m = 0.99,
        damping = 0.001,
        damping_decay = 100,
        damping_min = 0.001,
        norm_constrain = 0.001,
        η = 0.95,
        res_path = res_path, 
        checkpoints = checkpoints
    )
    Δt = optimizer.Δt
    acc = 0.0
    acc_opt = fill(0.0, optimizer.acc_step)
    x0, _theta, _acc = init_walkers(mol, model_list[1], ps_list[1], st_list[1],
                                    optimizer.burnin, optimizer.nchains * worldsize, Δt)
    for trial = 1:200
        x0, _theta, _acc = init_walkers(mol, model_list[1], ps_list[1], st_list[1],
                                    10, optimizer.nchains * worldsize, Δt)
        acc = mean(_acc)
        @printf("Try %2d: Δt = %.5f | acc = %.4f\n", trial, Δt, acc)
        push!(acc_opt, acc)
        deleteat!(acc_opt, 1)

        if acc < optimizer.acc_range[1]
            Δt *= exp(1/10 * (acc - optimizer.acc_range[1]) / optimizer.acc_range[1])
            @printf("acc too low, decreasing Δt → %.5f\n", Δt)
        elseif acc > optimizer.acc_range[2]
            Δt *= exp(1/10 * (acc - optimizer.acc_range[2]) / optimizer.acc_range[2])
            @printf("acc too high, decreasing Δt → %.5f\n", Δt)
        else
            break
        end
    end

    @printf("Initialize MCMC: Δt = %.5f, accRate = %.5f \n", Δt, acc)
    optimizer.Δt = Δt
    mkpath(optimizer.res_path)
    clean_TD = replace("$(TD)", r"[^A-Za-z0-9]" => "")
    res_path = "$mol_name/"
    save_path = joinpath(res_path, "$(mol_name)_$clean_TD.jld2")
    x0 = split_x0(x0, optimizer.nchains, worldsize)
    @save save_path x0 optimizer model_list ps_list st_list spec_list spec1p_list totdeg_list ν_list

    return nothing
end

function split_x0(x0::Vector, nchains::Int, worldsize)
    nworkers = worldsize
    @assert length(x0) == nchains * nworkers "x0 length must be equal to nchains * worldsize"
    return [x0[(i-1)*nchains+1 : i*nchains] for i in 1:nworkers]
end

function load_setup(mol_name::String, TD)
    clean_TD = replace("$(TD)", r"[^A-Za-z0-9]" => "")
    res_path = "$mol_name/"
    file_path = joinpath(res_path, "$(mol_name)_$clean_TD.jld2")

    @info "Loading setup from: $file_path"

    @load file_path x0 optimizer model_list ps_list st_list spec_list spec1p_list totdeg_list ν_list

    return x0, optimizer, model_list, ps_list, st_list, spec_list, spec1p_list, totdeg_list, ν_list
end

function get_resume_index(checkpoints_by_layer, res_path::String)
    latest_file = nothing
    latest_layer = 0
    latest_stop = 0

    for (i, layer_ckpts) in enumerate(checkpoints_by_layer)
        for (j, (start, stop)) in enumerate(layer_ckpts)
            l = [start, stop]
            filename = joinpath(res_path, "$(i)_$(l).jld2")
            if isfile(filename)
                @info "Found checkpoint: $filename"
                if i > latest_layer || (i == latest_layer && stop > latest_stop)
                    latest_file = filename
                    latest_layer = i
                    latest_stop = stop
                end
            else
                @info "Resuming from layer $i, checkpoint $j → missing file: $filename"
                return i, j, latest_file
            end
        end
    end

    @info "All checkpoints found. Nothing to resume."
    return length(checkpoints_by_layer) + 1, 1, latest_file
end

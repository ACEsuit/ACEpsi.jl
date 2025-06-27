using JSON
using StaticArrays
using ForwardDiff
using Optimisers: destructure
using LinearAlgebra

export clipE, test_wavefunction
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
        @assert norm(s(d) - gradx(model, X, ps, st)) < 1e-8
        p, s = destructure(ps)
        d = ForwardDiff.gradient(p -> model(X, s(p), st)[1], p)   # Compute ∇_θ log|ψ| using ForwardDiff
        @assert norm(d - ACEpsi.gradp(model, X, ps, st)) < 1e-3
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

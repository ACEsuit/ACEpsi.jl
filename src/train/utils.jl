using JSON
export clipE
function acc_adjust(k::Int, Δt::Number, acc_opt::AbstractVector, acc_range::AbstractVector, acc_step::Int)
    if mod(k, acc_step) == 0
        if mean(acc_opt) < acc_range[1]
            Δt *= exp(1/10 * (mean(acc_opt) - acc_range[1])/acc_range[1])
        elseif mean(acc_opt) > acc_range[2]
            Δt *= exp(1/10 * (mean(acc_opt) - acc_range[2])/acc_range[2])
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


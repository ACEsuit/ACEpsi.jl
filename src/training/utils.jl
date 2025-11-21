using JSON

function acc_adjust(Δt::T, k::Integer, acc_hist::AbstractVector;
                    acc_step::Integer = 50,
                    acc_range::Tuple{<:Real,<:Real} = (0.25, 0.45),
                    η::Real = 0.10,
                    window::Symbol = :tail,
                    Δt_min::Real = T(1e-4),
                    Δt_max::Real = T(1.0),
                   ) where {T<:Real}
    n = length(acc_hist)
    n == 0 && return Δt
    if window === :tail
        i0 = max(1, n - acc_step + 1)
        m = mean(@view acc_hist[i0:n])
    elseif window === :all
        m = mean(acc_hist)
    else
        error("window must be :tail or :all")
    end
    lo, hi = acc_range
    newΔt = Δt
    if m < lo
        newΔt = Δt * exp(η * (m - lo) / lo)
    elseif m > hi
        newΔt = Δt * exp(η * (m - hi) / hi)
    end
    return T(clamp(newΔt, T(Δt_min), T(Δt_max)))
end

function clipE(x::AbstractArray{<:Real}, ā::Real)
    T = promote_type(eltype(x), typeof(ā))
    aa = T(ā)
    y = similar(x, T)
    @. y = ifelse(abs(x) <= aa,
                  T(x),
                  aa * sign(x) * (one(T) + log((one(T) + (abs(x)/aa)^2) / T(2))))
    return y
end

function InverseLR(k, lr, lr_dc)
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

using JLD2, JSON
function init_hf(mol_name, ps, basis_set)
    data = JSON.parsefile("hf.json")
    C = reduce(hcat, data[basis_set][mol_name]["C_occ"])' |> x->permutedims(x)
    ps1 = deepcopy(ps)
    ps1.branch.bf.dense.weight .= C
    return ps1
end



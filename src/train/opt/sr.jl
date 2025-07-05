export opts!

function opts!(i, OptParams::OPTPARAMS, optimizer::DirectSolver, Eloc, o::Matrix{T}, nchains, damping::Float64, dim_ps::Int64, η::Float64, norm_constrain, γ, m) where {T}
    s = o * o' # O * O'
    @inbounds @simd for i = 1:dim_ps
        s[i, i] += damping
    end
    if OptParams.s_prev[1] != 0.0
        s .*= (1 - η)
        s .+= η * OptParams.s_prev
    end
    OptParams.s_prev .= s 
    fill!(OptParams.dw_tot, zero(T))
    OptParams.dw_tot = s \ OptParams.f
    ∇clip!(OptParams.dw_tot, OptParams.f, norm_constrain, γ)   
    lmul!(-γ, OptParams.dw_tot) 
    return OptParams.dw_tot, length(OptParams.f), norm(OptParams.f)
end

function opts!(i, OptParams::OPTPARAMSPRING, optimizer::SPRINGSolver, Eloc, o::Matrix{T}, nchains, damping::Float64, dim_ps::Int64, η::Float64, norm_constrain, γ, m) where {T}
    res = norm(OptParams.f)
    lmul!(-γ, Eloc) # -delta tau * (E - E_mean)
    s = o' * o
    
    s .+= 1/(nchains)
    Tvals, Tvecs = eigen(Symmetric(s))
    Tvals = max.(Tvals, 0.0) .+ damping

    mul!(OptParams.dow, transpose(o), OptParams.dw_tot) 
    epsilon_tilde = Eloc .- η * OptParams.dow

    mul!(OptParams.dow, Tvecs', epsilon_tilde)
    ldiv!(Diagonal(Tvals), OptParams.dow)
    OptParams.dow = Tvecs * OptParams.dow
    OptParams.dow .-= mean(OptParams.dow)

    mul!(OptParams.f, o, OptParams.dow)
    OptParams.dw_tot .= η * OptParams.dw_tot .+ OptParams.f / sqrt(nchains) 
    OptParams.dw_tot .*= min(1, sqrt(norm_constrain)/norm(OptParams.dw_tot))
    return OptParams.dw_tot, length(OptParams.f), res
end

function opts!(i, OptParams::OPTPARAMMINSR, optimizer::MINSRSolver, Eloc, o::Matrix{T}, nchains, damping::Float64, dim_ps::Int64, η::Float64, norm_constrain, γ, m) where {T}
    res = norm(OptParams.f)
    lmul!(-γ, Eloc) # -delta tau * (E - E_mean)
    s = o' * o
    
    Tvals, Tvecs = eigen(Symmetric(s))
    Tvals = max.(Tvals, 0.0) .+ damping

    mul!(OptParams.dow, Tvecs', Eloc)
    ldiv!(Diagonal(Tvals), OptParams.dow)
    OptParams.dow = Tvecs * OptParams.dow

    mul!(OptParams.f, o, OptParams.dow)
    OptParams.dw_tot .= η * OptParams.dw_tot .+ (1-η) * OptParams.f / sqrt(nchains) 
    OptParams.dw_tot .*= min(1, sqrt(norm_constrain)/norm(OptParams.dw_tot))
    return OptParams.dw_tot, length(OptParams.f), res
end

using LowRankApprox

function opts!(i, OptParams::OPTPARAMSKETCH, optimizer::SketchSolver, Eloc, o::Matrix{T}, nchains, damping::Float64, dim_ps::Int64, η::Float64, norm_constrain, γ, m) where {T}
    sr_rank0 = optimizer.sr_rank0::Int64
    sr_rank = optimizer.sr_rank::Int64
    sketch_opts = LRAOptions(sketch = :srft, rank = min(dim_ps, sr_rank) )
    res = norm(OptParams.f)
    lmul!(-γ, Eloc)

    if OptParams.sr_o[1] == 0.0
        oa = o::Matrix{T}
        ea = Eloc::Vector{T}
    else
        lmul!(sqrt(η), OptParams.sr_o)
        lmul!(sqrt(1 - η), o)
        lmul!(sqrt(η), OptParams.ek)
        lmul!(sqrt(1 - η), Eloc)
        oa = hcat(OptParams.sr_o[:, 1:sr_rank0], o)::Matrix{T}
        ea = vcat(OptParams.ek[1:sr_rank0], Eloc)::Vector{T}
    end
    svdo = psvdfact(oa, sketch_opts) # :none, :randn, :sub, :srft, :sprn

    ind = searchsortedlast(svdo[:S] / svdo[:S][1], damping, rev = true)
    sigma0 = 1 / (damping * abs(svdo[:S][1]))^2

    u = svdo[:U][:, 1:ind]::Matrix{T}
    s = svdo[:S][1:ind]::Vector{T}
    v = svdo[:V][:, 1:ind]::Matrix{T}

    mul!(OptParams.f, oa, ea) # np * 1, g
    uf = u' * OptParams.f::Vector{T} # u' * g
    uf .*= ((1 ./ s).^2 .- Ref(sigma0)) # (1/s^2 - 1/s0^2) * u' * g

    mul!(OptParams.dw_tot, u, uf) # u * (1/s^2 - 1/s0^2) * u' * g = u * 1/s^2 * u' * g - u * 1/s0^2 * u' * g

    OptParams.dw_tot .+= sigma0 * OptParams.f # 1/s0^2 * g

    fill!(OptParams.sr_o, zero(eltype(oa))) 
    @views mul!(OptParams.sr_o[:, 1:ind], u, Diagonal(s))
    fill!(OptParams.ek, zero(eltype(oa))) 
    @views mul!(OptParams.ek[1:ind], v', ea) 
    
    optimizer.sr_rank0 = ind
    
    if ind == sr_rank
        optimizer.sr_rank = min(Int(ceil(sr_rank * optimizer.sr_scale)), optimizer.sr_rank_max)
    end
    res_sf = norm(OptParams.dw_tot)
    ∇clip!(OptParams.dw_tot, norm_constrain, res_sf)  
    return OptParams.dw_tot, ind, res
end

function opts!(i, OptParams::OPTPARAMSVD, optimizer::SVDSolver, Eloc, o::Matrix{T}, nchains, damping::Float64, dim_ps::Int64, η::Float64, norm_constrain, γ, m) where {T}
    sr_rank0 = optimizer.sr_rank0::Int64
    sr_rank = optimizer.sr_rank::Int64
    res = norm(OptParams.f)
    lmul!(-γ, Eloc)
    maxit = 5
    if OptParams.sr_o[1] == 0.0
        r = min(min(dim_ps, nchains), sr_rank) 
        oa = o::Matrix{T}
        svdo = lmsvd(oa, r; maxit = 300)
        ea = Eloc::Vector{T}
    else
        lmul!(sqrt(η), OptParams.sr_o)
        lmul!(sqrt(1 - η), o)
        lmul!(sqrt(η), OptParams.ek)
        lmul!(sqrt(1 - η), Eloc)
        oa = hcat(OptParams.sr_o[:, 1:sr_rank0], o)
        ea = vcat(OptParams.ek[1:sr_rank0], Eloc)
        r = min(min(dim_ps, nchains), sr_rank) 
        if sr_rank0 < r
            maxit = 20
        end
        if i == 1
            svdo = lmsvd(oa, r; maxit = 300, X = OptParams.u[:, 1:r])
        else
            svdo = ssisvd(oa, r; maxit = maxit, X = OptParams.u[:, 1:r])
        end
    end
    ind = searchsortedlast(svdo.S / svdo.S[1], damping, rev = true)
    sigma0 = 1 / (damping * abs(svdo.S[1]))^2
    OptParams.u[:, 1:length(svdo.S)] .= svdo.U

    u = svdo.U[:, 1:ind]
    s = svdo.S[1:ind]
    v = svdo.V[:, 1:ind]

    fill!(OptParams.sr_o, zero(eltype(oa))) 
    @views mul!(OptParams.sr_o[:, 1:ind], u, Diagonal(s))
    fill!(OptParams.ek, zero(eltype(oa))) 
    @views mul!(OptParams.ek[1:ind], v', ea) 

    optimizer.sr_rank0 = ind
    if ind == sr_rank
        optimizer.sr_rank = min(Int(ceil(sr_rank * optimizer.sr_scale)), optimizer.sr_rank_max)
    end

    mul!(OptParams.f, oa, ea) # np * 1, g
    uf = u' * OptParams.f # u' * g
    uf .*= ((1 ./ s).^2 .- Ref(sigma0)) # (1/s^2 - 1/s0^2) * u' * g

    mul!(OptParams.dw_tot, u, uf) # u * (1/s^2 - 1/s0^2) * u' * g = u * 1/s^2 * u' * g - u * 1/s0^2 * u' * g
    axpy!(sigma0, OptParams.f, OptParams.dw_tot) # 1/s0^2 * g

    res_sf = norm(OptParams.dw_tot)
    ∇clip!(OptParams.dw_tot, norm_constrain, res_sf)  
    return OptParams.dw_tot, ind, res_sf
end
















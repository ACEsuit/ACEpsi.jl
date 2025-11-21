function opts!(i,
               OptParams,
               optimizer::DirectSolver,
               force, 
               Eloc::AbstractVector{T},
               o::AbstractMatrix{T},
               nbatch::Integer,
               damping::T,
               dim_ps::Integer,
               η::T,
               norm_constrain,
               γ,
               m) where {T}

    res = norm(force)
    # Use `invs` as a scratch buffer to hold the SR matrix S
    S = OptParams.invs
    fill!(S, zero(T))

    # S = o * o'  (no allocations: S ← 1*o*o' + 0*S)
    mul!(S, o, transpose(o), one(T), zero(T))

    # Diagonal damping to ensure SPD
    d = diagind(S)
    @inbounds @simd for k in eachindex(d)
        S[d[k]] += T(damping)
    end

    # Exponential moving average to stabilize the estimate
    if !OptParams.first
        @inbounds @simd for j in eachindex(S)
            S[j] = (one(T) - T(η)) * S[j] + T(η) * OptParams.s_prev[j]
        end
    end
    # Prefer in-place Cholesky; if not PD, jitter the diagonal and retry.
    F = cholesky!(Symmetric(S, :L); check=false)
    if !isposdef(F)
        jit = max(T(1e-12), T(damping) * T(1e-2))
        @inbounds @simd for k in eachindex(d)
            S[d[k]] += jit
        end
        F = cholesky!(Symmetric(S, :L); check=false)
    end

    ldiv!(F, force)
    lmul!(-T(γ), force)

    nrm = T(sqrt(norm_constrain))/norm(force)
    force .*= min(1, nrm)
    copy!(OptParams.s_prev, S)
    OptParams = (; OptParams..., first = false)
    return OptParams, force, length(force), res
end

function opts!(i,
               OptParams,
               optimizer::DirectSolver,
               force::CuArray{T,1},
               Eloc::CuArray{T,1},
               o::CuArray{T,2},
               nbatch::Integer,
               damping::T,
               dim_ps::Integer,
               η::T,
               norm_constrain,
               γ,
               m, sr_st) where {T<:AbstractFloat}
    res = norm(force)
    need_new = !(eltype(OptParams.invs) == T)
    S = need_new ? similar(OptParams.invs, T) : OptParams.invs
    fill!(S, zero(T))
    OptParams = need_new ? merge(OptParams, (; invs = S)) : OptParams
    OptParams = need_new ? (; OptParams..., first = false) : OptParams
    
    # S ← 0
    
    mul!(S, o, transpose(o), one(T), zero(T))

    @views view(S, diagind(S)) .+= T(damping)

    if !OptParams.first
        @. S = (one(T) - T(η)) * S + T(η) * OptParams.s_prev
    end

    F = lu!(S, RowMaximum(); check=false, allowsingular=false)

    ldiv!(F, force)
    @. force *= -T(γ)

    nrm = T(sqrt(norm_constrain))/norm(force)
    force .*= min(1, nrm)
    copy!(OptParams.s_prev, S)
    return OptParams, force, length(force), res, sr_st
end


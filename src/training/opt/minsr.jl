
"""
- Build `S := o' * o` (nbatch×nbatch) and add diagonal damping.
- Eigendecompose `S` and solve for `z` in sample space:
    z = Q * Diag(1 ./ clamp(λ, 0, ∞) .+ damping) * Q' * (-γ * Eloc)
- Map back to parameter space: `f = o * z`.
- EMA on `dw_tot`: `dw_tot ← η * dw_tot + (1 - η) * f / √nbatch`.
- ℓ2 clip on `dw_tot`.

All updates are in-place; fields are not rebound.
"""
function opts!(i,
               OptParams,
               ::MINSRSolver,
               force, 
               Eloc,
               o::AbstractMatrix{T},
               nbatch,
               damping::Real,
               dim_ps::Integer,
               η::Real,
               norm_constrain,
               γ::Real,
               m) where {T<:Real}

    @assert size(o,1) == dim_ps
    @assert size(o,2) == nbatch

    # Reported residual keeps prior semantics
    res = norm(force)

    # S = o' * o  (nbatch×nbatch)
    S = OptParams.s
    mul!(S, transpose(o), o)  # S ← o' * o

    # Eigendecomposition
    F = eigen!(Symmetric(S))
    λ = F.values
    Q = F.vectors

    # Regularize eigenvalues: max(λ, 0) + damping
    @inbounds @simd for k in eachindex(λ)
        λ[k] = max(λ[k], zero(T)) + T(damping)
    end

    # Workspace z := OptParams.dow (length nbatch)
    z = OptParams.dow

    # z = -γ * Eloc
    @inbounds @simd for k in eachindex(z)
        z[k] = -T(γ) * Eloc[k]
    end

    # z ← Q * Diag(1./λ) * Q' * z
    mul!(z, transpose(Q), z)         # ẑ = Q' * z
    @inbounds @simd for k in eachindex(z)
        z[k] /= λ[k]
    end
    mul!(z, Q, z)                    # z = Q * ẑ

    # f = o * z
    f = OptParams.f
    mul!(f, o, z)

    # EMA update of dw_tot with normalization by sqrt(nbatch)
    scale = inv(sqrt(T(nbatch)))
    dw_tot = OptParams.dw_tot
    @inbounds @simd for k in eachindex(OptParams.dw_tot)
        dw_tot[k] = T(η) * OptParams.dw_tot[k] + (one(T) - T(η)) * scale * OptParams.f[k]
    end

    # ℓ2 clipping of dw_tot
    nrm = T(sqrt(norm_constrain))/norm(dw_tot)
    dw_tot .*= min(1, nrm)

    copy!(OptParams.dw_tot, dw_tot)
    return OptParams, dw_tot, length(OptParams.f), res
end


function opts!(i,
               OptParams,
               ::MINSRSolver,
               force, 
               Eloc,
               o::AbstractGPUArray,
               nbatch,
               damping::Real,
               dim_ps::Integer,
               η::Real,
               norm_constrain,
               γ::Real,
               m, sr_st)

    @assert size(o,1) == dim_ps
    @assert size(o,2) == nbatch

    # Residual reported
    res = norm(force)
    T = eltype(o)

    # S = o' * o  (nbatch×nbatch)
    need_new = !(eltype(OptParams.s) == T)
    S = need_new ? similar(o, nbatch, nbatch) : OptParams.s
    z = need_new ? similar(o, nbatch) : OptParams.dow
    f = need_new ? similar(o, dim_ps) : OptParams.f
    dw_tot = need_new ? T.(OptParams.dw_tot) : OptParams.dw_tot
    OptParams = need_new ? merge(OptParams, (; s = S, dow = z, f = f, dw_tot = dw_tot)) : OptParams

    need_new = sr_st == nothing
    sr_st = need_new ? (;z1 = similar(o, nbatch)) : sr_st
    z1 = sr_st.z1
    
    mul!(S, transpose(o), o)               # S ← o' * o

    # Eigendecomposition in sample space  
    w, info = CUSOLVER.syevd!('V', 'U', S)
    F = Eigen(w, S)

    λ = F.values
    Q = F.vectors
    # Ensure positivity (jitter + damping)
    λ .= clamp.(λ, zero(T), typemax(T)) .+ damping

    # z = o' * dw_tot
    mul!(z, transpose(o), dw_tot)   # z ← o' * dw_tot

    # z = -η * (o' * dw_tot)  then add -γ * Eloc
    Eloc *= -γ   
    # Solve in eigenbasis: z ← Q * Diag(1./λ) * Q' * z
    # ẑ = Q' * z
    mul!(z1, transpose(Q), Eloc)  # z1 = Q' * z
    @. z1 /= λ                 # z1 = Diag(1./λ) * (Q' * z)
    mul!(z, Q, z1)             # y  = Q * z1 = (S+λI)^(-1) z

    # f = O * y
    mul!(f, o, z)

    # Exponential moving update of dw_tot with normalization by sqrt(nbatch)
    @. dw_tot = η * dw_tot + (1 - η) * f

    # ℓ2 clipping
    nrm = sqrt(norm_constrain)/norm(dw_tot)
    @. dw_tot *= min(1, nrm)

    copy!(OptParams.dw_tot, dw_tot)
    return OptParams, dw_tot, length(OptParams.f), res, sr_st
end



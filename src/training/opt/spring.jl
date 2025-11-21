using LinearAlgebra

"""
- Build S := o' * o  (size nbatch×nbatch), add diagonal regularization (1/nbatch) and damping.
- Compute eigendecomposition of Symmetric(S).
- Form ε̃ = -γ * Eloc - η * (o' * dw_tot) in-place using `OptParams.dow` as workspace.
- Solve S * z = ε̃ via eigendecomp: z = Q * Diagonal(1./λ) * Q' * ε̃.
- Center z by removing its mean; then f = o * z.
- Exponential moving update of dw_tot and ℓ₂-norm clipping.
"""
function opts!(i,
               OptParams,
               ::SPRINGSolver,
               force, 
               Eloc,
               o::Matrix,
               nbatch,
               damping::Real,
               dim_ps::Integer,
               η::Real,
               norm_constrain,
               γ::Real,
               m)

    @assert size(o,1) == dim_ps
    @assert size(o,2) == nbatch

    # Residual reported
    res = norm(force)
    T = eltype(o)

    S = o' * o  # (nbatch×nbatch)
    S = OptParams.s
    mul!(S, transpose(o), o)               # S ← o' * o

    # Add diagonal regularization: 1/nbatch and damping
    invn = one(T) / T(nbatch)
    @. S += invn

    # Eigendecomposition in sample space
    F = eigen(S)
    λ = F.values
    Q = F.vectors
    # Ensure positivity (jitter + damping)
    @inbounds @simd for k in eachindex(λ)
        λ[k] = max(λ[k], zero(T)) + T(damping)
    end

    # Workspace z := OptParams.dow  (length nbatch)
    z = OptParams.dow
    # z = o' * dw_tot
    mul!(z, transpose(o), OptParams.dw_tot)   # z ← o' * dw_tot

    # z = -η * (o' * dw_tot)  then add -γ * Eloc
    @inbounds @simd for k in eachindex(z)
        z[k] = -T(η) * z[k]
    end
    @inbounds @simd for k in eachindex(z)
        z[k] += -T(γ) * Eloc[k]
    end # epsilon_tilde

    # Solve in eigenbasis: z ← Q * Diag(1./λ) * Q' * z
    # ẑ = Q' * z
    z = transpose(Q) * z # reuse z as ẑ
    # ẑ ./= λ
    @inbounds @simd for k in eachindex(z)
        z[k] /= λ[k]
    end
    # z = Q * ẑ
    z = Q * z

    # Center z
    μ = mean(z)
    @inbounds @simd for k in eachindex(z)
        z[k] -= μ
    end

    # f = o * z
    f = OptParams.f
    mul!(f, o, z)

    # Exponential moving update of dw_tot with normalization by sqrt(nbatch)
    scale = inv(sqrt(T(nbatch)))
    dw_tot = OptParams.dw_tot
    @inbounds @simd for k in eachindex(dw_tot)
        dw_tot[k] = T(η) * dw_tot[k] + scale * f[k]
    end

    # ℓ2 clipping
    nrm = T(sqrt(norm_constrain))/norm(dw_tot)
    dw_tot .*= min(1, nrm)
    copy!(OptParams.dw_tot, dw_tot)
    return OptParams, dw_tot, length(OptParams.f), res
end


function opts!(i,
               OptParams,
               ::SPRINGSolver,
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
    
    # Add diagonal regularization: 1/nbatch and damping
    invn = one(T) / T(nbatch)
    @. S += invn

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
    z = Eloc - η * z

    # Solve in eigenbasis: z ← Q * Diag(1./λ) * Q' * z
    # ẑ = Q' * z
    mul!(z1, transpose(Q), z)  # z1 = Q' * z
    @. z1 /= λ                 # z1 = Diag(1./λ) * (Q' * z)
    mul!(z, Q, z1)             # y  = Q * z1 = (S+λI)^(-1) z

    # f = O * y
    mul!(f, o, z)

    # Exponential moving update of dw_tot with normalization by sqrt(nbatch)
    @. dw_tot = η * dw_tot + f

    # ℓ2 clipping
    nrm = sqrt(norm_constrain)/norm(dw_tot)
    @. dw_tot *= min(1, nrm)

    copy!(OptParams.dw_tot, dw_tot)
    return OptParams, dw_tot, length(OptParams.f), res, sr_st
end



using Polynomials4ML
using LuxCore
using StaticArrays
import KernelAbstractions as KA
using ChainRulesCore: NoTangent, unthunk
using ChainRulesCore
# -------------------------------------------------------------------------
# Orbital families
# -------------------------------------------------------------------------
abstract type OrbitalFamily end
struct Gaussian <: OrbitalFamily end
struct Slater   <: OrbitalFamily end
struct Plain   <: OrbitalFamily end
export Gaussian, Slater, Plain

# -------------------------------------------------------------------------
# AtomicOrbitals layer
# -------------------------------------------------------------------------
struct AtomicOrbitals{F<:OrbitalFamily, LEN, TD, TY, SPEC_T} <: AbstractLuxLayer
    Dn::TD
    Ylm::TY
    spec::SVector{LEN, SPEC_T}
    specidx::Vector{NTuple{3,Int}}
end

AtomicOrbitals(::Type{F}, Dn, Ylm, spec, specidx) where {F<:OrbitalFamily} =
    AtomicOrbitals{F, length(spec), typeof(Dn), typeof(Ylm), eltype(spec)}(Dn, Ylm, spec, specidx)

# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------
LuxCore.initialparameters(::AbstractRNG, l::AtomicOrbitals{Gaussian}) =
    (; ζ = Matrix(l.Dn.ζ), D = Matrix(l.Dn.D))

LuxCore.initialparameters(::AbstractRNG, l::AtomicOrbitals{Slater}) = begin
    ζ = Matrix(l.Dn.ζ); D = Matrix(l.Dn.D)
    D_s, α_s = ACEpsi.fit_all_rows(D, ζ)
    (; ζ = α_s)
end

LuxCore.initialparameters(::AbstractRNG, l::AtomicOrbitals{Plain}) = (; ζ = [1.0])
# -------------------------------------------------------------------------
# States
# -------------------------------------------------------------------------
LuxCore.initialstates(::AbstractRNG, l::AtomicOrbitals{<:OrbitalFamily}) = (;
    Dspec     = Tuple(l.Dn.spec),
    spec      = Tuple(l.spec),
    specn1vec = collect(getfield.(l.Dn.spec, Ref(:n1))),
    specidx   = Tuple(((i,j) for (_,i,j) in l.specidx)),
    backend   = nothing,
    dY_buf    = nothing,
    X_buf     = nothing,
    Y_buf     = nothing,   # (nX, KY)
    P_buf     = nothing,   # (nX, Nζ)
    out_buf   = nothing,   # (N, B, I)
    Rhat      = nothing,   # Vector{SVector{3,T}} length nX
    r_buf     = nothing,   # Vector{T} length nX
    ΔP        = nothing,
    ΔY        = nothing,
    ΔR        = nothing,
    Δx        = nothing
)

# -------------------------------------------------------------------------
# Radial kernels (device)
# -------------------------------------------------------------------------
@kernel function dbasis_kernel_gaussian!(P, @Const(X), @Const(specn1), @Const(ζ), @Const(D), @Const(K))
    i, n = @index(Global, NTuple)
    @inbounds begin
        x  = X[i]
        fx = x[1]*x[1] + x[2]*x[2] + x[3]*x[3]  # ‖x‖²
        s  = zero(eltype(P))
        for m in 1:K
            s += D[n,m] * exp(-ζ[n,m] * fx)
        end
        P[i, n] = s
    end
end

@kernel function dbasis_kernel_slater!(P, @Const(X), @Const(specn1), @Const(ζ), @Const(K))
    i, n = @index(Global, NTuple)
    n1 = specn1[n]
    @inbounds begin
        x  = X[i]
        fx = sqrt(x[1]^2 + x[2]^2 + x[3]^2)  # ‖x‖
        s  = zero(eltype(P))
        for m in 1:K
            s += fx^(n1-1) * exp(-ζ[n,m] * fx)
        end
        P[i, n] = s
    end
end

@inline function chebyshev_T(t, k)
    # k ≥ 0
    k == 0 && return one(t)
    k == 1 && return t
    Tkm1 = one(t)   # T_0
    Tk   = t        # T_1
    @inbounds for _ = 2:k
        Tkp1 = 2t*Tk - Tkm1
        Tkm1 = Tk
        Tk   = Tkp1
    end
    return Tk
end


@kernel function dbasis_kernel_plain!(P, @Const(X), @Const(specn1), @Const(ζ))
    i, n = @index(Global, NTuple)
    n1 = specn1[n]
    a  = ζ[1]
    @inbounds begin
        x  = X[i]
        fx = sqrt(x[1]^2 + x[2]^2 + x[3]^2)  # ‖x‖

        T = typeof(fx)
        # t = (4/π)*atan(ρ) - 1  ∈ [-1,1)
        t  = (T(4)/T(pi)) * atan(a * fx) - T(1)
        P[i, n] = chebyshev_T(t, n1 - 1)
    end
end
# -------------------------------------------------------------------------
# Geometry helpers (device)
# -------------------------------------------------------------------------
@kernel function normalize_kernel!(Rhat, r, @Const(X))
    i = @index(Global)
    @inbounds begin
        x   = X[i]
        rr  = sqrt(x[1]*x[1] + x[2]*x[2] + x[3]*x[3])
        rr  = ifelse(rr == zero(rr), eps(eltype(rr)), rr)
        r[i] = rr
        Rhat[i] = x/rr
    end
end

@kernel function pack_svec3!(X_buf, R, N, B)
    i = @index(Global)
    b = (i - 1) ÷ N + 1
    n = i - (b - 1) * N
    X_buf[i] = SVector{3, eltype(R)}(
        R[1, n, b],
        R[2, n, b],
        R[3, n, b],
    )
end

@kernel function combine!(out, P3v, Y3v, specidx, N, B)
    i, j = @index(Global, NTuple)
    b = (i - 1) ÷ N + 1
    n = i - (b - 1)*N
    n_idx, y_idx = specidx[j]
    out[n,b,j] = P3v[n,b,n_idx] * Y3v[n,b,y_idx]
end

using KernelAbstractions: @atomic

@kernel function accum_grad!(ΔP, ΔY, Δout, P3v, Y3v, specidx, N, B)
    i, j = @index(Global, NTuple)
    k = (i - 1) ÷ (N*B) + 1
    rem = i - (k - 1)*(N*B)
    b = (rem - 1) ÷ N + 1
    n = rem - (b - 1)*N
    n_idx, y_idx = specidx[j]
    @atomic ΔP[n,b,n_idx] += Δout[n,b,j] * Y3v[n,b,y_idx]
    @atomic ΔY[n,b,y_idx] += Δout[n,b,j] * P3v[n,b,n_idx]
end

@kernel function _pb_Ylm!(∂x, @Const(ΔY), @Const(dYlm), @Const(Rhat), @Const(r))
    a = @index(Global)
    nfeat = size(ΔY, 2)
    t = @inbounds ΔY[a,1] * dYlm[a,1]
    @inbounds for k in 2:nfeat
        t += ΔY[a,k] * dYlm[a,k]
    end
    r̂ = @inbounds Rhat[a]
    proj = r̂ * (r̂ ⋅ t)
    @inbounds ∂x[a] = (t - proj) / r[a]
end

# -------------------------------------------------------------------------
# Radial dispatch (host): forward/backward/branch
# -------------------------------------------------------------------------
@inline function radial_forward!(::Type{Gaussian}, backend, groupsize,
                                 P_buf, X_buf, specn1, ps, Kexp, nX, Nζ)
    dbasis_kernel_gaussian!(backend, groupsize)(
        P_buf, X_buf, specn1, ps.ζ, ps.D, Kexp; ndrange = (nX, Nζ)
    )
end

@inline function radial_forward!(::Type{Slater}, backend, groupsize,
                                 P_buf, X_buf, specn1, ps, Kexp, nX, Nζ)
    dbasis_kernel_slater!(backend, groupsize)(
        P_buf, X_buf, specn1, ps.ζ, Kexp; ndrange = (nX, Nζ)
    )
end

@inline function radial_forward!(::Type{Plain}, backend, groupsize,
                                 P_buf, X_buf, specn1, ζ, nX, Nζ)
    dbasis_kernel_plain!(backend, groupsize)(
        P_buf, X_buf, specn1, ζ; ndrange = (nX, Nζ)
    )
end

function radial_backprop!(::Type{Gaussian}, ΔP, R, ps, st, r_buf, Rhat, ΔR)
    N, B = size(R,2), size(R,3)
    Nζ, Kexp = size(ps.ζ)

    ζ4  = reshape(ps.ζ, Nζ, Kexp, 1, 1)
    D4  = reshape(ps.D, Nζ, Kexp, 1, 1)
    fx  = @. R[1, :, :]^2 + R[2, :, :]^2 + R[3, :, :]^2
    fx4 = reshape(fx, 1, 1, N, B)
    E   = @. exp(-ζ4 * fx4)

    ΔP_Nζ_N_B = permutedims(ΔP, (3,1,2))
    S = dropdims(sum((ζ4 .* D4) .* E; dims=2), dims=2)
    Δfx = -dropdims(sum(ΔP_Nζ_N_B .* S; dims=1), dims=1)

    @. ΔR[1, :, :] += 2 * R[1, :, :] * Δfx
    @. ΔR[2, :, :] += 2 * R[2, :, :] * Δfx
    @. ΔR[3, :, :] += 2 * R[3, :, :] * Δfx

    ΔP4   = reshape(ΔP_Nζ_N_B, Nζ, 1, N, B)
    ΔD_ps = dropdims(sum(ΔP4 .* E; dims=(3,4)), dims=(3,4))
    Δζ_ps = dropdims(sum(ΔP4 .* (-fx4) .* D4 .* E; dims=(3,4)), dims=(3,4))

    return (; ζ = Δζ_ps, D = ΔD_ps)
end

function radial_backprop!(::Type{Slater}, ΔP, R, ps, st, r_buf, Rhat, ΔR)
    N, B = size(R,2), size(R,3)
    Nζ, Kexp = size(ps.ζ)

    ζ4  = reshape(ps.ζ, Nζ, Kexp, 1, 1)
    r   = reshape(r_buf, N, B)
    r4  = reshape(r, 1, 1, N, B)
    E   = @. exp(-ζ4 * r4)

    n1_vec = st.specn1vec
    n1_4   = reshape(n1_vec, Nζ, 1, 1)
    G     = @. r4^(n1_4 - 1)
    Gm1   = @. (n1_4 - 1) * r4^(n1_4 - 2)

    S0 = dropdims(sum(E; dims=2), dims=2)
    Sζ = dropdims(sum((ζ4) .* E; dims=2), dims=2)

    ΔP_Nζ_N_B = permutedims(ΔP, (3,1,2))
    S0_4 = reshape(S0, Nζ, 1, N, B)
    Sζ_4 = reshape(Sζ, Nζ, 1, N, B)
    dPdr = @. Gm1 * S0_4 - G * Sζ_4
    dPdr_3 = reshape(dPdr, Nζ, N, B)

    Δr = dropdims(sum(ΔP_Nζ_N_B .* dPdr_3; dims=1), dims=1)
    Rhat1 = reshape(getindex.(Rhat, 1), N, B)
    Rhat2 = reshape(getindex.(Rhat, 2), N, B)
    Rhat3 = reshape(getindex.(Rhat, 3), N, B)
    @. ΔR[1, :, :] += Rhat1 * Δr
    @. ΔR[2, :, :] += Rhat2 * Δr
    @. ΔR[3, :, :] += Rhat3 * Δr

    ΔP4   = reshape(ΔP_Nζ_N_B, Nζ, 1, N, B)
    Δζ_ps = dropdims(sum(ΔP4 .* (-r4) .* G .* E; dims=(3,4)), dims=(3,4))

    return (; ζ = Δζ_ps)
end
@inline function chebyshev_U_scalar(t, m::Int)
    m <= 0 && return one(t) * (m == 0)   # U_0 = 1
    Um1 = one(t)     # U_0
    U0  = 2*t        # U_1
    @inbounds for _ = 2:m
        U1 = 2*t*U0 - Um1
        Um1 = U0
        U0  = U1
    end
    return U0
end

@kernel function reduce_dPdr_cheb_atan_a!(
    Δρ, @Const(ΔP_Nζ_N_B), @Const(n1_vec),
    @Const(t), @Const(dt_dρ)
)
    nz, n, b = @index(Global, NTuple)
    k = @inbounds n1_vec[nz] - 1
    if k > 0
        tt   = @inbounds t[n, b]
        dtdr = @inbounds dt_dρ[n, b]
        dPdr = k * chebyshev_U_scalar(tt, k - 1) * dtdr
        contrib = @inbounds ΔP_Nζ_N_B[nz, n, b] * dPdr
        @atomic Δρ[n, b] += contrib
    end
end

@kernel function reduce_dPda_cheb_atan!(
    Δa_accum, @Const(ΔP_Nζ_N_B), @Const(n1_vec),
    @Const(t), @Const(dt_da)
)
    nz, n, b = @index(Global, NTuple)
    k = @inbounds n1_vec[nz] - 1
    if k > 0
        tt    = @inbounds t[n, b]
        dtda  = @inbounds dt_da[n, b]
        dPda  = k * chebyshev_U_scalar(tt, k - 1) * dtda
        contrib = @inbounds ΔP_Nζ_N_B[nz, n, b] * dPda
        @atomic Δa_accum[1] += contrib
    end
end

function radial_backprop!(::Type{Plain}, ΔP, R, ps, st, r_buf, Rhat, ΔR, backend, groupsize)
    # R :: (3, N, B)
    N, B = size(R,2), size(R,3)
    n1_vec = st.specn1vec
    Tρ = eltype(r_buf)

    ρ = reshape(r_buf, N, B)                 # (N,B)
    a = ps.ζ

    # t(ρ),  dt/dρ, dt/da
    aρ    = a .* ρ                           # (N,B)
    fourπ = Tρ(4)/Tρ(pi)
    t     = fourπ .* atan.(aρ) .- one(Tρ)    # (N,B)
    dt_dρ = fourπ .* (a ./ (one(Tρ) .+ aρ.^2))        # (N,B)
    dt_da = fourπ .* (ρ ./ (one(Tρ) .+ aρ.^2))        # (N,B)

    # ΔP: (N,B,Nζ) -> (Nζ,N,B)
    ΔP_Nζ_N_B = permutedims(ΔP, (3,1,2))

    Δρ = similar(ρ); fill!(Δρ, zero(Tρ))
    reduce_dPdr_cheb_atan_a!(backend, groupsize)(
        Δρ, ΔP_Nζ_N_B, n1_vec, t, dt_dρ;
        ndrange = (length(n1_vec), N, B)
    )
    KA.synchronize(backend)

    Rhat1 = reshape(getindex.(Rhat, 1), N, B)
    Rhat2 = reshape(getindex.(Rhat, 2), N, B)
    Rhat3 = reshape(getindex.(Rhat, 3), N, B)
    @. ΔR[1, :, :] += Rhat1 * Δρ
    @. ΔR[2, :, :] += Rhat2 * Δρ
    @. ΔR[3, :, :] += Rhat3 * Δρ

    Δa_accum = similar(ps.ζ); fill!(Δa_accum, zero(eltype(ps.ζ)))
    reduce_dPda_cheb_atan!(backend, groupsize)(
        Δa_accum, ΔP_Nζ_N_B, n1_vec, t, dt_da;
        ndrange = (length(n1_vec), N, B)
    )
    KA.synchronize(backend)

    return (; ζ = Δa_accum)
end




function branch_param_grads(::Type{Gaussian}, ΔP4, ps, x_embed, r_buf)
    Nζ, Kexp, _, B = size(ΔP4)
    N, B2 = size(x_embed, 2), size(x_embed, 3); @assert B == B2

    D4  = reshape(ps.D, Nζ, Kexp, 1, 1)
    ζ4  = reshape(ps.ζ, Nζ, Kexp, 1, 1)
    fx  = @. x_embed[1, :, :]^2 + x_embed[2, :, :]^2 + x_embed[3, :, :]^2
    fx4 = reshape(fx, 1, 1, N, B)
    E   = @. exp(-ζ4 * fx4)

    ΔD_per = ΔP4 .* E
    Δζ_per = ΔP4 .* (-fx4) .* D4 .* E

    ΔD_b = dropdims(sum(ΔD_per; dims=3), dims=3) # Nζ×Kexp×B
    Δζ_b = dropdims(sum(Δζ_per; dims=3), dims=3) # Nζ×Kexp×B

    GD = reshape(ΔD_b, :, B)
    GZ = reshape(Δζ_b, :, B)
    return vcat(GZ, GD)    # (2*Nζ*Kexp)×B
end

function branch_param_grads(::Type{Slater}, ΔP4, ps, x_embed, r_buf; n1_vec::AbstractVector)
    Nζ, Kexp, _, B = size(ΔP4)
    N, B2 = size(x_embed, 2), size(x_embed, 3); @assert B == B2

    ζ4  = reshape(ps.ζ, Nζ, Kexp, 1, 1)
    r   = reshape(r_buf, N, B)
    r4  = reshape(r, 1, 1, N, B)
    n1_4 = reshape(n1_vec, Nζ, 1, 1, 1)
    G    = @. r4^(n1_4 - 1)
    E    = @. exp(-ζ4 * r4)

    Δζ_per = ΔP4 .* (-r4) .* G .* E
    Δζ_b = dropdims(sum(Δζ_per; dims=3), dims=3) # Nζ×Kexp×B
    return reshape(Δζ_b, :, B)                   # (Nζ*Kexp)×B
end

@inline function _branch_param_grads(::Type{Gaussian}, ΔP4, ps_branch, x_embed, r_buf, st_branch)
    branch_param_grads(Gaussian, ΔP4, ps_branch, x_embed, r_buf)
end
@inline function _branch_param_grads(::Type{Slater}, ΔP4, ps_branch, x_embed, r_buf, st_branch)
    branch_param_grads(Slater, ΔP4, ps_branch, x_embed, r_buf; n1_vec = st_branch.specn1vec)
end

# -------------------------------------------------------------------------
# Public call entry
# -------------------------------------------------------------------------
(l::AtomicOrbitals{<:OrbitalFamily})(R, ps, st) = evaluate(l, R, ps, st)

function evaluate(l::AtomicOrbitals{F, LEN,TD,TY,SPEC_T},
                  R::AbstractArray{T,3}, ps, st) where {F<:OrbitalFamily, LEN,TD,TY,SPEC_T,T<:Number}
    @assert size(R,1) == 3
    N, B = size(R,2), size(R,3)
    nX   = N*B

    KY   = length(l.Ylm)
    Nζ   = size(ps.ζ, 1)
    Kexp = size(ps.ζ, 2)

    Tbuf = promote_type(eltype(R), eltype(ps.ζ))
    need_new = (st.Y_buf === nothing || eltype(st.Y_buf) !== Tbuf)
    backend   = st.backend === nothing ? KA.get_backend(R) : st.backend
    groupsize = KA.isgpu(backend) ? 128 : 128

    X_buf = st.X_buf   === nothing || eltype(st.X_buf)   !== SVector{3,Tbuf} ? similar(R, SVector{3,Tbuf}, nX)             : st.X_buf
    Y_buf = st.Y_buf   === nothing || eltype(st.Y_buf)   !== Tbuf            ? similar(R, Tbuf, (nX, KY))                   : st.Y_buf
    P_buf = st.P_buf   === nothing || eltype(st.P_buf)   !== Tbuf            ? similar(R, Tbuf, (nX, Nζ))                   : st.P_buf
    out   = st.out_buf === nothing || eltype(st.out_buf) !== Tbuf            ? similar(R, Tbuf, (N, B, length(st.specidx))) : st.out_buf
    r_buf = st.r_buf   === nothing || eltype(st.r_buf)   !== Tbuf            ? similar(R, Tbuf, (nX,))                      : st.r_buf
    Rhat  = st.Rhat    === nothing || eltype(st.Rhat)    !== SVector{3,Tbuf} ? similar(R, SVector{3,Tbuf}, nX)              : st.Rhat

    ΔY = st.ΔY === nothing || eltype(st.ΔY) !== Tbuf ? similar(R, Tbuf, (N, B, KY)) : st.ΔY
    ΔP = st.ΔP === nothing || eltype(st.ΔP) !== Tbuf ? similar(R, Tbuf, (N, B, Nζ)) : st.ΔP
    ΔR = st.ΔR === nothing || eltype(st.ΔR) !== Tbuf ? similar(R, Tbuf, (3, N, B))  : st.ΔR

    pack_svec3!(backend, groupsize)(X_buf, R, N, B; ndrange = nX)
    KA.synchronize(backend)

    normalize_kernel!(backend, groupsize)(Rhat, r_buf, X_buf; ndrange = (nX,))
    KA.synchronize(backend)

    radial_forward!(F, backend, groupsize, P_buf, X_buf, st.specn1vec, ps, Kexp, nX, Nζ)
    KA.synchronize(backend)
    P3v = reshape(P_buf, N, B, Nζ)

    Polynomials4ML.evaluate!(Y_buf, l.Ylm, Rhat)
    KA.synchronize(backend)
    Y3v = reshape(Y_buf, N, B, KY)

    combine!(backend, groupsize)(out, P3v, Y3v, st.specidx, N, B; ndrange = (nX, length(st.specidx)))
    KA.synchronize(backend)

    st′ = need_new ? merge(st, (; backend, X_buf, Y_buf, P_buf, out_buf = out, Rhat, r_buf, ΔP = ΔP, ΔY = ΔY, ΔR = ΔR)) : st
    return out, st′
end


function evaluate(l::AtomicOrbitals{Plain, LEN,TD,TY,SPEC_T},
                  R::AbstractArray{T,3}, ps, st) where {LEN,TD,TY,SPEC_T,T<:Number}
    @assert size(R,1) == 3
    N, B = size(R,2), size(R,3)
    nX   = N*B

    KY   = length(l.Ylm)
    Nζ   = length(st.specn1vec)

    Tbuf = eltype(R)
    need_new = (st.Y_buf === nothing || eltype(st.Y_buf) !== Tbuf)
    backend   = st.backend === nothing ? KA.get_backend(R) : st.backend
    groupsize = KA.isgpu(backend) ? 128 : 128

    X_buf = st.X_buf   === nothing || eltype(st.X_buf)   !== SVector{3,Tbuf} ? similar(R, SVector{3,Tbuf}, nX)             : st.X_buf
    Y_buf = st.Y_buf   === nothing || eltype(st.Y_buf)   !== Tbuf            ? similar(R, Tbuf, (nX, KY))                   : st.Y_buf
    P_buf = st.P_buf   === nothing || eltype(st.P_buf)   !== Tbuf            ? similar(R, Tbuf, (nX, Nζ))                   : st.P_buf
    out   = st.out_buf === nothing || eltype(st.out_buf) !== Tbuf            ? similar(R, Tbuf, (N, B, length(st.specidx))) : st.out_buf
    r_buf = st.r_buf   === nothing || eltype(st.r_buf)   !== Tbuf            ? similar(R, Tbuf, (nX,))                      : st.r_buf
    Rhat  = st.Rhat    === nothing || eltype(st.Rhat)    !== SVector{3,Tbuf} ? similar(R, SVector{3,Tbuf}, nX)              : st.Rhat

    ΔY = st.ΔY === nothing || eltype(st.ΔY) !== Tbuf ? similar(R, Tbuf, (N, B, KY)) : st.ΔY
    ΔP = st.ΔP === nothing || eltype(st.ΔP) !== Tbuf ? similar(R, Tbuf, (N, B, Nζ)) : st.ΔP
    ΔR = st.ΔR === nothing || eltype(st.ΔR) !== Tbuf ? similar(R, Tbuf, (3, N, B))  : st.ΔR

    pack_svec3!(backend, groupsize)(X_buf, R, N, B; ndrange = nX)
    KA.synchronize(backend)

    normalize_kernel!(backend, groupsize)(Rhat, r_buf, X_buf; ndrange = (nX,))
    KA.synchronize(backend)

    radial_forward!(Plain, backend, groupsize, P_buf, X_buf, st.specn1vec, ps.ζ, nX, Nζ)
    KA.synchronize(backend)
    P3v = reshape(P_buf, N, B, Nζ)

    Polynomials4ML.evaluate!(Y_buf, l.Ylm, Rhat)
    KA.synchronize(backend)
    Y3v = reshape(Y_buf, N, B, KY)

    combine!(backend, groupsize)(out, P3v, Y3v, st.specidx, N, B; ndrange = (nX, length(st.specidx)))
    KA.synchronize(backend)

    st′ = need_new ? merge(st, (; backend, X_buf, Y_buf, P_buf, out_buf = out, Rhat, r_buf, ΔP = ΔP, ΔY = ΔY, ΔR = ΔR)) : st
    return out, st′
end
# -------------------------------------------------------------------------
# Internal buffer allocator for rrule
# -------------------------------------------------------------------------
@inline function _alloc_buffers!(st, R, KY, Nζ, Tbuf)
    N, B = size(R,2), size(R,3)
    nX   = N*B
    X_buf = st.X_buf === nothing || eltype(st.X_buf) !== SVector{3,Tbuf} ? similar(R, SVector{3,Tbuf}, nX) : st.X_buf
    Y_buf = st.Y_buf === nothing || eltype(st.Y_buf) !== Tbuf            ? similar(R, Tbuf, (nX, KY))       : st.Y_buf
    P_buf = st.P_buf === nothing || eltype(st.P_buf) !== Tbuf            ? similar(R, Tbuf, (nX, Nζ))       : st.P_buf
    out   = st.out_buf === nothing || eltype(st.out_buf) !== Tbuf        ? similar(R, Tbuf, (N, B, length(st.specidx))) : st.out_buf
    r_buf = st.r_buf === nothing || eltype(st.r_buf) !== Tbuf            ? similar(R, Tbuf, (nX,))          : st.r_buf
    Rhat  = st.Rhat  === nothing || eltype(st.Rhat)  !== SVector{3,Tbuf} ? similar(R, SVector{3,Tbuf}, nX)  : st.Rhat
    dY_buf = st.dY_buf === nothing || eltype(st.dY_buf) !== SVector{3,Tbuf} ? similar(R, SVector{3,Tbuf}, nX, KY) : st.dY_buf
    ΔY    = st.ΔY === nothing || eltype(st.ΔY) !== Tbuf ? similar(R, Tbuf, (N, B, KY)) : st.ΔY
    ΔP    = st.ΔP === nothing || eltype(st.ΔP) !== Tbuf ? similar(R, Tbuf, (N, B, Nζ)) : st.ΔP
    ΔR    = st.ΔR === nothing || eltype(st.ΔR) !== Tbuf ? similar(R, Tbuf, (3, N, B))  : st.ΔR
    Δx    = st.Δx === nothing || eltype(st.Δx) !== SVector{3,Tbuf} ? similar(R, SVector{3,Tbuf}, nX) : st.Δx
    return X_buf, Y_buf, P_buf, out, r_buf, Rhat, dY_buf, ΔY, ΔP, ΔR, Δx
end

# -------------------------------------------------------------------------
# rrule for evaluate
# -------------------------------------------------------------------------
function ChainRulesCore.rrule(::typeof(evaluate),
               l::AtomicOrbitals{F},
               R::AbstractArray{T,3}, ps, st) where {F<:OrbitalFamily, T}
    @assert size(R,1) == 3
    N, B = size(R,2), size(R,3)
    nX   = N*B
    KY   = length(l.Ylm)
    Nζ, Kexp = size(ps.ζ)

    Tbuf = promote_type(eltype(R), eltype(ps.ζ))
    backend   = st.backend === nothing ? KA.get_backend(R) : st.backend
    groupsize = KA.isgpu(backend) ? 128 : 128

    X_buf, Y_buf, P_buf, out, r_buf, Rhat, dY_buf, ΔY, ΔP, ΔR, Δx = _alloc_buffers!(st, R, KY, Nζ, Tbuf)
    st′ = merge(st, (; backend, dY_buf, Y_buf, P_buf, out_buf = out, Rhat, r_buf, ΔP, ΔY, ΔR, Δx))

    pack_svec3!(backend, groupsize)(X_buf, R, N, B; ndrange = nX); KA.synchronize(backend)
    normalize_kernel!(backend, groupsize)(Rhat, r_buf, X_buf; ndrange = (nX,)); KA.synchronize(backend)

    radial_forward!(F, backend, groupsize, P_buf, X_buf, st.specn1vec, ps, Kexp, nX, Nζ); KA.synchronize(backend)
    P3v = reshape(P_buf, N, B, Nζ)

    Polynomials4ML.evaluate_ed!(Y_buf, dY_buf, l.Ylm, Rhat); KA.synchronize(backend)
    Y3v = reshape(Y_buf, N, B, KY)

    combine!(backend, groupsize)(out, P3v, Y3v, st.specidx, N, B; ndrange = (nX, length(st.specidx))); KA.synchronize(backend)

    fill!(ΔP, zero(eltype(ΔP))); fill!(ΔY, zero(eltype(ΔY))); fill!(ΔR, zero(eltype(ΔR))); fill!(Δx, zero(eltype(Δx)))

    function pullback(Δout̄)
        Δout = unthunk(Δout̄[1])
        accum_grad!(backend, groupsize)(ΔP, ΔY, Δout, P3v, Y3v, st.specidx, N, B; ndrange = (nX, length(st′.specidx)))
        KA.synchronize(backend)

        Δps_named = radial_backprop!(F, ΔP, R, ps, st′, r_buf, Rhat, ΔR)

        ΔYy = reshape(ΔY, nX, KY)
        _pb_Ylm!(backend, groupsize)(Δx, ΔYy, dY_buf, Rhat, r_buf; ndrange=(nX,)); KA.synchronize(backend)
        ΔR .+= reshape(reinterpret(T, Δx), 3, N, B)

        return NoTangent(), NoTangent(), ΔR, Δps_named, NoTangent()
    end

    return (out, st′), pullback
end


function ChainRulesCore.rrule(::typeof(evaluate),
               l::AtomicOrbitals{Plain},
               R::AbstractArray{T,3}, ps, st) where {T}
    @assert size(R,1) == 3
    N, B = size(R,2), size(R,3)
    nX   = N*B
    KY   = length(l.Ylm)
    Nζ   = length(st.specn1vec)

    Tbuf = eltype(R)
    backend   = st.backend === nothing ? KA.get_backend(R) : st.backend
    groupsize = KA.isgpu(backend) ? 128 : 128

    X_buf, Y_buf, P_buf, out, r_buf, Rhat, dY_buf, ΔY, ΔP, ΔR, Δx = _alloc_buffers!(st, R, KY, Nζ, Tbuf)
    st′ = merge(st, (; backend, dY_buf, Y_buf, P_buf, out_buf = out, Rhat, r_buf, ΔP, ΔY, ΔR, Δx))

    pack_svec3!(backend, groupsize)(X_buf, R, N, B; ndrange = nX); KA.synchronize(backend)
    normalize_kernel!(backend, groupsize)(Rhat, r_buf, X_buf; ndrange = (nX,)); KA.synchronize(backend)

    radial_forward!(Plain, backend, groupsize, P_buf, X_buf, st.specn1vec, ps.ζ, nX, Nζ); KA.synchronize(backend)
    P3v = reshape(P_buf, N, B, Nζ)

    Polynomials4ML.evaluate_ed!(Y_buf, dY_buf, l.Ylm, Rhat); KA.synchronize(backend)
    Y3v = reshape(Y_buf, N, B, KY)

    combine!(backend, groupsize)(out, P3v, Y3v, st.specidx, N, B; ndrange = (nX, length(st.specidx))); KA.synchronize(backend)

    fill!(ΔP, zero(eltype(ΔP))); fill!(ΔY, zero(eltype(ΔY))); fill!(ΔR, zero(eltype(ΔR))); fill!(Δx, zero(eltype(Δx)))

    function pullback(Δout̄)
        Δout = unthunk(Δout̄[1])
        accum_grad!(backend, groupsize)(ΔP, ΔY, Δout, P3v, Y3v, st.specidx, N, B; ndrange = (nX, length(st′.specidx)))
        KA.synchronize(backend)

        Δps_named = radial_backprop!(Plain, ΔP, R, ps, st′, r_buf, Rhat, ΔR, backend, groupsize)

        ΔYy = reshape(ΔY, nX, KY)
        _pb_Ylm!(backend, groupsize)(Δx, ΔYy, dY_buf, Rhat, r_buf; ndrange=(nX,)); KA.synchronize(backend)
        ΔR .+= reshape(reinterpret(T, Δx), 3, N, B)

        return NoTangent(), NoTangent(), ΔR, Δps_named, NoTangent()
    end

    return (out, st′), pullback
end

# -------------------------------------------------------------------------
# Branch mode
# -------------------------------------------------------------------------
function branch_blocks(l_branch::AtomicOrbitals{F},
                       dx_pooling,
                       x_embed::AbstractArray{T,3},
                       ps_branch, st_branch) where {F<:OrbitalFamily, T<:Real}
    N, B = size(x_embed,2), size(x_embed,3)
    nX   = N*B
    KY   = length(l_branch.Ylm)
    Nζ, Kexp = size(ps_branch.ζ)

    backend   = st_branch.backend === nothing ? KA.get_backend(x_embed) : st_branch.backend
    groupsize = KA.isgpu(backend) ? 128 : 128

    X_buf = st_branch.X_buf
    pack_svec3!(backend, groupsize)(X_buf, x_embed, N, B; ndrange = nX); KA.synchronize(backend)

    r_buf, Rhat = st_branch.r_buf, st_branch.Rhat
    normalize_kernel!(backend, groupsize)(Rhat, r_buf, X_buf; ndrange = (nX,)); KA.synchronize(backend)

    Y_buf, P_buf = st_branch.Y_buf, st_branch.P_buf
    dY_buf = st_branch.dY_buf === nothing ? similar(x_embed, SVector{3,T}, nX, KY) : st_branch.dY_buf

    radial_forward!(F, backend, groupsize, P_buf, X_buf, st_branch.specn1vec, ps_branch, Kexp, nX, Nζ)
    KA.synchronize(backend)
    P3v = reshape(P_buf, N, B, Nζ)

    Polynomials4ML.evaluate_ed!(Y_buf, dY_buf, l_branch.Ylm, Rhat)
    KA.synchronize(backend)
    Y3v = reshape(Y_buf, N, B, KY)

    ΔY, ΔP = st_branch.ΔY, st_branch.ΔP
    fill!(ΔP, zero(eltype(ΔP))); fill!(ΔY, zero(eltype(ΔY)))

    accum_grad!(backend, groupsize)(ΔP, ΔY, dx_pooling, P3v, Y3v, st_branch.specidx, N, B; ndrange = (nX, length(st_branch.specidx)))
    KA.synchronize(backend)

    ΔP_Nζ_N_B = permutedims(ΔP, (3,1,2))     # Nζ×N×B
    ΔP4 = reshape(ΔP_Nζ_N_B, Nζ, 1, N, B)

    Gps = _branch_param_grads(F, ΔP4, ps_branch, x_embed, r_buf, st_branch)
    return 2 * Gps
end

@kernel function reduce_dPda_cheb_atan_batch!(
    Δa_batch, @Const(ΔP_Nζ_N_B),
    @Const(n1_vec), @Const(t), @Const(dt_da)
)
    nz, n, b = @index(Global, NTuple)
    k = @inbounds n1_vec[nz] - 1
    if k > 0
        tt    = @inbounds t[n, b]
        dtda  = @inbounds dt_da[n, b]
        dPda  = k * chebyshev_U_scalar(tt, k - 1) * dtda
        contrib = @inbounds ΔP_Nζ_N_B[nz, n, b] * dPda
        @atomic Δa_batch[b] += contrib
    end
end


function branch_blocks(l_branch::AtomicOrbitals{Plain},
                       dx_pooling,
                       x_embed::AbstractArray{T,3},
                       ps_branch, st_branch) where {T<:Real}
    N, B = size(x_embed,2), size(x_embed,3)
    nX   = N*B
    KY = length(l_branch.Ylm)
    Nζ = length(st_branch.specn1vec)

    backend   = st_branch.backend === nothing ? KA.get_backend(x_embed) : st_branch.backend
    groupsize = KA.isgpu(backend) ? 128 : 128

    X_buf = st_branch.X_buf
    pack_svec3!(backend, groupsize)(X_buf, x_embed, N, B; ndrange = nX); KA.synchronize(backend)

    r_buf, Rhat = st_branch.r_buf, st_branch.Rhat
    normalize_kernel!(backend, groupsize)(Rhat, r_buf, X_buf; ndrange = (nX,)); KA.synchronize(backend)

    Y_buf, P_buf = st_branch.Y_buf, st_branch.P_buf
    dY_buf = st_branch.dY_buf === nothing ? similar(x_embed, SVector{3,T}, nX, KY) : st_branch.dY_buf

    radial_forward!(Plain, backend, groupsize, P_buf, X_buf, st_branch.specn1vec, ps_branch.ζ, nX, Nζ)
    KA.synchronize(backend)
    P3v = reshape(P_buf, N, B, Nζ)

    Polynomials4ML.evaluate_ed!(Y_buf, dY_buf, l_branch.Ylm, Rhat)
    KA.synchronize(backend)
    Y3v = reshape(Y_buf, N, B, KY)

    ΔY, ΔP = st_branch.ΔY, st_branch.ΔP
    fill!(ΔP, zero(eltype(ΔP))); fill!(ΔY, zero(eltype(ΔY)))

    accum_grad!(backend, groupsize)(ΔP, ΔY, dx_pooling, P3v, Y3v, st_branch.specidx, N, B; ndrange = (nX, length(st_branch.specidx)))
    KA.synchronize(backend)

    ΔP_Nζ_N_B = permutedims(st_branch.ΔP, (3,1,2))  # (Nζ,N,B)
    ρ = reshape(st_branch.r_buf, N, B)
    a = ps_branch.ζ

    aρ    = a .* ρ                           # (N,B)
    Tρ = eltype(r_buf)
    fourπ = Tρ(4)/Tρ(pi)
    t     = fourπ .* atan.(aρ) .- one(Tρ)    # (N,B)
    dt_dρ = fourπ .* (a ./ (one(Tρ) .+ aρ.^2))        # (N,B)
    dt_da = fourπ .* (ρ ./ (one(Tρ) .+ aρ.^2))        # (N,B)

    Δa_batch = similar(ρ, Tρ, B)          # length B
    fill!(Δa_batch, zero(Tρ))

    reduce_dPda_cheb_atan_batch!(backend, groupsize)(
        Δa_batch, ΔP_Nζ_N_B, st_branch.specn1vec, t, dt_da;
        ndrange = (length(st_branch.specn1vec), N, B)
    )
    KA.synchronize(backend)

    return 2 * reshape(Δa_batch, 1, B)
end

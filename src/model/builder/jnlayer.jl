using ChainRulesCore
using CUDA
using Lux
using KernelAbstractions
const KA = KernelAbstractions

struct JNEnvelope{Nx} <: AbstractLuxLayer
    nuc::Int              # M = number of centers
    Σ::NTuple{Nx, Char}   # length == Nel
    Nel::Int              # Nel
end

# ---------- parameters & states ----------
function Lux.initialparameters(rng::AbstractRNG, l::JNEnvelope)
    M, Nel = l.nuc, l.Nel
    # P[m,i,s], L[:,:,m,i,s]
    P = 10.0f0 .* randn(rng, Float32, M, Nel, 2)
    L = rand(Float32, 3, 3, M, Nel, 2)
    @inbounds for m in 1:M, j in 1:Nel, s in 1:2
        L[1,1,m,j,s] = 1.0f0; L[2,2,m,j,s] = 1.0f0; L[3,3,m,j,s] = 1.0f0
    end
    return (P = P, L = L)
end

function Lux.initialstates(::AbstractRNG, l::JNEnvelope)
    ΣA = map(spin2idx, collect(l.Σ))  # Vector{Int}
    return (; ΣA = ΣA, backend = nothing, out_buf = nothing, grad_buf = nothing, gradout_buf = nothing)
end

@inline function mul3x3_vec(L, x1, x2, x3, m, i, s)
    # L :: (3,3,M,Nel,2); use i instead of j
    y1 = L[1,1,m,i,s]*x1 + L[1,2,m,i,s]*x2 + L[1,3,m,i,s]*x3
    y2 = L[2,1,m,i,s]*x1 + L[2,2,m,i,s]*x2 + L[2,3,m,i,s]*x3
    y3 = L[3,1,m,i,s]*x1 + L[3,2,m,i,s]*x2 + L[3,3,m,i,s]*x3
    return y1, y2, y3
end

@kernel function envelope_kernel!(
    out,      # (Nel, Nel, B)
    X,        # (3, Nel, B, M)
    P,        # (M, Nel, 2)
    L,        # (3,3,M,Nel,2)
    ΣA::AbstractVector{Int},  # (Nel,)
    M::Int,
)
    (i, j, z) = @index(Global, NTuple)

    Nel, B = size(out,1), size(out,3)
    s_idx = ΣA[i]               # 1 or 2
    acc = zero(eltype(out))

    @inbounds for m in 1:M
        # d = X[:, j, z, m]
        x1 = X[1, j, z, m]
        x2 = X[2, j, z, m]
        x3 = X[3, j, z, m]
        # A = L[:,:,m,i,s_i]
        y1, y2, y3 = mul3x3_vec(L, x1, x2, x3, m, i, s_idx)
        ρ = sqrt(y1*y1 + y2*y2 + y3*y3)
        acc += P[m, i, s_idx] * exp(-ρ)
    end

    out[i, j, z] = acc
end

function (l::JNEnvelope)(x, ps, st)
    return evaluate(l, x, ps, st)
end

function evaluate(l::JNEnvelope, x, ps, st)
    Nel = l.Nel
    M   = l.nuc

    # (3,Nel,B,M)
    X4 = cat(x...; dims=4)

    T = promote_type(eltype(X4), eltype(ps.P), eltype(ps.L))
    _, Nelx, B, Mx = size(X4)
    @assert Nelx == Nel && Mx == M
    backend = (st.backend === nothing) ? KA.get_backend(X4) : st.backend

    need_new = (st.out_buf === nothing ||
                !(eltype(st.out_buf) === T && ndims(st.out_buf) == 3 &&
                  size(st.out_buf,1) == Nel && size(st.out_buf,2) == Nel && size(st.out_buf,3) == B))
    out_buf = need_new ? similar(X4, T, Nel, Nel, B) : st.out_buf
    fill!(out_buf, zero(T))

    ΣA = st.ΣA

    ker = envelope_kernel!(backend)
    ker(out_buf, X4, ps.P, ps.L, ΣA, M; ndrange = (Nel, Nel, B))
    KA.synchronize(backend)

    st2 = need_new ? (; st..., backend=backend, out_buf=out_buf) : st
    return out_buf, st2
end

# --------------- backward kernels (no atomics needed) ---------------

# ΔX: thread = (j,z,m), loop over i
@kernel function gradX_kernel!(
    ΔX4, dY, X4, P, L, ΣA::AbstractVector{Int}
)
    (j, z, m) = @index(Global, NTuple)
    Nel = size(X4,2); B = size(X4,3)
    # d = X4[:, j, z, m]
    d1 = X4[1,j,z,m]; d2 = X4[2,j,z,m]; d3 = X4[3,j,z,m]
    acc1 = zero(eltype(ΔX4)); acc2 = acc1; acc3 = acc1
    epsρ = sqrt(eps(eltype(acc1)))

    @inbounds for i in 1:Nel
        s = ΣA[i]
        # A = L[:,:,m,i,s]
        a11=L[1,1,m,i,s]; a12=L[1,2,m,i,s]; a13=L[1,3,m,i,s]
        a21=L[2,1,m,i,s]; a22=L[2,2,m,i,s]; a23=L[2,3,m,i,s]
        a31=L[3,1,m,i,s]; a32=L[3,2,m,i,s]; a33=L[3,3,m,i,s]

        y1 = a11*d1 + a12*d2 + a13*d3
        y2 = a21*d1 + a22*d2 + a23*d3
        y3 = a31*d1 + a32*d2 + a33*d3

        ρ  = sqrt(y1*y1 + y2*y2 + y3*y3)
        e  = exp(-ρ)
        w  = P[m,i,s]
        g  = dY[i,j,z]
        invρ = inv(ρ + epsρ)

        # A' * y
        t1 = a11*y1 + a21*y2 + a31*y3
        t2 = a12*y1 + a22*y2 + a32*y3
        t3 = a13*y1 + a23*y2 + a33*y3
        coeff = g * w * e * (-invρ)
        acc1 += coeff * t1
        acc2 += coeff * t2
        acc3 += coeff * t3
    end

    ΔX4[1,j,z,m] += acc1
    ΔX4[2,j,z,m] += acc2
    ΔX4[3,j,z,m] += acc3
end

# ΔP: thread = (m,i), loop over j,z
@kernel function gradP_kernel!(
    ΔP, dY, X4, P, L, ΣA::AbstractVector{Int}
)
    (m, i) = @index(Global, NTuple)
    Nel = size(X4,2)
    B   = size(X4,3)
    T   = eltype(ΔP)
    epsρ = sqrt(eps(T))
    s = ΣA[i]  # fixed for this (m,i)

    acc = zero(T)

    @inbounds for j in 1:Nel, z in 1:B
        # d = X4[:, j, z, m]
        d1 = X4[1,j,z,m]; d2 = X4[2,j,z,m]; d3 = X4[3,j,z,m]
        # A = L[:,:,m,i,s]
        a11=L[1,1,m,i,s]; a12=L[1,2,m,i,s]; a13=L[1,3,m,i,s]
        a21=L[2,1,m,i,s]; a22=L[2,2,m,i,s]; a23=L[2,3,m,i,s]
        a31=L[3,1,m,i,s]; a32=L[3,2,m,i,s]; a33=L[3,3,m,i,s]

        y1 = a11*d1 + a12*d2 + a13*d3
        y2 = a21*d1 + a22*d2 + a23*d3
        y3 = a31*d1 + a32*d2 + a33*d3

        ρ  = sqrt(y1*y1 + y2*y2 + y3*y3)
        e  = exp(-ρ)
        g  = dY[i,j,z]
        acc += g * e
    end
    ΔP[m,i,s] += acc
end

# ΔL: thread = (m,i), loop over j,z -> 3x3
@kernel function gradL_kernel!(
    ΔL, dY, X4, P, L, ΣA::AbstractVector{Int}
)
    (m, i) = @index(Global, NTuple)
    Nel = size(X4,3 - 1)  # == size(X4,2)
    B   = size(X4,3)
    T   = eltype(ΔL)

    epsρ = sqrt(eps(T))
    s = ΣA[i]
    w = P[m,i,s]

    s11=T(0); s12=T(0); s13=T(0);
    s21=T(0); s22=T(0); s23=T(0);
    s31=T(0); s32=T(0); s33=T(0);

    @inbounds for j in 1:Nel, z in 1:B
        # d = X4[:, j, z, m]
        d1 = X4[1,j,z,m]; d2 = X4[2,j,z,m]; d3 = X4[3,j,z,m]

        # A = L[:,:,m,i,s]
        a11=L[1,1,m,i,s]; a12=L[1,2,m,i,s]; a13=L[1,3,m,i,s]
        a21=L[2,1,m,i,s]; a22=L[2,2,m,i,s]; a23=L[2,3,m,i,s]
        a31=L[3,1,m,i,s]; a32=L[3,2,m,i,s]; a33=L[3,3,m,i,s]

        y1 = a11*d1 + a12*d2 + a13*d3
        y2 = a21*d1 + a22*d2 + a23*d3
        y3 = a31*d1 + a32*d2 + a33*d3

        ρ  = sqrt(y1*y1 + y2*y2 + y3*y3)
        e  = exp(-ρ)
        g  = dY[i,j,z]
        cL = -(g * w * e) / (ρ + epsρ)

        s11 += cL*y1*d1; s12 += cL*y1*d2; s13 += cL*y1*d3
        s21 += cL*y2*d1; s22 += cL*y2*d2; s23 += cL*y2*d3
        s31 += cL*y3*d1; s32 += cL*y3*d2; s33 += cL*y3*d3
    end

    @inbounds begin
        ΔL[1,1,m,i,s] += s11; ΔL[1,2,m,i,s] += s12; ΔL[1,3,m,i,s] += s13
        ΔL[2,1,m,i,s] += s21; ΔL[2,2,m,i,s] += s22; ΔL[2,3,m,i,s] += s23
        ΔL[3,1,m,i,s] += s31; ΔL[3,2,m,i,s] += s32; ΔL[3,3,m,i,s] += s33
    end
end

# --------------------------- rrule ---------------------------
function ChainRulesCore.rrule(::typeof(evaluate), l::JNEnvelope, x, ps, st)
    y, st2 = evaluate(l, x, ps, st)

    X4 = cat(x...; dims=4)
    Nel = l.Nel
    M   = l.nuc
    _, Nelx, B, Mx = size(X4)
    @assert Nelx == Nel && Mx == M
    ΣA = st2.ΣA
    backend = st2.backend
    need_new = st2.grad_buf === nothing
    grad_buf = need_new ? similar(X4) : st2.grad_buf
    gradout_buf = need_new ? map(similar, x)  : st2.gradout_buf
    st3 = need_new ? (; st2..., grad_buf=grad_buf, gradout_buf=gradout_buf) : st2
    
    function pullback(ȳ_st)
        dY, _ = ȳ_st
        dY = unthunk(dY)
        T = promote_type(eltype(X4), eltype(ps.P), eltype(ps.L), eltype(dY))
        fill!(grad_buf, zero(T))
        ΔP  = similar(ps.P, T); fill!(ΔP, zero(T))
        ΔL  = similar(ps.L, T); fill!(ΔL, zero(T))

        # gradX: ndrange (j, z, m)
        gradX_kernel!(backend)(grad_buf, dY, X4, ps.P, ps.L, ΣA; ndrange=(Nel, B, M))
        KA.synchronize(backend)
        # gradP: ndrange (m, i)
        gradP_kernel!(backend)(ΔP,  dY, X4, ps.P, ps.L, ΣA; ndrange=(M, Nel))
        KA.synchronize(backend)
        # gradL: ndrange (m, i)
        gradL_kernel!(backend)(ΔL, dY, X4, ps.P, ps.L, ΣA; ndrange=(M, Nel))
        KA.synchronize(backend)

        @inbounds for m in 1:M
            @views gradout_buf[m] .= grad_buf[:,:,:,m]
        end

        Δps = (P = ΔP, L = ΔL)
        return NoTangent(), NoTangent(), gradout_buf, Δps, NoTangent()
    end

    return (y, st3), pullback
end

@kernel function gradP_perbatch_kernel!(
    ΔPnb,       # (M, Nel, 2, B)  -- second dim is i
    dY,         # (Nel, Nel, B)
    X4,         # (3, Nel, B, M)
    L,          # (3,3,M,Nel,2)
    ΣA          # (Nel,)
)
    (m, i, z) = @index(Global, NTuple)

    Nel = size(X4, 2)
    T   = eltype(ΔPnb)
    epsρ = sqrt(eps(T))
    s = ΣA[i]

    acc = zero(T)

    @inbounds for j in 1:Nel
        # d = X4[:, j, z, m]
        d1 = X4[1,j,z,m]; d2 = X4[2,j,z,m]; d3 = X4[3,j,z,m]
        # A = L[:,:,m,i,s]
        a11=L[1,1,m,i,s]; a12=L[1,2,m,i,s]; a13=L[1,3,m,i,s]
        a21=L[2,1,m,i,s]; a22=L[2,2,m,i,s]; a23=L[2,3,m,i,s]
        a31=L[3,1,m,i,s]; a32=L[3,2,m,i,s]; a33=L[3,3,m,i,s]

        y1 = a11*d1 + a12*d2 + a13*d3
        y2 = a21*d1 + a22*d2 + a23*d3
        y3 = a31*d1 + a32*d2 + a33*d3

        ρ  = sqrt(y1*y1 + y2*y2 + y3*y3)
        e  = exp(-ρ)
        g  = dY[i,j,z]
        acc += g * e
    end
    ΔPnb[m,i,s,z] = acc
end

# per-batch ∂L: ΔLnb :: (3,3,M,Nel,2,B)
@kernel function gradL_perbatch_kernel!(
    ΔLnb,      # (3,3,M,Nel,2,B) -- second dim is i
    dY,        # (Nel, Nel, B)
    X4,        # (3, Nel, B, M)
    P,         # (M, Nel, 2)
    L,         # (3,3,M,Nel,2)
    ΣA         # (Nel,)
)
    (m, i, z) = @index(Global, NTuple)

    Nel = size(X4, 2)
    T   = eltype(ΔLnb)
    epsρ = sqrt(eps(T))
    s = ΣA[i]
    w = P[m,i,s]

    s11=T(0); s12=T(0); s13=T(0);
    s21=T(0); s22=T(0); s23=T(0);
    s31=T(0); s32=T(0); s33=T(0);

    @inbounds for j in 1:Nel
        # d = X4[:, j, z, m]
        d1 = X4[1,j,z,m]; d2 = X4[2,j,z,m]; d3 = X4[3,j,z,m]

        # A = L[:,:,m,i,s]
        a11=L[1,1,m,i,s]; a12=L[1,2,m,i,s]; a13=L[1,3,m,i,s]
        a21=L[2,1,m,i,s]; a22=L[2,2,m,i,s]; a23=L[2,3,m,i,s]
        a31=L[3,1,m,i,s]; a32=L[3,2,m,i,s]; a33=L[3,3,m,i,s]

        y1 = a11*d1 + a12*d2 + a13*d3
        y2 = a21*d1 + a22*d2 + a23*d3
        y3 = a31*d1 + a32*d2 + a33*d3

        ρ  = sqrt(y1*y1 + y2*y2 + y3*y3)
        e  = exp(-ρ)
        g  = dY[i,j,z]
        cL = -(g * w * e) / (ρ + epsρ)

        s11 += cL*y1*d1; s12 += cL*y1*d2; s13 += cL*y1*d3
        s21 += cL*y2*d1; s22 += cL*y2*d2; s23 += cL*y2*d3
        s31 += cL*y3*d1; s32 += cL*y3*d2; s33 += cL*y3*d3
    end

    @inbounds begin
        ΔLnb[1,1,m,i,s,z] += s11; ΔLnb[1,2,m,i,s,z] += s12; ΔLnb[1,3,m,i,s,z] += s13
        ΔLnb[2,1,m,i,s,z] += s21; ΔLnb[2,2,m,i,s,z] += s22; ΔLnb[2,3,m,i,s,z] += s23
        ΔLnb[3,1,m,i,s,z] += s31; ΔLnb[3,2,m,i,s,z] += s32; ΔLnb[3,3,m,i,s,z] += s33
    end
end

function gradPL_per_batch(x, P, L, ΣA, dY)
    # X4 : (3, Nel, B, M)
    X4 = cat(x...; dims=4)
    M   = size(X4,4)
    Nel = size(X4,2)
    B   = size(X4,3)

    backend = KA.get_backend(X4)
    T = promote_type(eltype(X4), eltype(L), eltype(P), eltype(dY))

    ΔPnb = similar(X4, T, M, Nel, 2, B)   # (M,Nel,2,B)
    fill!(ΔPnb, zero(T))

    gradP_perbatch_kernel!(backend)(ΔPnb, dY, X4, L, ΣA; ndrange=(M, Nel, B))
    KA.synchronize(backend)

    X4  = cat(x...; dims=4)         # (3,Nel,B,M)
    M   = size(X4,4)
    Nel = size(X4,2)
    B   = size(X4,3)

    ΔLnb = similar(X4, T, 3, 3, M, Nel, 2, B)  # (3,3,M,Nel,2,B)
    fill!(ΔLnb, zero(T))

    gradL_perbatch_kernel!(backend)(ΔLnb, dY, X4, P, L, ΣA; ndrange=(M, Nel, B))
    KA.synchronize(backend)

    ΔP_flat = 2 * reshape(ΔPnb, M*Nel*2, B)
    ΔL_flat = 2 * reshape(ΔLnb, 9*M*Nel*2, B)
    return vcat(ΔP_flat, ΔL_flat)
end

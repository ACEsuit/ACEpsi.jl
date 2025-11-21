using LinearAlgebra: dot
using Zygote
export SumH, Elocal, compute_Eloc_dp

struct SumH{Nnuc, Nx, T,TT} <:AbstractLuxLayer
    mol::Molecule{Nnuc, Nx, T,TT}
end

LuxCore.initialparameters(::AbstractRNG, l::SumH) = NamedTuple()

function LuxCore.initialstates(::AbstractRNG, l::SumH)
    M = length(l.mol.nuclei)
    rr = reduce(hcat, collect([l.mol.nuclei[i].rr for i = 1:M]))
    Z = Tuple(collect([l.mol.nuclei[i].charge for i = 1:M]))
    return (; M = M, rr = rr, Z = Z)
end

# Electron–electron repulsion per batch: out[z] = Σ_{i<j} 1/|r_i - r_j|
# R: (3, N, B)  coordinates; out: (B,)
@kernel function vee_kernel!(out, @Const(R), @Const(N))
    z = @index(Global)
    acc = zero(eltype(out))
    @inbounds for i = 1:N-1
        for j = i+1:N
            dx = R[1, i, z] - R[1, j, z]
            dy = R[2, i, z] - R[2, j, z]
            dz = R[3, i, z] - R[3, j, z]
            r2 = dx*dx + dy*dy + dz*dz
            acc += inv(sqrt(r2))
        end
    end
    out[z] = acc
end

function Vee(R::AbstractArray{T, 3}) where {T}
    N  = size(R, 2)
    B  = size(R, 3)
    out = similar(R, T, B)
    fill!(out, zero(T))

    backend = KA.get_backend(R)
    vee_kernel!(backend)(out, R, N; ndrange=B)
    KA.synchronize(backend)
    return out
end

# External potential per batch:
# V_ext(z) = - ∑_{i=1..N} ∑_{j=1..M} Z[j] / |r_i(z) - R_nuc[j]|
# R  :: (3, N, B)   electron coords
# rr :: (3, M)      nuclear coords
# Z  :: (M,)        nuclear charges
@kernel function vext_kernel!(out, @Const(R), @Const(rr), @Const(Z), @Const(N))
    z = @index(Global)
    T   = eltype(out)
    acc = zero(T)
    @inbounds for i = 1:N
        @inbounds for j = 1:size(rr, 2)
            dx = R[1, i, z] - rr[1, j]
            dy = R[2, i, z] - rr[2, j]
            dz = R[3, i, z] - rr[3, j]
            r2 = dx*dx + dy*dy + dz*dz
            acc -= Z[j] / sqrt(r2)
        end
    end
    out[z] = acc
end

function Vext(R::AbstractArray{T,3}, rr::AbstractArray{T2,2}, Z) where {T,T2}
    N  = size(R, 2)
    B  = size(R, 3)

    Tout = promote_type(T, T2)
    out  = similar(R, Tout, B)

    backend = KA.get_backend(R)
    vext_kernel!(backend)(out, R, rr, Z, N; ndrange=B)
    KA.synchronize(backend)
    return out
end

"""
Compute local energy using SumH Hamiltonian:
    E_loc = Vext + Vee - (1/4) ∇² log|ψ| - (1/8) |∇ log|ψ||²
Ref: https://arxiv.org/abs/2105.08351
"""
norm2_per_batch(G) = vec(sum(abs2, G; dims=(1,2)))

function Elocal(H::SumH, ps_H, st_H, model, ps, st_eval, st_lap, R::AbstractArray{T,3}) where {T}
    Vext_b = Vext(R, st_H.rr, st_H.Z)      # B
    Vee_b  = Vee(R)                        # B
    G = gradx_batch(model, R, ps, st_eval)
    g2_b = norm2_per_batch(G)
    lap_b, _ = laplacian(model, R, ps, st_lap)
    return Vext_b .+ Vee_b .- T(0.25).*lap_b .- T(0.125).*g2_b
end

function compute_Eloc_dp(H::SumH, ps_H, st_H, model, ps, st_eval, st_lap, X::AbstractArray{TX,3}) where {TX}
    Vext_b = Vext(X, st_H.rr, st_H.Z)      # B
    Vee_b  = Vee(X)                        # B
    T = eltype(Vext_b)

    G = gradx_batch(model, X, ps, st_eval)
    g2_b = norm2_per_batch(G)
    O = gradp_batch(model, X, ps, st_eval)
    lap_b, st_lap = laplacian(model, X, ps, st_lap)


    elocs = Vext_b .+ Vee_b .- T(0.25).*lap_b .- T(0.125).*g2_b
    return elocs, T(0.5) * O, st_eval, st_lap
end

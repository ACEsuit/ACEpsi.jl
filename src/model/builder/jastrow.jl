import KernelAbstractions as KA
using ChainRulesCore

struct JastrowLayer{Nx} <: AbstractLuxLayer 
    Σ::NTuple{Nx, Char}
end

LuxCore.initialparameters(::AbstractRNG, ::JastrowLayer) = NamedTuple()

function LuxCore.initialstates(::AbstractRNG, l::JastrowLayer)
    Σ = spin2idx.(l.Σ)
    Ne = length(Σ)
    Σv = collect(Σ)
    same = reshape(Σv, Ne, 1) .== reshape(Σv, 1, Ne)
    cij_T  = 0.5 .- 0.25 .* same   #
    return (; Σ = Σ, same = same, cij = cij_T, backend = nothing, out_buf = nothing, dR_buf = nothing)
end

(l::JastrowLayer)(X::AbstractArray{T,3}, ps, st) where {T} = evaluate(l, X, ps, st)
function evaluate(l::JastrowLayer, X::AbstractArray{T,3}, ps, st) where {T}
    B  = size(X, 3)
    Ne = size(X, 2)

    backend = st.backend === nothing ? KA.get_backend(X) : st.backend
    groupsize = KernelAbstractions.isgpu(backend) ? 256 : 256

    need_new = st.out_buf == nothing
    out_buf = st.out_buf == nothing ? similar(X, T, (B,)) : st.out_buf
    dR_buf = st.dR_buf == nothing ? similar(X) : st.dR_buf
    st2 = need_new ? merge(st, (; out_buf = out_buf, dR_buf = dR_buf, cij = T.(st.cij), backend = backend)) : st

    cij_T = st2.cij

    kernel! = jastrow_kernel!(backend, groupsize)
    kernel!(out_buf, X, cij_T, Ne; ndrange = B)
    KA.synchronize(backend)
    return out_buf, st2
end

# -------------------- Forward kernel --------------------
@kernel function jastrow_kernel!(out, R::AbstractArray{T,3}, cij, Ne::Int) where {T}
    s = @index(Global)
    γ = zero(T)
    @inbounds for i in 1:Ne-1
        xi = R[1,i,s]; yi = R[2,i,s]; zi = R[3,i,s]
        for j in i+1:Ne
            dx = xi - R[1,j,s]; dy = yi - R[2,j,s]; dz = zi - R[3,j,s]
            dist = sqrt(dx*dx + dy*dy + dz*dz)
            γ -= cij[i,j] / (one(T) + dist)
        end
    end
    out[s] = γ
end

# rrule for the layer call, so we can use st buffers
function ChainRulesCore.rrule(::typeof(evaluate), l::JastrowLayer, X::AbstractArray{T,3}, ps, st) where {T}
    y, st2 = l(X, ps, st)
    B, Ne = size(X,3), size(X,2)
    backend = st2.backend
    groupsize = KernelAbstractions.isgpu(backend) ? 128 : 128
    dR = st2.dR_buf
    fill!(dR, zero(T))
    cij = st2.cij
    
    kernel! = jastrow_pullback!(backend, groupsize)
    function pullback(ȳ_raw)
        ȳ, st̄ = ȳ_raw
        dout = unthunk(ȳ) 
        kernel!(dout, X, cij, Ne, B, dR; ndrange=B)
        KA.synchronize(backend)
        return (NoTangent(), NoTangent(), dR, NoTangent(), NoTangent())
    end
    return (y, st2), pullback
end

# y_s = - Σ_{i<j} cij[i,j]/(1+r_ij)
# dy/dr = + cij/(1+r)^2,  dr/dri = (ri - rj)/r
# => dL/dri += dout[s] * cij[i,j] * (ri-rj)/( r*(1+r)^2 )
@kernel function jastrow_pullback!(dout, R::AbstractArray{T,3}, cij, Ne::Int, B::Int, dR::AbstractArray{T,3}) where {T}
    s = @index(Global)
    @inbounds for i in 1:Ne-1
        xi = R[1,i,s]; yi = R[2,i,s]; zi = R[3,i,s]
        for j in i+1:Ne
            dx = xi - R[1,j,s]
            dy = yi - R[2,j,s]
            dz = zi - R[3,j,s]
            dist = sqrt(dx*dx + dy*dy + dz*dz) + eps(T)
            invd  = inv(dist)
            inv1p = inv(one(T) + dist)
            fac = dout[s] * cij[i,j] * invd * (inv1p * inv1p)

            dR[1,i,s] += fac*dx;  dR[2,i,s] += fac*dy;  dR[3,i,s] += fac*dz
            dR[1,j,s] -= fac*dx;  dR[2,j,s] -= fac*dy;  dR[3,j,s] -= fac*dz
        end
    end
end


import KernelAbstractions as KA
using ChainRulesCore
using HyperDualNumbers: Hyper
using ForwardDiff: Dual

# Y_masked[i, j, b] = δ_{Σ_i, Σ_j} * sum_{l=1}^L ( W[i, l] * x_corr[b, j, l] )
struct FusedMaskPermuteLayer{Nx} <: AbstractLuxLayer
    Σ::NTuple{Nx, Char}
end

LuxCore.initialparameters(::AbstractRNG, ::FusedMaskPermuteLayer) = NamedTuple()

function LuxCore.initialstates(::AbstractRNG, l::FusedMaskPermuteLayer{Nx}) where {Nx}
    Σv = collect(Int32.(l.Σ))
    M  = reshape(Σv, Nx, 1) .== reshape(Σv, 1, Nx)   # Nx×Nx
    M_ = reshape(M, 1, Nx, Nx)                       # 1×Nx×Nx
    return (; ΣA = M_, backend = nothing, out_buf = nothing, dx_buf = nothing)
end

(l::FusedMaskPermuteLayer)(x::AbstractArray, ps, st) = evaluate(l, x, ps, st)

@kernel function fuse_mask_permute!(
    out,          # (Nel, Nel, B)
    xmat,         # (Nel, Nel*B)
    ΣA,           # (1, Nel, Nel) of Bool
    Nel::Int, B::Int)

    gid = @index(Global)                      # 1..Nel*Nel*B
    i = 1 + (gid - 1) % Nel
    j = 1 + ((gid - 1) ÷ Nel) % Nel
    b = 1 + (gid - 1) ÷ (Nel*Nel)

    if ΣA[1, i, j]
        ℓ = i + (b - 1) * Nel + (j - 1) * Nel * B
        p = 1 + (ℓ - 1) ÷ Nel
        @inbounds out[i, j, b] = xmat[i, p]
    else
        @inbounds out[i, j, b] = zero(eltype(out))
    end
end

@kernel function fuse_mask_permute_pullback!(
    dx,           # (Nel, Nel*B)
    dY,           # (Nel, Nel, B)
    ΣA,           # (1, Nel, Nel)
    Nel::Int, B::Int)

    gid = @index(Global)

    i = 1 + (gid - 1) % Nel
    j = 1 + ((gid - 1) ÷ Nel) % Nel
    b = 1 + (gid - 1) ÷ (Nel*Nel)

    if ΣA[1, i, j]
        ℓ = i + (b - 1) * Nel + (j - 1) * Nel * B
        p = 1 + (ℓ - 1) ÷ Nel
        @inbounds dx[i, p] = dY[i, j, b]
    end
end

function evaluate(l::FusedMaskPermuteLayer{Nx},
                  x::AbstractArray{T,2},
                  ps,
                  st) where {Nx,T}

    @assert size(x,1) == Nx
    @assert size(x,2) % Nx == 0

    Nel = Nx
    B   = size(x,2) ÷ Nel
    ndr = Nel*Nel*B

    backend = st.backend === nothing ? KA.get_backend(x) : st.backend
    groupsize = KernelAbstractions.isgpu(backend) ? 128 : 128
    ΣA = st.ΣA
    need_new = (st.out_buf === nothing || !(eltype(st.out_buf) === T && ndims(st.out_buf) == 3 &&
                                size(st.out_buf,1) == Nel && size(st.out_buf,2) == Nel && size(st.out_buf,3) == B))
    out_buf = need_new ? similar(x, T, Nel, Nel, B) : st.out_buf 
    st2 = need_new ? merge(st, (; backend = backend, out_buf = out_buf, ΣA = ΣA)) : st
    fill!(out_buf, zero(T))
    
    kernel! = fuse_mask_permute!(backend, groupsize)
    kernel!(out_buf, x, ΣA, Nel, B; ndrange = ndr)
    KA.synchronize(backend)
    return out_buf, st2
end

function ChainRulesCore.rrule(::typeof(evaluate),
                              l::FusedMaskPermuteLayer{Nx},
                              x::AbstractArray{T,2},
                              ps,
                              st) where {Nx,T}

    y, st2 = evaluate(l, x, ps, st)

    Nel = Nx
    @assert size(x,1) == Nel
    @assert size(x,2) % Nel == 0
    B   = size(x,2) ÷ Nel
    ndr = Nel*Nel*B

    backend = st2.backend === nothing ? KA.get_backend(x) : st2.backend
    groupsize = KernelAbstractions.isgpu(backend) ? 128 : 128
    ΣA = st2.ΣA

    need_new = (st2.dx_buf === nothing || !(eltype(st2.dx_buf) === T && ndims(st2.dx_buf) == 2 &&
                           size(st2.dx_buf,1) == Nel && size(st2.dx_buf,2) == Nel*B ))

    dx = need_new ? similar(x) : st2.dx_buf
    st3 = need_new ? merge(st2, (; dx_buf = dx)) : st2

    kernel_pb! = fuse_mask_permute_pullback!(backend, groupsize)
    fill!(dx, zero(T))
    function pullback(ȳ_raw)
        ȳ, st̄ = ȳ_raw
        dY = unthunk(ȳ)
        kernel_pb!(dx, dY, ΣA, Nel, B; ndrange = ndr)
        KA.synchronize(backend)
        return NoTangent(), NoTangent(), dx, NoTangent(), NoTangent()
    end
    return (y, st3), pullback
end

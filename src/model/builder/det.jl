using CUDA
using CUDA.CUBLAS 
using HyperDualNumbers
using HyperDualNumbers: Hyper
using ForwardDiff: Dual

struct detlayer <: AbstractLuxLayer end
LuxCore.initialparameters(::AbstractRNG, ::detlayer) = NamedTuple()

function LuxCore.initialstates(::AbstractRNG, ::detlayer)
    return (; backend = nothing,
             out_buf  = nothing,
             Afact    = nothing,
             dX_buf   = nothing, 
             streams  = nothing, 
             streams_2 = nothing, 
             g1 = nothing, g2 = nothing, g12 = nothing, tmp = nothing, 
             X1 = nothing, X2 = nothing, X12 = nothing, P = nothing)
end

# ---------------- forward call ----------------
(l::detlayer)(X, ps, st) = evaluate(l, X, ps, st)

function evaluate(::detlayer, X::Array{T,3}, ps, st) where {T<:AbstractFloat}
    y = map(first ∘ logabsdet, eachslice(X; dims=3))
    return y, st
end

function evaluate(::detlayer, X::AbstractGPUArray{T,3}, ps, st) where {T<:AbstractFloat}
    N, _, B = size(X)
    need_new = (st.Afact === nothing || typeof(st.Afact) !== typeof(X) || size(st.Afact) != size(X))
    backend = st.backend === nothing ? KA.get_backend(X) : st.backend
    groupsize = KernelAbstractions.isgpu(backend) ? 128 : 128
    out_buf = need_new ? similar(X, T, (B,)) : st.out_buf
    Afact = st.Afact === nothing || typeof(st.Afact) !== typeof(X) || size(st.Afact) != size(X) ? similar(X) : st.Afact
    dX_buf = st.dX_buf == nothing || typeof(st.dX_buf) !== typeof(X) || size(st.dX_buf) != size(X) ? similar(X) : st.dX_buf
    st2 = need_new ? merge(st, (; out_buf = out_buf, dX_buf = dX_buf, Afact = Afact, backend = backend)) : st

    copyto!(Afact, X)  # Afact .= X
    CUDA.@sync begin pivot, info = CUBLAS.getrf_strided_batched!(Afact, true) end
    kernel! = _sum_logabs_diag_kernel!(backend, groupsize)
    kernel!(Afact, out_buf, size(Afact, 1); ndrange = B)
    KA.synchronize(backend)
    return out_buf, st2
end

@kernel function _sum_logabs_diag_kernel!(Afact, out, N::Int)
    b = @index(Global)
    acc = zero(eltype(out))
    @inbounds for k in 1:N
        acc += log(abs(Afact[k, k, b]))
    end
    out[b] = acc
end

@kernel function set_scaled_eye!(rhs, ybar, N::Int)
    b = @index(Global)
    val = ybar[b]
    @inbounds for k in 1:N
        rhs[k, k, b] = val
    end
end

# Δx[b] = ȳ[b] * M^{-T}
function ChainRulesCore.rrule(::typeof(evaluate), l::detlayer, X::AbstractGPUArray{T,3}, ps, st) where {T<:AbstractFloat}
    N, _, B = size(X)
    need_new = (st.Afact === nothing || typeof(st.Afact) !== typeof(X) || size(st.Afact) != size(X))
    backend = st.backend === nothing ? KA.get_backend(X) : st.backend
    groupsize = KernelAbstractions.isgpu(backend) ? 128 : 128
    out_buf = need_new ? similar(X, T, (B,)) : st.out_buf
    Afact = st.Afact === nothing || typeof(st.Afact) !== typeof(X) || size(st.Afact) != size(X) ? similar(X) : st.Afact
    dX_buf = st.dX_buf == nothing || typeof(st.dX_buf) !== typeof(X) || size(st.dX_buf) != size(X) ? similar(X) : st.dX_buf
    st2 = need_new ? merge(st, (; out_buf = out_buf, dX_buf = dX_buf, Afact = Afact, backend = backend)) : st

    copyto!(Afact, X)  # Afact .= X
    CUDA.@sync begin pivot, info = CUBLAS.getrf_strided_batched!(Afact, true) end
    kernel! = _sum_logabs_diag_kernel!(backend, groupsize)
    kernel!(Afact, out_buf, size(Afact, 1); ndrange = B)
    KA.synchronize(backend)
    
    fill!(dX_buf, zero(T))
    function pullback(ȳ_raw)
        ȳ, st̄ = ȳ_raw
        ybar = unthunk(ȳ) 
        kernel! = set_scaled_eye!(backend, groupsize)
        kernel!(dX_buf, ybar, N; ndrange = B)
        KA.synchronize(backend)
        CUDA.@sync begin 
            info = CUBLAS.getrs_strided_batched!('T', Afact, dX_buf, pivot)
        end
        return (NoTangent(), NoTangent(), dX_buf, NoTangent(), NoTangent())
    end

    return (out_buf, st2), pullback
end

function logabsdet_lu(A)
    F = lu(A, check=false)
    dU = diag(F.factors)
    return sum(log.(abs.(dU)))
end

function evaluate(l::detlayer, x::Array{T, 3}, ps, st) where {T <: Union{Hyper, Dual}}
    return map(logabsdet_lu, eachslice(x; dims=(3))), st
end

function split_hyper(Xh)
    return (
        HyperDualNumbers.value.(Xh),
        HyperDualNumbers.epsilon1.(Xh),
        HyperDualNumbers.epsilon2.(Xh),
        HyperDualNumbers.epsilon12.(Xh),
    )
end

@kernel function _trace_kernel!(out, M, N::Int)
    b = @index(Global)
    acc = zero(eltype(out))
    @inbounds for i in 1:N
        acc += M[i,i,b]
    end
    out[b] += acc
end

@kernel function _tr_AD1AD2_kernel!(acc, X1, X2T, N::Int)
    b = @index(Global)
    s = zero(eltype(acc))
    @inbounds for i in 1:N, j in 1:N
        s += X1[i,j,b] * X2T[j,i,b]
    end
    acc[b] += s
end

function evaluate(l::detlayer, X::AbstractGPUArray{Hyper{T},3}, ps, st) where {T}
    N, _, B = size(X)

    # backend
    backend = st.backend === nothing ? KA.get_backend(X) : st.backend
    groupsize = KA.isgpu(backend) ? 128 : 128

    # ----- buffer plumbing -----
    # scalars per-batch
    need_new = st.out_buf === nothing || !(st.out_buf isa typeof(similar(X, T, (B,)))) || length(st.out_buf) != B
    f0  = st.out_buf === nothing || !(st.out_buf isa typeof(similar(X, T, (B,)))) || length(st.out_buf) != B ? similar(X, T, (B,)) : st.out_buf
    g1  = st.g1 === nothing     || !(st.g1     isa typeof(similar(X, T, (B,)))) || length(st.g1)     != B ? similar(X, T, (B,)) : st.g1
    g2  = st.g2 === nothing     || !(st.g2     isa typeof(similar(X, T, (B,)))) || length(st.g2)     != B ? similar(X, T, (B,)) : st.g2
    g12 = st.g12 === nothing    || !(st.g12    isa typeof(similar(X, T, (B,)))) || length(st.g12)    != B ? similar(X, T, (B,)) : st.g12
    tmp = st.tmp === nothing    || !(st.tmp    isa typeof(similar(X, T, (B,)))) || length(st.tmp)    != B ? similar(X, T, (B,)) : st.tmp

    # matrices per-batch
    Afact = st.Afact === nothing || typeof(st.Afact) !== typeof(X) || size(st.Afact) != size(X) ? similar(X, T, size(X)...) : st.Afact
    X1    = st.X1    === nothing || typeof(st.X1)    !== typeof(X) || size(st.X1)    != size(X) ? similar(X, T, size(X)...) : st.X1
    X2    = st.X2    === nothing || typeof(st.X2)    !== typeof(X) || size(st.X2)    != size(X) ? similar(X, T, size(X)...) : st.X2
    X12   = st.X12   === nothing || typeof(st.X12)   !== typeof(X) || size(st.X12)   != size(X) ? similar(X, T, size(X)...) : st.X12
    Pbuf  = st.P     === nothing || typeof(st.P)     !== typeof(X) || size(st.P)     != size(X) ? similar(X, T, size(X)...) : st.P

    st2 = need_new ? merge(st, (; backend = backend,
                      out_buf = f0, g1 = g1, g2 = g2, g12 = g12, tmp = tmp,
                      Afact = Afact, X1 = X1, X2 = X2, X12 = X12, P = Pbuf)) : st
    
    # ----- split Hyper into four real tensors -----
    M0, D1, D2, D12 = split_hyper(X)

    # ----- batched LU factorization of A0 -----
    copy!(Afact, M0)
    CUDA.@sync begin 
        pivot, info = CUBLAS.getrf_strided_batched!(Afact, true)  # Afact gets LU, pivot carries ipiv
    end

     # ----- f0 = sum(log|diag(U)|) i.e., log|det(A0)| -----
    fill!(f0, zero(T))
    kernel_logabs! = _sum_logabs_diag_kernel!(backend, groupsize)
    kernel_logabs!(Afact, f0, N; ndrange=B)
    KA.synchronize(backend)

    # ----- X1 = A0^{-1}A1, X2 = A0^{-1}A2 -----
    copy!(X1, D1)
    copy!(X2, D2)
    CUBLAS.getrs_strided_batched!('N', Afact, X1, pivot)
    CUDA.synchronize()
    CUBLAS.getrs_strided_batched!('N', Afact, X2, pivot)
    CUDA.synchronize()

    # g1 = tr(X1), g2 = tr(X2)
    fill!(g1, zero(T)); fill!(g2, zero(T))
    kernel_trace! = _trace_kernel!(backend, groupsize)
    kernel_trace!(g1, X1, N; ndrange=B)
    KA.synchronize(backend)
    kernel_trace!(g2, X2, N; ndrange=B)
    KA.synchronize(backend)

    # ----- X12 = A0^{-1}A12 -----
    copy!(X12, D12)
    CUDA.@sync CUBLAS.getrs_strided_batched!('N', Afact, X12, pivot)

    # g12 <- tr(X12)
    fill!(g12, zero(T))
    kernel_trace!(g12, X12, N; ndrange=B)
    KA.synchronize(backend)

    # ----- subtract tr(X1*X2) in a stable way -----
    # Pbuf = X1 * X2  (batched GEMM, no elementwise nonsense)
    CUDA.@sync begin
        CUBLAS.gemm_strided_batched!(
        'N', 'N',
        one(T),  # alpha
        X1, X2,
        zero(T), # beta
        Pbuf)
    end

    fill!(tmp, zero(T))
    kernel_trace!(tmp, Pbuf, N; ndrange=B)
    KA.synchronize(backend)

    # g12 = tr(A0^{-1}A12) - tr(A0^{-1}A1 A0^{-1}A2)
    g12 .-= tmp

    # ----- pack HyperDual in log-domain -----
    # out = log|det(A0)| + ε1*tr(A0^{-1}A1) + ε2*tr(A0^{-1}A2) + ε1ε2*(...)
    out_h = Hyper.(f0, g1, g2, g12)
    return out_h, st2
end


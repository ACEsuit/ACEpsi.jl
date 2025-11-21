import KernelAbstractions as KA
using ChainRulesCore
using CUDA

struct pooling{NN, Nx} <: AbstractLuxLayer
    spec::NTuple{NN, @NamedTuple{I::Int64, n1::Int64, n2::Int64, l::Int64, m::Int64}}
    Σ::NTuple{Nx, Char}
end

LuxCore.initialparameters(::AbstractRNG, ::pooling) = NamedTuple()

function LuxCore.initialstates(::AbstractRNG, l::pooling)
    return (; Σ_idx = spin2idx.(l.Σ),
              spec   = l.spec,
              backend = nothing,
              Aall_buf = nothing,
              A_buf    = nothing,
              Sup_buf  = nothing,
              Sdn_buf  = nothing,
              dX_buf   = nothing)
end

(l::pooling)(X, ps, st) = evaluate(l, X, ps, st)

# ---------------------- kernels ----------------------
# Aall[z, 2*(k-1)+spin] = ∑_{i with sig_idx[i]==spin} X[i,z,k],  spin∈{1,2}
@kernel function pool_stage1!(Aall, @Const(X), @Const(sig_idx),
                              Nel::Int, Nb::Int, K::Int)
    z, k, spin = @index(Global, NTuple)
    T = eltype(Aall)
    s = zero(T)
    @inbounds for i in 1:Nel
        if sig_idx[i] == spin
            s += X[i, z, k]
        end
    end
    @inbounds Aall[z, 2*(k-1) + spin] = s
end

# A[z,i, 3*(k-1)+3] = X[i,z,k]
# A[z,i, 3*(k-1)+1] = sum_up_except_i
# A[z,i, 3*(k-1)+2] = sum_dn_except_i
@kernel function pool_stage2!(A, @Const(X), @Const(Aall), @Const(sig_idx),
                              Nel::Int, Nb::Int, K::Int)
    z, i, k = @index(Global, NTuple)
    T  = eltype(A)
    x  = X[i, z, k]
    up = Aall[z, 2*(k-1) + 1]
    dn = Aall[z, 2*(k-1) + 2]
    @inbounds begin
        A[z, i, 3*(k-1) + 3] = x
        A[z, i, 3*(k-1) + 1] = up - (sig_idx[i]==1 ? x : zero(T))
        A[z, i, 3*(k-1) + 2] = dn - (sig_idx[i]==2 ? x : zero(T))
    end
end

# ---------- adjoint kernels ----------
@kernel function pool_adj_stage1!(S_up, S_dn, @Const(dA), Nb::Int, K::Int, Nel::Int)
    z, k = @index(Global, NTuple)
    Tu = eltype(S_up); Td = eltype(S_dn)
    su = zero(Tu); sd = zero(Td)
    @inbounds for j in 1:Nel
        su += dA[z, j, 3*(k-1) + 1]
        sd += dA[z, j, 3*(k-1) + 2]
    end
    @inbounds begin
        S_up[z, k] = su
        S_dn[z, k] = sd
    end
end

@kernel function pool_adj_stage2!(dX, @Const(dA), @Const(S_up), @Const(S_dn),
                                  @Const(sig_idx), Nel::Int, Nb::Int, K::Int)
    z, i, k = @index(Global, NTuple)
    T = eltype(dX)
    up_i = dA[z, i, 3*(k-1) + 1]
    dn_i = dA[z, i, 3*(k-1) + 2]
    base = dA[z, i, 3*(k-1) + 3]
    s = sig_idx[i]
    extra = (s==1 ? (S_up[z,k] - up_i) : zero(T)) +
            (s==2 ? (S_dn[z,k] - dn_i) : zero(T))
    @inbounds dX[i, z, k] = base + extra
end

# ---------------------- evaluate ----------------------
function evaluate(l::pooling{Nnlm, Nel},
                  X::AbstractArray{T,3},
                  ps,
                  st) where {Nnlm, Nel, T}

    @assert size(X,1) == Nel
    @assert size(X,3) == Nnlm
    Nb = size(X,2); K = size(X,3)

    backend = st.backend === nothing ? KA.get_backend(X) : st.backend
    groupsize = KernelAbstractions.isgpu(backend) ? 128 : 128
    Σ = st.Σ_idx

    Aall = st.Aall_buf
    need_Aall = (Aall === nothing) ||
                !(eltype(Aall)===T && size(Aall)==(Nb, 2K) && typeof(Aall)===typeof(X))
    Aall = need_Aall ? similar(X, T, Nb, 2K) : Aall

    A = st.A_buf
    need_A = (A === nothing) ||
             !(eltype(A)===T && size(A)==(Nb, Nel, 3K) && typeof(A)===typeof(X))
    A = need_A ? similar(X, T, Nb, Nel, 3K) : A

    # launch
    pool_stage1!(backend, groupsize)(Aall, X, Σ, Nel, Nb, K; ndrange=(Nb, K, 2))
    KA.synchronize(backend)
    pool_stage2!(backend, groupsize)(A, X, Aall, Σ, Nel, Nb, K; ndrange=(Nb, Nel, K))
    KA.synchronize(backend)

    st2 = need_Aall ? merge(st, (; backend=backend, Aall_buf=Aall, A_buf=A)) : st
    return A, st2
end

function ChainRulesCore.rrule(::typeof(evaluate),
               l::pooling{Nnlm,Nel},
               X::AbstractArray{T,3},
               ps,
               st) where {Nnlm,Nel,T}

    Y, st2 = evaluate(l, X, ps, st)
    Nb, K = size(X,2), size(X,3)
    backend  = st2.backend === nothing ? KA.get_backend(X) : st2.backend
    groupsize = KernelAbstractions.isgpu(backend) ? 128 : 128
    Σ = st2.Σ_idx

    dX = st2.dX_buf
    need_dX = (dX === nothing) || !(eltype(dX)===T && size(dX)==size(X) && typeof(dX)===typeof(X))
    dX = need_dX ? similar(X) : dX

    S_up = st2.Sup_buf
    need_Sup = (S_up === nothing) || !(eltype(S_up)===T && size(S_up)==(Nb, K) && typeof(S_up)===typeof(X))
    S_up = need_Sup ? similar(X, T, Nb, K) : S_up

    S_dn = st2.Sdn_buf
    need_Sdn = (S_dn === nothing) || !(eltype(S_dn)===T && size(S_dn)==(Nb, K) && typeof(S_dn)===typeof(X))
    S_dn = need_Sdn ? similar(X, T, Nb, K) : S_dn

    st3 = need_dX ? merge(st2, (; dX_buf = dX, Sup_buf = S_up, Sdn_buf = S_dn)) : st2

    function pullback(Ȳ_raw)
        Ȳ, st̄ = Ȳ_raw
        dA = unthunk(Ȳ)

        pool_adj_stage1!(backend, groupsize)(S_up, S_dn, dA, Nb, K, Nel; ndrange=(Nb, K))
        KA.synchronize(backend)
        pool_adj_stage2!(backend, groupsize)(dX, dA, S_up, S_dn, Σ, Nel, Nb, K; ndrange=(Nb, Nel, K))
        KA.synchronize(backend)
        return NoTangent(), NoTangent(), dX, NoTangent(), NoTangent()
    end

    return (Y, st3), pullback
end
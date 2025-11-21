
using LowRankApprox
using CUDA
function opts!(i,
               OptParams,
               optimizer::SketchSolver,
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

    # ---- unpack & local opts ----
    sr_rank0 = Int(optimizer.sr_rank0)
    sr_rank  = Int(optimizer.sr_rank)
    sketch_opts = LRAOptions(sketch = :srft, rank = min(dim_ps, sr_rank))
    res = norm(force)

    lmul!(-γ, Eloc)

    oa = o
    ea = Eloc
    if !OptParams.first
        @views begin
            oa = hcat(sqrt(η) * OptParams.sr_o[:, 1:sr_rank0],
                      sqrt(1 - η) * o)::Matrix{T}
            ea = vcat(sqrt(η) * OptParams.ek[1:sr_rank0],
                      sqrt(1 - η) * Eloc)::Vector{T}
        end
    end
    svdo = psvdfact(oa, sketch_opts)
    S = svdo[:S]
    U = svdo[:U]
    V = svdo[:V]
    
    ind = searchsortedlast(S / abs(S[1]), damping; rev = true)
    sigma0 = inv(damping * abs(S[1]))^2

    @views u = U[:, 1:ind]
    @views s = S[1:ind]
    @views v = V[:, 1:ind]

    f = OptParams.f
    mul!(f, oa, ea)           # ĝ
    # uf = u' * f
    uf = adjoint(u) * f
    # uf .*= (1/s^2 - 1/s0^2)
    @inbounds for j in eachindex(s, uf)
        invs2 = inv(s[j])^2
        uf[j] *= (invs2 - sigma0)
    end
    dw_tot = OptParams.dw_tot
    # dw_tot = u * uf + 1/s0^2 * f
    mul!(dw_tot, u, uf)        # u * (...)
    @. dw_tot += sigma0 * f

    sr_o = OptParams.sr_o
    fill!(sr_o, zero(T))
    @views begin
        for j in 1:ind
            @inbounds sr_o[:, j] .= u[:, j] .* s[j]
        end
    end

    ek = OptParams.ek
    fill!(ek, zero(T))
    @views mul!(ek[1:ind], adjoint(v), ea)

    optimizer.sr_rank0 = ind
    if ind == sr_rank
        optimizer.sr_rank = min(
            Int(ceil(sr_rank * optimizer.sr_scale)),
            optimizer.sr_rank_max
        )
    end

    nrm = T(sqrt(norm_constrain))/norm(dw_tot)
    dw_tot .*= min(1, nrm)

    copy!(OptParams.sr_o, sr_o)
    copy!(OptParams.ek, ek)
    OptParams = (; OptParams..., first = false)
    return OptParams, dw_tot, ind, res
end

function opts!(i,
               OptParams,
               optimizer::SketchSolver,
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
    sr_rank0 = Int(optimizer.sr_rank0)
    sr_rank  = Int(optimizer.sr_rank)
    sr_rank = min(min(Int(dim_ps), Int(nbatch)), sr_rank)
    sketch_opts = LRAOptions(sketch = :srft, rank = sr_rank)
    res = norm(force)
    @. Eloc *= -γ
    
    need_new = !(eltype(OptParams.f) == T)
    f = need_new ? similar(o, dim_ps) : OptParams.f
    dw_tot = need_new ? similar(OptParams.dw_tot, T) : OptParams.dw_tot
    sr_o = need_new ? similar(OptParams.sr_o, T) : OptParams.sr_o
    ek = need_new ? similar(OptParams.ek, T) : OptParams.ek  # CuVector{T}
    fill!(sr_o, zero(T))
    fill!(ek, zero(T))
    OptParams = need_new ? merge(OptParams, (; f = f, dw_tot = dw_tot, sr_o = sr_o, ek = ek)) : OptParams
    OptParams = need_new ? (; OptParams..., first = false) : OptParams

    oa = o
    ea = Eloc
    if !OptParams.first
        @views begin
            oa = hcat(sqrt(η)     * OptParams.sr_o[:, 1:sr_rank0],
                          sqrt(1 - η) * o)
            ea = vcat(sqrt(η)     * OptParams.ek[1:sr_rank0],
                          sqrt(1 - η) * Eloc)
        end
    end
    m, n = size(oa)
    oversample = 10
    l = min(sr_rank + oversample, min(m, n))
    need_new = sr_st == nothing || size(sr_st.Y, 2) != l
    Y = need_new ? similar(o, m, l) : sr_st.Y
    sr_st = need_new ? (; Y = Y) : sr_st

    U, S, V = sketch_svd_gpu(oa, sr_rank, Y; oversample = oversample)
    #svdo = psvdfact(Matrix(oa), sketch_opts)
    #S = svdo[:S]
    #U = svdo[:U]
    #V = svdo[:V]
    
    SS = Array(S)
    ind = searchsortedlast(SS ./ SS[1], damping; rev = true)
    S1 = abs(SS[1])
    sigma0 = T(inv(damping * S1)^2)

    r0 = min(min(Int(dim_ps), Int(nbatch)), sr_rank0)
    r0 = max(r0, 1)
    if ind <= r0
        ind = r0
    else
        optimizer.sr_rank0 = ind
    end

    #S = cu(S)
    #U = cu(U)
    #V = cu(Matrix(V))
    
    @views uu = U[:, 1:ind]
    @views s = S[1:ind]
    @views v = V[:, 1:ind]

    need_new = !hasproperty(sr_st, :y) 
    sr_st = need_new ? merge(sr_st, (; tmp_US = similar(o, size(uu,1), ind), y = similar(o, ind), uf = similar(o, ind))) : sr_st
    
    need_new = size(sr_st.y, 1) != ind
    tmp_US = need_new ? similar(o, size(uu,1), ind) : sr_st.tmp_US
    y = need_new ? similar(o, ind) : sr_st.y
    uf = need_new ? similar(o, ind) : sr_st.uf
    sr_st = need_new ? merge(sr_st, (; tmp_US = tmp_US, y = y, uf = uf)) : sr_st

    # f = oa * ea                      
    mul!(f, oa, ea)                   # ĝ
    mul!(uf, adjoint(uu), f)

    # uf .*= (1/s^2 - 1/s0^2)  
    @. uf *= (1 / s^2 - sigma0)      

    # dw_tot = u * uf + 1/s0^2 * f
    mul!(dw_tot, uu, uf)
    @. dw_tot += sigma0 * f

    # CuMatrix{T}
    mul!(tmp_US, uu, Diagonal(s))
    @views sr_o[:, 1:ind] .= tmp_US
    mul!(y, adjoint(v), ea)            
    @views ek[1:ind] .= y

    if ind == optimizer.sr_rank
        optimizer.sr_rank = min(Int(ceil(optimizer.sr_rank * optimizer.sr_scale)), optimizer.sr_rank_max)
    end

    nrm = sqrt(norm_constrain)/norm(dw_tot)
    dw_tot .*= min(1, nrm)
    
    copy!(OptParams.sr_o, sr_o)
    copy!(OptParams.ek, ek)
    return OptParams, dw_tot, ind, res, sr_st
end

using CUDA, LinearAlgebra, Random

function sketch_svd_gpu(A::CuArray, k::Int, Y; oversample::Int=10)
    m, n = size(A)
    l = min(k + oversample, min(m, n))
    T = eltype(A)

    Ω = CUDA.randn(T, n, l)
    mul!(Y, A, Ω)
    F = qr(Y)
    Qbuf = F.factors
    CUDA.CUSOLVER.orgqr!(Qbuf, F.τ)
    B = adjoint(Qbuf) * A
    F = svd(B)
    Ub = F.U
    U = Qbuf * Ub 
    k_eff = min(k, length(F.S))
    return U[:, 1:k_eff], F.S[1:k_eff], copy(F.V[:, 1:k_eff])
end
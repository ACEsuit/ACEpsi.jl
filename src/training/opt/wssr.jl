
function opts!(i,
               OptParams,
               optimizer::SVDSolver,
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

    # ----- ranks & bookkeeping -----
    sr_rank0 = Int(optimizer.sr_rank0)
    sr_rank  = Int(optimizer.sr_rank)
    res = norm(force)
    lmul!(-γ, Eloc)

    # ----- assemble oa, ea with mixing -----
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

    # ----- requested SVD rank -----
    r = min(min(Int(dim_ps), Int(nbatch)), sr_rank)
    r = max(r, 1)

    u = OptParams.u
    if size(u, 2) < r
        mm = size(u, 1)
        k  = r - size(u, 2)
        @views u[:, size(OptParams.u, 2)+1:r] .= randn(mm, k)
    end

    # ----- partial SVD with warm start -----
    if i == 1
        U, S, V = lmsvd(oa, r; maxit = 300, X = u[:, 1:r])
    else
        U, S, V = ssisvd(oa, r; maxit = 10,  X0 = u[:, 1:r])
    end

    # ----- rank selection by damping -----
    ind = searchsortedlast(S ./ S[1], damping; rev = true)
    sigma0 = inv(damping * abs(S[1]))^2

    @views u[:, 1:length(S)] .= U
    @views uu = U[:, 1:ind]
    @views s = S[1:ind]
    @views v = V[:, 1:ind]

    # ----- direction -----
    # f = oa * ea
    f = OptParams.f
    mul!(f, oa, ea)
    uf = adjoint(uu) * f

    # uf .*= (1/s^2 - 1/s0^2)
    @inbounds for j in eachindex(s, uf)
        invs2 = inv(s[j])^2
        uf[j] *= (invs2 - sigma0)
    end

    # dw_tot = u * uf + 1/s0^2 * f
    dw_tot = OptParams.dw_tot
    mul!(dw_tot, uu, uf)
    @. dw_tot += sigma0 * f

    # ----- store history for next iter -----
    # sr_o[:,1:ind] = u * diagm(s)
    sr_o = OptParams.sr_o
    fill!(sr_o, zero(T))
    @views begin
        for j in 1:ind
            sr_o[:, j] .= uu[:, j] .* s[j]
        end
    end

    ek = OptParams.ek
    fill!(ek, zero(T))
    @views mul!(ek[1:ind], adjoint(v), ea)

    # ----- update ranks -----
    optimizer.sr_rank0 = ind
    if ind == optimizer.sr_rank
        optimizer.sr_rank = min(Int(ceil(optimizer.sr_rank * optimizer.sr_scale)),
                                optimizer.sr_rank_max)
    end

    # ----- norm constraint -----
    nrm = T(sqrt(norm_constrain))/norm(dw_tot)
    dw_tot .*= min(1, nrm)

    copy!(OptParams.sr_o, sr_o)
    copy!(OptParams.ek, ek)
    copy!(OptParams.u, u)
    OptParams = (; OptParams..., first = false)
    return OptParams, dw_tot, ind, res
end



# ---------- Utility: finite guards ----------
function finite_guard!(M::CuArray, where::AbstractString)
    if any(isnan.(M)) || any(isinf.(M))
        error("Non-finite values detected at $where")
    end
    return nothing
end

@inline function assemble_oa_ea!(
    oa_buf,
    ea_buf, 
    sr_o_view::CuArray{T,2},    # m×sr_cols
    o::CuArray{T,2},            # m×nb
    ek_view::CuArray{T,1},      # sr_cols
    Eloc::CuArray{T,1},         # nb
    η::T
) where {T}
    m, sr_cols = size(sr_o_view)
    nb = size(o,2)
    need_cols = sr_cols + nb
    fill!(oa_buf, zero(T))
    fill!(ea_buf, zero(T))
    oa = @view oa_buf[:, 1:need_cols]
    ea = @view ea_buf[1:need_cols]

    @views oa[:, 1:sr_cols] .= sqrt(η) .* sr_o_view
    @views ea[1:sr_cols]     .= sqrt(η) .* ek_view

    @views oa[:, sr_cols+1:end] .= sqrt(1-η) .* o
    @views ea[sr_cols+1:end]    .= sqrt(1-η) .* Eloc

    return oa, ea
end

Base.@kwdef mutable struct OptBufs{T} 
    oa_buf::CuArray{T,2} = CuArray{T}(undef, 0, 0) 
    ea_buf::CuArray{T,1} = CuArray{T}(undef, 0) 
    sr_o::CuArray{T,2} = CuArray{T}(undef, 0, 0)
    tmp_US::CuArray{T,2} = CuArray{T}(undef, 0, 0)
    y::CuArray{T,1} = CuArray{T}(undef, 0) 
    uf::CuArray{T,1} = CuArray{T}(undef, 0) 
end

function opts!(i,
               OptParams,
               optimizer::SVDSolver,
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

    if i == 1
        sr_st  = OptBufs{T}() 
        sr_st.oa_buf = similar(o, size(o, 1), Int(ceil(1.5 * optimizer.sr_rank0)) + nbatch)
        sr_st.ea_buf = similar(o, Int(ceil(1.5 * optimizer.sr_rank0)) + nbatch)
    end

    need_new = !(eltype(OptParams.f) == T)
    f = need_new ? T.(OptParams.f) : OptParams.f
    dw_tot = need_new ? T.(OptParams.dw_tot) : OptParams.dw_tot
    sr_o = need_new ? T.(OptParams.sr_o) : OptParams.sr_o
    ek = need_new ? T.(OptParams.ek) : OptParams.ek  # CuVector{T}
    u = need_new ? T.(OptParams.u) : OptParams.u  # CuVector{T}
    OptParams = need_new ? merge(OptParams, (; u = u, f = f, dw_tot = dw_tot, sr_o = sr_o, ek = ek)) : OptParams
    OptParams = need_new ? (; OptParams..., first = false) : OptParams

    sr_rank0 = Int(optimizer.sr_rank0)
    sr_rank  = Int(optimizer.sr_rank)
    sr_rank = min(min(Int(dim_ps), Int(nbatch)), sr_rank)
    r = max(min(min(Int(dim_ps), Int(nbatch)), sr_rank), 1)
    res = norm(force)
    Eloc *= -γ   
    oa = o
    ea = Eloc
    if !OptParams.first
        @views sr_o_hist = OptParams.sr_o[:, 1:sr_rank0]
        @views ek_hist   = OptParams.ek[1:sr_rank0]
        oa, ea = assemble_oa_ea!(sr_st.oa_buf, sr_st.ea_buf, sr_o_hist, o, ek_hist, Eloc, η)
    end
    
    ur = size(OptParams.u, 2)
    if ur < r
        mm = size(OptParams.u, 1)
        k  = r - ur
        @views u[:, ur+1:r] .= CUDA.randn(T, mm, k)
    end

    if i == 1
        F = svd(oa)                  # F <: LinearAlgebra.SVD
        U = F.U[:, 1:r]              # m×r 
        S = F.S[1:r]                 # r
        V = F.V[:, 1:r]            # n×r
    else
        U, S, V = ssisvd(oa, r; maxit = optimizer.svd_iteration, X0 = u[:, 1:r])
    end

    SS = Array(S)
    ind = searchsortedlast(SS / SS[1], damping; rev = true)
    S1 = abs(SS[1])
    sigma0 = inv(damping * S1)^2

    r0 = min(min(Int(dim_ps), Int(nbatch)), sr_rank0)
    r0 = max(r0, 1)
    if ind <= r0
        ind = r0
    else
        optimizer.sr_rank0 = ind
    end
    @views u[:, 1:length(S)] .= U
    @views uu = U[:, 1:ind]
    @views s = S[1:ind]
    @views v = V[:, 1:ind]

    if size(sr_st.y, 1) != ind
        sr_st.tmp_US = similar(o, size(uu,1), ind)
        sr_st.y = similar(o, ind)
        sr_st.uf = similar(o, ind) 
    end

    # f = oa * ea
    mul!(f, oa, ea)                   # ĝ
    mul!(sr_st.uf, adjoint(uu), f)

    # uf .*= (1/s^2 - 1/s0^2)  
    @. sr_st.uf *= (1 / s^2 - sigma0)      

    # dw_tot = u * uf + 1/s0^2 * f
    mul!(dw_tot, uu, sr_st.uf)
    @. dw_tot += sigma0 * f

    @views copyto!(sr_st.tmp_US, uu)
    @views sr_st.tmp_US .= sr_st.tmp_US .* permutedims(s)
    #mul!(tmp_US, uu, Diagonal(s))
    @views sr_o[:, 1:ind] .= sr_st.tmp_US

    mul!(sr_st.y, adjoint(v), ea)            
    @views ek[1:ind] .= sr_st.y

    if ind == optimizer.sr_rank
        optimizer.sr_rank = min(Int(ceil(optimizer.sr_rank * optimizer.sr_scale)),
                                optimizer.sr_rank_max)
        sr_st.oa_buf = similar(o, size(o, 1), Int(ceil(1.5 * optimizer.sr_rank)) + nbatch)
        sr_st.ea_buf = similar(o, Int(ceil(1.5 * optimizer.sr_rank)) + nbatch)
    end

    nrm = sqrt(norm_constrain)/norm(dw_tot)
    dw_tot .*= min(1, nrm)
    
    copy!(OptParams.sr_o, sr_o)
    copy!(OptParams.ek, ek)
    copy!(OptParams.u, u)
    return OptParams, dw_tot, ind, res, sr_st
end


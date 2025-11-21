export burnin!, distributed_sampling!, initialize_around_nuclei

function initialize_around_nuclei(mol::Molecule{Nnuc,Nx, T,TT}, nchains::Integer;
                                  Δt::Real = 0.08, device::Symbol = :auto, σ::Union{Nothing,Real}=nothing) where {Nnuc,Nx, T,TT}
    nucs = mol.nuclei 
    Nel = sum([mol.nuclei[i].charge for i = 1:length(mol.nuclei)])
    charges = map(_charge, nucs)
    tot = sum(charges)
    Nel <= tot || throw(ArgumentError("Nel=$Nel exceeds total nuclear charge $tot"))

    inuc = Vector{Int}(undef, Nel)
    k = 1
    @inbounds for (i, q) in enumerate(charges)
        for _ = 1:q
            if k > Nel; break; end
            inuc[k] = i; k += 1
        end
        if k > Nel; break; end
    end
    r0 = Matrix{T}(undef, 3, Nel)
    @inbounds for j = 1:Nel
        r0[:, j] = _rr(nucs[inuc[j]])
    end

    use_gpu = device == :gpu || (device == :auto && CUDA.has_cuda())
    noiseσ = isnothing(σ) ? sqrt(T(Δt)) : T(σ)

    if use_gpu
        r0_d = CUDA.CuArray(r0)
        ξ     = CUDA.randn(T, 3, Nel, nchains)     # 3×Nel×nchains
        return reshape(r0_d, 3, Nel, 1) .+ noiseσ .* ξ
    else
        ξ     = randn(T, 3, Nel, nchains)
        return reshape(r0, 3, Nel, 1) .+ noiseσ .* ξ
    end
end

function burnin!(X, model, ps, st, burnin::Int64; Δt = 0.08, xprop_buf = nothing, logu_buf  = nothing)
    theta, st = model(X, ps, st)
    acc = zeros(burnin)
    fill!(acc, 0.0)
    X, theta, st, acc = distributed_sampling!(X, model, ps, st, theta, acc, Δt, burnin; xprop_buf = xprop_buf, logu_buf = logu_buf)
    return X, theta, st, acc 
end

function distributed_sampling!(X, model, ps, st, theta::AbstractArray, acc::AbstractArray, Δt::TN, T::Int64; xprop_buf = nothing, logu_buf  = nothing) where {TN <: Float64}
    nchains = size(X, 3)
    @assert length(theta) == nchains
    @assert length(acc) == T
    for i = 1:T
        X, theta, st, a = distributed_mhsteps!(model, ps, st, X, theta, Δt, i; xprop_buf = xprop_buf, logu_buf = logu_buf)
        acc[i] = a
    end
    return X, theta, st, acc
end

function distributed_mhsteps!(model, ps, st, X::AbstractArray{Tx,3}, theta::AbstractVector{Ty}, 
    σ::T, iter::Int; xprop_buf = nothing, logu_buf  = nothing) where {Tx, Ty, T<:AbstractFloat}

    Xprop = xprop_buf === nothing ? similar(X) : xprop_buf
    @assert size(Xprop) == size(X)

    ξ = similar(X)
    randn!(ξ)
    @. Xprop = X + Tx(σ) * ξ
    θprop, st = model(Xprop, ps, st)
    
    logu = logu_buf === nothing ? similar(theta) : logu_buf
    rand!(logu)
    @. logu = log(logu)
    logA = @. θprop - theta
    accept = @. logu ≤ logA

    mask = reshape(accept, 1, 1, :)
    X .= ifelse.(mask, Xprop, X)
    theta .= ifelse.(accept, θprop, theta)

    return X, theta, st, count(accept) / length(accept)
end

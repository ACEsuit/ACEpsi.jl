using LinearAlgebra
using CUDA

# ---------- Orthogonalize W against X ----------

"""
    orth_against!(X, W)

Make W orthogonal to columns of X, then re-orthogonalize W.
"""
function orth_against!(X::CuArray{T,2}, W::CuArray{T,2}) where {T}
    C = CUDA.zeros(T, size(X,2), size(W,2))
    # C = X' * W
    CUDA.CUBLAS.gemm!('T','N', one(T), X, W, zero(T), C)
    # W := W - X*C
    CUDA.CUBLAS.gemm!('N','N', -one(T), X, C, one(T), W)
    Fq = qr!(W)
    Qbuf = Fq.factors
    CUDA.CUSOLVER.orgqr!(Qbuf, Fq.τ)
    return Qbuf
end

# ---------- SS-Iteration SVD (GPU) ----------
function ssisvd(A::CuArray{T,2}, r::Integer;
                X0=nothing, maxit::Integer=2) where {T<:AbstractFloat}
    m, n = size(A)
    r > 0 || throw(ArgumentError("r must be ≥ 1"))
    qtarget = min(r, m, n)

    # Scale-guard for GEMMs: we multiply by alpha = 1/sA in every A*· or A'*·
    sA = maximum(abs, A)
    sA = sA == 0 ? one(T) : sA
    α = one(T) / sA

    # Initialize subspace X (m×q)
    X = if X0 === nothing
        Ω  = CUDA.randn(T, n, qtarget)
        X1 = CUDA.zeros(T, m, qtarget)
        # X1 = α * A * Ω
        CUDA.CUBLAS.gemm!('N','N', α, A, Ω, zero(T), X1)
        Fq = qr!(X1)
        X1 = Fq.factors
        CUDA.CUSOLVER.orgqr!(X1, Fq.τ)
    else
        size(X0,1) == m || throw(DimensionMismatch("X0 rows must equal size(A,1)"))
        Xb = copy(X0)                   # avoid mutating caller
        Fq = qr!(Xb)
        Xb = Fq.factors
        CUDA.CUSOLVER.orgqr!(Xb, Fq.τ)
        q0 = size(Xb,2)
        if q0 < qtarget
            add = qtarget - q0
            Ω  = CUDA.randn(T, n, add)
            W  = CUDA.zeros(T, m, add)
            CUDA.CUBLAS.gemm!('N','N', α, A, Ω, zero(T), W)
            W = orth_against!(Xb, W)
            hcat(Xb, W)                 # size m×qtarget
        else
            Xb                          # truncate to target size
        end
    end

    q = size(X,2)

    # Power iteration with orthogonalization of both Y and X
    Y = CUDA.zeros(T, n, q)
    for _ = 1:maxit
        # Y = α * A' * X
        CUDA.CUBLAS.gemm!('C','N', α, A, X, zero(T), Y)
        Fq = qr!(Y)
        Y = Fq.factors
        CUDA.CUSOLVER.orgqr!(Y, Fq.τ)
        # X = α * A * Y
        CUDA.CUBLAS.gemm!('N','N', α, A, Y, zero(T), X)
        Fq = qr!(X)
        X = Fq.factors
        CUDA.CUSOLVER.orgqr!(X, Fq.τ)
    end

    B = CUDA.zeros(T, q, n)                     # q×n
    CUDA.CUBLAS.gemm!('T','N', α, X, A, zero(T), B) # B = (X' * A_scaled)
    F = svd(B; full=false)                      # on GPU q×n

    Ur = F.U[:, 1:r]                            # q×r
    Sr = F.S[1:r] .* sA                         # rescale σ back
    Vr = F.V[:, 1:r]                          # n×r
    
    # U = X * Ur
    U  = CUDA.zeros(T, m, r)
    CUDA.CUBLAS.gemm!('N','N', one(T), X, Ur, zero(T), U)

    return U, Sr, Vr
end

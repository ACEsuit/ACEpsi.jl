using LinearAlgebra

"""
    lmsvd(A::Matrix{T}, r; X = nothing, tol = 1e-8, maxit = 10, memo = 3) where {T}

Computes an approximate rank-`r` singular value decomposition (SVD) of the matrix `A` using Limited Memory Block Krylov Subspace scheme.

# Arguments
- **A**: A matrix of type `Matrix{T}` whose SVD is to be approximated.
- **r**: The target rank for the truncated SVD.
- **X** (optional): An initial approximation matrix for the left singular subspace. If not provided, a random initialization is used.
- **tol** (optional): Convergence tolerance (default is 1e-8).
- **maxit** (optional): Maximum number of iterations for the subspace iteration solver (default is 10).
- **memo** (optional): Memory (default is 3).

# Returns
An `LMSVD{T}` object containing:
- **U**: Approximate left singular vectors.
- **S**: Approximate singular values (in descending order).
- **V**: Approximate right singular vectors.
  
# Example
```julia
A = rand(100, 50)
svd_result = lmsvd(A, 10)
```

# Reference
Xin Liu, Zaiwen Wen, and Yin Zhang, "Limited Memory Block Krylov Subspace Optimization for Computing Dominant Singular Value Decompositions," SIAM Journal on Scientific Computing, 35-3 (2013), pp. A1641–A1668.
"""
function lmsvd(A::AbstractMatrix{T}, r; X = nothing, tol = 1e-8, maxit = 10, memo = 3) where {T}
    m, n = size(A)
    r > 0 || throw(ArgumentError("r must be ≥ 1"))
    r = min(r, m, n)
    q = min(min(2*r, r + 10), min(m, n))

    if X == nothing
        Y = randn_like(A, n, q)
        # X ← A * Y
        X = similar(A, T, m, q)
        mul!(X, A, Y)
    else
        size(X, 1) == m || throw(DimensionMismatch("X rows must equal size(A,1)"))
        q = size(X, 2)
        Y = similar(A, T, n, q)
        mul!(Y, adjoint(A), X)
        mul!(X, A, Y)
    end
    F = qr(X)
    X = CuMatrix(F.Q)
    Y = adjoint(A) * X
    
    # Call solver
    X, Y = lm_lbo(A, X, Y, r, tol, maxit, memo)
    # Generate SVD
    U1, S1, V1 = get_svd(X, Y)

    U = copy(@view U1[:, 1:r])
    V = copy(@view V1[:, 1:r])
    S = S1[1:r]
    return U, S, V
end

function get_svd(X, Y)
    F = qr!(Y)
    Rt = copy(transpose(F.R)) 
    W, S, Z = svd(Rt)  

    U = X * W
    Q = CuMatrix(F.Q)
    V = Q * Z
    return U, S, V
end


"""
    Solve min ||XY' - A||_F, s.t. X'X = I using limited memory look-back optimization (LBO).

    The problem is equivalent to max ||A' * X||_F,  s.t. X'*X = I.
    
    Inputs:
        A     -- an (m by n) matrix or a struct
        X     -- an (m by k) matrix with X'X = I
        Y     -- an (n by k) matrix where Y = A'X
        r     -- number of leading singular triplets
        tol   -- tolerance for convergence
        maxit -- maximum number of iterations
        memo  -- memory for look-back optimization

    Outputs:
        X, Y -- updated X and Y matrices
"""
function lm_lbo(A, X::AbstractMatrix{RT}, Y, r, tol, maxit, memo) where {RT}
    m, n, k = size(X,1), size(Y,1), size(Y,2)
    mn = min(m, n)
    Lm = k

    rvr   = similar(A, RT, r)
    rvr0  = similar(A, RT, r)
    fill!(rvr, zero(RT))
    fill!(rvr0, zero(RT))
    chg_rvr = one(RT)

    chgv = zeros(RT, maxit)
    kktc = zeros(RT, maxit)
    xtrm = zeros(RT, maxit)

    qtol = eps(RT)^min(mn/40/k, one(RT))
    rtol = RT(5) * max(sqrt(tol*qtol), RT(5) * eps(RT))
    ptol = RT(5) * max(tol, sqrt(eps(RT)))

    if k < r
        error("Working size too small: k=$k < r=$r")
    end

    Xm = similar(X, m, (1 + memo)*k)
    fill!(Xm, zero(RT))
    Ym = similar(Y, n, (1 + memo)*k)
    fill!(Ym, zero(RT))

    @views Xm[:, k+1:2k] .= X
    @views Ym[:, k+1:2k] .= Y

    AY = similar(X, m, k)
    SX = similar(X, m, k)

    for iter = 1:maxit
        @. SX = X

        # Subspace iteration
        mul!(AY, A, Y)                   # AY = A*Y (GPU)
        F = qr(AY) 
        X = CuMatrix(F.Q) 
        Y = adjoint(A) * X              # Y = A'*X
        if Lm == 0 || iter <= 3
            # SYTY = X' * A * Y 
            SYTY = adjoint(SX) * AY
            @. SYTY = (SYTY .+ SYTY') .* RT(0.5)
            teig = eigen(SYTY)
            tU, tE = teig.vectors, teig.values

            @. rvr0 = rvr
            rvr = similar(A, RT, r)            # CuArray
            fill!(rvr, zero(RT))
            @views rvr .= tE[end-r+1:end] 
            chg_rvr = norm(rvr0 - rvr) / max(norm(rvr), eps(RT))

            AY = AY * tU
            SX = SX * tU
        end

        xtrm[iter] = Lm / k
        chgv[iter] = chg_rvr

        if chg_rvr < rtol
            @views begin
                # kkt = AY(:, end-r+1:end) - SX(:, end-r+1:end) * diag(rvr)
                AYr = AY[:, end-r+1:end]
                SXr = SX[:, end-r+1:end]
                kkt = AYr .- SXr .* rvr'
                kktcheck = maximum(sqrt.(sum(kkt .^ 2, dims = 1))) / max(tol, sum(@views rvr[end:end]))
            end
            if kktcheck < ptol
                break
            end
            kktc[iter] = kktcheck
        else
            kktc[iter] = iter == 1 ? zero(RT) : kktc[iter-1]
        end
        xtrm[iter] = Lm / k

        # Look-back optimization
        if Lm == 0
            continue
        end

        @views begin
            Xm[:, 1:k] .= X
            Ym[:, 1:k] .= Y
        end

        Im = k+1 : k+Lm
        @views T  = adjoint(X) * Xm[:, Im]      # k×Lm
        @views Px = Xm[:, Im] .- X * T
        @views Py = Ym[:, Im] .- Y * T

        # T = Px'Px
        T = adjoint(Px) * Px
        if Lm > 50
            dT = diag(T)
            sdT = sort(dT, rev = true)
            idx = sortperm(dT, rev = true)
            L = Int(sum(sdT .> 5e-8))
            if L < 0.95 * Lm
                Lm = L
                @views Icut = idx[1:Lm]
                @views Py = Py[:, Icut]
                @views T = T[Icut, Icut]
            end
        end
        TA = Symmetric(copy(T))
        evU = eigen(TA)
        e_tol = min(sqrt(eps(RT)), tol)
        ev, U = evU.values, evU.vectors
        cut = findfirst(>(e_tol), ev)

        if isnothing(cut)
            Lm = 0
            continue
        end

        L = Lm - cut + 1
        dv = 1 ./ sqrt.(ev[cut:end])
        T_1 = U[:, cut:end] * Diagonal(dv) 
        Yo = hcat(Y, copy(Py) * T_1)

        # T2 = Yo'Yo  ( (k+L)×(k+L) ) → CPU
        T2 = adjoint(Yo) * Yo
        T2 .= (T2 .+ T2') .* RT(0.5)
        DU2 = eigen(T2)
        D, U2 = DU2.values, DU2.vectors

        idx = size(T2,1) - k + 1 : size(T2,1)
        Y .= Yo * U2[:, idx]

        Lm = max(0, round(Int, L / k)) * k
        if iter < memo
            Lm += k
        end

        if Lm > 0
            @views begin
                Xm[:, k+1:k+Lm] .= Xm[:, 1:Lm]
                Ym[:, k+1:k+Lm] .= Ym[:, 1:Lm]
            end
        end

        rvr0 .= rvr
        rvr   .= D[end-r+1:end]
        chg_rvr = norm(rvr - rvr0) / max(norm(rvr), eps(RT))
    end

    return X, Y
end




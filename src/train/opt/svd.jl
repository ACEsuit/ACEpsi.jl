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

struct LMSVD{T} 
    U::Matrix{T}
    S::Vector{T}
    V::Matrix{T}
end

function lmsvd(A::Matrix{T}, r; X = nothing, tol = 1e-8, maxit = 10, memo = 3) where {T}
    m, n = size(A)
    if X == nothing
        Y = randn(n, min(2r, r + 10, m, n))
        X = A * Y;
    else
        Y = A' * X
        mul!(X, A, Y)
    end
    F = qr!(X)
    X = Matrix(F.Q)
    mul!(Y, A', X)
    
    # Call solver
    X, Y = lm_lbo(A, X, Y, r, tol, maxit, memo)
    # Generate SVD
    U, S, V = get_svd(X, Y)
    return LMSVD{T}(U[:, 1:r], S[1:r], V[:, 1:r])
end

function get_svd(X, Y)
    F = qr!(Y)
    W, S, Z = svd(transpose(F.R))
    U = X * W
    V = Matrix(F.Q) * Z
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
function lm_lbo(A, X::Matrix{RT}, Y, r, tol, maxit, memo) where {RT}
    m, n, k = size(X, 1), size(Y, 1), size(Y, 2)
    mn = min(m, n)
    Lm = k
    rvr = zeros(RT, r)
    rvr0 = zeros(RT, r)
    chg_rvr = one(RT)
    chgv, kktc, xtrm = zeros(RT, maxit), zeros(RT, maxit), zeros(RT, maxit);
    qtol = eps(RT)^min(mn/40/k,1)
    rtol = 5 * max(sqrt(tol*qtol), 5 * eps())
    ptol = 5 * max(tol,sqrt(eps()))

    if k < r
        error("Working size too small")
    end

    Xm = zeros(RT, (m, (1 + memo) * k)) 
    Ym = zeros(RT, (n, (1 + memo) * k)) 

    Xm[:, k+1:2*k] .= X # [AX0^i, ..., AX0^i-p]
    Ym[:, k+1:2*k] .= Y # [A'AX0^i, ..., A'AX0^i-p]

    for iter = 1:maxit
        SX = X
        # Subspace iteration
        AY = (A * Y)::Matrix{RT}
        F = qr(AY) 
        X = Matrix(F.Q)
        Y = A' * X 
        
        if Lm == 0 || iter <= 3
            SYTY = SX' * AY # X' * A * Y
            SYTY = Symmetric(((0.5*(SYTY+SYTY'))))::Symmetric{RT, Matrix{RT}}
            teigen = eigen!(SYTY)
            tU = teigen.vectors
            tE = teigen.values
            rvr0 = rvr; 
            rvr = tE[end - r + 1:end]
            chg_rvr = norm(rvr0 - rvr) / norm(rvr)
            AY = AY * tU
            SX = SX * tU
        end
        xtrm[iter] = Lm/k
        chgv[iter] = chg_rvr
        if chg_rvr < rtol
            kkt = AY[:, end - r + 1:end] .- SX[:, end - r + 1:end] .* rvr'
            # kkt = AY[:, end - r + 1:end] - (SX[:, end - r + 1:end]) * diagm(rvr)
            kktcheck = maximum(sqrt.(sum(kkt .^ 2, dims = 1))) / max(tol, rvr[end])
            if kktcheck < ptol
                break
            end
            kktc[iter] = kktcheck
        else
            if iter == 1; kktc[iter] = 0; else kktc[iter] = kktc[iter-1]; end
        end

        xtrm[iter] = Lm / k
        # Look-back optimization
        if Lm == 0
            continue
        end

        Xm[:, 1:k] .= X # [X^i, X^i-1, ..., X^i-p]
        Ym[:, 1:k] .= Y # [Y^i, Y^i-1, ..., Y^i-p]

        Im = k + 1:k + Lm 
        T = X' * Xm[:, Im]
        # Px = Xm[:, Im] - X * T
        Xm_sub = @view Xm[:, Im] 
        XT =  X * T
        Px = Xm_sub .- XT # Xm[:, Im] - X * X' * Xm[:, Im]
        
        # Py = Ym[:, Im]  - Y * T       # Ym[:, Im] - (A' * X) * (X' * Xm[:, Im])
        Ym_sub = @view Ym[:, Im]  
        YT = Y * T           
        Py = Ym_sub .- YT


        T = Px' * Px # Px' * Px

        if Lm > 50
            dT = diag(T)
            sdT = sort(dT, rev = true)
            idx = sortperm(dT, rev = true)
            L = Int(sum(sdT .> 5e-8))
            if L < 0.95 * Lm
                Lm = L
                Icut = idx[1:Lm]
                Py = Py[:, Icut]
                T = T[Icut, Icut]
            end
        end
        ev, U = eigen!(Symmetric(T)) # Px' * Px = U * Lambda * U'  
        e_tol = min(sqrt(eps()), tol)
        cut = findfirst(x -> x > e_tol, ev) # step 2 

        if isnothing(cut)
            Lm = 0
            continue
        end

        L = Lm - cut + 1
        dv = 1 ./ sqrt.(ev[cut:end]) # Lambda^{-1/2}
        T_1 = U[:, cut:end] * Diagonal(dv) # U * Lambda^{-1/2}
        Yo = hcat(Y, Py * T_1) # R = [Y^i, Py * U * Lambda^{-1/2}]
        T_2 = Symmetric(Yo' * Yo) # R' * R
        D, U = eigen!(T_2) # R' R = U D U' 

        Y = Yo * U[:, end - k + 1:end] # R * U
        Lm = max(0, round(Int, L / k)) * k
        if iter < memo
            Lm += k
        end

        if Lm > 0
            @views Xm[:, k + 1:k + Lm] .= Xm[:, 1:Lm]
            @views Ym[:, k + 1:k + Lm] .= Ym[:, 1:Lm]
        end

        rvr0 = rvr
        rvr = D[end - r + 1:end]
        chg_rvr = norm(rvr-rvr0)/norm(rvr);
    end
    return X, Y
end




using LinearAlgebra

"""
    ssisvd(A::Matrix{T}, r; X = nothing, maxit = 10) where {T}

Computes an approximate rank-`r` singular value decomposition (SVD) of the matrix `A` using a simple subspace iteration scheme.

# Arguments
- **A**: A matrix of type `Matrix{T}` whose SVD is to be approximated.
- **r**: The target rank for the truncated SVD.
- **X** (optional): An initial approximation matrix for the left singular subspace. If not provided, a random initialization is used.
- **maxit** (optional): Maximum number of iterations for the subspace iteration solver (default is 10).

# Returns
An `LMSVD{T}` object containing:
- **U**: Approximate left singular vectors.
- **S**: Approximate singular values (in descending order).
- **V**: Approximate right singular vectors.

# Method Overview
1. **Initialization**:
   - If no initial `X` is provided, a random matrix `Y` is generated with dimensions determined by `min(2r, r + 10, m, n)`, and `X` is computed as `A * Y`.
   - If an initial `X` is provided, `Y` is computed as `A' * X` and then `X` is updated to `A * Y`.
2. **Preprocessing**:
   - A QR factorization is applied to `X` to obtain an orthonormal basis.
   - `Y` is then updated as the product `A' * Q`, where `Q` is the orthonormal factor from the QR factorization.
3. **Subspace Iteration**:
   - The function calls `SSI`, which refines the approximations of `X` and `Y` via iterative updates.
4. **Final SVD Extraction**:
   - A QR factorization is performed on the refined `Y` to obtain an orthogonal matrix `Q` and an upper triangular matrix `R`.
   - The diagonal elements of `R` (after adjusting for sign) serve as singular value estimates.
   - The singular values are sorted in descending order and the corresponding columns from `X` and `Q` are selected as the left and right singular vectors, respectively.
   - The result is returned in an `LMSVD{T}` structure.

# Example
```julia
A = rand(100, 50)
svd_result = ssisvd(A, 10)
```
"""

function ssisvd(A::Matrix{T}, r; X = nothing, maxit = 10) where {T}
    m, n = size(A)
    if X == nothing
        Y = randn(n, min(2r, r + 10, m, n))
        X = A * Y;
    else
        Y = A' * X
        mul!(X, A, Y)
    end
    F = qr!(X)
    Y = A' * Matrix(F.Q)
    
    # Call solver
    X, Y = SSI(A, X, Y, maxit)

    #U, S, V = get_svd(X, Y)
    # Generate SVD
    F = qr!(Y) 
    R = diag(F.R)
    Q = Matrix(F.Q)
    sign_R = sign.(R)
    R .= abs.(R)
    sorted_indices = sortperm(R, rev = true)[1:r]
    X .*= sign_R'
    U = copy(X[:, sorted_indices])
    S = copy(R[sorted_indices])
    V = copy(Q[:, sorted_indices])
    return LMSVD{T}(U, S, V)
end

"""
    SSI(A, X::Matrix{RT}, Y, maxit; k = 3) where {RT}

Performs simple subspace iteration (SSI) of `AA'`.

# Arguments
- **A**: The input matrix, subspaces of AA' are to be approximated.
- **X**: A matrix representing an initial approximation of the subspace.
- **Y**: A container holding the product A'X.
- **maxit**: The maximum number of iterations to perform in the subspace iteration process.
- **k** (keyword, optional): The interval (in iterations) at which reorthogonalization of `X` is performed. Defaults to 3.

# Returns
A tuple `(X, Y)` where:
- **X**: The refined approximation of the subspace.
- **Y**: A'X.

# Method Overview
1. **Iterative Refinement**:
   - For each iteration from 1 to `maxit`, the algorithm updates `X` and `Y` alternately:
     - **Update X**: Compute the product `A * Y` and store the result in `X` using in-place multiplication.
     - **Reorthogonalization**: Every `k` iterations or at the final iteration, reorthogonalize `X` using QR factorization.
     - **Update Y**: Compute the product `A' * X` and store the result in `Y` using in-place multiplication.
2. **Return**: After `maxit` iterations, the refined matrices `X` and `Y` are returned.

This iterative process enhances the accuracy of the approximate subspaces, preparing them for subsequent SVD extraction steps.
"""
function SSI(A, X::Matrix{RT}, Y, maxit; k = 3) where {RT}
    for iter = 1:maxit
        mul!(X, A, Y)
        if iter % k == 0 || iter == maxit
            F = qr!(X) 
            X = Matrix(F.Q)
        end
        mul!(Y, A', X)
    end
    return X, Y
end

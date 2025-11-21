using SpecialFunctions     # erfc
using Optim                # Brent
using Printf
const π = Base.MathConstants.pi

# I2(α, β) = ∫_0^∞ r^2 * exp(-β r^2 - α r) dr
@inline function I2(α::T, β::T) where {T<:Real}
    if β <= eps(T)
        return zero(T)
    end
    t = α / (2*sqrt(β))
    term1 = (sqrt(π) * (α^2 + 2β) * erfcx(t)) / (8 * β^(T(5)/2))  # β^(2.5)
    term2 = α / (4 * β^2)
    return term1 - term2
end


# 〈G, e^{-α r}〉 with 4π r^2 weight
@inline function inner_GS(α, Drow::AbstractVector, ζrow::AbstractVector)
    T = promote_type(eltype(Drow), eltype(ζrow), typeof(α))
    s = zero(T)
    @inbounds @simd for m in eachindex(Drow, ζrow)
        β = ζrow[m]; Dm = Drow[m]
        if β > eps(T) && Dm != 0
            s += Dm * I2(α, β)
        end
    end
    return 4π * s
end

@inline normS2(α) = π/α^3
@inline Q(α, Drow, ζrow) = abs2(inner_GS(α, Drow, ζrow)) / normS2(α)


"""
    fit_one_slater(Drow, ζrow; α_lo, α_hi) -> (A*, α*)
"""
function fit_one_slater(Drow::AbstractVector, ζrow::AbstractVector; α_hi_factor::Real=50)
    mask = ζrow .> eps(eltype(ζrow))
    if !any(mask)
        return (zero(eltype(Drow)), one(eltype(ζrow)))
    end
    Dpos = view(Drow, mask); ζpos = view(ζrow, mask)

    βmin = minimum(ζpos)
    α_lo = max(eltype(βmin)(1e-6), 0.02*sqrt(βmin))
    α_hi = α_hi_factor * sqrt(βmin)
    obj(u) = -Q(exp(u), Dpos, ζpos)
    res = optimize(obj, log(α_lo), log(α_hi))
    ustar = Optim.minimizer(res)
    αstar = exp(ustar)
    Astar = inner_GS(αstar, Dpos, ζpos) / normS2(αstar)
    return Astar, αstar
end

"""
    fit_all_rows(D, ζ) -> (D_s, α_s)
"""
function fit_all_rows(D::AbstractMatrix, ζ::AbstractMatrix)
    N, K = size(D)
    D_s  = similar(D, N, 1)
    α_s  = similar(ζ, N, 1)
    @inbounds for n in 1:N
        D1 = view(D,n,:)
        ζ1 = view(ζ,n,:)
        D2 = D1[findall(x -> abs(x) > 0, D1)]
        ζ2 = ζ1[findall(x -> abs(x) > 0, D1)]
        A, α = fit_one_slater(D2, ζ2)
        D_s[n,1] = A
        α_s[n,1] = α
    end
    return D_s, α_s
end

# G(r) = Σ_m D[n,m] * exp(-ζ[n,m]*r^2)
@inline function eval_gauss_sum(r::Real, Drow, ζrow)
    s = zero(promote_type(eltype(Drow), eltype(ζrow), Float64))
    @inbounds @simd for m in eachindex(Drow, ζrow)
        s += Drow[m] * exp(-ζrow[m]*r*r)
    end
    return s
end

# Slater：S(r) = A * exp(-α r)
@inline eval_slater(r, A, α) = A * exp(-α*r)

# ∫ 4π r^2 (S−G)^2 dr
function weighted_rmse(Drow, ζrow, A, α; rmax=6.0, M=2000)
    rs = range(0.0, rmax; length=M)
    acc = 0.0
    for j in eachindex(rs)
        r = rs[j]
        g = eval_gauss_sum(r, Drow, ζrow)
        s = eval_slater(r, A, α)
        w = 4π*r*r
        acc += w * (s - g)^2
    end
    Δ = step(rs)
    val = acc * Δ
    return sqrt(val)
end

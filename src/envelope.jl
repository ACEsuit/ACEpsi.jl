export envelopefunc
using LinearAlgebra

"""
decay(x) = e^{-ξ⋅f(x)}
"""
mutable struct envelopefunc
    f::Function
    ξ::Float64
end
envelopefcn(f::Function, ξ::Float64) = envelopefunc(f, ξ)
(Φ::envelopefunc)(args...) = evaluate(Φ, args...)

function evaluate(U::envelopefunc, X::AbstractVector, args...)
    x = norm(X)
    val = exp(-U.ξ * U.f(x))
    return val
end

# -------- envelopefcn function parameter wraging --------

function set_params!(U::envelopefunc, c::Float64)
    U.ξ = c
    return U
 end

function set_params!(U::envelopefunc, c::AbstractVector)
    U.ξ = c[1]
    return U
 end

 
function gradp_env(U::envelopefunc, X::AbstractVector, args...)
    x = norm(X)
    return -evaluate(U, X)  * U.f(x)
end

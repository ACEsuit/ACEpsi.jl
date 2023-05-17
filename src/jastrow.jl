# This should probably be re-implemented properly and be made performant. 
# the current version is just for testing 
using ACEpsi.AtomicOrbitals: Nuc
using LuxCore: AbstractExplicitLayer
using Random: AbstractRNG

import ChainRules
import ChainRules: rrule, NoTangent
mutable struct Jastrow{T}
    nuclei::Vector{Nuc{T}}  # nuclei
end

(f::Jastrow)(args...) = evaluate(f, args...)

## F_2(x) = -1/2\sum_{l=1}^L \sum_{i=1}^N Z_l|yi,l|+1/4\sum_{1\leq i<j\leq N}|x_i-x_j|
## F_3(x) = C_0\sum_{l=1}^L  \sum_{1\leq i<j\leq N} Z_l (yil * yjl) * ln(|yil|^2+|yjl|^2)

function evaluate(f::Jastrow, X::AbstractVector, Σ) 
    nuc = f.nuclei
    Nnuc = length(nuc)
    Nel = size(X, 1)
    T = promote_type(eltype(X[1]))
    XN = zeros(T, (Nnuc, Nel))

    C0 = (2-pi)/(12*pi)
    F2 = zero(T)
    F3 = zero(T)

    # trans
    for I = 1:Nnuc, i = 1:Nel
        XN[I, i] = norm(X[i] - nuc[I].rr)
        F2 += -1/2 * nuc[I].charge * XN[I, i]
    end

    for i = 1:Nel-1, j = i+1:Nel
        F2 += 1/4 * norm(X[i] - X[j])
    end

    for i = 1:Nnuc, j = 1:Nel-1, k = j+1: Nel 
        F3 += C0 * nuc[i].charge * XN[i, j] * XN[i, k] * log(XN[i, j]^2 + XN[i, k]^2)
    end
    return exp(F2 + F3)
end


# --------------------- connect with Lux 

struct JastrowLayer <: AbstractExplicitLayer 
   basis::Jastrow
end

lux(basis::Jastrow) = JastrowLayer(basis)

initialparameters(rng::AbstractRNG, l::JastrowLayer) = _init_luxparams(rng, l.basis)

initialstates(rng::AbstractRNG, l::JastrowLayer) = _init_luxstate(rng, l.basis)


# This should be removed later and replace by ObejctPools
(l::JastrowLayer)(X, ps, st) = 
      evaluate(l.basis, X, st.Σ), st

function rrule(::typeof(evaluate), js::Jastrow, X::AbstractVector, Σ::AbstractVector) 
    J = evaluate(js, X, Σ) 
 
    function pb(dJ)
        @assert dJ isa Real  
        return NoTangent(), NoTangent(), X, NoTangent()
    end
    
    return J, pb 
 end 
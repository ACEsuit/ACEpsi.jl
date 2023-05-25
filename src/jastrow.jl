# This should probably be re-implemented properly and be made performant. 
# the current version is just for testing 
using ACEpsi.AtomicOrbitals: Nuc
using LuxCore: AbstractExplicitLayer
using Random: AbstractRNG

abstract type Jastrow end

(f::Jastrow)(args...) = evaluate(f, args...)

mutable struct JFournais{T}<:Jastrow
    nuclei::Vector{Nuc{T}}  # nuclei
end
# Fournais et al: https://link.springer.com/article/10.1007/s00220-004-1257-6
# F_2(x) = -1/2\sum_{l=1}^L \sum_{i=1}^N Z_l|yi,l|+1/4\sum_{1\leq i<j\leq N}|x_i-x_j|
# F_3(x) = C_0\sum_{l=1}^L  \sum_{1\leq i<j\leq N} Z_l (yil * yjl) * ln(|yil|^2+|yjl|^2)

# I think there is a mistake here^^^ 

function evaluate(f::JFournais, X::AbstractVector, Σ) 
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
        # this is probably a mistake? I think the orginal paper had a dot product
        F3 += C0 * nuc[i].charge * XN[i, j] * XN[i, k] * log(XN[i, j]^2 + XN[i, k]^2)
    end
    return exp(F2 + F3)
end

mutable struct JPauliNet{T}<:Jastrow{T}
    """ currently don't plan to mimic PauliNet's e-n cusp
    """
    nuclei::Vector{Nuc{T}}
end

## PauliNet
## γ(R) := ∑_{i<j} -c_{ij}/(1+|r_i-r_j|)
## c_{ij} = 1/2 if antiparallel, 1/4 if parallel

function evaluate(f::JPauliNet, X::AbstractVector, Σ)
    Nel = size(X, 1)
    T = promote_type(eltype(X[1]))
    γ = zero(T)
    for i = 1:Nel-1, j = i+1:Nel
        if Σ[i] != Σ[j] # anti-parallel
            γ += -(1/2) / (1+norm(X[i] - X[j]))
        else # parallel
            γ += -(1/4) / (1+norm(X[i] - X[j]))
        end
    end
end

# mutable struct YserentantAbs{T}<:Jastrow
#     """ note: we will include (or will we?) parallel-spin cusps, which Yserentant's paper did not consider
#     """
#     nuclei::Vector{Nuc{T}}
# end

# ## YserentantAbs: http://www.numdam.org/item/M2AN_2011__45_5_803_0/
# ## see ending remarks of Section 3
# ## 

# function evaluate()

# mutable struct YserentantLog{T}<:Jastrow
#     """ note: we will include parallel-spin cusps, which Yserentant's paper did not consider
#     """
#     nuclei::Vector{Nuc{T}}
# end

# ## YserentantLog: http://www.numdam.org/item/M2AN_2011__45_5_803_0/
# ## see ending remarks of Section 3
# ## 

# mutable struct JCASINO3B{T}
#     nuclei::Vector{Nuc{T}}
#     ord::Integer
# end

## CASINO 
##

function evaluate()
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

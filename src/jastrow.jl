# This should probably be re-implemented properly and be made performant. 
# the current version is just for testing 
using ACEpsi.AtomicOrbitals: Nuc
using LuxCore
using LuxCore: AbstractExplicitLayer
using Random: AbstractRNG
using Zygote: Buffer

mutable struct Jastrow{T}
    nuclei::Vector{Nuc{T}}  # nuclei
end

(f::Jastrow)(args...) = evaluate(f, args...)

## F_2(x) = -1/2\sum_{l=1}^L \sum_{i=1}^N Z_l|yi,l|+1/4\sum_{1\leq i<j\leq N}|x_i-x_j|
## F_3(x) = C_0\sum_{l=1}^L  \sum_{1\leq i<j\leq N} Z_l (yil * yjl) * ln(|yil|^2+|yjl|^2)

function evaluate(f::Jastrow, X::AbstractVector, Σ) 
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

    # trans

    return exp(γ)
end


# --------------------- connect with Lux 

struct JastrowLayer <: AbstractExplicitLayer 
   basis::Jastrow
end

lux(basis::Jastrow) = JastrowLayer(basis)

LuxCore.initialparameters(rng::AbstractRNG, l::JastrowLayer) = NamedTuple()

# This should be removed later and replace by ObejctPools
(l::JastrowLayer)(X, ps, st) = 
      evaluate(l.basis, X, st.Σ), st



## Bernie's code from Jastrow branch

mutable struct JPauliNet{T}
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
    return exp(γ)
end

struct JPauliNetLayer <: AbstractExplicitLayer 
    basis::JPauliNet
end
 
lux(basis::JPauliNet) = JPauliNetLayer(basis)
 
(l::JPauliNetLayer)(X, ps, st) = 
      evaluate(l.basis, X, st.Σ), st
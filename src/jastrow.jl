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

function evaluate(f::Jastrow, X::AbstractVector, Σ) 
    return 1.0
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

mutable struct JPauliNet{T}
    nuclei::Vector{Nuc{T}}
end

## PauliNet
## γ(R) := ∑_{i<j} -c_{ij}/(1+|r_i-r_j|)
## c_{ij} = 1/2 if antiparallel, 1/4 if parallel

function evaluate(f::JPauliNet, X::AbstractVector, Σ, b::Vector{<:TT}) where {TT}
    Nel = size(X, 1)
    T = promote_type(eltype(X[1]))
    γ = zero(T)
    for i = 1:Nel-1, j = i+1:Nel
        if Σ[i] != Σ[j] # anti-parallel
            γ -= 1/2 / (1 + norm(X[i] - X[j]))
        else # parallel
            γ -= 1/4 / (1 + norm(X[i] - X[j]))
        end
    end
    for i = 1:Nel, j = 1:length(f.nuclei)
        γ -= -2/ (1+norm(X[i] - f.nuclei[j].rr))
    end
    return exp(γ)
end

struct JPauliNetLayer <: AbstractExplicitLayer 
    basis::JPauliNet
end
 
lux(basis::JPauliNet) = JPauliNetLayer(basis)
 
LuxCore.initialparameters(rng::AbstractRNG, l::JPauliNetLayer) = (b = rand(3), )

(l::JPauliNetLayer)(X, ps, st) = 
      evaluate(l.basis, X, st.Σ, ps.b), st
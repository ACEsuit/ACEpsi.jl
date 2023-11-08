# This should probably be re-implemented properly and be made performant. 
# the current version is just for testing 
using ACEpsi.AtomicOrbitals: Nuc
using LuxCore
using LuxCore: AbstractExplicitLayer
using Random: AbstractRNG
using Zygote: Buffer
using LinearAlgebra
using Polynomials4ML: RTrigBasis, MonoBasis, LinearLayer

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



# === WIP: various JS factor from other architectures ===

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
    return 1.0
end

struct JPauliNetLayer <: AbstractExplicitLayer 
    basis::JPauliNet
end
 
lux(basis::JPauliNet) = JPauliNetLayer(basis)
 
(l::JPauliNetLayer)(X, ps, st) = 
      evaluate(l.basis, X, st.Σ), st


mutable struct JCasino1dVb{T}
    Lu::T # cutoff
    Nu::Int64
    Np::Int64
    L::T # cell-size
end

function _getXineqj(Xs)
    T = eltype(Xs[1])
    Nel = length(Xs)
    allXineqj = Zygote.Buffer(zeros(T, Nel * (Nel - 1)))
    idx = 0
    for j = 1:Nel
        for i = 1:Nel
            if i ≠ j
                idx += 1
                allXineqj[idx] = Xs[i][j]
            end
        end
    end
    @assert idx == Nel * (Nel - 1)
    return copy(allXineqj)
end

## CASINO trainable Jastrow for jellium
function JCasinoChain(J::JCasino1dVb)
    # coordinate trans
    L = J.L

    # cos 
    Np = J.Np
    l_trig = Polynomials4ML.lux(RTrigBasis(Np))
    getXineqj = WrappedFunction(Xs -> _getXineqj(Xs))

    #l_trigs = Tuple(collect(l_trig for _ = 2:2:2*Np+1)) 
    CosChain = Chain(; getXineqj_cos = getXineqj, SINCOS = l_trig, getcos = WrappedFunction(x -> x[:, 2:2:2*Np+1])) # picking out only cosines
    @assert length(2:2:2*Np+1) == Np

    # cusp?
    Nu = J.Nu
    l_mono = Polynomials4ML.lux(MonoBasis(Nu))
    # l_monos = Tuple(collect(l_mono for _ = 1:Nu))
    # MonoChain = Chain(; MONO = Lux.BranchLayer(l_monos...), hidden_J_Mono = LinearLayer(Np))
    
    # cut-off
    Lu = J.Lu
    
    ab_trans = Lux.WrappedFunction(x -> abs.(x)) # not sure if CASINO used absolute value or not to feed into monimials
    # Θ(x) = x < zero(eltype(x)) ? zero(eltype(x)) : one(eltype(x)) # Heaviside
    # approxθ(x) = 0.5 + 0.5 * tanh(10.0 * x)
    # cut = Lux.WrappedFunction(x -> Θ.(Lu .- x) .* ((x .- Lu) .^ 3))

    # cut = WrappedFunction(x -> (x .< Lu) .* ((x .- Lu) .^ 3))
    # we need to "unstransform" coordinates
    cusp_cut_Chain = Chain(; getXineqj_mono = getXineqj, abs = ab_trans , untrans = WrappedFunction(x -> x .* L ./ (2pi)), to_be_prod = l_mono)# , prod = WrappedFunction(x -> x[1] .* x[2]))

    return Chain(; combine = Lux.BranchLayer(CosChain, cusp_cut_Chain), pool_and_clean = WrappedFunction(x -> (hcat(sum(x[1], dims = 1), sum(x[2], dims = 1)))), hidden_J = LinearLayer(Np+Nu+1, 1))
end
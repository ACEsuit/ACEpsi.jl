import ForwardDiff
using StaticArrays
using Lux
using LuxCore: AbstractLuxLayer
using Random: AbstractRNG
using ChainRulesCore: NoTangent
import ChainRulesCore

# Diff layer
struct Diff_layer{Nnuc, T} <: AbstractLuxLayer 
   nuc::SVector{Nnuc, Nuc{Float64, T}}
end 

(l::Diff_layer)(X, ps, st) = evaluate(l, X, ps, st)

# jastrow layer
struct JastrowLayer <: AbstractLuxLayer 
    Σ::Vector{Char}
end

(l::JastrowLayer)(X, ps, st) = evaluate(l, X, l.Σ), ps, st

# masklayer
struct MaskLayer <: AbstractLuxLayer
    nX::Int64
    Σ::Vector{Char}
end

(l::MaskLayer)(Φ, ps, st) = begin 
    T = eltype(Φ)
    A::Matrix{Bool} = [l.Σ[i] == l.Σ[j] for j = 1:l.nX, i = 1:l.nX] 
    val::Matrix{T} = Matrix(Φ) .* A
    return val, st
end

struct BackflowPoolingLayer{TT}<: AbstractLuxLayer
    spec::Vector{TT}
    Σ::Vector{Char}
end

(l::BackflowPoolingLayer)(x, ps, st) = evaluate(l, x, l.Σ), ps, st

function evaluate(l::Diff_layer{Nnuc}, X::Vector{SVector{3, T}}, ps, st) where {Nnuc, T}
   return ntuple(i -> _getdiff(X, l.nuc[i].rr), Val(Nnuc)), st
end

function evaluate(jl::JastrowLayer, X::Vector{SVector{3, TX}}, Σ::Vector{Char}) where {TX}
    N = length(X)
    γ = zero(TX)
    for i in 1:N-1, j in i+1:N
        dist = norm(X[i] - X[j])

        c_ij = Σ[i] == Σ[j] ? 0.25 : 0.5

        γ += -c_ij / (1 + dist)
    end
    return exp(γ)
end

struct BackflowPoolingLayer_TD{TT}<: AbstractLuxLayer
    spec::Vector{TT}
    Σ::Vector{Char}
end

(l::BackflowPoolingLayer_TD)(x, ps, st) = evaluate(l, x, l.Σ), ps, st

function evaluate(l::BackflowPoolingLayer, x, Σ::Vector{Char})
    T = promote_type(eltype(x[1]))
    Nnlm = length(l.spec)
    Nel = length(Σ)

    @assert spin2idx(↑) == 1
    @assert spin2idx(↓) == 2
    @assert spin2idx(∅) == 3

    # Flattened arrays
    A = zeros(T, Nel, 3 * Nnlm)      # A[i, 3(k-1)+j] where j=1:3 (∅, ↑, ↓)
    Aall = zeros(T, 2 * Nnlm)        # Aall[2(k-1)+iσ] where iσ=1:2 (↑, ↓)

    @inbounds begin
        for k = 1:Nnlm
            @simd ivdep for i = 1:Nel
                iσ = spin2idx(Σ[i])
                Aall[2*(k - 1) + iσ] += x[i, k]                      # accumulate for ↑, ↓
                A[i, 3*(k - 1) + 3] = x[i, k]                        # j = 3 ↔ ∅ channel
            end
        end

        for k = 1:Nnlm
            @simd ivdep for iσ = 1:2
                σ = idx2spin(iσ)
                for i = 1:Nel
                    c = 3*(k - 1) + iσ                               # j = iσ ∈ {1,2}
                    A[i, c] = Aall[2*(k - 1) + iσ] - (Σ[i] == σ) * x[i, k]
                end
            end
        end
    end

    return A  # shape: (Nel, 3 * Nnlm)
end

function evaluate(l::BackflowPoolingLayer_TD, x, Σ::Vector{Char})
    T = promote_type(eltype(x[1]))
    Nnlm = length(l.spec)
    Nel = length(Σ)

    @assert spin2idx(↑) == 1
    @assert spin2idx(↓) == 2
    @assert spin2idx(∅) == 3

    A = zeros(T, Nel, 3, Nnlm)       # (Nel, 3 channel, Nnlm)
    Aall = zeros(T, 2, Nnlm)         # (spin channel ∈ {↑, ↓}, Nnlm)

    @inbounds begin
        for k = 1:Nnlm
            @simd ivdep for i = 1:Nel
                iσ = spin2idx(Σ[i])
                if iσ ≤ 2
                    Aall[iσ, k] += x[i, k]
                end
                A[i, 3, k] = x[i, k]
            end
        end

        for k = 1:Nnlm
            @simd ivdep for iσ = 1:2
                σ = idx2spin(iσ)
                for i = 1:Nel
                    A[i, iσ, k] = Aall[iσ, k] - (Σ[i] == σ ? x[i, k] : zero(T))
                end
            end
        end
    end

    return A  # shape: (Nel, 3, Nnlm)
end


function ChainRulesCore.rrule(::typeof(evaluate), l::Diff_layer{Nnuc}, X::Vector{SVector{3, TX}}, ps::NamedTuple, st::NamedTuple) where {Nnuc, TX}
   val = ntuple(i -> _getdiff(X, l.nuc[i].rr), Val(Nnuc))
   function pb(dA)
      return NoTangent(), NoTangent(), sum(dA[1]), NoTangent(), NoTangent()
   end
   return (val, st), pb
end

function ChainRulesCore.rrule(::typeof(Lux.apply), l::MaskLayer, Φ, ps, st) 
    T = eltype(Φ)
    A::Matrix{Bool} = [l.Σ[i] == l.Σ[j] for j = 1:l.nX, i = 1:l.nX]
    val::Matrix{T} = Matrix(Φ) .* A
    function pb(dΦ)
       return NoTangent(), NoTangent(), dΦ[1] .* A, NoTangent(), NoTangent()
    end
    return (val, st), pb
end
 
function ChainRulesCore.rrule(::typeof(evaluate), pooling::BackflowPoolingLayer, x, Σ::Vector{Char}) 
    A = evaluate(pooling, x, pooling.Σ)
    function pb(∂A)
        return NoTangent(), NoTangent(), _pullback_evaluate(∂A, pooling, x, Σ), NoTangent()
    end
    return A, pb
end 

function ChainRulesCore.rrule(::typeof(evaluate), pooling::BackflowPoolingLayer_TD, x, Σ::Vector{Char}) 
    A = evaluate(pooling, x, pooling.Σ)
    function pb(∂A)
        return NoTangent(), NoTangent(), _pullback_evaluate(∂A, pooling, x, Σ), NoTangent()
    end
    return A, pb
end 

# helper function

function _getdiff(X::AbstractArray{SVector{3, T}}, d::SVector{3, TT}) where {T, TT}
   result = similar(X)
   return _getdiff!(result, X, d)
end

function _getdiff!(result::AbstractArray{SVector{3, T}}, X::AbstractArray{SVector{3, TX}}, d::SVector{3, TD}) where {T, TX, TD}
   @inbounds for i in eachindex(X)
      result[i] = X[i] .- d
   end
   return result
end

function _pullback_evaluate(∂A, l::BackflowPoolingLayer, x, Σ::Vector{Char})
    TA = eltype(x[1])
    Nel = length(Σ)
    Nnlm = length(l.spec)

    ∂x = zeros(TA, Nel, Nnlm)

    @inbounds begin
        for k = 1:Nnlm
            for i = 1:Nel
                # j = 3 is ∅ channel
                ∂x[i, k] += ∂A[i, 3*(k - 1) + 3]
                @simd for j = 1:Nel
                    iσ = spin2idx(Σ[i])
                    ∂x[i, k] += ∂A[j, 3*(k - 1) + iσ] * (j != i)
                end
            end
        end
    end

    return ∂x
end

function _pullback_evaluate(∂A, l::BackflowPoolingLayer_TD, x, Σ::Vector{Char})
    TA = eltype(x[1])
    Nel = length(Σ)
    Nnlm = length(l.spec)

    ∂x = zeros(TA, Nel, Nnlm)

    @inbounds begin
        for k = 1:Nnlm
            for i = 1:Nel
                σi = Σ[i]
                iσ = spin2idx(σi)

                ∂x[i, k] += ∂A[i, 3, k]

                if iσ ≤ 2
                    for j = 1:Nel
                        if j != i && Σ[j] == σi
                            ∂x[i, k] += ∂A[j, iσ, k]
                        end
                    end
                end
            end
        end
    end

    return ∂x
end


function LuxCore.initialparameters(rng::AbstractRNG, d::Dense)
    weight = if d.init_weight === nothing
        Lux.kaiming_uniform(
            rng,
            Float64,
            d.out_dims,
            d.in_dims;
            gain= Lux.Utils.calculate_gain(d.activation, √5.0f0),
        )
    else
        d.init_weight(rng, d.out_dims, d.in_dims)
    end
    Lux.has_bias(d) || return (; weight)
    return (; weight, bias=Lux.init_linear_bias(rng, d.init_bias, d.in_dims, d.out_dims))
end
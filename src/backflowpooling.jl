using ACEpsi.AtomicOrbitals: AtomicOrbitalsBasisLayer
using LuxCore: AbstractExplicitLayer
using Random: AbstractRNG
using ChainRulesCore
using ChainRulesCore: NoTangent
using Polynomials4ML: _make_reqfields, @reqfields, POOL, TMP, META
using ObjectPools: acquire!

mutable struct BackflowPooling
   basis::AtomicOrbitalsBasisLayer
   @reqfields
end

function BackflowPooling(basis::AtomicOrbitalsBasisLayer)
   return BackflowPooling(basis, _make_reqfields()...)
end

(pooling::BackflowPooling)(args...) = evaluate(pooling, args...)

Base.length(pooling::BackflowPooling) = 3 * length(pooling.basis) # length(spin()) * length(1pbasis)
function evaluate(pooling::BackflowPooling, ϕnlm::Vector{TI}, Σ::AbstractVector) where {TI}
   Nel = length(Σ)
   Nnuc = length(ϕnlm)
   T = promote_type(eltype(ϕnlm[1]))
   Nnlm = size.(ϕnlm, 2)

   # evaluate the pooling operation
   #                spin  I    k = (nlm)

   Aall = acquire!(pooling.tmp, :Aall, (2, sum(Nnlm)), T)
   fill!(Aall, zero(T))
   @inbounds begin
      for I = 1:Nnuc 
         for k = 1:Nnlm[I]
            @simd ivdep for i = 1:Nel
               iσ = spin2idx(Σ[i])
               Aall[iσ, _ind(I, k, Nnlm)] += ϕnlm[I][i, k]
            end
         end
      end
   end # inbounds

   # now correct the pooling Aall and write into A^(i)
   # with do it with i leading so that the N-correlations can 
   # be parallelized over i 
   #
   # A[i, :] = A-basis for electron i, with channels, s, I, k=nlm 
   # A[i, ∅, I, k] = ϕnlm[I, i, k]
   # for σ = ↑ or ↓ we have
   # A[i, σ, I, k] = ∑_{j ≂̸ i : Σ[j] == σ}  ϕnlm[I, j, k]
   #               = ∑_{j : Σ[j] == σ}  ϕnlm[I, j, k] - (Σ[i] == σ) * ϕnlm[I, i, k]
   #
   #
   # TODO: discuss - this could be stored much more efficiently as a 
   #       lazy array. Could that have advantages? 
   #

   @assert spin2idx(↑) == 1
   @assert spin2idx(↓) == 2
   @assert spin2idx(∅) == 3

   A = acquire!(pooling.pool, :A, (Nel, 3, sum(Nnlm)), T)
   fill!(A, zero(T))

   @inbounds begin
      for I = 1:Nnuc 
         for k = 1:Nnlm[I]
            @simd ivdep for i = 1:Nel             
               A[i, 3, _ind(I, k, Nnlm)] = ϕnlm[I][i, k]
            end
            @simd ivdep for iσ = 1:2
               σ = idx2spin(iσ)
               for i = 1:Nel
                  A[i, iσ, _ind(I, k, Nnlm)] = Aall[iσ, _ind(I, k, Nnlm)] - (Σ[i] == σ) * ϕnlm[I][i, k]
               end
            end
         end
      end
   end # inbounds
   release!(Aall)
   
   return A
end

function _ind(ii::Integer, k::Integer, Nnlm::Vector{TI}) where {TI} 
   return sum(Nnlm[1:ii-1]) + k
end
# --------------------- connect with ChainRule
function ChainRulesCore.rrule(::typeof(evaluate), pooling::BackflowPooling, ϕnlm, Σ::AbstractVector) 
   A = pooling(ϕnlm, Σ)
   function pb(∂A)
      return NoTangent(), NoTangent(), _pullback_evaluate(∂A, pooling, ϕnlm, Σ), NoTangent()
   end
   return A, pb
end 

function _rrule_evaluate(pooling::BackflowPooling, ϕnlm, Σ)
   A = pooling(ϕnlm, Σ)
   return A, ∂A -> _pullback_evaluate(∂A, pooling, ϕnlm, Σ)
end

function _pullback_evaluate(∂A, pooling::BackflowPooling, ϕnlm::Vector{TI}, Σ::AbstractVector) where {TI}
   TA = eltype(ϕnlm[1])
   ∂ϕnlm = [acquire!(pooling.pool, Symbol("∂ϕnlm$i"), size(ϕnlm[i]), TA) for i = 1:length(ϕnlm)]
   for j = 1:length(ϕnlm) 
      fill!(∂ϕnlm[j], zero(TA)) 
   end
   _pullback_evaluate!(∂ϕnlm, ∂A, pooling, ϕnlm, Σ)
   return ∂ϕnlm
end

function _pullback_evaluate!(∂ϕnlm, ∂A, pooling::BackflowPooling, ϕnlm::Vector{TI}, Σ) where {TI}
   Nel, Nnuc, Nnlm = size(ϕnlm[1], 1), length(ϕnlm), size.(ϕnlm, 2)
   for I = 1:Nnuc
      for i = 1:Nel
         for k = 1:Nnlm[I]
            ∂ϕnlm[I][i, k] += ∂A[i, 3, _ind(I, k, Nnlm)]
            for ii = 1:Nel
               ∂ϕnlm[I][i, k] += ∂A[ii, spin2idx(Σ[i]), _ind(I, k, Nnlm)] .* (i != ii)
            end
         end
      end
   end
   return nothing 
end
# --------------------- connect with Lux 

struct BackflowPoolingLayer <: AbstractExplicitLayer 
   basis::BackflowPooling
end

lux(basis::BackflowPooling) = BackflowPoolingLayer(basis)

initialparameters(rng::AbstractRNG, l::BackflowPoolingLayer) = _init_luxparams(rng, l.basis)
initialstates(rng::AbstractRNG, l::BackflowPoolingLayer) = _init_luxstate(rng, l.basis)


# This should be removed later and replace by ObejctPools
(l::BackflowPoolingLayer)(ϕnlm, ps, st) = 
      evaluate(l.basis, ϕnlm, st.Σ), st

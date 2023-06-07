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

function evaluate(pooling::BackflowPooling, ϕnlm::AbstractArray, Σ::AbstractVector)   
   Nnuc, _, Nnlm = size(ϕnlm)
   Nel = length(Σ)
   T = promote_type(eltype(ϕnlm))

    # evaluate the pooling operation
   #                spin  I    k = (nlm) 

   Aall = acquire!(pooling.pool, :Aall, (2, Nnuc, Nnlm), T)
   fill!(Aall, 0)

   @inbounds begin
      for k = 1:Nnlm
         for I = 1:Nnuc 
            @simd ivdep for i = 1:Nel
               iσ = spin2idx(Σ[i])
               Aall[iσ, I, k] += ϕnlm[I, i, k]
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

   A = acquire!(pooling.tmp, :Aall, (Nel, 3, Nnuc, Nnlm), T)
   fill!(A, 0)

   @inbounds begin
      for k = 1:Nnlm
         for I = 1:Nnuc 
            @simd ivdep for i = 1:Nel             
               A[i, 3, I, k] = ϕnlm[I, i, k]
            end
            @simd ivdep for iσ = 1:2
               σ = idx2spin(iσ)
               for i = 1:Nel
                  A[i, iσ, I, k] = Aall[iσ, I, k] - (Σ[i] == σ) * ϕnlm[I, i, k]
               end
            end
         end
      end
   end # inbounds

   release!(Aall)
   
   return A
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

function _pullback_evaluate(∂A, pooling::BackflowPooling, ϕnlm, Σ)
   TA = promote_type(eltype.(ϕnlm)...)
   ∂ϕnlm = zeros(TA, size(ϕnlm))
   _pullback_evaluate!(∂ϕnlm, ∂A, pooling, ϕnlm, Σ)
   return ∂ϕnlm
end


function _pullback_evaluate!(∂ϕnlm, ∂A, pooling::BackflowPooling, ϕnlm, Σ)
   Nnuc, Nel, Nnlm = size(ϕnlm)
   basis = pooling.basis

   @assert Nnlm == length(basis.prodbasis.layers.ϕnlms.basis.spec)
   @assert Nel == length(Σ)
   @assert size(∂ϕnlm) == (Nnuc, Nel, Nnlm)
   @assert size(∂A) == (Nel, 3, Nnuc, Nnlm)

   for I = 1:Nnuc
      for i = 1:Nel
         for k = 1:Nnlm
            ∂ϕnlm[I, i, k] += ∂A[i, 3, I, k]
            for ii = 1:Nel
               ∂ϕnlm[I, i, k] += ∂A[ii, spin2idx(Σ[i]), I, k] .* (i != ii)
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


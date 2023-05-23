using ACEpsi.AtomicOrbitals: AtomicOrbitalsBasis
using LuxCore: AbstractExplicitLayer
using Random: AbstractRNG
using ChainRulesCore
using ChainRulesCore: NoTangent

mutable struct BackflowPooling
   basis::AtomicOrbitalsBasis
end

(pooling::BackflowPooling)(args...) = evaluate(pooling, args...)

Base.length(pooling::BackflowPooling) = 3 * length(pooling.basis) # length(spin()) * length(1pbasis)

function evaluate(pooling::BackflowPooling, ϕnlm, Σ::AbstractVector)
   basis = pooling.basis
   nuc = basis.nuclei 
   Nnuc = length(nuc)
   Nnlm = length(basis.prodbasis.sparsebasis.spec) 
   Nel = length(Σ)
   T = promote_type(eltype(ϕnlm))

    # evaluate the pooling operation
   #                spin  I    k = (nlm) 

   Aall = zeros(T, (2, Nnuc, Nnlm))
   for k = 1:Nnlm
      for i = 1:Nel 
         iσ = spin2idx(Σ[i])
         for I = 1:Nnuc 
            Aall[iσ, I, k] += ϕnlm[I, i, k]
         end
      end
   end

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
   A = zeros(T, ((Nel, 3, Nnuc, Nnlm)))
   for k = 1:Nnlm 
      for I = 1:Nnuc 
         for i = 1:Nel             
            A[i, 3, I, k] = ϕnlm[I, i, k]
         end
         for iσ = 1:2 
            σ = idx2spin(iσ)
            for i = 1:Nel 
               A[i, iσ, I, k] = Aall[iσ, I, k] - (Σ[i] == σ) * ϕnlm[I, i, k]
            end
         end
      end
   end

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

   @assert Nnlm == length(basis.prodbasis.spec1)
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

# ----- ObejctPools
# (l::BackflowPoolingLayer)(args...) = evaluate(l, args...)

# function evaluate(l::BackflowPoolingLayer, ϕnlm_Σ::SINGLE, ps, st)
#    B = acquire!(st.cache, :B, (length(l.basis), ), _valtype(l.basis, x))
#    evaluate!(parent(B), l.basis, ϕnlm_Σ[1], ϕnlm_Σ[2])
#    return B 
# end 


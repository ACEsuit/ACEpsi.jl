using ACEpsi.AtomicOrbitals: AtomicOrbitalsBasisLayer
using LuxCore: AbstractExplicitLayer
using Random: AbstractRNG
using ChainRulesCore
using ChainRulesCore: NoTangent
using Polynomials4ML: _make_reqfields, @reqfields, POOL, TMP, META
using ObjectPools: acquire!

mutable struct BackflowPooling1d
   @reqfields
end

function BackflowPooling1d()
   return BackflowPooling1d(_make_reqfields()...)
end

(pooling::BackflowPooling1d)(args...) = evaluate(pooling, args...)

Base.length(pooling::BackflowPooling1d) = 3 * length(pooling.basis) # length(spin()) * length(1pbasis)

function evaluate(pooling::BackflowPooling1d, ϕnlm::AbstractArray, Σ::AbstractVector)   
   Nel, n = size(ϕnlm)
   T = promote_type(eltype(ϕnlm))

    # evaluate the pooling operation
   #                spin  I    k = (nlm)
   Aall = acquire!(pooling.tmp, :Aall, (2, n), T) # 2 * n
   fill!(Aall, zero(T))

   @inbounds begin
      for k = 1:n
        @simd ivdep for i = 1:Nel
            iσ = spin2idx(Σ[i])
            Aall[iσ, k] += ϕnlm[i, k]
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

   A = acquire!(pooling.pool, :Aall, (Nel, 3, n), T) # Nel, 3, n
   fill!(A, zero(T))

   @inbounds begin
      for k = 1:n
            @simd ivdep for i = 1:Nel             
               A[i, 3, k] = ϕnlm[i, k]
            end
            @simd ivdep for iσ = 1:2
               σ = idx2spin(iσ)
               for i = 1:Nel
                  A[i, iσ, k] = Aall[iσ, k] - (Σ[i] == σ) * ϕnlm[i, k]
               end
            end
      end
   end # inbounds

   release!(Aall)
   
   return A
end

# --------------------- connect with ChainRule
function ChainRulesCore.rrule(::typeof(evaluate), pooling::BackflowPooling1d, ϕnlm, Σ::AbstractVector) 
   A = pooling(ϕnlm, Σ)
   function pb(∂A)
      return NoTangent(), NoTangent(), _pullback_evaluate(∂A, pooling, ϕnlm, Σ), NoTangent()
   end
   return A, pb
end 

function _rrule_evaluate(pooling::BackflowPooling1d, ϕnlm, Σ)
   A = pooling(ϕnlm, Σ)
   return A, ∂A -> _pullback_evaluate(∂A, pooling, ϕnlm, Σ)
end

function _pullback_evaluate(∂A, pooling::BackflowPooling1d, ϕnlm, Σ)
   TA = eltype(ϕnlm)
   ∂ϕnlm = acquire!(pooling.pool, :∂ϕnlm, size(ϕnlm), TA)
   fill!(∂ϕnlm, zero(TA))
   _pullback_evaluate!(∂ϕnlm, ∂A, pooling, ϕnlm, Σ)
   return ∂ϕnlm
end


function _pullback_evaluate!(∂ϕnlm, ∂A, pooling::BackflowPooling1d, ϕnlm, Σ)
   Nel, Nnlm = size(ϕnlm)
   #basis = pooling.basis

   #@assert Nnlm == length(basis.prodbasis.layers.ϕnlms.basis.spec)
   @assert Nel == length(Σ)
   @assert size(∂ϕnlm) == (Nel, Nnlm)
   @assert size(∂A) == (Nel, 3, Nnlm)

      for i = 1:Nel
         for k = 1:Nnlm
            ∂ϕnlm[i, k] += ∂A[i, 3, k]
            for ii = 1:Nel
               ∂ϕnlm[i, k] += ∂A[ii, spin2idx(Σ[i]), k] .* (i != ii)
            end
         end
      end
 
   return nothing 
end


# --------------------- connect with Lux 

struct BackflowPooling1dLayer <: AbstractExplicitLayer 
   basis::BackflowPooling1d
end

lux(basis::BackflowPooling1d) = BackflowPooling1dLayer(basis)

initialparameters(rng::AbstractRNG, l::BackflowPooling1dLayer) = _init_luxparams(rng, l.basis)

initialstates(rng::AbstractRNG, l::BackflowPooling1dLayer) = _init_luxstate(rng, l.basis)


# This should be removed later and replace by ObejctPools
(l::BackflowPooling1dLayer)(ϕnlm, ps, st) = 
      evaluate(l.basis, ϕnlm, st.Σ), st

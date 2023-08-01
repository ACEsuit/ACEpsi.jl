using ACEpsi.AtomicOrbitals: AtomicOrbitalsBasisLayer
using LuxCore: AbstractExplicitLayer
using Random: AbstractRNG
using ChainRulesCore
using ChainRulesCore: NoTangent
using Polynomials4ML: _make_reqfields, @reqfields, POOL, TMP, META
using ObjectPools: acquire!

mutable struct BackflowPooling1dps
   @reqfields
end

function BackflowPooling1dps()
   return BackflowPooling1dps(_make_reqfields()...)
end

(pooling::BackflowPooling1dps)(args...) = evaluate(pooling, args...)

Base.length(pooling::BackflowPooling1dps) = 3 * length(pooling.basis) # length(spin()) * length(1pbasis)

function evaluate(pooling::BackflowPooling1dps, Ps, Σ::AbstractVector)   

   Nel, n = size(Ps[1])
   T = promote_type(eltype(Ps[1]))

    # evaluate the pooling operation
   #                spin  I    k = (nlm)
   Aall = [acquire!(pooling.tmp, Symbol("Aall$i"), (2, n), T) for i in 1:Nel] # Nel * 2 * n
   for j = 1:Nel
      fill!(Aall[j], zero(T))
   end

   @inbounds begin
      for j = 1:Nel
         for k = 1:n
            @simd ivdep for i = 1:Nel
                  iσ = spin2idx(Σ[i])
                  Aall[j][iσ, k] += Ps[j][i, k]
            end
         end
      end
   end # inbounds

   # now correct the pooling Aall and write into A^(i)
   # with do it with i leading so that the N-correlations can 
   # be parallelized over i 
   #
   # A[i, :] = A-basis for electron i, with channels, s, I, k=nlm 
   # A[i, ∅, I, k] = Ps[I, i, k]
   # for σ = ↑ or ↓ we have
   # A[i, σ, I, k] = ∑_{j ≂̸ i : Σ[j] == σ}  Ps[I, j, k]
   #               = ∑_{j : Σ[j] == σ}  Ps[I, j, k] - (Σ[i] == σ) * Ps[I, i, k]
   #
   #
   # TODO: discuss - this could be stored much more efficiently as a 
   #       lazy array. Could that have advantages? 
   #

   @assert spin2idx(↑) == 1
   @assert spin2idx(↓) == 2
   @assert spin2idx(∅) == 3

   A = acquire!(pooling.pool, :A, (Nel, 3, n), T) # Nel, 3, n
   fill!(A, zero(T))

   @inbounds begin
      for k = 1:n
            @simd ivdep for i = 1:Nel             
               A[i, 3, k] = Ps[i][i, k]
            end
            @simd ivdep for iσ = 1:2
               σ = idx2spin(iσ)
               for i = 1:Nel
                  A[i, iσ, k] = Aall[i][iσ, k] - (Σ[i] == σ) * Ps[i][i, k]
               end
            end
      end
   end # inbounds

   release!(Aall)
   
   return A
end

# --------------------- connect with ChainRule
function ChainRulesCore.rrule(::typeof(evaluate), pooling::BackflowPooling1dps, Ps, Σ::AbstractVector) 
   A = pooling(Ps, Σ)
   function pb(∂A)
      return NoTangent(), NoTangent(), _pullback_evaluate(∂A, pooling, Ps, Σ), NoTangent()
   end
   return A, pb
end 

# function _rrule_evaluate(pooling::BackflowPooling1dps, Ps, Σ)
#    A = pooling(Ps, Σ)
#    return A, ∂A -> _pullback_evaluate(∂A, pooling, Ps, Σ)
# end

function _pullback_evaluate(∂A, pooling::BackflowPooling1dps, Ps, Σ)
   TA = eltype(Ps[1])
   ∂Ps = Tuple([acquire!(pooling.pool, Symbol("∂Ps$i"), size(Ps[1]), TA) for i = 1:length(Σ)])
   for j = 1:length(Σ)
      fill!(∂Ps[j], zero(TA))
   end
   _pullback_evaluate!(∂Ps, ∂A, pooling, Ps, Σ)
   return ∂Ps
end


function _pullback_evaluate!(∂Ps, ∂A, pooling::BackflowPooling1dps, Ps, Σ)
   Nel, Nnlm = size(Ps[1])
   #basis = pooling.basis

   #@assert Nnlm == length(basis.prodbasis.layers.Pss.basis.spec)
   @assert Nel == length(Σ)
   @assert size(∂Ps[1]) == (Nel, Nnlm)
   @assert size(∂A) == (Nel, 3, Nnlm)

   
   for i = 1:Nel
      for k = 1:Nnlm
         ∂Ps[i][i, k] += ∂A[i, 3, k]
         for ii = 1:Nel
            ∂Ps[ii][i, k] += ∂A[ii, spin2idx(Σ[i]), k] .* (i != ii)
         end
      end
   end
 
   return nothing 
end


# # --------------------- connect with Lux 

struct BackflowPooling1dpsLayer <: AbstractExplicitLayer 
   basis::BackflowPooling1dps
end

lux(basis::BackflowPooling1dps) = BackflowPooling1dpsLayer(basis)

initialparameters(rng::AbstractRNG, l::BackflowPooling1dpsLayer) = _init_luxparams(rng, l.basis)

initialstates(rng::AbstractRNG, l::BackflowPooling1dpsLayer) = _init_luxstate(rng, l.basis)


# This should be removed later and replace by ObejctPools
(l::BackflowPooling1dpsLayer)(Ps, ps, st) = 
      evaluate(l.basis, Ps, st.Σ), st

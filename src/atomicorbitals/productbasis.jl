using Lux: WrappedFunction
using Lux
using Polynomials4ML: SparseProduct, AbstractPoly4MLBasis, release!
import LuxCore 
import LuxCore: initialparameters, initialstates, AbstractExplicitLayer
using Random: AbstractRNG

function _invmap(a::AbstractVector)
   inva = Dict{eltype(a), Int}()
   for i = 1:length(a) 
      inva[a[i]] = i 
   end
   return inva 
end

function dropnames(namedtuple::NamedTuple, names::Tuple{Vararg{Symbol}}) 
   keepnames = Base.diff_names(Base._nt_names(namedtuple), names)
   return NamedTuple{keepnames}(namedtuple)
end
struct ProductBasis_ATOLayer <: AbstractExplicitLayer 
   sparsebasis::SparseProduct
   bRnl::AbstractPoly4MLBasis
   bYlm::AbstractPoly4MLBasis
   L::Int
end

struct ProductBasis_STOLayer <: AbstractExplicitLayer 
   sparsebasis::SparseProduct
   bRnl::AbstractPoly4MLBasis
   bYlm::AbstractPoly4MLBasis
   L::Int
end

ProductBasisLayer(spec1::Vector,bRnl::AbstractPoly4MLBasis,bYlm::AbstractPoly4MLBasis) = begin
   spec1idx = Vector{Tuple{Int, Int}}(undef, length(spec1))
   spec_Rnl = natural_indices(bRnl); inv_Rnl = _invmap(spec_Rnl)
   spec_Ylm = natural_indices(bYlm); inv_Ylm = _invmap(spec_Ylm)
 
   spec1idx = Vector{Tuple{Int, Int}}(undef, length(spec1))
   for (i, b) in enumerate(spec1)
      spec1idx[i] = (inv_Rnl[dropnames(b,(:m,))], inv_Ylm[(l=b.l, m=b.m)])
   end
   sparsebasis = SparseProduct(spec1idx)

   if bRnl.Dn isa Union{Polynomials4ML.GaussianBasis, Polynomials4ML.SlaterBasis}
      return ProductBasis_ATOLayer(sparsebasis, bRnl, bYlm, length(spec1))
   elseif bRnl.Dn isa Polynomials4ML.STO_NG
      return ProductBasis_STOLayer(sparsebasis, bRnl, bYlm, length(spec1))
   end
end

initialparameters(rng::AbstractRNG, l::ProductBasis_ATOLayer) = ( ζ = l.bRnl.Dn.ζ, )
initialparameters(rng::AbstractRNG, l::ProductBasis_STOLayer) = NamedTuple()

initialstates(rng::AbstractRNG, l::ProductBasis_ATOLayer) = NamedTuple()
initialstates(rng::AbstractRNG, l::ProductBasis_STOLayer) = (ζ = l.bRnl.Dn.ζ, )

function evaluate(l::ProductBasis_ATOLayer, X::Vector{SVector{3, T}}, ps, st) where {T}
   R = norm.(X)
   l.bRnl.Dn.ζ = ps[1] 
   _bRnl = evaluate(l.bRnl, R)
   _Ylm = evaluate(l.bYlm, X)
   _ϕnlm = evaluate(l.sparsebasis,(_bRnl, _Ylm))
   release!(_bRnl)
   release!(_Ylm)
   return _ϕnlm, st
end

(l::ProductBasis_ATOLayer)(X, ps, st) = evaluate(l, X, ps, st)

function evaluate(l::ProductBasis_STOLayer, X::Vector{SVector{3, T}}, ps, st) where {T}
   R = norm.(X)
   l.bRnl.Dn.ζ = st[1] 
   _bRnl = evaluate(l.bRnl, R)
   _Ylm = evaluate(l.bYlm, X)
   _ϕnlm = evaluate(l.sparsebasis,(_bRnl, _Ylm))
   release!(_bRnl)
   release!(_Ylm)
   return _ϕnlm, st
end

(l::ProductBasis_STOLayer)(X, ps, st) = evaluate(l, X, ps, st)

using ChainRulesCore

function ChainRulesCore.rrule(::typeof(evaluate), l::ProductBasis_ATOLayer, X::Vector{SVector{3, T}}, ps, st) where {T}
   R = norm.(X)
   dnorm = X ./ R
   _bRnl, dR = evaluate_ed(l.bRnl, R)
   _bYlm, dX = evaluate_ed(l.bYlm, X)
   val = evaluate(l.sparsebasis, (_bRnl, _bYlm))
   ∂R = similar(R)
   ∂X_bYlm = similar(X)
   ∂X_bRnl = similar(X)
   ∂X = similar(X)
   dζ = Polynomials4ML.pb_params(ps.ζ, l.bRnl, R)
   ∂ζ = similar(l.bRnl.Dn.ζ)
   function pb(Δ)
      ∂BB = Polynomials4ML._pullback_evaluate(Δ[1], l.sparsebasis, (_bRnl, _bYlm))
      for i = 1:length(X)
         ∂X_bYlm[i] = sum([∂BB[2][i,j] * dX[i,j] for j = 1:length(dX[i,:])])
         ∂R[i] = dot(@view(∂BB[1][i, :]), @view(dR[i, :]))
         ∂X_bRnl[i] = ∂R[i] * dnorm[i]
         ∂X[i] = ∂X_bYlm[i] +  ∂X_bRnl[i]
      end
      for i = 1:length(l.bRnl.Dn.ζ)
         ∂ζ[i] = dot(@view(∂BB[1][:, i]), @view(dζ[:, i]))
      end
      return NoTangent(), NoTangent(), ∂X, (ζ = ∂ζ,), NoTangent()
   end
   return (val, st), pb
end 

function ChainRulesCore.rrule(::typeof(evaluate), l::ProductBasis_STOLayer, X::Vector{SVector{3, T}}, ps, st) where {T}
   R = norm.(X)
   dnorm = X ./ R
   _bRnl, dR = evaluate_ed(l.bRnl, R)
   _bYlm, dX = evaluate_ed(l.bYlm, X)
   val = evaluate(l.sparsebasis, (_bRnl, _bYlm))
   ∂R = similar(R)
   ∂X_bYlm = similar(X)
   ∂X_bRnl = similar(X)
   ∂X = similar(X)
   function pb(Δ)
      ∂BB = Polynomials4ML._pullback_evaluate(Δ[1], l.sparsebasis, (_bRnl, _bYlm))
      for i = 1:length(X)
         ∂X_bYlm[i] = sum([∂BB[2][i,j] * dX[i,j] for j = 1:length(dX[i,:])])
         ∂R[i] = dot(@view(∂BB[1][i, :]), @view(dR[i, :]))
         ∂X_bRnl[i] = ∂R[i] * dnorm[i]
         ∂X[i] = ∂X_bYlm[i] +  ∂X_bRnl[i]
      end
      for i = 1:length(l.bRnl.ζ)
         ∂ζ[i] = dot(@view(∂BB[1][:, i]), @view(dζ[:, i]))
      end
      return NoTangent(), NoTangent(), ∂X, NoTangent(), NoTangent()
   end
   return (val, st), pb
end 


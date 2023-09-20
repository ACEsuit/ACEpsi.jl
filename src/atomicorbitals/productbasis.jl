using Lux: WrappedFunction
using Lux
using Polynomials4ML: SparseProduct, AbstractPoly4MLBasis, release!, GaussianBasis, SlaterBasis, STO_NG, AtomicOrbitalsRadials, SVecPoly4MLBasis
import LuxCore 
import LuxCore: initialparameters, initialstates, AbstractExplicitLayer
using Random: AbstractRNG

AOR_type(TP, T, TI, Dn) = AtomicOrbitalsRadials{TP, Dn{T}, TI}
const ATOLayer = Union{GaussianBasis, SlaterBasis}

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

struct ProductBasis{TDN, TP, T, TT, TI, NB} <: AbstractExplicitLayer 
   sparsebasis::SparseProduct{NB}
   bRnl::Union{AOR_type(TP, T, TI, STO_NG), AOR_type(TP, T, TI, GaussianBasis), AOR_type(TP, T, TI, SlaterBasis)}
   bYlm::Union{RYlmBasis{TT}, RRlmBasis{TT}}
   L::Int
   Dn::TDN
end

ProductBasisLayer(spec1::Vector, bRnl::AbstractPoly4MLBasis, bYlm::AbstractPoly4MLBasis) = begin
   spec1idx = Vector{Tuple{Int, Int}}(undef, length(spec1))
   spec_Rnl = natural_indices(bRnl); inv_Rnl = _invmap(spec_Rnl)
   spec_Ylm = natural_indices(bYlm); inv_Ylm = _invmap(spec_Ylm)
 
   spec1idx = Vector{Tuple{Int, Int}}(undef, length(spec1))
   for (i, b) in enumerate(spec1)
      spec1idx[i] = (inv_Rnl[dropnames(b,(:m,))], inv_Ylm[(l=b.l, m=b.m)])
   end
   sparsebasis = SparseProduct(spec1idx)
   return ProductBasis(sparsebasis, bRnl, bYlm, length(spec1), bRnl.Dn)
end

initialparameters(rng::AbstractRNG, l::ProductBasis{ATOLayer{T}, TP, T, TT, TI, NB}) where {TP, T, TT, TI, NB} = ( ζ = l.bRnl.Dn.ζ, )
initialparameters(rng::AbstractRNG, l::ProductBasis{STO_NG{T}, TP, T, TT, TI, NB}) where {TP, T, TT, TI, NB} = NamedTuple()

initialstates(rng::AbstractRNG, l::ProductBasis{ATOLayer{T}, TP, T, TT, TI, NB}) where {TP, T, TT, TI, NB} = NamedTuple()
initialstates(rng::AbstractRNG, l::ProductBasis{STO_NG{T}, TP, T, TT, TI, NB}) where {TP, T, TT, TI, NB} = (ζ = l.bRnl.Dn.ζ, )

(l::ProductBasis)(X, ps, st) = evaluate(l, X, ps, st)

function evaluate(l::ProductBasis{ATOLayer{T}, TP, T, TT, TI, NB}, X::Vector{SVector{3, TX}}, ps, st) where {TP, T, TT, TI, NB, TX}
   R = norm.(X)
   l.bRnl.Dn.ζ = ps[1] 
   _bRnl = evaluate(l.bRnl, R)
   _Ylm = evaluate(l.bYlm, X)
   _ϕnlm = evaluate(l.sparsebasis,(_bRnl, _Ylm))
   release!(_bRnl)
   release!(_Ylm)
   return _ϕnlm, st
end

function evaluate(l::ProductBasis{STO_NG{T}, TP, T, TT, TI, NB}, X::Vector{SVector{3, TX}}, ps, st) where {TP, T, TT, TI, NB, TX}
   R = norm.(X)
   l.bRnl.Dn.ζ = st[1] 
   _bRnl = evaluate(l.bRnl, R)
   _Ylm = evaluate(l.bYlm, X)
   _ϕnlm = evaluate(l.sparsebasis,(_bRnl, _Ylm))
   release!(_bRnl)
   release!(_Ylm)
   return _ϕnlm, st
end

using ChainRulesCore

function ChainRulesCore.rrule(::typeof(evaluate), l::ProductBasis{ATOLayer{T}, TP, T, TT, TI, NB}, X::Vector{SVector{3, TX}}, ps, st) where {TP, T, TT, TI, NB, TX}
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

function ChainRulesCore.rrule(::typeof(evaluate), l::ProductBasis{STO_NG{T}, TP, T, TT, TI, NB}, X::Vector{SVector{3, TX}}, ps, st) where {TP, T, TT, TI, NB, TX}
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
      return NoTangent(), NoTangent(), ∂X, NoTangent(), NoTangent()
   end
   return (val, st), pb
end 

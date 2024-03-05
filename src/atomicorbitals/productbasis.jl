using Lux: WrappedFunction
using Lux
using Polynomials4ML: SparseProduct, AbstractPoly4MLBasis, release!, GaussianBasis, SlaterBasis, STO_NG, AtomicOrbitalsRadials, SVecPoly4MLBasis
import LuxCore 
import LuxCore: initialparameters, initialstates, AbstractExplicitLayer
using Random: AbstractRNG
using ObjectPools: unwrap

AOR_type(TP, T, TI, Dn) = AtomicOrbitalsRadials{TP, Dn{T}, TI}

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

struct ProductBasis{TDN, TP, T, TT, TI, TL, NB} <: AbstractExplicitLayer 
   sparsebasis::Vector{SparseProduct{NB}}
   speclist::Vector{TL}
   bRnl::Union{Vector{AOR_type(TP, T, TI, SlaterBasis)}, Vector{AOR_type(TP, T, TI, STO_NG)}, Vector{AOR_type(TP, T, TI, GaussianBasis)}}
   bYlm::Union{RYlmBasis{TT}, RRlmBasis{TT}}
   Dn::TDN
end

ProductBasisLayer(speclist::Vector{TI}, bRnl::Vector{TB}, bYlm::AbstractPoly4MLBasis, totdeg::TT) where {TB, TI, TT}= begin
   Nnuc = length(speclist)
   spec_Ylm = natural_indices(bYlm); inv_Ylm = _invmap(spec_Ylm)
   _spec1idx = []
   for i = 1:Nnuc
      spec1 = make_nlms_spec(bRnl[speclist[i]], bYlm, totaldegree = totdeg)
      spec1idx = Vector{Tuple{Int, Int}}(undef, length(spec1))
      spec_Rnl = natural_indices(bRnl[speclist[i]]); inv_Rnl = _invmap(spec_Rnl)
      for (z, b) in enumerate(spec1)
         spec1idx[z] = (inv_Rnl[dropnames(b,(:m,))], inv_Ylm[(l=b.l, m=b.m)])
      end
      push!(_spec1idx, spec1idx)
   end
   sparsebasis = [SparseProduct(_spec1idx[i]) for i = 1:Nnuc]
   return ProductBasis(sparsebasis, speclist, bRnl, bYlm, bRnl[1].Dn)
end

initialparameters(rng::AbstractRNG, l::ProductBasis{GaussianBasis{T}, TP, T, TT, TI, NB}) where {TP, T, TT, TI, NB} = ( ζ = [l.bRnl[i].Dn.ζ for i = 1:length(l.bRnl)], )
initialparameters(rng::AbstractRNG, l::ProductBasis{SlaterBasis{T}, TP, T, TT, TI, NB}) where {TP, T, TT, TI, NB} = ( ζ = [l.bRnl[i].Dn.ζ for i = 1:length(l.bRnl)], )
initialparameters(rng::AbstractRNG, l::ProductBasis{STO_NG{T}, TP, T, TT, TI, NB}) where {TP, T, TT, TI, NB} = NamedTuple()

initialstates(rng::AbstractRNG, l::ProductBasis{GaussianBasis{T}, TP, T, TT, TI, NB}) where {TP, T, TT, TI, NB} = NamedTuple()
initialstates(rng::AbstractRNG, l::ProductBasis{SlaterBasis{T}, TP, T, TT, TI, NB}) where {TP, T, TT, TI, NB} = NamedTuple()
initialstates(rng::AbstractRNG, l::ProductBasis{STO_NG{T}, TP, T, TT, TI, NB}) where {TP, T, TT, TI, NB} = ( ζ = [l.bRnl[i].Dn.ζ for i = 1:length(l.bRnl)], )

(l::ProductBasis)(X, ps, st) = evaluate(l, X, ps, st)

function evaluate(l::ProductBasis{SlaterBasis{T}, TP, T, TT, TI, NB}, X::NTuple{Nnuc, Vector{SVector{3, TX}}}, ps, st) where {TP, T, TT, TI, NB, Nnuc, TX}
   RT = promote_type(T, TT, TX)
   Nel = length(X[1])
   R = acquire!(l.bRnl[1].pool, :R, (Nel,), RT)
   # z: 1:Nnuc 
   # (X - R[1], X - R[2], X - R[3]) -> (ϕnlm1, ϕnlm2, ϕnlm3)
   ϕnlm = [acquire!(l.bRnl[1].pool, Symbol("ϕ$z"), (Nel, length(l.sparsebasis[l.speclist[z]].spec)), RT) for z = 1:Nnuc]
   for z = 1:Nnuc
      # norm(X - R[z])
      @simd ivdep for i = 1:Nel
         R[i] = norm(X[z][i])
      end
      # Ylm(X - R[z])
      _Ylm = evaluate(l.bYlm, X[z])
      
      # Rnl
      i = l.speclist[z]
      l.bRnl[i].Dn.ζ = ps[1][i] 
      _bRnl = evaluate(l.bRnl[i], R)
      ϕnlm[z] = evaluate(l.sparsebasis[i],(_bRnl, _Ylm))
   end
   return ϕnlm, ps, st
end

function evaluate(l::ProductBasis{GaussianBasis{T}, TP, T, TT, TI, NB}, X::NTuple{Nnuc, Vector{SVector{3, TX}}}, ps, st) where {TP, T, TT, TI, NB, Nnuc, TX}
   RT = promote_type(T, TT, TX)
   Nel = length(X[1])
   R = acquire!(l.bRnl[1].pool, :R, (Nel,), RT)
   # z: 1:Nnuc 
   # (X - R[1], X - R[2], X - R[3]) -> (ϕnlm1, ϕnlm2, ϕnlm3)
   ϕnlm = [acquire!(l.bRnl[1].pool, Symbol("ϕ$z"), (Nel, length(l.sparsebasis[l.speclist[z]].spec)), RT) for z = 1:Nnuc]
   for z = 1:Nnuc
      # norm(X - R[z])
      @simd ivdep for i = 1:Nel
         R[i] = norm(X[z][i])
      end
      # Ylm(X - R[z])
      _Ylm = evaluate(l.bYlm, X[z])
      
      # Rnl
      i = l.speclist[z]
      l.bRnl[i].Dn.ζ = ps[1][i] 
      _bRnl = evaluate(l.bRnl[i], R)
      ϕnlm[z] = evaluate(l.sparsebasis[i],(_bRnl, _Ylm))
   end
   return ϕnlm, ps, st
end

function evaluate(l::ProductBasis{STO_NG{T}, TP, T, TT, TI, NB}, X::NTuple{Nnuc, Vector{SVector{3, TX}}}, ps, st) where {TP, T, TT, TI, NB, Nnuc, TX}
   RT = promote_type(T, TT, TX)
   Nel = length(X[1])
   R = acquire!(l.bRnl[1].pool, :R, (Nel,), RT)
   # z: 1:Nnuc 
   # (X - R[1], X - R[2], X - R[3]) -> (ϕnlm1, ϕnlm2, ϕnlm3)
   ϕnlm = [acquire!(l.bRnl[1].pool, Symbol("ϕ$z"), (Nel, length(l.sparsebasis[l.speclist[z]].spec)), RT) for z = 1:Nnuc]
   for z = 1:Nnuc
      # norm(X - R[z])
      @simd ivdep for i = 1:Nel
         R[i] = norm(X[z][i])
      end
      # Ylm(X - R[z])
      _Ylm = evaluate(l.bYlm, X[z])
      
      # Rnl
      i = l.speclist[z]
      l.bRnl[i].Dn.ζ = st[1][i] 
      _bRnl = evaluate(l.bRnl[i], R)
      ϕnlm[z] = evaluate(l.sparsebasis[i],(_bRnl, _Ylm))
   end
   return ϕnlm, ps, st
end

using ChainRulesCore

function ChainRulesCore.rrule(::typeof(evaluate), l::ProductBasis{SlaterBasis{T}, TP, T, TT, TI, NB}, X::NTuple{Nnuc, Vector{SVector{3, TX}}}, ps, st) where {TP, T, TT, TI, NB, Nnuc, TX}
   RT = promote_type(T, TT, TX)
   Nel = length(X[1])
   R = acquire!(l.bRnl[1].pool, :R, (Nel,), RT)
   ϕnlm = [acquire!(l.bRnl[1].pool, Symbol("ϕ$z"), (Nel, length(l.sparsebasis[l.speclist[z]].spec)), RT) for z = 1:Nnuc]
   dnorm = acquire!(l.bYlm.pool, :norm, (Nel, Nnuc), Polynomials4ML._gradtype(l.bYlm, X[1]))
   bYlm = acquire!(l.bRnl[1].pool, :Ylm, (Nel, length(l.bYlm), Nnuc), RT)
   dX = acquire!(l.bYlm.pool, :dX, (Nel, length(l.bYlm), Nnuc), Polynomials4ML._gradtype(l.bYlm, X[1]))
   dζ = [acquire!(l.bRnl[1].pool, Symbol("dζ$z"), (3, length(l.bRnl[l.speclist[z]])), RT) for z = 1:Nnuc] 
   dR = [acquire!(l.bRnl[1].pool, Symbol("dR$z"), (3, length(l.bRnl[l.speclist[z]])), RT) for z = 1:Nnuc]
   bRnl = [acquire!(l.bRnl[1].pool, Symbol("bRnl$z"), (3, length(l.bRnl[l.speclist[z]])), RT) for z = 1:Nnuc] 
   for z = 1:Nnuc
      # norm(X - R[z])
      @simd ivdep for i = 1:Nel
         R[i] = norm(X[z][i])
      end
      dnorm[:,z] = X[z] ./ R
      # Ylm(X - R[z])
      bYlm[:,:,z], dX[:,:,z] = evaluate_ed(l.bYlm, X[z])
      
      # Rnl
      i = l.speclist[z]
      l.bRnl[i].Dn.ζ = ps[1][i] 
      bRnl[z], dR[z], dζ[z] = Polynomials4ML.evaluate_ed_dp(l.bRnl[i], R)
      ϕnlm[z] = evaluate(l.sparsebasis[i],(bRnl[z], bYlm[:,:,z]))
   end
   ∂X = Tuple([zero(X[z]) for z = 1:Nnuc])
   function pb(Δ)
      ∂ζ = [zeros(promote_type(eltype(l.bRnl[i].Dn.ζ), eltype(Δ[1][1]), eltype(bRnl[1]), eltype(bYlm)), size(l.bRnl[i].Dn.ζ)) for i = 1:length(l.bRnl)]
      for z = 1:Nnuc
         n = l.speclist[z]
         ∂BB = Polynomials4ML._pullback_evaluate(Δ[1][z], l.sparsebasis[n], (bRnl[z], bYlm[:,:,z]))
         for i = 1:Nel
            ∂X[z][i] = dot(@view(∂BB[1][i, :]), @view(dR[z][i, :])) * dnorm[i,z]
            for j = 1:length(dX[i,:,z])
               ∂X[z][i] = muladd(∂BB[2][i,j], dX[i,j,z], ∂X[z][i])
            end
         end
         for i = 1:length(l.bRnl[n].Dn.ζ)
            ∂ζ[n][i] += dot(@view(∂BB[1][:, i]), @view(dζ[n][:, i]))
         end
      end
      return NoTangent(), NoTangent(), ∂X, (ζ = ∂ζ,), NoTangent()
   end
   release!(R);release!(dnorm);release!(bYlm);release!(dX);release!(dζ);release!(dR);release!(bRnl)
   return (ϕnlm, ps, st), pb
end 

function ChainRulesCore.rrule(::typeof(evaluate), l::ProductBasis{GaussianBasis{T}, TP, T, TT, TI, NB}, X::NTuple{Nnuc, Vector{SVector{3, TX}}}, ps, st) where {TP, T, TT, TI, NB, Nnuc, TX}
   RT = promote_type(T, TT, TX)
   Nel = length(X[1])
   R = acquire!(l.bRnl[1].pool, :R, (Nel,), RT)
   ϕnlm = [acquire!(l.bRnl[1].pool, Symbol("ϕ$z"), (Nel, length(l.sparsebasis[l.speclist[z]].spec)), RT) for z = 1:Nnuc]
   dnorm = acquire!(l.bYlm.pool, :norm, (Nel, Nnuc), Polynomials4ML._gradtype(l.bYlm, X[1]))
   bYlm = acquire!(l.bRnl[1].pool, :Ylm, (Nel, length(l.bYlm), Nnuc), RT)
   dX = acquire!(l.bYlm.pool, :dX, (Nel, length(l.bYlm), Nnuc), Polynomials4ML._gradtype(l.bYlm, X[1]))
   dζ = [acquire!(l.bRnl[1].pool, Symbol("dζ$z"), (3, length(l.bRnl[l.speclist[z]])), RT) for z = 1:Nnuc] 
   dR = [acquire!(l.bRnl[1].pool, Symbol("dR$z"), (3, length(l.bRnl[l.speclist[z]])), RT) for z = 1:Nnuc]
   bRnl = [acquire!(l.bRnl[1].pool, Symbol("bRnl$z"), (3, length(l.bRnl[l.speclist[z]])), RT) for z = 1:Nnuc] 
   for z = 1:Nnuc
      # norm(X - R[z])
      @simd ivdep for i = 1:Nel
         R[i] = norm(X[z][i])
      end
      dnorm[:,z] = X[z] ./ R
      # Ylm(X - R[z])
      bYlm[:,:,z], dX[:,:,z] = evaluate_ed(l.bYlm, X[z])
      
      # Rnl
      i = l.speclist[z]
      l.bRnl[i].Dn.ζ = ps[1][i] 
      bRnl[z], dR[z], dζ[z] = Polynomials4ML.evaluate_ed_dp(l.bRnl[i], R)
      ϕnlm[z] = evaluate(l.sparsebasis[i],(bRnl[z], bYlm[:,:,z]))
   end
   ∂X = Tuple([zero(X[z]) for z = 1:Nnuc])
   ∂ζ = [zero(l.bRnl[i].Dn.ζ) for i = 1:length(l.bRnl)]
   function pb(Δ)
      for z = 1:Nnuc
         n = l.speclist[z]
         ∂BB = Polynomials4ML._pullback_evaluate(Δ[1][z], l.sparsebasis[n], (bRnl[z], bYlm[:,:,z]))
         for i = 1:Nel
            ∂X[z][i] = dot(@view(∂BB[1][i, :]), @view(dR[z][i, :])) * dnorm[i,z]
            for j = 1:length(dX[i,:,z])
               ∂X[z][i] = muladd(∂BB[2][i,j], dX[i,j,z], ∂X[z][i])
            end
         end
         for i = 1:length(l.bRnl[n].Dn.ζ)
            ∂ζ[n][i] += dot(@view(∂BB[1][:, i]), @view(dζ[n][:, i]))
         end
      end
      return NoTangent(), NoTangent(), ∂X, (ζ = ∂ζ,), NoTangent()
   end
   release!(R);release!(dnorm);release!(bYlm);release!(dX);release!(dζ);release!(dR);release!(bRnl)
   return (ϕnlm, ps, st), pb
end 

function ChainRulesCore.rrule(::typeof(evaluate), l::ProductBasis{STO_NG{T}, TP, T, TT, TI, NB}, X::NTuple{Nnuc, Vector{SVector{3, TX}}}, ps, st) where {TP, T, TT, TI, NB, Nnuc, TX}
   RT = promote_type(T, TT, TX)
   Nel = length(X[1])
   R = acquire!(l.bRnl[1].pool, :R, (Nel,), RT)
   ϕnlm = [acquire!(l.bRnl[1].pool, Symbol("ϕ$z"), (Nel, length(l.sparsebasis[l.speclist[z]].spec)), RT) for z = 1:Nnuc]
   dnorm = acquire!(l.bYlm.pool, :norm, (Nel, Nnuc), Polynomials4ML._gradtype(l.bYlm, X[1]))
   bYlm = acquire!(l.bRnl[1].pool, :Ylm, (Nel, length(l.bYlm), Nnuc), RT)
   dX = acquire!(l.bYlm.pool, :dX, (Nel, length(l.bYlm), Nnuc), Polynomials4ML._gradtype(l.bYlm, X[1]))
   dR = [acquire!(l.bRnl[1].pool, Symbol("dR$z"), (3, length(l.bRnl[l.speclist[z]])), RT) for z = 1:Nnuc]
   bRnl = [acquire!(l.bRnl[1].pool, Symbol("bRnl$z"), (3, length(l.bRnl[l.speclist[z]])), RT) for z = 1:Nnuc] 
   for z = 1:Nnuc
      # norm(X - R[z])
      @simd ivdep for i = 1:Nel
         R[i] = norm(X[z][i])
      end
      dnorm[:,z] = X[z] ./ R
      # Ylm(X - R[z])
      bYlm[:,:,z], dX[:,:,z] = evaluate_ed(l.bYlm, X[z])
   
      # Rnl
      i = l.speclist[z]
      l.bRnl[i].Dn.ζ = st[1][i] 
      bRnl[z], dR[z] = Polynomials4ML.evaluate_ed(l.bRnl[i], R)
      ϕnlm[z] = evaluate(l.sparsebasis[i],(bRnl[z], bYlm[:,:,z]))
   end
   ∂X = Tuple([zero(X[z]) for z = 1:Nnuc])
   function pb(Δ)
      for z = 1:Nnuc
         n = l.speclist[z]
         ∂BB = Polynomials4ML._pullback_evaluate(Δ[1][z], l.sparsebasis[n], (bRnl[z], bYlm[:,:,z]))
         for i = 1:Nel
            ∂X[z][i] = dot(@view(∂BB[1][i, :]), @view(dR[z][i, :])) * dnorm[i,z]
            for j = 1:length(dX[i,:,z])
               ∂X[z][i] = muladd(∂BB[2][i,j], dX[i,j,z], ∂X[z][i])
            end
         end
      end
      return NoTangent(), NoTangent(), ∂X, NoTangent(), NoTangent()
   end
   release!(R);release!(dnorm);release!(bYlm);release!(dX);release!(dR);release!(bRnl)
   return (ϕnlm, ps, st), pb
end


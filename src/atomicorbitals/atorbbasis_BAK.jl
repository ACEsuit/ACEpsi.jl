import Polynomials4ML: evaluate

using ACEpsi: ↑, ↓, ∅, spins, extspins, Spin, spin2idx, idx2spin
using Polynomials4ML: SparseProduct, _make_reqfields
using LuxCore: AbstractExplicitLayer, AbstractExplicitContainerLayer, apply
using Random: AbstractRNG

using Polynomials4ML
using StaticArrays
using LinearAlgebra: norm 

using ChainRulesCore
using ChainRulesCore: NoTangent
using Zygote

using Lux: Chain

using Polynomials4ML: _make_reqfields, @reqfields, POOL, TMP, META
using ObjectPools: acquire!

struct Nuc{T}
   rr::SVector{3, T}
   charge::T   # should this be an integer? 
end 

#
# Ordering of the embedding 
# nuc | 1 2 3  1 2 3  1 2 3
#   k | 1 1 1  2 2 2  2 2 2
#

const NTRNL1 = NamedTuple{(:n, :l, :m), Tuple{Int, Int, Int}}
const NTRNLIS = NamedTuple{(:I, :s, :n, :l, :m), Tuple{Int, Spin, Int, Int, Int}}

"""
This constructs the specification of all the atomic orbitals for one
nucleus. 

* bRnl : radial basis 
* Ylm : angular basis, assumed to be spherical harmonics 
* admissible : a filter, default is a total degree 
"""
function make_nlms_spec(bRnl, bYlm;
            totaldegree::Integer = -1,
            admissible = ( (br, by) -> degree(bRnl, br) + 
                           degree(bYlm, by) <= totaldegree), 
            nnuc = 0)
   
   spec_Rnl = natural_indices(bRnl)
   spec_Ylm = natural_indices(bYlm)
   
   spec1 = []
   for (iR, br) in enumerate(spec_Rnl), (iY, by) in enumerate(spec_Ylm)
      if br.l != by.l 
         continue 
      end
      if admissible(br, by) 
         push!(spec1, (br..., m = by.m))
      end
   end
   return spec1 
end


# mutable struct AtomicOrbitalsBasis{NB, T}
#    prodbasis::ProductBasis{NB}
#    nuclei::Vector{Nuc{T}}  # nuclei (defines the shifted orbitals)
# end

# (aobasis::AtomicOrbitalsBasis)(args...) = evaluate(aobasis, args...)
# Base.length(aobasis::AtomicOrbitalsBasis) = length(aobasis.prodbasis.spec1) * length(aobasis.nuclei)

# function AtomicOrbitalsBasis(bRnl, bYlm; 
#                totaldegree=3, 
#                nuclei = Nuc{Float64}[], 
#                )
#    spec1 = make_nlms_spec(bRnl, bYlm; 
#                           totaldegree = totaldegree)
#    prodbasis = ProductBasis(spec1, bRnl, bYlm)
#    return AtomicOrbitalsBasis(prodbasis, nuclei)
# end


# function evaluate(basis::AtomicOrbitalsBasis, X::AbstractVector{<: AbstractVector}, Σ)
#    nuc = basis.nuclei
#    Nnuc = length(nuc)
#    Nel = size(X, 1)
#    T = promote_type(eltype(X[1]))
   
#    # XX = zeros(VT, (Nnuc, Nel))
   
#    # # trans
#    # for I = 1:Nnuc, i = 1:Nel
#    #    XX[I, i] = X[i] - nuc[I].rr
#    # end

#    Nnlm = length(basis.prodbasis.sparsebasis.spec) 
#    ϕnlm = Zygote.Buffer(zeros(T, (Nnuc, Nel, Nnlm)))

#    # Think how to prevent this extra FLOPS here while keeping it Zygote-friendly
#    for I = 1:Nnuc
#       ϕnlm[I,:,:] = evaluate(basis.prodbasis, map(x -> x - nuc[I].rr, X))
#    end

#    return copy(ϕnlm)
# end

# # ------------ utils for AtomicOrbitalsBasis ------------
# function set_nuclei!(basis::AtomicOrbitalsBasis, nuclei::AbstractVector{<: Nuc})
#    basis.nuclei = copy(collect(nuclei))
#    return nothing
# end


# function get_spec(basis::AtomicOrbitalsBasis) 
#    spec = []
#    Nnuc = length(basis.nuclei)

#    spec = Array{Any}(undef, (3, Nnuc, length(basis.prodbasis.spec1)))

#    for (k, nlm) in enumerate(basis.prodbasis.spec1)
#       for I = 1:Nnuc 
#          for (is, s) in enumerate(extspins())
#             spec[is, I, k] = (I = I, s=s, nlm...)
#          end
#       end
#    end

#    return spec 
# end


# # ------------ Evaluation kernels 


# # ------------ connect with ChainRulesCore

# # Placeholder for now, fix this later after making sure Zygote is done correct with Lux
# # function ChainRulesCore.rrule(::typeof(evaluate), basis::AtomicOrbitalsBasis, X::AbstractVector{<: AbstractVector}, Σ)
# #    val = evaluate(basis, X, Σ)
# #    dB = similar(X)
# #    function pb(dA)
# #       return NoTangent(), NoTangent(), dB, NoTangent()
# #    end
# #    return val, pb
# # end

# # ------------ connect with Lux
# struct AtomicOrbitalsBasisLayer{TB} <: AbstractExplicitLayer
#    basis::TB
#    meta::Dict{String, Any}
# end

# Base.length(l::AtomicOrbitalsBasisLayer) = length(l.basis)

# lux(basis::AtomicOrbitalsBasis) = AtomicOrbitalsBasisLayer(basis, Dict{String, Any}())

# initialparameters(rng::AbstractRNG, l::AtomicOrbitalsBasisLayer) = _init_luxparams(rng, l.basis)

# initialstates(rng::AbstractRNG, l::AtomicOrbitalsBasisLayer) = _init_luxstate(rng, l.basis)

# (l::AtomicOrbitalsBasisLayer)(X, ps, st) = 
#       evaluate(l.basis, X, st.Σ), st


struct AtomicOrbitalsBasisLayer{L, T} <: AbstractExplicitContainerLayer{(:prodbasis, )}
   prodbasis::L
   nuclei::Vector{Nuc{T}}
   @reqfields
end

Base.length(l::AtomicOrbitalsBasisLayer) = length(l.prodbasis.layer.ϕnlms.basis.spec) * length(l.nuclei)

function AtomicOrbitalsBasisLayer(prodbasis, nuclei)
   return AtomicOrbitalsBasisLayer(prodbasis, nuclei, _make_reqfields()...)
end

function (l::AtomicOrbitalsBasisLayer)(X, ps, st)
   nuc = l.nuclei
   Nnuc = length(nuc)
   Nel = size(X, 1)
   T = promote_type(eltype(X[1]))
   Nnlm = length(l.prodbasis.layers.ϕnlms.basis.spec)

   # ϕnlm = acquire!(l.pool, :ϕnlm, (Nnuc, Nel, Nnlm), T)
   ϕnlm = zeros(T, (Nnuc, Nel, Nnlm))

   for I = 1:Nnuc
      ϕnlm[I,:,:], _ = l.prodbasis(map(x -> x - nuc[I].rr, X), ps, st)
   end

   return ϕnlm, st
end


# (l::AtomicOrbitalsBasisLayer)(X, ps, st) = 
#       evaluate(l, X, ps, st)

# chho: I did eval here twice but I just want to make sure it works first
function ChainRulesCore.rrule(::typeof(apply), l::AtomicOrbitalsBasisLayer, X, ps, st)
   val = l(X, ps, st)
   nuc = l.nuclei
   Nnuc = length(nuc)
   Nel = length(X)
   # T = promote_type(eltype(X[1]))
   XX = zeros(typeof(X[1]), (Nnuc, Nel))
   
   # trans
   for I = 1:Nnuc, i = 1:Nel
      XX[I, i] = X[i] - nuc[I].rr
   end

   # pbI = [(val_I, pb_I) for I = 1:Nnuc] where val_I = (A, st)
   pbI = [Zygote.pullback(X -> l.prodbasis(X, ps, st), XX[I, :]) for I = 1:Nnuc]
   
   function pb(∂ϕnlm)
      dB = similar(X)
      for I = 1:Nnuc
         # pbI[2] is the pb, pbI[I][1][2] is the state, this pb returns a tuple of (A, st) so we need to get [1]
         dB += pbI[I][2]((∂ϕnlm[1][I, :, :], pbI[I][1][2]))[1]
      end
      return NoTangent(), NoTangent(), dB, NoTangent(), NoTangent()
   end

   return val, pb
end

# ------------ utils for AtomicOrbitalsBasis ------------
function set_nuclei!(basis::AtomicOrbitalsBasisLayer, nuclei::AbstractVector{<: Nuc})
   basis.nuclei = copy(collect(nuclei))
   return nothing
end


function get_spec(l::AtomicOrbitalsBasisLayer, spec1p) 
   spec = []
   Nnuc = length(l.nuclei)

   spec = Array{Any}(undef, (3, Nnuc, length(spec1p)))

   for (k, nlm) in enumerate(spec1p)
      for I = 1:Nnuc 
         for (is, s) in enumerate(extspins())
            spec[is, I, k] = (I = I, s=s, nlm...)
         end
      end
   end

   return spec 
end
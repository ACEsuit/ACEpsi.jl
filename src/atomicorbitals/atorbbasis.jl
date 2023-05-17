import Polynomials4ML: evaluate

using ACEpsi: ↑, ↓, ∅, spins, extspins, Spin, spin2idx, idx2spin
using Polynomials4ML: SparseProduct, ProductBasis
using LuxCore: AbstractExplicitLayer
using Random: AbstractRNG

using Polynomials4ML
using StaticArrays
using LinearAlgebra: norm 

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

function ProductBasis(spec1, bRnl, bYlm)
   spec1idx = Vector{Tuple{Int, Int}}(undef, length(spec1)) 
   spec_Rnl = bRnl.spec; inv_Rnl = _invmap(spec_Rnl)
   spec_Ylm = natural_indices(bYlm); inv_Ylm = _invmap(spec_Ylm)

   spec1idx = Vector{Tuple{Int, Int}}(undef, length(spec1))
   for (i, b) in enumerate(spec1)
      spec1idx[i] = (inv_Rnl[dropnames(b,(:m,))], inv_Ylm[(l=b.l, m=b.m)])
   end
   sparsebasis = SparseProduct(spec1idx)
   return ProductBasis(spec1, bRnl, bYlm, sparsebasis)
end


mutable struct AtomicOrbitalsBasis{NB, T}
   prodbasis::ProductBasis{NB}
   nuclei::Vector{Nuc{T}}  # nuclei (defines the shifted orbitals)
end

(aobasis::AtomicOrbitalsBasis)(args...) = evaluate(aobasis, args...)


function AtomicOrbitalsBasis(bRnl, bYlm; 
               totaldegree=3, 
               nuclei = Nuc{Float64}[], 
               )
   spec1 = make_nlms_spec(bRnl, bYlm; 
                          totaldegree = totaldegree)
   prodbasis = ProductBasis(spec1, bRnl, bYlm)
   return AtomicOrbitalsBasis(prodbasis, nuclei)
end


function evaluate(basis::AtomicOrbitalsBasis, X::AbstractVector{<: AbstractVector}, Σ)
   nuc = basis.nuclei 
   Nnuc = length(nuc)
   Nel = size(X, 1)
   T = promote_type(eltype(X[1]))
   VT = SVector{3, T}
   XX = zeros(VT, (Nnuc, Nel))
   
   # trans
   for I = 1:Nnuc, i = 1:Nel
      XX[I, i] = X[i] - nuc[I].rr
   end

   Nnlm = length(basis.prodbasis.sparsebasis.spec) 
   ϕnlm = zeros(T, (Nnuc, Nel, Nnlm))

   for I = 1:Nnuc
      ϕnlm[I,:,:] = evaluate(basis.prodbasis, XX[I,:])
   end

   return ϕnlm
end

# ------------ utils for AtomicOrbitalsBasis ------------
function set_nuclei!(basis::AtomicOrbitalsBasis, nuclei::AbstractVector{<: Nuc})
   basis.nuclei = copy(collect(nuclei))
   return nothing
end


function get_spec(basis::AtomicOrbitalsBasis) 
   spec = []
   Nnuc = length(basis.nuclei)

   spec = Array{3}(undef, (3, Nnuc, length(basis.prodbasis.spec1)))

   for (k, nlm) in enumerate(basis.prodbasis.spec1)
      for I = 1:Nnuc 
         for (is, s) in enumerate(extspins())
            spec[is, I, k] = (I = I, s=s, nlm...)
         end
      end
   end

   return spec 
end


# ------------ Evaluation kernels 


# ------------ connect with Lux
struct AtomicOrbitalsBasisLayer{TB} <: AbstractExplicitLayer
   basis::TB
   meta::Dict{String, Any}
end

Base.length(l::AtomicOrbitalsBasisLayer) = length(l.basis)

lux(basis::AtomicOrbitalsBasis) = AtomicOrbitalsBasisLayer(basis, Dict{String, Any}())

initialparameters(rng::AbstractRNG, l::AtomicOrbitalsBasisLayer) = _init_luxparams(rng, l.basis)

initialstates(rng::AbstractRNG, l::AtomicOrbitalsBasisLayer) = _init_luxstate(rng, l.basis)

(l::AtomicOrbitalsBasisLayer)(X, ps, st) = 
      evaluate(l.basis, X, st.Σ), st



# ----- ObejctPools
# (l::AtomicOrbitalsBasisLayer)(args...) = evaluate(l, args...)

# function evaluate(l::AtomicOrbitalsBasisLayer, x::SINGLE, ps, st)
#    B = acquire!(st.cache, :B, (length(l.basis), ), _valtype(l.basis, x))
#    evaluate!(parent(B), l.basis, x)
#    return B 
# end 

# function evaluate(l::AtomicOrbitalsBasisLayer, X::AbstractArray{<: SINGLE}, ps, st)
#    B = acquire!(st.cache[:Bbatch], (length(l.basis), length(X)), _valtype(l.basis, X[1]))
#    evaluate!(parent(B), l.basis, X)
#    return B 
# end

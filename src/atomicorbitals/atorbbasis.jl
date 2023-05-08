
using ACEpsi: ↑, ↓, ∅, spins, extspins, Spin, spin2idx, idx2spin
import Polynomials4ML
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

mutable struct ProductBasis{NB, TR, TY}
   sparsebasis::SparseProduct{NB}
   bRnl::TR
   bYlm::TY
end

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
   
   spec1 = NTRNL1[]
   for (iR, br) in enumerate(spec_Rnl), (iY, by) in enumerate(spec_Ylm)
      if br.l != by.l 
         continue 
      end
      if admissible(br, by) 
         push!(spec1, (n=br.n, l = br.l, m = by.m))
      end
   end
   return spec1 
end


function evaluate(basis::ProductBasis, X::AbstractVector{<: AbstractVector}, Σ)
   Nel = length(X)
   T = promote_type(eltype(X[1]))
   VT = SVector{3, T}
   @show VT
   
   # create all the shifted configurations 
   xx = zeros(eltype(VT), Nel)
   for i = 1:Nel
      xx[i] = norm(X[i])
   end

   # evaluate the radial and angular components on all the shifted particles 
   Rnl = reshape(evaluate(basis.bRnl, xx[:]), (Nel, length(basis.bRnl)))
   Ylm = reshape(evaluate(basis.bYlm, X[:]), (Nel, length(basis.bYlm)))

   # evaluate all the atomic orbitals as ϕ_nlm = Rnl * Ylm 
   ϕnlm = evaluate(basis.sparsebasis, (Rnl, Ylm))

   return ϕnlm
end

mutable struct AtomicOrbitalsBasis{NB, T}
   prodbasis::ProductBasis{NB}
   nuclei::Vector{Nuc{T}}  # nuclei (defines the shifted orbitals)
end

function evaluate(basis::AtomicOrbitalsBasis, X::AbstractVector{<: AbstractVector}, Σ)
   nuc = basis.nuclei 
   Nnuc = length(nuc)
   
   
   XX = zeros(VT, (Nnuc, Nel))
   
   for I = 1:Nnuc, i = 1:Nel
      XX[I, i] = X[i] - nuc[I].rr
   end

   Nnlm = length(basis.prodbasis.sparsebasis.spec) 

   ϕnlm = zeros(TA, (Nnuc, Nel, Nnlm))

   for I = 1:Nnuc 
      ϕnlm[I,:,:] = evaluate(basis.prodbasis, XX[I,:], Σ)
   end

   # evaluate the pooling operation
   #                spin  I    k = (nlm) 
   Aall = zeros(TA, (2, Nnuc, Nnlm))
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
   A = zeros(TA, ((Nel, 3, Nnuc, Nnlm)))
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







function _invmap(a::AbstractVector)
   inva = Dict{eltype(a), Int}()
   for i = 1:length(a) 
      inva[a[i]] = i 
   end
   return inva 
end

function _invmap(a::AbstractVector)
   inva = Dict{eltype(a), Int}()
   for i = 1:length(a) 
      inva[a[i]] = i 
   end
   return inva 
end

"""

function AtomicOrbitalsBasis(bRnl, bYlm; 
               totaldegree=3, 
               nuclei = Nuc{Float64}[], 
               )
   spec1 = make_nlms_spec(bRnl, bYlm; 
                          totaldegree = totaldegree) 
   return AtomicOrbitalsBasis(bRnl, bYlm, spec1, nuclei)
end




function set_nuclei!(basis::AtomicOrbitalsBasis, nuclei::AbstractVector{<: Nuc})
   basis.nuclei = copy(collect(nuclei))
   return nothing 
end


function get_spec(basis::AtomicOrbitalsBasis) 
   spec = NTRNLIS[]
   Nnuc = length(basis.nuclei)

   spec = Array{NTRNLIS, 3}(undef, (3, Nnuc, length(basis.spec1)))

   for (k, nlm) in enumerate(basis.spec1)
      for I = 1:Nnuc 
         for (is, s) in enumerate(extspins())
            spec[is, I, k] = (I = I, s=s, nlm...)
         end
      end
   end

   return spec 
end


# ------------ Evaluation kernels 


"""
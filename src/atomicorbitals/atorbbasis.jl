
using ACEpsi: ↑, ↓, spins, extspins, Spin, spin2idx, idx2spin
import Polynomials4ML
using StaticArrays
using LinearAlgebra: norm 

struct Nuc{T}
   rr::SVector{3, T}
   charge::T 
end 


#
# Ordering of the embedding 
# nuc | 1 2 3  1 2 3  1 2 3
#   k | 1 1 1  2 2 2  2 2 2
#

const NTRNL1 = NamedTuple{(:n, :l, :m), Tuple{Int, Int, Int}}
const NTRNLIS = NamedTuple{(:I, :s, :n, :l, :m), Tuple{Int, Spin, Int, Int, Int}}

mutable struct AtomicOrbitalsBasis{TR, TY, T}
   bRnl::TR
   bYlm::TY
   # ------- specification of the atomic orbitals Rnl * Ylm
   spec1::Vector{NTRNL1}                # human readable spec
   spec1idx::Vector{Tuple{Int, Int}}    # indices into Rnl, Ylm 
   nuclei::Vector{Nuc{T}}               # nuclei (defines the shifted orbitals)
   # ------- specification of the pooling operations A_{sInlm}
   # spec::Vector{NTRNLIS}                # human readable spec, definition is 
                                        # implicit in iteration protocol. 
end


function _invmap(a::AbstractVector)
   inva = Dict{eltype(a), Int}()
   for i = 1:length(a) 
      inva[a[i]] = i 
   end
   return inva 
end

function AtomicOrbitalsBasis(bRnl, bYlm, spec1::Vector{NTRNL1}, 
                             nuclei::Vector{<: Nuc}) 

   spec1idx = Vector{Tuple{Int, Int}}(undef, length(spec1)) 
   spec_Rnl = bRnl.spec; inv_Rnl = _invmap(spec_Rnl)
   spec_Ylm = Polynomials4ML.natural_indices(bYlm); inv_Ylm = _invmap(spec_Ylm)

   spec1idx = Vector{Tuple{Int, Int}}(undef, length(spec1))
   for (i, b) in enumerate(spec1)
      spec1idx[i] = (inv_Rnl[(n=b.n, l=b.l)], inv_Ylm[(l=b.l, m=b.m)])
   end

   # spec = NTRNLIS[]
   basis = AtomicOrbitalsBasis(bRnl, bYlm, spec1, spec1idx, eltype(nuclei)[])
   # set_nuclei!(basis, nuclei)
   return basis 
end


function AtomicOrbitalsBasis(bRnl, bYlm; 
               totaldegree=3, 
               nuclei = Nuc{Float64}[], 
               )
   spec1 = make_nlms_spec(bRnl, bYlm; 
                          totaldegree = totaldegree) 
   return AtomicOrbitalsBasis(bRnl, bYlm, spec1, nuclei)
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


function set_nuclei!(basis::AtomicOrbitalsBasis, nuclei::AbstractVector{<: Nuc})
   basis.nuclei = copy(collect(nuclei))
   Nnuc = length(basis.nuclei)

   spec = NTRNLIS[] 
   for b in basis.spec1, I = 1:Nnuc, s in extspins()
      push!(spec, (I = I, s=s, b...))
   end

   basis.spec = spec

   # bA = _build_pooling(spec1, Nnuc)
   # basis.bA = bA 
   return nothing 
end


# ------------ Evaluation kernels 

"""
This function return correct Si for pooling operation.
"""
function onehot_spin!(Si, i, Σ)
   Si .= 0
   for k = 1:length(Σ)
      Si[k, spin2num(Σ[k])] = 1
   end
   # set current electron to ϕ, also remove their contribution in the sum of ↑ or ↓ basis
   Si[i, 1] = 1 
   Si[i, 2] = 0
   Si[i, 3] = 0
end


function proto_evaluate(basis::AtomicOrbitalsBasis, 
                        X::AbstractVector{<: AbstractVector}, 
                        Σ)
   nuc = basis.nuclei 
   Nnuc = length(nuc)
   Nel = length(X)
   Nnlm = length(basis.spec1)
   VT = promote_type(eltype(nuc[1].rr), eltype(X[1]))
   @show VT
   
   # create all the shifted configurations 
   # this MUST be done in the format (I, i)
   XX = zeros(VT, (Nnuc, Nel))
   xx = zeros(eltype(VT), (Nnuc, Nel))
   for I = 1:Nnuc, i = 1:Nel
      XX[I, i] = X[i] - nuc[I].rr
      xx[I, i] = norm(XX[I, i])
   end

   # evaluate the radial and angular components on all the shifted particles 
   Rnl = reshape(evaluate(basis.bRnl, xx[:]), (Nnuc, Nel))
   Ylm = reshape(evaluate(basis.bYlm, XX[:]), (Nnuc, Nel))

   # evaluate all the atomic orbitals as ϕ_nlm = Rnl * Ylm 
   TA = promote_type(eltype(Rnl), eltype(Ylm))
   @show TA 
   ϕnlm = zeros(TA, (Nnuc, Nel, Nnlm))
   for (k, (iR, iY)) in enumerate(basis.spec1idx)
      for i = 1:Nel, I = 1:Nnuc
         ϕnlm[I, i, k] = Rnl[I, i, iR] * Ylm[I, i, iY]
      end
   end

   # evaluate the pooling operation
   #                spin  I    k = (nlm) 
   Aall = zeros(TA, (2, Nnuc, Nnlm))
   for k = 1:Nnlm
      for i = 1:Nel 
         iσ = spin2num(Σ[i])
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


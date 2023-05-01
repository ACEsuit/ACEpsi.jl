
using ACEpsi: ↑, ↓, spins, Spin
using StaticArrays

struct Nuc{T}
   rr::SVector{3, T}
   charge::T 
end 


#
# Ordering of the embedding 
# nuc | 1 2 3  1 2 3  1 2 3
#   k | 1 1 1  2 2 2  2 2 2
#

const NTRNL1 = NamedTuple{(:n, :l, :m, :s), Tuple{Int, Int, Int, Spin}}
const NTRNLI = NamedTuple{(:I, :n, :l, :m, :s), Tuple{Int, Int, Int, Int, Spin}}

mutable struct AtomicOrbitalsBasis{TR, TY, T}
   bRnl::TR
   bYlm::TY
   spec1::Vector{NTRNL1}
   spec::Vector{NTRNLI}
   nuclei::Vector{Nuc{T}} 
   # pooling::PooledSparseProduct{3}
end

function AtomicOrbitalsBasis(bRnl, bYlm, spec1::Vector{NTRNL1}, 
                             nuclei::Vector{<: Nuc}) 
   spec = NTRNLI[]
   basis = AtomicOrbitalsBasis(bRnl, bYlm, spec1, spec, eltype(nuclei)[])
   set_nuclei!(basis, nuclei)
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
         for s in spins 
            push!(spec1, (n=br.n, l = br.l, m = by.m, s = s))
         end
      end
   end
   return spec1 
end


function set_nuclei!(basis::AtomicOrbitalsBasis, nuclei::AbstractVector{<: Nuc})
   basis.nuclei = copy(collect(nuclei))
   Nnuc = length(basis.nuclei)

   spec = NTRNLI[] 
   for b in basis.spec1, I = 1:Nnuc 
      push!(spec, (I = I, b...))
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


# function evaluate(basis::AtomicOrbitalsBasis, 
#                   X::AbstractVector{<: AbstractVector}, 
#                   Σ)
#    nuc = basis.nuclei 
#    Nnuc = length(nuc)
#    Nel = length(X)
#    VT = promote_type(eltype(nuc), eltype(X))
   
#    # create all the shifted configurations 
#    XX = zeros(VT, Nel)
#    Rnl = 
   
# end


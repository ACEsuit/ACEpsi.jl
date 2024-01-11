using Polynomials4ML, Random, ACEpsi
using Polynomials4ML: OrthPolyBasis1D3T, LinearLayer, PooledSparseProduct, SparseSymmProdDAG, SparseSymmProd, release!
using Polynomials4ML.Utils: gensparse
using LinearAlgebra: qr, I, logabsdet, pinv, mul!, dot , tr, det
using LuxCore: AbstractExplicitLayer
using Lux: Chain, WrappedFunction, BranchLayer
using ChainRulesCore: NoTangent
using ChainRulesCore
# ----------------------------------------
# some quick hacks that we should take care in P4ML later with careful thoughts
using StrideArrays
using ObjectPools: unwrap,ArrayPool, FlexArray,acquire!
using Lux

using ACEpsi.AtomicOrbitals: make_nlms_spec
using ACEpsi.TD: No_Decomposition, Tucker
using ACEpsi: ↑, ↓, ∅, spins, extspins, Spin, spin2idx, idx2spin

# ----------------- utils ------------------
function get_spec(nuclei::Vector{Nuc{TN}}, speclist::Vector{TS}, bRnl, bYlm, totdeg) where {TN, TS}
   Nnuc = length(nuclei)
   spec1p = [make_nlms_spec(bRnl[speclist[i]], bYlm, totaldegree = totdeg) for i = 1:Nnuc]
   Nnlm = length.(spec1p)

   spec = Array{Any}(undef, (3, sum(Nnlm)))
   for (is, s) in enumerate(ACEpsi.extspins())
       t = 0
       for i = 1:Nnuc
           for nlm in spec1p[i]
               t += 1
               spec[is, t] = (s=s, I = i, nlm...)
           end
       end
   end
   return spec[:]
end

function displayspec(spec, spec1p)
   nicespec = []
   for k = 1:length(spec)
      push!(nicespec, ([spec1p[spec[k][j]] for j = 1:length(spec[k])]))
   end
   return nicespec
end

# ----------------- usual BF without tensor decomposition ------------------
function BFwf_lux(Nel::Integer, Nbf::Integer, speclist::Vector{Int}, bRnl, bYlm, nuclei, TD::No_Decomposition; totdeg = 100, 
   ν = 3, sd_admissible = bb -> prod(b.s != '∅' for b in bb) == 0, 
   js = JPauliNet(nuclei)) 
   # ----------- Lux connections ---------
   # X -> (X-R[1], X-R[2], X-R[3])
   embed_layer = embed_diff_layer(nuclei)
   # X -> (ϕ1(X-R[1]), ϕ2(X-R[2]), ϕ3(X-R[3])
   prodbasis_layer = ACEpsi.AtomicOrbitals.ProductBasisLayer(speclist, bRnl, bYlm, totdeg)
   
   # BackFlowPooling: (length(nuclei), nX, length(spec1 from totaldegree)) -> (nX, 3, length(nuclei) * length(spec1))
   aobasis_layer = ACEpsi.AtomicOrbitals.AtomicOrbitalsBasisLayer(prodbasis_layer, nuclei)
   pooling = BackflowPooling(aobasis_layer)
   pooling_layer = ACEpsi.lux(pooling)

   spec1p = get_spec(nuclei, speclist, bRnl, bYlm, totdeg)
   tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
   default_admissible = bb -> (length(bb) == 0) || (sum(b.n1 - 1 for b in bb ) <= totdeg)
   specAA = gensparse(; NU = ν, tup2b = tup2b, admissible = default_admissible,
                        minvv = fill(0, ν), 
                        maxvv = fill(length(spec1p), ν), 
                        ordered = true)
   spec = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]
   
   # further restrict
   spec = [t for t in spec if sd_admissible([spec1p[t[j]] for j = 1:length(t)])]
   
   # define n-correlation
   corr1 = Polynomials4ML.SparseSymmProd(spec)

   # (nX, 3, length(nuclei), length(spec1 from totaldegree)) -> (nX, length(spec))
   corr_layer = Polynomials4ML.lux(corr1)

   l_hidden = Tuple(collect(Chain(; hidden = LinearLayer(length(corr1), Nel), Mask = ACEpsi.MaskLayer(Nel), det = WrappedFunction(x::Matrix -> det(x))) for i = 1:Nbf))
   
   jastrow_layer = ACEpsi.lux(js)

   BFwf_chain = Chain(; diff = embed_layer, Pds = prodbasis_layer, 
                        bA = pooling_layer, reshape = myReshapeLayer((Nel, 3 * sum(length.(prodbasis_layer.sparsebasis)))), 
                        bAA = corr_layer, hidden = BranchLayer(l_hidden...),
                        sum = WrappedFunction(sum))
   return Chain(; branch = BranchLayer(; js = jastrow_layer, bf = BFwf_chain, ), prod = WrappedFunction(x -> x[1] * x[2]), logabs = WrappedFunction(x -> 2 * log(abs(x))) ), spec, spec1p
end

# ----------------- custom layers ------------------
import ChainRulesCore: rrule
import ForwardDiff
struct embed_diff_layer <: AbstractExplicitLayer
   nuc::Vector{Nuc{Float64}}
end

function evaluate(l::embed_diff_layer, X, ps, st)
   return ntuple(i -> X .- Ref(l.nuc[i].rr), length(l.nuc)), ps, st
end

(l::embed_diff_layer)(X, ps, st) = evaluate(l, X, ps, st)

function ChainRulesCore.rrule(::typeof(evaluate), l::embed_diff_layer, X, ps, st)
   val = ntuple(i -> X .- Ref(l.nuc[i].rr), length(l.nuc))
   function pb(dA)
      return NoTangent(), NoTangent(), sum(dA[1]), NoTangent(), NoTangent()
   end
   return (val, st), pb
end

struct MaskLayer <: AbstractExplicitLayer 
   nX::Int64
end

(l::MaskLayer)(Φ, ps, st) = begin 
   T = eltype(Φ)
   A::Matrix{Bool} = [st.Σ[i] == st.Σ[j] for j = 1:l.nX, i = 1:l.nX] 
   val::Matrix{T} = Φ .* A
   release!(Φ)
   return val, st
end

function rrule(::typeof(Lux.apply), l::MaskLayer, Φ, ps, st) 
   T = eltype(Φ)
   A::Matrix{Bool} = [st.Σ[i] == st.Σ[j] for j = 1:l.nX, i = 1:l.nX]
   val::Matrix{T} = Φ .* A
   function pb(dΦ)
      return NoTangent(), NoTangent(), dΦ[1] .* A, NoTangent(), NoTangent()
   end
   release!(Φ)
   return (val, st), pb
end

##

struct myReshapeLayer{N} <: AbstractExplicitLayer
   dims::NTuple{N, Int}
end

@inline function (r::myReshapeLayer)(x::AbstractArray, ps, st::NamedTuple)
   return reshape(unwrap(x), r.dims), st
end

function rrule(::typeof(LuxCore.apply), l::myReshapeLayer{N}, X, ps, st) where {N}
   val = l(X, ps, st)
   function pb(dϕnlm) # dA is of a tuple (dAmat, st), dAmat is of size (Nnuc, Nel, Nnlm)
      A = reshape(unwrap(dϕnlm[1]), size(X))
      return NoTangent(), NoTangent(), A, NoTangent(), NoTangent()
   end
   return val, pb
end

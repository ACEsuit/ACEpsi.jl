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
using ACEpsi.TD: No_Decomposition, Tucker, SCP
using ACEpsi: ↑, ↓, ∅, spins, extspins, Spin, spin2idx, idx2spin

# ----------------- usual BF without tensor decomposition ------------------
function BFwf_lux(Nel::Integer, Nbf::Integer, speclist::Vector{Int}, bRnl, bYlm, nuclei, TD::No_Decomposition; totdeg = 100, cluster = Nel, 
   ν = 3, sd_admissible = bb -> sum(b.s == '∅' for b in bb) == 1, disspec = [],
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
   tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]]
   default_admissible = bb -> (length(bb) <= 1) || (sum([abs(sort(bb, by = b -> b.I)[1].I - sort(bb, by = b -> b.I)[end].I)]) <= cluster)

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

   l_hidden = Tuple(collect(Chain(; hidden1 = LinearLayer(length(corr1), Nel; use_cache = false), Mask = ACEpsi.MaskLayer(Nel), det = WrappedFunction(x::Matrix -> det(x))) for i = 1:Nbf))
   
   jastrow_layer = ACEpsi.lux(js)

   BFwf_chain = Chain(; diff = embed_layer, Pds = prodbasis_layer, 
                        bA = pooling_layer, reshape = myReshapeLayer((Nel, 3 * sum(length.(prodbasis_layer.sparsebasis)))), 
                        bAA = corr_layer, 
                        hidden = l_hidden[1], # BranchLayer(l_hidden...),
                        sum = WrappedFunction(sum))
   return Chain(; branch = BranchLayer(; js = jastrow_layer, bf = BFwf_chain, ), prod = WrappedFunction(x -> x[1] * x[2]), logabs = WrappedFunction(x -> 2 * log(abs(x))) ), spec, spec1p, disspec
end

# ----------------- custom layers ------------------
import ChainRulesCore: rrule
import ForwardDiff
using StaticArrays

struct embed_diff_layer{NNuc} <: AbstractExplicitLayer
   nuc::SVector{NNuc, Nuc{Float64}}
end

_getdiff(_X::AbstractArray{SVector{3, T}}, _d::SVector{3, Float64}) where T = begin
   return [SVector{3, T}(_x -_d) for _x in _X]
end

function evaluate(l::embed_diff_layer{NNuc}, X::AbstractVector, ps, st) where {NNuc}
   return ntuple(i -> _getdiff(X, l.nuc[i].rr), Val(NNuc)), st
end

(l::embed_diff_layer)(X, ps, st) = evaluate(l, X, ps, st)

function ChainRulesCore.rrule(::typeof(evaluate), l::embed_diff_layer{NNuc}, X::AbstractVector, ps::NamedTuple, st::NamedTuple) where {NNuc}
   val = ntuple(i -> _getdiff(X, l.nuc[i].rr), Val(NNuc))
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
   val::Matrix{T} = Matrix(Φ) .* A
   release!(Φ)
   return val, st
end

function rrule(::typeof(Lux.apply), l::MaskLayer, Φ, ps, st) 
   T = eltype(Φ)
   A::Matrix{Bool} = [st.Σ[i] == st.Σ[j] for j = 1:l.nX, i = 1:l.nX]
   val::Matrix{T} = Matrix(Φ) .* A
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

# ----------------- utils ------------------
function get_spec(nuclei::SVector{NNuc, Nuc{TN}}, speclist::Vector{TS}, bRnl, bYlm, totdeg) where {NNuc, TN, TS}
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

function get_spec(TD::Union{Tucker, SCP})  
   spec = Array{Any}(undef, (3, TD.P))
 
   for k = 1:TD.P
      for (is, s) in enumerate(extspins())
            spec[is, k] = (s=s, P = k)
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


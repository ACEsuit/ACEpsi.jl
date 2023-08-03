
using Polynomials4ML, Random 
using Polynomials4ML: OrthPolyBasis1D3T
using Polynomials4ML: PooledSparseProduct, SparseSymmProdDAG, SparseSymmProd
using ObjectPools: release!
using Polynomials4ML.Utils: gensparse
using LinearAlgebra: qr, I, logabsdet, pinv, mul!, dot , tr, det
import ForwardDiff
using ACEpsi.AtomicOrbitals: make_nlms_spec
using ACEpsi: ↑, ↓, ∅, spins, extspins, Spin, spin2idx, idx2spin
using ACEpsi
using LuxCore: AbstractExplicitLayer
using LuxCore
using Lux
using Lux: Chain, WrappedFunction, BranchLayer
using ChainRulesCore
using ChainRulesCore: NoTangent
# ----------------------------------------

# ----------------- custom layers ------------------
struct MaskLayer <: AbstractExplicitLayer 
   nX::Integer
end

(l::MaskLayer)(Φ, ps, st) = Φ .* [st.Σ[i] == st.Σ[j] for j = 1:l.nX, i = 1:l.nX], st

##

struct DenseLayer <: AbstractExplicitLayer 
   in_dim::Integer
   out_dim::Integer
end

function (l::DenseLayer)(x::AbstractMatrix, ps, st)
   return parent(x) * ps.W, st
end

# Jerry: Maybe we should use Glorot Uniform if we have no idea about what we should use?
LuxCore.initialparameters(rng::AbstractRNG, l::DenseLayer) = ( W = randn(rng, l.out_dim, l.in_dim), )
LuxCore.initialstates(rng::AbstractRNG, l::DenseLayer) = NamedTuple()

function ChainRulesCore.rrule(::typeof(Lux.apply), l::DenseLayer, x::AbstractMatrix, ps, st)
   val = l(x, ps, st)
   function pb(A)
      return NoTangent(), NoTangent(), A[1] * ps.W', (W = x' * A[1],), NoTangent()
   end
   return val, pb
end

##

function get_spec(nuclei, spec1p) 
   spec = []
   Nnuc = length(nuclei)

   spec = Array{Any}(undef, (3, Nnuc, length(spec1p)))

   for (k, nlm) in enumerate(spec1p)
      for I = 1:Nnuc 
         for (is, s) in enumerate(extspins())
            spec[is, I, k] = (s=s, I = I, nlm...)
         end
      end
   end

   return spec[:]
end

# ----------------- custom layers ------------------

function BFwf_lux(Nel::Integer, bRnl, bYlm, nuclei; totdeg = 15, 
   ν = 3, T = Float64, 
   sd_admissible = bb -> prod(b.s != '∅' for b in bb) == 0) 

   spec1p = make_nlms_spec(bRnl, bYlm; 
                          totaldegree = totdeg)

   # ----------- Lux connections ---------
   # AtomicOrbitalsBasis: (X, Σ) -> (length(nuclei), nX, length(spec1))
   prodbasis_layer = ACEpsi.AtomicOrbitals.ProductBasisLayer(spec1p, bRnl, bYlm)
   aobasis_layer = ACEpsi.AtomicOrbitals.AtomicOrbitalsBasisLayer(prodbasis_layer, nuclei)

   # BackFlowPooling: (length(nuclei), nX, length(spec1 from totaldegree)) -> (nX, 3, length(nuclei), length(spec1))
   pooling = BackflowPooling(aobasis_layer)
   pooling_layer = ACEpsi.lux(pooling)

   spec1p = get_spec(nuclei, spec1p)
   # define sparse for n-correlations
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
   corr_layer = Polynomials4ML.lux(corr1; use_cache = false)

   js = Jastrow(nuclei)
   jastrow_layer = ACEpsi.lux(js)

   reshape_func = x -> reshape(x, (size(x, 1), prod(size(x)[2:end])))

   _det = x -> size(x) == (1, 1) ? x[1,1] : det(Matrix(x))
   BFwf_chain = Chain(; ϕnlm = aobasis_layer, bA = pooling_layer, reshape = WrappedFunction(reshape_func), 
                        bAA = corr_layer, hidden1 = DenseLayer(Nel, length(corr1)), 
                        Mask = ACEpsi.MaskLayer(Nel), det = WrappedFunction(x -> _det(x)))
   return Chain(; branch = BranchLayer(; js = jastrow_layer, bf = BFwf_chain, ), prod = WrappedFunction(x -> prod(x)), logabs = WrappedFunction(x -> 2 * log(abs(x))) ), spec, spec1p
end

function displayspec(spec, spec1p)
   nicespec = []
   for k = 1:length(spec)
      push!(nicespec, ([spec1p[spec[k][j]] for j = 1:length(spec[k])]))
   end
   return nicespec
end
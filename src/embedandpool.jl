import ACEpsi: BFwf_lux
using ACEpsi: No_Decomposition, embed_diff_layer, get_spec, myReshapeLayer
using Polynomials4ML.Utils: gensparse
using Polynomials4ML: LinearLayer
using Polynomials4ML
using StaticArrays
using ACEpsi

using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: Nuc, make_nlms_spec, evaluate
using ACEpsi: BackflowPooling, BFwf_lux, setupBFState, Jastrow, displayspec
using ACEpsi.vmc: gradx, laplacian, grad_params, EmbeddingW!, _invmap, VMC_multilevel, wf_multilevel, VMC, gd_GradientByVMC, gd_GradientByVMC_multilevel, AdamW, SR, SumH, MHSampler
using ACEbase.Testing: print_tf, fdtest
using LuxCore
using Lux
using Zygote
using Optimisers
using Random
using Printf
using LinearAlgebra
using BenchmarkTools
using HyperDualNumbers: Hyper
using StaticArrays
# EMBED : NamedTuple
#
#
module ABC
   using Polynomials4ML: _make_reqfields, @reqfields
   using ACEpsi
   using ACEpsi.AtomicOrbitals: Nuc, make_nlms_spec, evaluate
   using ACEpsi: BackflowPooling, BFwf_lux, setupBFState, Jastrow, displayspec
   using LuxCore: AbstractExplicitContainerLayer

   mutable struct EmbedAndPool{ENV, NAMES, EMBED} <: AbstractExplicitContainerLayer{(:environ, embeddings, pooling)}

      environ::ENV
      embeddings::NamedTuple{NAMES, EMBED}
      pooling::BackflowPooling
      #---
      @reqfields
   end

   function EmbedAndPool(nuclei, speclist, bRnl, bYlm, totdeg)
      environ = embed_diff_layer(nuclei)
      l_pb = ProductBasisLayer(speclist, bRnl, bYlm, totdeg)
      aobasis_layer = ACEpsi.AtomicOrbitals.AtomicOrbitalsBasisLayer(l_pb, nuclei)
      pooling = BackflowPooling(aobasis_layer)
      return EmbedAndPool(environ, l_pb, pooling, _make_reqfields()...)
   end

   (pooling::BackflowPooling)(args...) = evaluate(pooling, args...)


   @generated function evaluate(embedpool::EmbedAndPool{ENV, NAMES, EMBED}, 
                        X::AbstractVector, ps, st) where {ENV, NAMES, EMBED}
      
      names = NAMES.parameters
      NE = length(names)
   
      quote
         # needs extra layer
         # X -> XX stores the environments of the electons
         XX = embedpool.environ(X, ps.environ, st.environ)
         # XX = Nin x Nel

         @nexprs $NE a -> begin
            ps_a = ps[$(names[a])]
            st_a = st[$(names[a])]
            # E_a = Nin x Nfeat_a x Nel
            E_a = evaluate(embedpool.embeddings[a], XX, ps_a, st_a)
         end

         # pooling 
         #EE = tuple(E_1,...., )
         A = evaluate(embedpool.pooling, EE, ps.pooling, st.pooling) 
         # A = Nfeat x Nel 

         return A 
      end
   end
   # @generated function pullback(embedpool::EmbedAndPool{NE, EMBED}, 
#                              X::AbstractVector, ps, st, ∂A)   where {NE, EMBED}
#    quote
#       @nexprs $NE i -> begin
#          ps_i = ps[$(NAMES[i])]
#          st_i = st[$(NAMES[i])]
#          E_i, dE_i = evaluate_ed(embedpool.embeddings[i], X, ps_i, st_i)
#       end

#       # pooling 
#       EE = tuple(E_1,...., )
#       # A = evaluate(embedpool.pooling, EE, ps.pooling, st.pooling) 
#       ∂EE = pullback(embedpool.pooling, EE, ps, st, ∂A)

#       @nexprs $NE i -> begin
#          ∂ps_i, ∂X_i = pullback(embedpool.embeddings[i], X, ps, st, ∂EE[i])
#       end

#       ∂ps = ∂ps_$NE 
#       ∂X = ∂X_$NE
#       @nexprs = $(NE-1) i -> begin
#          ∂ps += ∂ps_i
#          ∂X += ∂X_i
#       end

#       return ∂X, ∂ps
#    end
# end
end

Nel = 4
X = randn(SVector{3, Float64}, Nel)
Σ = [↑,↑,↓,↓]


spec_ = [(n1 = 1, n2 = 1, l = 0), 
        (n1 = 1, n2 = 2, l = 0), 
        (n1 = 1, n2 = 3, l = 0), 
        (n1 = 1, n2 = 1, l = 1), 
        (n1 = 1, n2 = 2, l = 1), 
        (n1 = 2, n2 = 1, l = 0), 
        (n1 = 2, n2 = 2, l = 0), 
        (n1 = 2, n2 = 3, l = 0), 
        (n1 = 2, n2 = 1, l = 1), 
        (n1 = 2, n2 = 2, l = 1), 
        (n1 = 3, n2 = 1, l = 0), 
        (n1 = 3, n2 = 2, l = 0), 
        (n1 = 3, n2 = 3, l = 0), 
        (n1 = 3, n2 = 1, l = 1), 
        (n1 = 3, n2 = 2, l = 1)
        ]
spec = [spec_]
        
n1 = 5
nuclei = SVector{1}([ Nuc(zeros(SVector{3, Float64}), Nel * 1.0)])

Pn = Polynomials4ML.legendre_basis(n1+1)
Ylmdegree = 2
totdegree = 20
ζ = 10.0 * rand(length(spec))
Dn = SlaterBasis(ζ)
bYlm = RRlmBasis(Ylmdegree)
totdegree = totdeg = 30

ν = 2

MaxIters = 10
_spec = [ spec[1][1:8]]
speclist  = [1]

bRnl = [AtomicOrbitalsRadials(Pn, SlaterBasis(10 * rand(length(_spec[j]))), _spec[speclist[j]]) for j = 1:length(_spec)]

embed_layer = embed_diff_layer(nuclei)
prodbasis_layer = ACEpsi.AtomicOrbitals.ProductBasisLayer(speclist, bRnl, bYlm, totdeg)
aobasis_layer = ACEpsi.AtomicOrbitals.AtomicOrbitalsBasisLayer(prodbasis_layer, nuclei)
pooling = BackflowPooling(aobasis_layer)
pooling_layer = ACEpsi.lux(pooling)


##


old_pooling = Chain(; diff = embed_layer,)# Pds = prodbasis_layer,)
#bA = pooling_layer)
ps, st = setupBFState(MersenneTwister(1234), old_pooling, Σ)
out1, _ = old_pooling(X, ps, st)
Zygote.gradient(x -> sum(sum(old_pooling(x, ps, st)[1][1])), X)

@time Zygote.gradient(x -> sum(sum(old_pooling(x, ps, st)[1][1])), X)

using ObjectPools: release!
@profview for i = 1:500
   #out, _ = old_pooling(X, ps, st)
   Zygote.gradient(x -> sum(sum(old_pooling(x, ps, st)[1][1])), X)
end


new_pooling = ABC.EmbedAndPool(nuclei, speclist, bRnl, bYlm, totdeg)















using Polynomials4ML, Random 
using Polynomials4ML: OrthPolyBasis1D3T, LinearLayer, PooledSparseProduct, SparseSymmProdDAG, SparseSymmProd, release!
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
# some quick hacks that we should take care in P4ML later with careful thoughts
using ObjectPools: acquire!
using StrideArrays
using ObjectPools: unwrap

function BFchain(Nel::Integer, bRnl, bYlm, nuclei; totdeg = 15, 
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
   BFwf_chain = Chain(; ϕnlm = aobasis_layer, bA = pooling_layer, reshape = myReshapeLayer((Nel, 3 * length(nuclei) * length(prodbasis_layer.sparsebasis))), 
                        bAA = corr_layer, hidden1 = LinearLayer(length(corr1), Nel), Mask = ACEpsi.MaskLayer(Nel), det = WrappedFunction(x::Matrix -> det(x)))
   return BFwf_chain, spec, spec1p
end

function mBFwf_sto(Nel::Integer, bRnl, bYlm, nuclei, Nbf::Integer; totdeg = 15, 
   ν = 3, T = Float64, 
   sd_admissible = bb -> prod(b.s != '∅' for b in bb) == 0, js = Jastrow(nuclei)) 

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
   l_det = Tuple(collect(Chain(; hidden1 = LinearLayer(length(corr1), Nel), Mask = ACEpsi.MaskLayer(Nel), det = WrappedFunction(x::Matrix -> det(x))) for i = 1:Nbf))

   jastrow_layer = ACEpsi.lux(js)

   BFwf_chain = Chain(; ϕnlm = aobasis_layer, bA = pooling_layer, reshape = ACEpsi.myReshapeLayer((Nel, 3 * length(nuclei) * length(prodbasis_layer.sparsebasis))), 
                        bAA = corr_layer, Pds = Lux.Parallel(nothing, l_det...), col = WrappedFunction(x -> collect(x)), hidden2 = LinearLayer(Nbf, 1), reduce = WrappedFunction(x -> x[1]))
   
   return Chain(; branch = BranchLayer(; js = jastrow_layer, bf = BFwf_chain, ), prod = WrappedFunction(x -> x[1] * x[2]), logabs = WrappedFunction(x -> 2 * log(abs(x))) ), spec, spec1p
end

function mBFwf(Nel::Integer, bRnl, bYlm, nuclei, Nbf::Integer; totdeg = 15, 
   ν = 3, T = Float64, 
   sd_admissible = bb -> prod(b.s != '∅' for b in bb) == 0, js = JPauliNet(nuclei)) 
   _, spec, spec1p =  BFchain(Nel, bRnl, bYlm, nuclei, totdeg = totdeg, 
                              ν = ν, sd_admissible = sd_admissible)
   l_bf = Tuple(collect(BFchain(Nel, bRnl, bYlm, nuclei, totdeg = totdeg, 
                                 ν = ν, sd_admissible = sd_admissible)[1] for i = 1:Nbf))

   jastrow_layer = ACEpsi.lux(js)

   BFwf_chain = Chain(; Pds = Lux.Parallel(nothing, l_bf...), col = WrappedFunction(x -> collect(x)), hidden2 = LinearLayer(Nbf, 1), reduce = WrappedFunction(x -> x[1]))
   return Chain(; branch = BranchLayer(; js = jastrow_layer, bf = BFwf_chain, ), prod = WrappedFunction(x -> x[1] * x[2]), logabs = WrappedFunction(x -> 2 * log(abs(x))) ), spec, spec1p
end
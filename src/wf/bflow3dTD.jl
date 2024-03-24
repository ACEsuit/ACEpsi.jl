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
using ACEpsi.TD: TDs
using ACEpsi: ↑, ↓, ∅, spins, extspins, Spin, spin2idx, idx2spin

# ----------------- usual BF without tensor decomposition ------------------
function BFwf_lux(Nel::Integer, Nbf::Integer, speclist::Vector{Int}, bRnl, bYlm, nuclei, TD::TDs; totdeg = 100, 
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

   extend!(x, n::Int) = append!(x, (1 for i in 1:n-length(x)))
   spec2 = deepcopy(spec)
   extend!.(spec2, ν)
   tt_layer = ACEpsi.TD.TDLayer(Nbf, TD.P, Nel, spec1p, stack(spec2, dims=1), TD)

   l_hidden = Chain(; mask = Lux.Parallel(nothing, (Mask = ACEpsi.MaskLayer(Nel) for j = 1:Nbf)...))
   l_det    = Chain(; d  = Lux.Parallel(+, (Det  = WrappedFunction(x -> det(x)) for j = 1:Nbf)...))
   
   jastrow_layer = ACEpsi.lux(js)

   BFwf_chain = Chain(; diff = embed_layer, Pds = prodbasis_layer, 
                        bA = pooling_layer, reshape = ACEpsi.myReshapeLayer((Nel, 3 * sum(length.(prodbasis_layer.sparsebasis)))), 
                        bAA = corr_layer, 
                        TK = tt_layer, 
                        hidden = l_hidden, 
                        det = l_det)
   return Chain(; branch = BranchLayer(; js = jastrow_layer, bf = BFwf_chain, ), prod = WrappedFunction(x -> x[1] * x[2]), logabs = WrappedFunction(x -> 2 * log(abs(x))) ), spec, spec1p, disspec
end
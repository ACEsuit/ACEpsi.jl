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
using ACEpsi.Cluster: _bf_orbital
using ACEpsi: ↑, ↓, ∅, spins, extspins, Spin, spin2idx, idx2spin

function BFwf_lux(Nel::Integer, Nbf::Integer, speclist::Vector{Int}, bRnl, bYlm, nuclei, TD::No_Decomposition, c::_bf_orbital; totdeg = 100, cluster = Nel, 
   ν = 3, sd_admissible = bb -> sum(b.s == '∅' for b in bb) == 1, nuc = get_nuc(nuclei, Nel), 
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
   index, disspec = sparse(spec, spec1p, Nel, nuc)

   l_hidden = Tuple(collect(Chain(; hidden = Lux.Parallel(nothing, (LinearLayer(length(disspec[j]), 1) for j = 1:Nel)...), l_concat = WrappedFunction(x -> hcat(x...)), Mask = ACEpsi.MaskLayer(Nel), det = WrappedFunction(x::Matrix -> det(x))) for i = 1:Nbf))
   jastrow_layer = ACEpsi.lux(js)

   BFwf_chain = Chain(; diff = embed_layer, Pds = prodbasis_layer, 
                        bA = pooling_layer, reshape = ACEpsi.myReshapeLayer((Nel, 3 * sum(length.(prodbasis_layer.sparsebasis)))), 
                        bAA = corr_layer, bf = WrappedFunction(A -> Tuple([A[:, index[i]] for i = 1:Nel])), 
                        hidden = BranchLayer(l_hidden...),
                        sum = WrappedFunction(sum))
   return Chain(; branch = BranchLayer(; js = jastrow_layer, bf = BFwf_chain, ), prod = WrappedFunction(x -> x[1] * x[2]), logabs = WrappedFunction(x -> 2 * log(abs(x))) ), spec, spec1p, disspec
end

get_charge(nuc::Nuc) = Int(nuc.charge)

function get_nuc(nuclei::Vector{Nuc{T}}, N::TN) where {T, TN <: Integer}
    nuc = cumsum(get_charge.(nuclei))
    out = zeros(Int, N); out[1] = 1; t = 1
    if N > 1
       for i in 2:N
           if cumsum(out[1:i-1])[1] == nuc[t]
               t = t+1
           end
           out[i] = t
       end
    end
    out[end] = length(nuc)
    return out
end

function sparse(spec, spec1p, Nel::Integer, nuc)
   disspec = displayspec(spec, spec1p)
   _spec = []
   index = []
   for i = 1:Nel
        push!(_spec, [;])
        push!(index, [;])
        for (j, bb) in enumerate(disspec)
            if sum([(bb[z].s == '∅') && bb[z].I == nuc[i] for z = 1:length(bb)]) > 0
                push!(index[i], j)
                push!(_spec[i], bb)
            end
        end
   end
   return index, _spec
end

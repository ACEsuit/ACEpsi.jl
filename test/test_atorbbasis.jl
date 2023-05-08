
using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: AtomicOrbitalsBasis, Nuc, make_nlms_spec, _invmap, ProductBasis, evaluate
using Polynomials4ML.Testing: print_tf 

totdeg = 3
bRnl = ACEpsi.AtomicOrbitals.RnlExample(totdeg)
bYlm = RYlmBasis(totdeg)
nuclei = [ Nuc(3 * rand(SVector{3, Float64}), 1.0) for _=1:3 ]

totaldegree = 4
spec1 = make_nlms_spec(bRnl, bYlm; totaldegree = totaldegree) 
spec1idx = Vector{Tuple{Int, Int}}(undef, length(spec1)) 
spec_Rnl = bRnl.spec; inv_Rnl = _invmap(spec_Rnl)
spec_Ylm = Polynomials4ML.natural_indices(bYlm); inv_Ylm = _invmap(spec_Ylm)

spec1idx = Vector{Tuple{Int, Int}}(undef, length(spec1))
for (i, b) in enumerate(spec1)
   spec1idx[i] = (inv_Rnl[(n=b.n, l=b.l)], inv_Ylm[(l=b.l, m=b.m)])
end
sparsebasis = SparseProduct(spec1idx)
prodbasis = ProductBasis(sparsebasis, bRnl, bYlm)

Nel = 5
X = randn(SVector{3, Float64}, Nel)
Σ = rand(spins(), Nel)

ϕnlm = evaluate(prodbasis, X, Σ)

bAnlm = AtomicOrbitalsBasis(prodbasis, nuclei)



A = evaluate(bAnlm, X, Σ)



# ## not test
# spec = ACEpsi.AtomicOrbitals.get_spec(bAnlm)

# ##






# ## test the Evaluation: 
# using LinearAlgebra: norm 
# bYlm_ = RYlmBasis(totdeg)
# Nnlm = length(bAnlm.spec1)
# Nnuc = length(bAnlm.nuclei)
# inv_nl = ACEpsi.AtomicOrbitals._invmap(bRnl.spec)
# inv_lm = ACEpsi.AtomicOrbitals._invmap(natural_indices(bYlm))

# for I = 1:Nnuc 
#    XI = X .- Ref(bAnlm.nuclei[I].rr)
#    xI = norm.(XI)
#    Rnl = evaluate(bRnl, xI)
#    Ylm = evaluate(bYlm_, XI)
#    for k = 1:Nnlm 
#       nlm = bAnlm.spec1[k] 
#       iR = inv_nl[(n = nlm.n, l = nlm.l)]
#       iY = inv_lm[(l = nlm.l, m = nlm.m)]

#       for i = 1:Nel 
#          for (is, s) in enumerate(ACEpsi.extspins())
#             a1 = A[i, is, I, k] 

#             if s in [↑, ↓]
#                a2 = sum( Rnl[j, iR] * Ylm[j, iY] * (Σ[j] == s) * (1 - (j == i)) for j = 1:Nel )
#             else # s = ∅
#                a2 = Rnl[i, iR] * Ylm[i, iY]
#             end
#             # println("(i=$i, σ=$s, I=$I, n=$(nlm.n), l=$(nlm.l), m=$(nlm.m)) -> ", abs(a1 - a2))
#             print_tf(@test abs(a1 - a2) < 1e-12)
#          end
#       end
#    end
# end
# println()

## 


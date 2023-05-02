
using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree
using ACEpsi.AtomicOrbitals: AtomicOrbitalsBasis, Nuc, proto_evaluate
using Polynomials4ML.Testing: print_tf 

##

totdeg = 3
bRnl = ACEpsi.AtomicOrbitals.RnlExample(totdeg)
bYlm = RYlmBasis(totdeg)
nuclei = [ Nuc(3 * rand(SVector{3, Float64}), 1.0) for _=1:3 ]

bAnlm = AtomicOrbitalsBasis(bRnl, bYlm; 
                            totaldegree = totdeg, 
                            nuclei = nuclei )

spec = ACEpsi.AtomicOrbitals.get_spec(bAnlm)

##

Nel = 5
X = randn(SVector{3, Float64}, Nel)
Σ = rand(spins(), Nel)

A = proto_evaluate(bAnlm, X, Σ)


## test the Evaluation: 
using LinearAlgebra: norm 
bYlm_ = RYlmBasis(totdeg)
Nnlm = length(bAnlm.spec1)
Nnuc = length(bAnlm.nuclei)
inv_nl = ACEpsi.AtomicOrbitals._invmap(bRnl.spec)
inv_lm = ACEpsi.AtomicOrbitals._invmap(natural_indices(bYlm))

for I = 1:Nnuc 
   XI = X .- Ref(bAnlm.nuclei[I].rr)
   xI = norm.(XI)
   Rnl = evaluate(bRnl, xI)
   Ylm = evaluate(bYlm_, XI)
   for k = 1:Nnlm 
      nlm = bAnlm.spec1[k] 
      iR = inv_nl[(n = nlm.n, l = nlm.l)]
      iY = inv_lm[(l = nlm.l, m = nlm.m)]

      for i = 1:Nel 
         for (is, s) in enumerate(ACEpsi.extspins())
            a1 = A[i, is, I, k] 

            if s in [↑, ↓]
               a2 = sum( Rnl[j, iR] * Ylm[j, iY] * (Σ[j] == s) * (1 - (j == i)) for j = 1:Nel )
            else # s = ∅
               a2 = Rnl[i, iR] * Ylm[i, iY]
            end
            # println("(i=$i, σ=$s, I=$I, n=$(nlm.n), l=$(nlm.l), m=$(nlm.m)) -> ", abs(a1 - a2))
            print_tf(@test abs(a1 - a2) < 1e-12)
         end
      end
   end
end
println()

## 


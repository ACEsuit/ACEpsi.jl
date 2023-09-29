using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: Nuc, make_nlms_spec, evaluate, AtomicOrbitalsBasisLayer
using ACEpsi: extspins
using ACEpsi: BackflowPooling
using ACEbase.Testing: print_tf, fdtest
using LuxCore
using Random
using Zygote 

# test configs
Rnldegree = 4
Ylmdegree = 4
totdegree = 8
Nel = 5
X = randn(SVector{3, Float64}, Nel)
Σ = rand(spins(), Nel)

nuclei = [ Nuc(3 * rand(SVector{3, Float64}), 1.0) for _=1:3 ]
##

# Defining AtomicOrbitalsBasis
n1 = 5
n2 = 1
Pn = Polynomials4ML.legendre_basis(n1+1)
spec = [(n1 = n1, n2 = n2, l = l) for n1 = 1:n1 for n2 = 1:n2 for l = 0:n1-1] 
ζ = 10 * rand(length(spec))
Dn = SlaterBasis(ζ)
bRnl = AtomicOrbitalsRadials(Pn, Dn, spec) 
bYlm = RYlmBasis(Ylmdegree)
spec1p = make_nlms_spec(bRnl, bYlm; totaldegree = totdegree) 

# define basis and pooling operations
prodbasis_layer = ACEpsi.AtomicOrbitals.ProductBasisLayer(spec1p, bRnl, bYlm)
aobasis_layer = ACEpsi.AtomicOrbitals.AtomicOrbitalsBasisLayer(prodbasis_layer, nuclei)

pooling = BackflowPooling(aobasis_layer)
pooling_layer = ACEpsi.lux(pooling)

println()
@info("Test evaluate ProductBasisLayer")
ps1, st1 = LuxCore.setup(MersenneTwister(1234), prodbasis_layer)
bϕnlm, st1 = prodbasis_layer(X, ps1, st1)

@info("Test evaluate AtomicOrbitalsBasis")
ps, st = LuxCore.setup(MersenneTwister(1234), aobasis_layer)
bϕnlm, st = aobasis_layer(X, ps, st)

@info("Test BackflowPooling")
A = pooling(bϕnlm, Σ)

println()


##
@info("Check get_spec is working")
spec = ACEpsi.AtomicOrbitals.get_spec(aobasis_layer, spec1p)


@info("Test evaluation by manual construction")
using LinearAlgebra: norm 
bYlm_ = RYlmBasis(totdegree)
Nnlm = length(aobasis_layer.prodbasis.sparsebasis)
Nnuc = length(aobasis_layer.nuclei)

for I = 1:Nnuc 
   XI = X .- Ref(aobasis_layer.nuclei[I].rr)
   xI = norm.(XI)
   Rnl = evaluate(bRnl, xI)
   Ylm = evaluate(bYlm_, XI)
   for k = 1:Nnlm 
      nlm = aobasis_layer.prodbasis.sparsebasis.spec[k]
      iR = nlm[1]
      iY = nlm[2]

      for i = 1:Nel 
         for (is, s) in enumerate(ACEpsi.extspins())
            a1 = A[i, is, I, k] 

            if s in [↑, ↓]
               a2 = sum( Rnl[j, iR] * Ylm[j, iY] * (Σ[j] == s) * (1 - (j == i)) for j = 1:Nel )
            else # s = ∅
               a2 = Rnl[i, iR] * Ylm[i, iY]
            end
            # println("(i=$i, σ=$s, I=$I, n=$(nlm.n), l=$(nlm.l), m=$(nlm.m)) -> ", abs(a1 - a2))
            print_tf(@test a1 ≈ a2)
         end
      end
   end
end
println()

# 
@info("---------- rrule tests ----------")
using LinearAlgebra: dot 

@info("BackFlowPooling rrule")
for ntest = 1:30
   local testϕnlm
   testϕnlm = randn(size(bϕnlm))
   bdd = randn(size(bϕnlm))
   _BB(t) = testϕnlm + t * bdd
   bA2 = pooling(testϕnlm, Σ)
   u = randn(size(bA2))
   F(t) = dot(u, pooling(_BB(t), Σ))
   dF(t) = begin
      val, pb = ACEpsi._rrule_evaluate(pooling, _BB(t), Σ)
      ∂BB = pb(u)
      return dot(∂BB, bdd)
   end
   print_tf(@test fdtest(F, dF, 0.0; verbose=false))
end
println()

@info("Checking Zygote running correctly")
val, pb = Zygote.pullback(pooling, bϕnlm, Σ)
val1, pb1 = ACEpsi._rrule_evaluate(pooling, bϕnlm, Σ)
@assert val1 ≈ val1
@assert pb1(val) ≈ pb(val)[1] # pb(val)[2] is for Σ with no pb
using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: AtomicOrbitalsBasis, Nuc, make_nlms_spec, ProductBasis, evaluate
using ACEpsi: BackflowPooling
using ACEbase.Testing: print_tf, fdtest
using LuxCore
using Random

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
bRnl = ACEpsi.AtomicOrbitals.RnlExample(Rnldegree)
bYlm = RYlmBasis(Ylmdegree)
spec1 = make_nlms_spec(bRnl, bYlm; totaldegree = totdegree) 

# define basis and pooling operations
prodbasis = ProductBasis(spec1, bRnl, bYlm)
aobasis = AtomicOrbitalsBasis(prodbasis, nuclei)
pooling = BackflowPooling(aobasis)

# we can also construct in this way which wraps the definition of product basis inside
aobasis2 = AtomicOrbitalsBasis(bRnl, bYlm; totaldegree = totdegree, nuclei = nuclei, )

@info("Checking two type of construction are same")
for ntest = 1:30
   local X = randn(SVector{3, Float64}, Nel)
   local Σ = rand(spins(), Nel)
   print_tf(@test evaluate(aobasis, X, Σ) ≈ evaluate(aobasis2, X, Σ))
end

println()

@info("Test evaluate ProductBasis")
ϕnlm = prodbasis(X)

@info("Test evaluate AtomicOrbitalsBasis")
bϕnlm = aobasis(X, Σ)

@info("Test BackflowPooling")
A = pooling(bϕnlm, Σ)

println()


##
@info("Check get_spec is working")
spec = ACEpsi.AtomicOrbitals.get_spec(aobasis)


@info("Test evaluation by manual construction")
using LinearAlgebra: norm 
bYlm_ = RYlmBasis(totdegree)
Nnlm = length(aobasis.prodbasis.sparsebasis.spec)
Nnuc = length(aobasis.nuclei)

for I = 1:Nnuc 
   XI = X .- Ref(aobasis.nuclei[I].rr)
   xI = norm.(XI)
   Rnl = evaluate(bRnl, xI)
   Ylm = evaluate(bYlm_, XI)
   for k = 1:Nnlm 
      nlm = aobasis.prodbasis.sparsebasis.spec[k]
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

@info("---------- Lux tests ----------")
aobasis_layer = ACEpsi.AtomicOrbitals.lux(aobasis)
pooling_layer = ACEpsi.lux(pooling)

# set up null states
ps, st = LuxCore.setup(MersenneTwister(1234), aobasis_layer)
st1 = (Σ = Σ, )

@info("Checking layers working fine seperately")
print_tf(@test aobasis_layer(X, ps, st1)[1] ≈ bϕnlm)
print_tf(@test pooling_layer(bϕnlm, ps, st1)[1] ≈ A)
println()

@info("Checking Chain towards A basis")
using Lux: Chain
tryChain = Chain(; aobasis = aobasis_layer, pooling = pooling_layer)
chain_ps, chain_st = LuxCore.setup(MersenneTwister(1234), tryChain)
chain_st = (aobasis = (Σ = Σ, ), pooling = (Σ = Σ, ))
# try Chain is as expected
print_tf(@test tryChain(X, chain_ps, chain_st)[1] ≈ A)

println()


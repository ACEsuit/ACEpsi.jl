using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: AtomicOrbitalsBasis, Nuc, make_nlms_spec, ProductBasis, evaluate
using ACEpsi: BackflowPooling, BFwf_lux, setupBFState,Jastrow
using ACEbase.Testing: print_tf
using LuxCore
using Lux
using Zygote
using Optimisers # mainly for the destrcuture(ps) functionx
using Random
using HyperDualNumbers: Hyper
using Printf
using LinearAlgebra
using ACEbase.Testing: fdtest

function grad_test2(f, df, X::AbstractVector)
   F = f(X) 
   ∇F = df(X)
   nX = length(X)
   EE = Matrix(I, (nX, nX))
   
   for h in 0.1.^(3:12)
      gh = [ (f(X + h * EE[:, i]) - F) / h for i = 1:nX ]
      @printf(" %.1e | %.2e \n", h, norm(gh - ∇F, Inf))
   end
end

Rnldegree = 4
Ylmdegree = 4
totdegree = 8
Nel = 5
X = randn(SVector{3, Float64}, Nel)

# wrap it as HyperDualNumbers
x2dualwrtj(x, j) = SVector{3}([Hyper(x[i], i == j, i == j, 0) for i = 1:3])

hX = [x2dualwrtj(x, 1) for x in X]
Σ = rand(spins(), Nel)

nuclei = [ Nuc(3 * rand(SVector{3, Float64}), 1.0) for _=1:3 ]
##

# Defining AtomicOrbitalsBasis
bRnl = ACEpsi.AtomicOrbitals.RnlExample(Rnldegree)
bYlm = RYlmBasis(Ylmdegree)

BFwf_chain = BFwf_lux(Nel, bRnl, bYlm, nuclei; totdeg = totdegree, ν = 2)
ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)

@info("Test evaluate")
A1 = BFwf_chain(X, ps, st)
hA1 = BFwf_chain(hX, ps, st)

@assert hA1[1].value ≈ A1[1]

@info("Test Zygote API")
ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)
y, st = Lux.apply(BFwf_chain, X, ps, st)

Zygote.gradient(x -> BFwf_chain(x, ps, st)[1], X)[1]
@info("Test ∇ψ w.r.t. X")
F(X) = BFwf_chain(X, ps, st)[1]
dF(X) = Zygote.gradient(x -> BFwf_chain(x, ps, st)[1], X)[1]
fdtest(F, dF, X, verbose = true)


## Pullback API to capture change in state
Zygote.gradient(p -> BFwf_chain(X, p, st)[1], ps)[1]
@info("Test ∇ψ w.r.t. parameters")
W0, re = destructure(ps)
Fp = w -> BFwf_chain(X, re(w), st)[1]
dFp = w -> ( gl = Zygote.gradient(p -> BFwf_chain(X, p, st)[1], ps)[1]; destructure(gl)[1])
grad_test2(Fp, dFp, W0)

@info("Test Δψ w.r.t. X")

@info("Test gradp Δψ")




using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: Nuc, make_nlms_spec, evaluate
using ACEpsi: BackflowPooling, BFwf_lux, setupBFState, JPauliNet 
using ACEbase.Testing: print_tf, fdtest
using ACEpsi.vmc: gradient, laplacian, grad_params
using ACEbase.Testing: print_tf, fdtest
using LuxCore
using Lux
using Zygote
using Optimisers # mainly for the destrcuture(ps) function
using Random
using Printf
using LinearAlgebra
using BenchmarkTools

using HyperDualNumbers: Hyper


Nel = 5
X = randn(SVector{3, Float64}, Nel)
Σ = rand(spins(), Nel)
nuclei = [ Nuc(3 * rand(SVector{3, Float64}), 1.0) for _=1:3 ]

# wrap it as HyperDualNumbers
x2dualwrtj(x, j) = SVector{3}([Hyper(x[i], i == j, i == j, 0) for i = 1:3])
hX = [x2dualwrtj(x, 0) for x in X]
hX[1] = x2dualwrtj(X[1], 1) # test eval for grad wrt x coord of first elec

js = JPauliNet(nuclei)
jastrow_layer = ACEpsi.lux(js)
ps, st = LuxCore.setup(MersenneTwister(1234), jastrow_layer)
st = (Σ = Σ,)

A1 = jastrow_layer(X, ps, st)
hA1 = jastrow_layer(hX, ps, st)
print_tf(@test hA1[1].value ≈ A1[1])

@info("Test ∇ψ w.r.t. X")
y, st = Lux.apply(jastrow_layer, X, ps, st)

F(X) = jastrow_layer(X, ps, st)[1]
dF(X) = Zygote.gradient(x -> jastrow_layer(x, ps, st)[1], X)[1]
fdtest(F, dF, X, verbose = true)


using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: AtomicOrbitalsBasis, Nuc, make_nlms_spec, ProductBasis, evaluate
using ACEpsi: BackflowPooling, BFwf_lux, setupBFState,Jastrow
using ACEbase.Testing: print_tf
using LuxCore
using Lux
using Zygote
using Optimisers # mainly for the destrcuture(ps) function
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

hX = [x2dualwrtj(x, 0) for x in X]
hX[1] = x2dualwrtj(X[1], 1) # test eval for grad wrt x coord of first elec

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

@info("Test consistency with HyperDualNumbers")
for _ = 1:30
   local X = randn(SVector{3, Float64}, Nel)
   local Σ = rand(spins(), Nel)
   local hdF = [zeros(3) for _ = 1:Nel]
   local hX = [x2dualwrtj(x, 0) for x in X]
   for i = 1:3
      for j = 1:Nel
         hX[j] = x2dualwrtj(X[j], i) # ∂Ψ/∂xj_{i}
         hdF[j][i] = BFwf_chain(hX, ps, st)[1].epsilon1
         hX[j] = x2dualwrtj(X[j], 0)
      end
   end
   print_tf(@test dF(X) ≈ hdF)
end
println()


## Pullback API to capture change in state
Zygote.gradient(p -> BFwf_chain(X, p, st)[1], ps)[1]
@info("Test ∇ψ w.r.t. parameters")
W0, re = destructure(ps)
Fp = w -> BFwf_chain(X, re(w), st)[1]
dFp = w -> ( gl = Zygote.gradient(p -> BFwf_chain(X, p, st)[1], ps)[1]; destructure(gl)[1])
grad_test2(Fp, dFp, W0)


@info("Test Δψ w.r.t. X using HyperDualNumbers")

X = [randn(3) for _ = 1:Nel]
hX = [x2dualwrtj(x, 0) for x in X]
Σ = rand(spins(), Nel)
F(x) = BFwf_chain(x, ps, st)[1]
function ΔF(x)
   ΔΨ = 0.0
   hX = [x2dualwrtj(xx, 0) for xx in x]
   for i = 1:3
      for j = 1:Nel
         hX[j] = x2dualwrtj(x[j], i) # ∂Φ/∂xj_{i}
         ΔΨ += BFwf_chain(hX, ps, st)[1].epsilon12
         hX[j] = x2dualwrtj(x[j], 0)
      end
   end
   return ΔΨ
end
Δ1 = ΔF(X)
f0 = F(X)

for h in  0.1.^(1:8)
   Δfh = 0.0
   for i = 1:Nel
      for j = 1:3
         XΔX_add, XΔX_sub = deepcopy(X), deepcopy(X)
         XΔX_add[i][j] += h
         XΔX_sub[i][j] -= h
         Δfh += (F(XΔX_add) - f0) / h^2
         Δfh += (F(XΔX_sub) - f0) / h^2
      end
   end
   @printf(" %.1e | %.2e \n", h, abs(Δfh - Δ1))
end

# [ tr( hessian(xx -> gfun(xx).dot.W[i], xx) ) 
# for i = 1:length(Δ1.dot.W) ]

@info("Test gradp Δψ")



### test
# using ACEpsi, Polynomials4ML, StaticArrays, Test 
# using Polynomials4ML: natural_indices, degree, SparseProduct
# using ACEpsi.AtomicOrbitals: AtomicOrbitalsBasis, Nuc, make_nlms_spec, ProductBasis, evaluate
# using ACEpsi: BackflowPooling, BFwf_lux, setupBFState,Jastrow
# using ACEbase.Testing: print_tf
# using LuxCore
# using Lux
# using Zygote
# using Optimisers # mainly for the destrcuture(ps) functionx
# using Random
# using HyperDualNumbers: Hyper
# using Printf
# using LinearAlgebra
# using ACEbase.Testing: fdtest

# function grad_test2(f, df, X::AbstractVector)
#    F = f(X) 
#    ∇F = df(X)
#    nX = length(X)
#    EE = Matrix(I, (nX, nX))
   
#    for h in 0.1.^(3:12)
#       gh = [ (f(X + h * EE[:, i]) - F) / h for i = 1:nX ]
#       @printf(" %.1e | %.2e \n", h, norm(gh - ∇F, Inf))
#    end
# end

# Rnldegree = 4
# Ylmdegree = 4
# totdegree = 8
# Nel = 5
# X = randn(SVector{3, Float64}, Nel)

# # wrap it as HyperDualNumbers
# x2dualwrtj(x, j) = SVector{3}([Hyper(x[i], i == j, i == j, 0) for i = 1:3])

# hX = [x2dualwrtj(x, 1) for x in X]
# Σ = rand(spins(), Nel)

# nuclei = [ Nuc(3 * rand(SVector{3, Float64}), 1.0) for _=1:3 ]
# ##

# # Defining AtomicOrbitalsBasis
# bRnl = ACEpsi.AtomicOrbitals.RnlExample(Rnldegree)
# bYlm = RYlmBasis(Ylmdegree)


# totdeg =  totdegree
# spec1p = make_nlms_spec(bRnl, bYlm; 
#                           totaldegree = totdeg)

# aobasis = AtomicOrbitalsBasis(bRnl, bYlm; totaldegree = totdeg, nuclei = nuclei, )
# pooling = BackflowPooling(aobasis)


# tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
# default_admissible = bb -> (length(bb) == 0) || (sum(b[1] - 1 for b in bb ) <= totdeg)
# using ACEcore.Utils: gensparse
# ν = 2
# specAA = gensparse(; NU = ν, tup2b = tup2b, admissible = default_admissible,
#                         minvv = fill(0, ν), 
#                         maxvv = fill(length(spec1p), ν), 
#                         ordered = true)
# spec = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]
# corr1 = Polynomials4ML.SparseSymmProd(spec)

# aobasis_layer = ACEpsi.AtomicOrbitals.lux(aobasis)
# pooling_layer = ACEpsi.lux(pooling)
# corr_layer = Polynomials4ML.lux(corr1)

# js = Jastrow(nuclei)
# jastrow_layer = ACEpsi.lux(js)
# reshape_func = x -> reshape(x, (size(x, 1), prod(size(x)[2:end])))
# bchain = Chain(; ϕnlm = aobasis_layer, bA = pooling_layer, reshape = WrappedFunction(reshape_func), 
#                         bAA = corr_layer, transpose_layer = WrappedFunction(transpose))
                        
#  #, hidden1 = Dense(length(corr1), Nel)
#   #, Mask = ACEpsi.MaskLayer(Nel), det = WrappedFunction(x -> det(x)))


# ps, st = setupBFState(MersenneTwister(1234), bchain, Σ)

# Zygote.jacobian(p -> bchain(hX, p, st)[1], ps)[1]

# Zygote.jacobian(x -> bchain(x, ps, st)[1], hX)[1]

# # the problem layer

#bchain = Chain(; ϕnlm = aobasis_layer, bA = pooling_layer, reshape = WrappedFunction(reshape_func), 
#                        bAA = corr_layer, transpose_layer = WrappedFunction(transpose), hidden1 = Dense(length(corr1), Nel))


#ps, st = setupBFState(MersenneTwister(1234), bchain, Σ)

#Zygote.jacobian(p -> bchain(hX, p, st)[1], ps)[1]

#Zygote.jacobian(x -> bchain(x, ps, st)[1], hX)[1]
 
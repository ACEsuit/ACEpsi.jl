using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: Nuc, make_nlms_spec, evaluate
using ACEpsi: BackflowPooling, BFwf_lux, setupBFState, Jastrow
using ACEpsi.vmc: evaluate, gradient, laplacian, grad_params
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

Rnldegree = n1 = 4
Ylmdegree = 4
totdegree = 8
Nel = 5
X = randn(SVector{3, Float64}, Nel)
Σ = rand(spins(), Nel)
nuclei = [ Nuc(3 * rand(SVector{3, Float64}), 1.0) for _=1:3 ]

# wrap it as HyperDualNumbers
x2dualwrtj(x, j) = SVector{3}([Hyper(x[i], i == j, i == j, 0) for i = 1:3])
hX = [x2dualwrtj(x, 0) for x in X]
hX[1] = x2dualwrtj(X[1], 1) # test eval for grad wrt x coord of first elec

##

# Defining AtomicOrbitalsBasis
n2 = 3 
Pn = Polynomials4ML.legendre_basis(n1+1)
spec = [(n1 = n1, n2 = n2, l = l) for n1 = 1:n1 for n2 = 1:n2 for l = 0:n1-1] 
ζ = rand(length(spec))
Dn = GaussianBasis(ζ)
bRnl = AtomicOrbitalsRadials(Pn, Dn, spec) 
bYlm = RYlmBasis(Ylmdegree)

# setup state
BFwf_chain = BFwf_lux(Nel, bRnl, bYlm, nuclei; totdeg = totdegree, ν = 2)
ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)

##

@info("Test evaluate")
A1 = BFwf_chain(X, ps, st)
hA1 = BFwf_chain(hX, ps, st)

print_tf(@test hA1[1].value ≈ A1[1])

println()

##
F(X) = BFwf_chain(X, ps, st)[1]

# @profview let  F = F, X = X
#    for i = 1:10_000
#        F(X)
#    end
# end

@btime F(X)


##

@info("Test ∇ψ w.r.t. X")
ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)
y, st = Lux.apply(BFwf_chain, X, ps, st)

F(X) = BFwf_chain(X, ps, st)[1]
dF(X) = Zygote.gradient(x -> BFwf_chain(x, ps, st)[1], X)[1]
fdtest(F, dF, X, verbose = true)

##

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

##

@info("Test ∇ψ w.r.t. parameters")
p = Zygote.gradient(p -> BFwf_chain(X, p, st)[1], ps)[1]
p, = destructure(p)

W0, re = destructure(ps)
Fp = w -> BFwf_chain(X, re(w), st)[1]
dFp = w -> ( gl = Zygote.gradient(p -> BFwf_chain(X, p, st)[1], ps)[1]; destructure(gl)[1])
grad_test2(Fp, dFp, W0)

##

@info("Test consistency when input isa HyperDualNumbers")
#hp = Zygote.gradient(p -> BFwf_chain(hX, p, st)[1], ps)[1]

#hp, = destructure(hp)
#P = similar(p)
#for i = 1:length(P)
#   P[i] = hp[i].value
#end

#print_tf(@test P ≈ p)

#println()

##

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

##

@info("Test gradp Δψ using HyperDualNumbers")
g_bchain = xx -> Zygote.gradient(p -> BFwf_chain(xx, p, st)[1], ps)[1]
g_bchain(hX)

using ACEpsi: zero!
using HyperDualNumbers

function grad_lap(g_bchain, x)
   function _mapadd!(f, dest::NamedTuple, src::NamedTuple) 
      for k in keys(dest)
         _mapadd!(f, dest[k], src[k])
      end
      return nothing 
   end
   _mapadd!(f, dest::Nothing, src) = nothing
   _mapadd!(f, dest::AbstractArray, src::AbstractArray) = 
            map!((s, d) -> d + f(s), dest, src, dest)

   Δ = zero!(g_bchain(x))
   hX = [x2dualwrtj(xx, 0) for xx in x]
   for i = 1:3
      for j = 1:length(x)
         hX[j] = x2dualwrtj(x[j], i)
         _mapadd!(ε₁ε₂part, Δ, g_bchain(hX))
         hX[j] = x2dualwrtj(x[j], 0)
      end
   end
   return Δ
end

function ΔF(x, ps)
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

function ∇ΔF(x, ps)
   g_bchain = xx -> Zygote.gradient(p -> BFwf_chain(xx, p, st)[1], ps)[1]
   p, = destructure(grad_lap(g_bchain, x))
   return p
end

#W0, re = destructure(ps)
#Fp = w -> ΔF(X, re(w))
#dFp = w -> ∇ΔF(X, re(w))
#fdtest(Fp, dFp, W0)
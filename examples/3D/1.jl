using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: Nuc, make_nlms_spec, evaluate
using ACEpsi: BackflowPooling, BFwf_lux, setupBFState, Jastrow, displayspec
using ACEpsi.vmc: gradient, laplacian, grad_params, EmbeddingW!, _invmap, VMC_multilevel, wf_multilevel, VMC, gd_GradientByVMC, gd_GradientByVMC_multilevel, AdamW, SR, SumH, MHSampler
using ACEbase.Testing: print_tf, fdtest
using LuxCore
using Lux
using Zygote
using Optimisers
using Random
using Printf
using LinearAlgebra
using BenchmarkTools
using HyperDualNumbers: Hyper

n1 = Rnldegree = 2
totdegree = 2
Nel = 4
X = randn(SVector{3, Float64}, Nel)
Σ = [↑,↑,↓,↓]
nuclei = [Nuc(SVector(0.0,0.0,-3.015/2), 2.0), Nuc(SVector(0.0,0.0,3.015/2), 1.0),Nuc(SVector(0.0,0.0,4.015/2), 1.0)]

# Defining AtomicOrbitalsBasis
n2 = 1
Pn = Polynomials4ML.legendre_basis(n1+1)
Ylmdegree = 2
spec = [[(n1 = 1, n2 = 1, l = 0), (n1 = 2, n2 = 1, l = 0), (n1 = 2, n2 = 1, l = 1)], [(n1 = 1, n2 = 1, l = 0), (n1 = 2, n2 = 1, l = 2)]]
speclist = [1,2,2]

bYlm = RRlmBasis(Ylmdegree)
bRnl = [AtomicOrbitalsRadials(Pn, SlaterBasis(10 * rand(length(spec[i]))), spec[speclist[i]]) for i = 1:length(spec)]
ν = 2    
Nbf = 3
BFwf_chain, spec, spec1p = wf, spec, spec1p = BFwf_lux(Nel, Nbf, speclist, bRnl, bYlm, nuclei, ACEpsi.TD.No_Decomposition())

ps, st = setupBFState(MersenneTwister(1234), wf, Σ)
wf(X, ps, st)


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

x2dualwrtj(x, j) = SVector{3}([Hyper(x[i], i == j, i == j, 0) for i = 1:3])
hX = [x2dualwrtj(x, 0) for x in X]
hX[1] = x2dualwrtj(X[1], 1) # test eval for grad wrt x coord of first elec


@info("Test evaluate")
A1 = BFwf_chain(X, ps, st)
hA1 = BFwf_chain(hX, ps, st)

print_tf(@test hA1[1].value ≈ A1[1])

println()

##
F(X) = BFwf_chain(X, ps, st)[1]

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
#grad_test2(Fp, dFp, W0)

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
X = randn(SVector{3, Float64}, Nel)
XX = [Vector(x) for x in X]
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
         XΔX_add, XΔX_sub = deepcopy(XX), deepcopy(XX)
         XΔX_add[i][j] += h
         XΔX_sub[i][j] -= h
         XΔX_add = [SVector{3, Float64}(x) for x in XΔX_add]
         XΔX_sub = [SVector{3, Float64}(x) for x in XΔX_sub]
         Δfh += (F(XΔX_add) - f0) / h^2
         Δfh += (F(XΔX_sub) - f0) / h^2
      end
   end
   @printf(" %.1e | %.2e \n", h, abs(Δfh - Δ1))
end

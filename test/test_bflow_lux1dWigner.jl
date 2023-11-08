using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: Nuc1d, make_nlms_spec, evaluate
using ACEpsi: BackflowPooling1d, setupBFState, Jastrow
using ACEpsi.vmc: gradient, laplacian, grad_params
using ACEpsi.vmc: grad, Eloc_Exp_TV_clip, rq_MC
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

totdegree = [34]
Nel = 30
rs = 1 # Wigner-Seitz radius r_s for 1D = 1/(2ρ); where ρ = N/L
ρ = 1 / (2 * rs) # (average density)
L = Nel / ρ # supercell size
b = 1 # harmonic trap strength (the larger the "flatter") or simply "width"

X = (rand(Nel).-1/2)*L # uniform distribution [-L/2,L/2]
Σ = Array{Char}(undef, Nel)
# paramagnetic 
for i = 1:Int(Nel / 2)
   Σ[i] = ↑
   Σ[Int(Nel / 2)+i] = ↓
end
hX = [Hyper(x, 0, 0, 0) for x in X]
hX[1] = Hyper(X[1], 1, 1, 0)

ord = length(totdegree)
Pn = Polynomials4ML.RTrigBasis(maximum(totdegree)+ord)
trans = x -> 2*pi*x/L
BF, spec, spec1p = ACEpsi.WIGwfTrig_lux(Nel, Pn; ν = ord, trans = trans, totdeg = totdegree[1])
ps, st = setupBFState(MersenneTwister(1234), BF, Σ)

## check spec if needed
function getnicespec(spec::Vector, spec1p::Vector)
    return [[spec1p[i] for i = spec[j]] for j = eachindex(spec)]
end
@show getnicespec(spec, spec1p);

A = BF(X, ps, st)
hA = BF(hX, ps, st)

print_tf(@test hA[1].value ≈ A[1])

println()

@info("Test ∇ψ w.r.t. X")
ps, st = setupBFState(MersenneTwister(1234), BF, Σ)
y, st = Lux.apply(BF, X, ps, st)

F(X) = BF(X, ps, st)[1]
dF(X) = Zygote.gradient(x -> BF(x, ps, st)[1], X)[1]
fdtest(F, dF, X, verbose = true)

@info("Test consistency with HyperDualNumbers")
for _ = 1:30
   local X = (rand(Nel).-1/2)*L # uniform distribution [-L/2,L/2]
   local Σ = rand(spins(), Nel)
   local hdF = randn(Nel)
   local hX = [Hyper(x, 0, 0, 0) for x in X]
   for i = 1:Nel
        hX[i] = Hyper(X[i], 1, 1, 0) # ∂Ψ/∂xj_{i}
        hdF[i] = BF(hX, ps, st)[1].epsilon1
        hX[i] = Hyper(X[i], 0, 0, 0)
   end
   print_tf(@test dF(X) ≈ hdF)
end
println()

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

@info("Test ∇ψ w.r.t. parameters")
p = Zygote.gradient(p -> BF(X, p, st)[1], ps)[1]
p, = destructure(p)

W0, re = destructure(ps)
Fp = w -> BF(X, re(w), st)[1]
dFp = w -> ( gl = Zygote.gradient(p -> BF(X, p, st)[1], ps)[1]; destructure(gl)[1])
grad_test2(Fp, dFp, W0)


@info("Test Δψ w.r.t. X using HyperDualNumbers")

X = (rand(Nel).-1/2)*L # uniform distribution [-L/2,L/2]
hX = [Hyper(x, 0, 0, 0) for x in X]
Σ = rand(spins(), Nel)
F(X) = BF(X, ps, st)[1]


function ΔF(X)
   ΔΨ = 0.0
   hX = [Hyper(x, 0, 0, 0) for x in X]
   for i = 1:Nel
         hX[i] = Hyper(X[i], 1, 1, 0) # ∂Φ/∂xj_{i}
         ΔΨ += BF(hX, ps, st)[1].epsilon12
         hX[i] = Hyper(X[i], 0, 0, 0) 
   end
   return ΔΨ
end

Δ1 = ΔF(X)
f0 = F(X)

for h in  0.1.^(1:13)
   Δfh = 0.0
   for i = 1:Nel
         XΔX_add, XΔX_sub = deepcopy(X), deepcopy(X)
         XΔX_add[i] += h
         XΔX_sub[i] -= h
         Δfh += (F(XΔX_add) - f0) / h^2
         Δfh += (F(XΔX_sub) - f0) / h^2
   end
   @printf(" %.1e | %.2e | %.10e \n", h, abs(Δfh - Δ1), Δfh)
end
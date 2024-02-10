using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: Nuc1d, make_nlms_spec, evaluate
using ACEpsi: BackflowPooling1d, setupBFState, Jastrow
using ACEpsi.vmc: gradient, Eloc_Exp_TV_clip, rq_MC
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

totdegree = [15, 5]
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
trans = x -> 2 * pi * x / L

J = ACEpsi.JCasino1dVb(0.04*L ,3, 3, L)
JS = ACEpsi.JCasinoChain(J)

_get_ord = bb -> sum([bb[i].n .!= 1 for i = 1:length(bb)]) == 0 ? 1 : sum([bb[i].n .!= 1 for i = 1:length(bb)])
    sd_admissible_func(ord,Deg) = bb -> (all([length(bb) == ord]) # must be of order ord (this truncation is allowed only because 1 is included in the basis expanding body-order 1), 
    # && (all([sum([bb[i].n .!= 1 for i = 1:length(bb)]) == 0]) # all of the basis in bb must be (1, σ)
        # || 
        && all([sum([ceil(((bb[i].n)-1)/2) for i = 2:length(bb)]) <= ceil((Deg[_get_ord(bb)]-1)/2)])#) # if the wave# of the basis is less than the max wave#
        && (bb[1].s == '∅') # ensure b-order=1 are of empty spin (can be deleted because I have enforced it in BFwfTrig_lux)
        && all([b.s != '∅' for b in bb[2:end] if b.n != 1]) # ensure (non-cst) b-order>1 term are of of non-empty spin
        && all([b.s == '∅' for b in bb[2:end] if b.n == 1])) # ensure cst is of ∅ spin to avoid repeats, e.g [(2,∅),(1,↑)] == [(2,∅),(1,↓)], but notice δ_σ↑ + δ_σ↓ == 1
sd_admissible = sd_admissible_func(ord, totdegree)

BF, spec, spec1p = ACEpsi.BFJwfTrig_lux(Nel, Pn, 2; ν = ord, trans = trans, totdeg = totdegree[1], sd_admissible = sd_admissible, Jastrow_chain = JS)
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
dF(X)

# dF(X) = begin
#    hdF = zeros(Nel)
#    for i = 1:Nel
#       hX[i] = Hyper(X[i], 1, 1, 0) # ∂Ψ/∂xj_{i}
#       hdF[i] = BF(hX, ps, st)[1].epsilon1
#       hX[i] = Hyper(X[i], 0, 0, 0)
#    end
#    hdF
# end

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
    @show norm(∇F, Inf)

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


# # pair potential
# using ACEpsi.vmc: gradient, laplacian, grad_params, SumH, MHSampler, VMC, gd_GradientByVMC, d1_lattice, adamW
# function v_ewald(x::AbstractFloat, b::Real, L::Real, M::Integer, K::Integer)

#    erfox(y) = (erf(y) - 2 / sqrt(pi) * y) / (y + eps(y)) + 2 / sqrt(pi)
#    f1(m) = (y = abs(x - m * L) / (2 * b); (sqrt(π) * erfcx(y) - erfox(y)) / (2 * b))
#    f2(n) = (G = 2 * π / L; expint((b * G * n)^2) * cos(G * n * x))

#    return sum(f1, -M:M) + sum(f2, 1:K) * 2 / L
# end
# M = 500
# K = 50
# vb(x) = v_ewald(x, b, L, M, K)
# V(X::AbstractVector) = sum(vb(X[i]-X[j]) for i = 1:length(X)-1 for j = i+1:length(X));
# # Mdelung energy
# Mad = (Nel / 2) * (vb(0.0) - sqrt(pi) / (2 * b))


# Kin(wf, X::AbstractVector, ps, st) = -0.5 * laplacian(wf, X, ps, st)
# Vext(wf, X::AbstractVector, ps, st) = 0.0
# Vee(wf, X::AbstractVector, ps, st) = V(X) + Mad

# # define lattice pts for sampler_restart # considering deleting this altogether
# spacing = L / Nel
# x0 = -L / 2 + spacing / 2
# Lattice = [x0 + (k - 1) * spacing for k = 1:Nel]
# d = d1_lattice(Lattice)

# burnin = 2000
# N_chain = 1000
# MaxIters = 200
# lr = 0.0002
# lr_dc = 100

# ham = SumH(Kin, Vext, Vee)
# sam = MHSampler(BF, Nel, Δt = 0.5*L , burnin = burnin, nchains = N_chain, d = d) # d = d for now

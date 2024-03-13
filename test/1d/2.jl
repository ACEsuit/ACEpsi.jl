using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: Nuc1d, make_nlms_spec, evaluate
using ACEpsi: BackflowPooling1d, setupBFState, Jastrow
using ACEpsi.vmc: gradient, Eloc_Exp_TV_clip, rq_MC, d1_lattice, EmbeddingW_J!,laplacian, grad_params, SumH, MHSampler, VMC, gd_GradientByVMC,adamW
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
using SpecialFunctions
using ForwardDiff: Dual
using Dates
import JSON3

totdegree = [3]
Nel = 6
Np = 16
Nu = 1
rs = 0.5 # Wigner-Seitz radius r_s for 1D = 1/(2ρ); where ρ = N/L
ρ = 1 / (2 * rs) # (average density)
L = Nel / ρ # supercell size
b = 1.0 # harmonic trap strength (the larger the "flatter") or simply "width"

X = (rand(Nel) .-1/2) * L # uniform distribution [-L/2,L/2]
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

J = ACEpsi.JCasino1dVb(0.3, Np, Nu, L)
JS = ACEpsi.JCasinoChain(J)

sd_admissible_func(ord, Deg) = bb -> (all([length(bb) == 1]) && (bb[1].s == '∅') )
sd_admissible = sd_admissible_func(ord, totdegree)

# BF, spec, spec1p = ACEpsi.BFwfTrig_lux(Nel, Pn; ν = ord, trans = trans,  totdeg = totdegree[1], sd_admissible = sd_admissible)

BF, spec, spec1p = ACEpsi.BFJwfTrig_lux(Nel, Pn, 2; ν = ord, trans = trans, totdeg = totdegree[1], sd_admissible = sd_admissible, Jastrow_chain = JS)
ps, st = setupBFState(MersenneTwister(1234), BF, Σ)

# ps.to_be_prod.layer_1.hidden1.W .= 0
# for i = 1:Int(Nel / 2)
#     ps.to_be_prod.layer_1.hidden1.W[i, i] = 1
#     ps.to_be_prod.layer_1.hidden1.W[Int(Nel / 2)+i, i] = 1
# end

## check spec if needed
getnicespec(spec::Vector, spec1p::Vector) = [[spec1p[i] for i = spec[j]] for j = eachindex(spec)]
@show getnicespec(spec, spec1p);

A = BF(X, ps, st)
hA = BF(hX, ps, st)
A = BF(X, ps, st)
hA = BF(hX, ps, st)
print_tf(@test hA[1].value ≈ A[1])


function v_ewald(x::AbstractFloat, b::Real, L::Real, M::Integer, K::Integer)

    erfox(y) = (erf(y) - 2 / sqrt(pi) * y) / (y + eps(y)) + 2 / sqrt(pi)
    f1(m) = (y = abs(x - m * L) / (2 * b); (sqrt(π) * erfcx(y) - erfox(y)) / (2 * b))
    f2(n) = (G = 2 * π / L; expint((b * G * n)^2) * cos(G * n * x))

    return sum(f1, -M:M) + sum(f2, 1:K) * 2 / L
end

M = 600
K = 40
vb(x) = v_ewald(x, b, L, M, K)
V(X::AbstractVector) = sum(vb(X[i]-X[j]) for i = 1:length(X)-1 for j = i+1:length(X));
# Mdelung energy
Mad = 0 # (Nel / 2) * (vb(0.0) - sqrt(pi) / (2 * b))

Kin(wf, X::AbstractVector, ps, st) = -0.5 * laplacian(wf, X, ps, st)
Vext(wf, X::AbstractVector, ps, st) = 0.0
Vee(wf, X::AbstractVector, ps, st) = V(X) + Mad

# # define lattice pts for sampler_restart # considering deleting this altogether
spacing = L / Nel
x0 = -L / 2 + spacing / 2
Lattice = [x0 + (k - 1) * spacing for k = 1:Nel]
d = d1_lattice(Lattice)

burnin = 1000
N_chain = 2000
MaxIters = 20000
Δt = 0.25 * L

ham = SumH(Kin, Vext, Vee)
sam = MHSampler(BF, Nel, Δt = Δt , burnin = burnin, nchains = N_chain, d = d) # d = d for now

lr_0  = 0.0015
lr_dc = 1000.0
epsilon = 1e-3
kappa_S = 0.95
kappa_m = 0.
opt_vmc = VMC(MaxIters, lr_0, ACEpsi.vmc.SR(0.0, epsilon, kappa_S, kappa_m, 
ACEpsi.vmc.QGT(), 
ACEpsi.vmc.no_scale(),
ACEpsi.vmc.no_constraint()), lr_dc = lr_dc)

@info("Running b=$(b) rs=$(rs) N=$(Nel) on serial")
# error function derived from MATH607
err_recip(K; L::Real=1, b::Real=1) = (1 / (pi * b)) * (L / (2 * pi * K * b))^3 * exp(-(2 * pi * b * K / L)^2)
err_real(M; L::Real=1, b::Real=1) = ((2 * b)^2 / (sqrt(2) * L^3)) * 1 / (M - 1)^2

@info("checking that error for this choice of truncation < 10^-8")
@assert err_recip(K; L=L, b=b) < 1e-6
@assert err_real(M; L=L, b=b) < 1e-6

BF, err_opt, ps = gd_GradientByVMC(opt_vmc, sam, ham, BF, ps, st) # remember to change index whenever needed

# Eref: -0.004626909921635635 N = 2, rs = 1.0, b = 1.0
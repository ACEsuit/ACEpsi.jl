using BenchmarkTools

using ACEpsi, StaticArrays, Test
using Polynomials4ML
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.vmc: d1_lattice, EmbeddingW_J!
using ACEpsi: BackflowPooling1d, setupBFState, BFJwfTrig_lux
using ACEpsi.vmc: gradient, laplacian, grad_params, SumH, MHSampler, VMC, gd_GradientByVMC, d1, adamW, sr
using LuxCore
using Lux
using Zygote
using Optimisers # mainly for the destrcuture(ps) function
using Random
using LinearAlgebra
using BenchmarkTools
using HyperDualNumbers: Hyper
using SpecialFunctions
using ForwardDiff: Dual
using Dates
import JSON3

Nel = N = 4
rs = 1.0 # Wigner-Seitz radius r_s for 1D = 1/(2ρ); where ρ = N/L
ρ = 1 / (2 * rs) # (average density)
L = Nel / ρ # supercell size

b = 1.0 # harmonic trap strength (the larger the "flatter") or simply "width"
Σ = Array{Char}(undef, Nel)
# paramagnetic 
for i = 1:Int(Nel / 2)
    Σ[i] = ↑
    Σ[Int(Nel / 2)+i] = ↓
end

# Defining OrbitalsBasis
totdegree = [5]
ord = length(totdegree)
Pn = Polynomials4ML.RTrigBasis(maximum(totdegree))
trans = (x -> (2 * pi * x / L)::Union{Float64, Dual{Nothing, Float64, 1}, Hyper{Float64}}) #::typeof(x))# ::Union{Float64, Dual{Nothing, Float64, 1}, Hyper{Float64}})

# @info("setting up old wf in new code")
_get_ord = bb -> sum([bb[i].n .!= 1 for i = 1:length(bb)]) == 0 ? 1 : sum([bb[i].n .!= 1 for i = 1:length(bb)])
sd_admissible_func(ord,Deg) = bb -> (all([length(bb) == ord]) # must be of order ord (this truncation is allowed only because 1 is included in the basis expanding body-order 1), 
    && all([sum([ceil(((bb[i].n)-1)/2) for i = 2:length(bb)]) <= ceil((Deg[_get_ord(bb)]-1)/2)])#) # if the wave# of the basis is less than the max wave#
    && (bb[1].s == '∅') # ensure b-order=1 are of empty spin (can be deleted because I have enforced it in BFwfTrig_lux)
    && all([b.s != '∅' for b in bb[2:end] if b.n != 1]) # ensure (non-cst) b-order>1 term are of of non-empty spin
    && all([b.s == '∅' for b in bb[2:end] if b.n == 1])) # ensure cst is of ∅ spin to avoid repeats, e.g [(2,∅),(1,↑)] == [(2,∅),(1,↓)], but notice δ_σ↑ + δ_σ↓ == 1
sd_admissible = sd_admissible_func(ord, totdegree)

# remember to switch to (or not) embed_diff_func 
wf, spec, spec1p = ACEpsi.BFwfTrig_lux(Nel, Pn; ν = ord, trans = trans,  totdeg = totdegree[1], sd_admissible = sd_admissible)
Wigner = false # this is always false in this script
diff_coord = true
ps, st = setupBFState(MersenneTwister(1234), wf, Σ)

# pair potential
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
lr = 0.01
lr_dc = 1000
Δt = 0.4*L

ham = SumH(Kin, Vext, Vee)
sam = MHSampler(wf, Nel, Δt=Δt , burnin = burnin, nchains = N_chain, d = d) # d = d for now

opt_vmc = VMC(MaxIters, lr, adamW(), lr_dc = lr_dc)
@info("Running b=$(b) rs=$(rs) N=$(Nel) on serial")

# error function derived from MATH607
err_recip(K; L::Real=1, b::Real=1) = (1 / (pi * b)) * (L / (2 * pi * K * b))^3 * exp(-(2 * pi * b * K / L)^2)
err_real(M; L::Real=1, b::Real=1) = ((2 * b)^2 / (sqrt(2) * L^3)) * 1 / (M - 1)^2

@info("checking that error for this choice of truncation < 10^-8")
@assert err_recip(K; L=L, b=b) < 1e-8
@assert err_real(M; L=L, b=b) < 1e-8

@info("Set-up done. Into VMC")
wf, err_opt, ps = gd_GradientByVMC(opt_vmc, sam, ham, wf, ps, st) # remember to change index whenever needed

## b=1 rs=1
UHF_minimalPW = -0.1594301
CASINO_VMC = -0.1710782

# ## b=1 rs=0.5
# UHF_minimalPW = 0.0891900
# CASINO_VMC = 0.0853158

# ## b=2 rs=1
# UHF_minimalPW = -0.05775642
# CASINO_VMC = -0.0610819

# ## b=1 rs=2
# UHF_minimalPW = -0.1716581
# CASINO_VMC = -0.2006410

# ## b=1 rs=5
# UHF_minimalPW = -0.1161756
# CASINO_VMC = -0.168710
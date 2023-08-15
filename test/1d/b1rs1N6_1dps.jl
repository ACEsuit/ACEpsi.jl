using ACEpsi, StaticArrays, Test
using Polynomials4ML
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.vmc: d1_lattice
using ACEpsi: BackflowPooling1d, BFwf1dps_lux, setupBFState, Jastrow
using ACEpsi.vmc: gradient, laplacian, grad_params, SumH, MHSampler, VMC, gd_GradientByVMC, d1, adamW, sr
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
using Dates, JLD

Nel = 6
rs = 1 # Wigner-Seitz radius r_s for 1D = 1/(2ρ); where ρ = N/L
ρ = 1 / (2 * rs) # (average density)
L = Nel / ρ # supercell size

b = 1 # harmonic trap strength (the larger the "flatter") or simply "width"
Σ = Array{Char}(undef, Nel)
# paramagnetic 
for i = 1:Int(Nel / 2)
    Σ[i] = ↑
    Σ[Int(Nel / 2)+i] = ↓
end

# Defining OrbitalsBasis
totdegree = [2]
ord = length(totdegree)
Pn = Polynomials4ML.RTrigBasis(maximum(totdegree)+ord)
trans = (x -> 2 * pi * x / L)

wf = BFwf1dps_lux(Nel, Pn; ν = ord, trans = trans)
ps, st = setupBFState(MersenneTwister(1234), wf, Σ)

p, = destructure(ps)
length(p)

# pair potential
function v_ewald(x::AbstractFloat, b::Real, L::Real, M::Integer, K::Integer)

    erfox(y) = (erf(y) - 2 / sqrt(pi) * y) / (y + eps(y)) + 2 / sqrt(pi)
    f1(m) = (y = abs(x - m * L) / (2 * b); (sqrt(π) * erfcx(y) - erfox(y)) / (2 * b))
    f2(n) = (G = 2 * π / L; expint((b * G * n)^2) * cos(G * n * x))

    return sum(f1, -M:M) + sum(f2, 1:K) * 2 / L
end
M = 500
K = 50

# error function derived from MATH607
err_recip(K; L::Real=1, b::Real=1) = (1 / (pi * b)) * (L / (2 * pi * K * b))^3 * exp(-(2 * pi * b * K / L)^2)
err_real(M; L::Real=1, b::Real=1) = ((2 * b)^2 / (sqrt(2) * L^3)) * 1 / (M - 1)^2

@info("checking that error for this choice of truncation < 10^-8")
@assert err_recip(K; L=L, b=b) < 1e-8
@assert err_real(M; L=L, b=b) < 1e-8

vb(x) = v_ewald(x, b, L, M, K)
V(X::AbstractVector) = sum(vb(X[i]-X[j]) for i = 1:length(X)-1 for j = i+1:length(X));
# Mdelung energy
Mad = (Nel / 2) * (vb(0.0) - sqrt(pi) / (2 * b))

Kin(wf, X::AbstractVector, ps, st) = -0.5 * laplacian(wf, X, ps, st)
Vext(wf, X::AbstractVector, ps, st) = 0.0
Vee(wf, X::AbstractVector, ps, st) = V(X) + Mad

# define lattice pts for sampler_restart
spacing = L / Nel
x0 = -L / 2 + spacing / 2
Lattice = [x0 + (k - 1) * spacing for k = 1:Nel]
d = d1_lattice(Lattice)

burnin = 10
nchains = 600
MaxIter = 600

ham = SumH(Kin, Vext, Vee)
sam = MHSampler(wf, Nel, Δt = 0.5, burnin = burnin, nchains = nchains, d = d)

opt_vmc = VMC(MaxIter, 0.02, adamW(), lr_dc = 100)

# # save_data
# results_dir = @__DIR__() * "/jellium_data/b1rs1N$(Nel)" * string(Dates.now()) * "/"
# mkpath(results_dir)
# ## save initial config
# @info("initial config saved at : ", results_dir)
# save(results_dir * "Config_b1rs1N$(Nel).jld", "Nel", Nel, "Ord" , ord, "Deg" , totdegree, "MaxIters" , MaxIter, "burnin" , burnin, "N_chain" , nchains)

@info("Set-up done. Into VMC")
wf, err_opt, ps = gd_GradientByVMC(opt_vmc, sam, ham, wf, ps, st)

E_RHF = -0.152250987350


# using Plots
# p = plot(err_opt/N, w = 3)
# hline!([E_HF], lw=3, label="RHF($E_RHF)"

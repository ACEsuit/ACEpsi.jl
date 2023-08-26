using ACEpsi, StaticArrays, Test
using Polynomials4ML
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.vmc: d1_lattice
using ACEpsi: BackflowPooling1d, BFwf1dps_lux, BFwf1dps_lux2,setupBFState, Jastrow
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
totdegree = [4]
ord = length(totdegree)
Pn = Polynomials4ML.RTrigBasis(maximum(totdegree) + ord)
length(Pn)
trans = (x -> 2 * pi * x / L)
# sd_admissible = bb -> (bb[1].s == '∅') && all([b.s != '∅' for b in bb[2:end]]) # what is intended for BFwf1dps_lux2
# copied from ACESchrodinger
_get_ord = bb -> sum([bb[i].n .!= 1 for i = 1:length(bb)]) == 0 ? 1 : sum([bb[i].n .!= 1 for i = 1:length(bb)])

sd_admissible_func(ord,Deg) = bb -> (all([length(bb) == ord]) # must be of order ord, and 
                                     && (all([sum([bb[i].n .!= 1 for i = 1:length(bb)]) == 0]) # all of the basis in bb must be (1, σ)
                                         || all([sum([bb[i].n for i = 1:length(bb)]) <= Deg[_get_ord(bb)] + ord]))) # if the degree of the basis is less then the maxdegree, "+ ord" since we denote degree 0 = 1
sd_admissible = sd_admissible_func(ord,totdegree[1])
wf, spec, spec1p = BFwf1dps_lux(Nel, Pn; ν = ord, trans = trans,  totdeg = totdegree[1], sd_admissible = sd_admissible)
ps, st = setupBFState(MersenneTwister(1234), wf, Σ)

using Random

X = [-0.7130696735938061
-0.9975858483890182
-0.12499593203823238
-1.3094355839415257
 0.2674274610827771
-1.1597926515238148]

wf(X, ps, st)

# ps.hidden1.W
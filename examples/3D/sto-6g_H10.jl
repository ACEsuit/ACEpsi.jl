using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: Nuc, make_nlms_spec, evaluate
using ACEpsi: BackflowPooling, BFwf_lux, setupBFState, Jastrow, displayspec
using ACEpsi.vmc: gradient, laplacian, grad_params, SumH, MHSampler, VMC, gd_GradientByVMC
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

n1 = Rnldegree = 1
Ylmdegree = 0
totdegree = 20
Nel = 10
X = randn(SVector{3, Float64}, Nel)
Σ = [↑,↑,↑,↑,↑,↓,↓,↓,↓,↓]
spacing = 1.0
nuclei = [Nuc(SVector(0.0,0.0,(i-1/2-Nel/2) * spacing), 1.0) for i = 1:Nel]
Pn = Polynomials4ML.legendre_basis(n1+1)
spec = [(n1 = 1, n2 = 1, l = 0)] 

# Ref: https://link.springer.com/book/10.1007/978-90-481-3862-3: P235
# STO: 0.7790 * e^(-1.24 * r)
# ϕ_1s(1, r) = \sum_(k = 1)^K d_1s,k g_1s(α_1k, r)
# g_1s(α, r) = (2α/π)^(3/4) * exp(-αr^2): α ∼ ζ, g ∼ D
# sto-3g: Ref: https://www.basissetexchange.org/
# BASIS SET: (3s) -> [1s]
# H    S
#      0.3425250914E+01       0.1543289673E+00
#      0.6239137298E+00       0.5353281423E+00
#      0.1688554040E+00       0.4446345422E+00

# ζ = [[0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00]]
# D = [[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]]
# D[1] = [(2 * ζ[1][i]/pi)^(3/4) * D[1][i] for i = 1:length(ζ[1])]
# P_0(x) = 1/sqrt(2)
# Y_0(x) = 1/(2*sqrt(pi))

# sto-6g: Ref: https://www.basissetexchange.org/
# BASIS SET: (6s) -> [1s]
# H    S
#      0.3552322122E+02       0.9163596281E-02
#      0.6513143725E+01       0.4936149294E-01
#      0.1822142904E+01       0.1685383049E+00
#      0.6259552659E+00       0.3705627997E+00
#      0.2430767471E+00       0.4164915298E+00
#      0.1001124280E+00       0.1303340841E+00

ζ = [[0.3552322122E+02, 0.6513143725E+01, 0.1822142904E+01,0.6259552659E+00, 0.2430767471E+00, 0.1001124280E+00]]
D = [[0.9163596281E-02, 0.4936149294E-01,0.1685383049E+00,0.3705627997E+00,  0.4164915298E+00, 0.1303340841E+00]]
D[1] = [(2 * ζ[1][i]/pi)^(3/4) * D[1][i] for i = 1:length(ζ[1])] * sqrt(2) * 2 * sqrt(pi)

Dn = STO_NG((ζ, D))
bRnl = AtomicOrbitalsRadials(Pn, Dn, spec) 
bYlm = RYlmBasis(Ylmdegree)

ord = 2
wf, spec, spec1p = BFwf_chain, spec, spec1p  = BFwf_lux(Nel, bRnl, bYlm, nuclei; totdeg = totdegree, ν = ord)

ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ) # ps.hidden1.W: Nels * basis
p, = destructure(ps)
length(p)

ham = SumH(nuclei)
sam = MHSampler(wf, Nel, nuclei, Δt = 0.5, burnin = 1000, nchains = 2000)

#using BenchmarkTools # N = 10, nuc = 10  # 10, 265, 3685
#@btime $wf($X, $ps, $st)               # ord = 1: 17.847 μs (27 allocations: 8.27 KiB)
                                        # ord = 2: 22.100 μs (27 allocations: 8.27 KiB)
                                        # ord = 3: 78.104 μs (27 allocations: 8.27 KiB)

#@btime $gradient($wf, $X, $ps, $st)    # ord = 1: 63.649 μs (305 allocations: 117.30 KiB)
                                        # ord = 2: 93.476 μs (307 allocations: 160.55 KiB)
                                        # ord = 3: 704.326 μs (307 allocations: 698.33 KiB)

#@btime $grad_params($wf, $X, $ps, $st) # ord = 1: 56.309 μs (282 allocations: 58.45 KiB)
                                        # ord = 2: 83.227 μs (284 allocations: 98.30 KiB)
                                        # ord = 3: 623.401 μs (284 allocations: 632.67 KiB)

#@btime $laplacian($wf, $X, $ps, $st)   # ord = 1: 762.276 μs (881 allocations: 837.47 KiB)
                                        # ord = 2: 1.763 ms (881 allocations: 837.47 KiB)
                                        # ord = 3: 20.683 ms (881 allocations: 837.47 KiB)

opt_vmc = VMC(5000, 0.015, ACEpsi.vmc.adamW(); lr_dc = 100.0)
wf, err_opt, ps = gd_GradientByVMC(opt_vmc, sam, ham, wf, ps, st)

## FCI: -23.1140: ord = 2: -23.3829
## UHF: -23.0414: ord = 1: -23.0432

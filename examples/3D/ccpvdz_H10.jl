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

n1 = Rnldegree = 2
Ylmdegree = 1
totdegree = 30
Nel = 10
X = randn(SVector{3, Float64}, Nel)
Σ = [↑,↑,↑,↑,↑,↓,↓,↓,↓,↓]
spacing = 1.0
nuclei = [Nuc(SVector(0.0,0.0,(i-1/2-Nel/2) * spacing), 1.0) for i = 1:Nel]
Pn = Polynomials4ML.legendre_basis(n1)
spec = [(n1 = 1, n2 = 1, l = 0), (n1 = 1, n2 = 2, l = 0), (n1 = 2, n2 = 1, l = 1)]

# Ref: http://www.grant-hill.group.shef.ac.uk/ccrepo/hydrogen/hbasis.php
# (4s,1p) -> [2s,1p]
# H    S
#      1.301000E+01           1.968500E-02           0.000000E+00
#      1.962000E+00           1.379770E-01           0.000000E+00
#      4.446000E-01           4.781480E-01           0.000000E+00
#      1.220000E-01           5.012400E-01           1.000000E+00
# H    P
#      7.270000E-01           1.0000000

ζ = [[1.301000E+01, 1.962000E+00, 4.446000E-01, 1.220000E-01], [1.220000E-01], [7.270000E-01]]
D = [[1.968500E-02, 1.379770E-01, 4.781480E-01, 5.012400E-01], [1.0000000], [1.0000000]]
D[1] = [(2 * ζ[1][i]/pi)^(3/4) * D[1][i] for i = 1:length(ζ[1])] * sqrt(2) * 2 * sqrt(pi)
D[2] = [(2 * ζ[2][i]/pi)^(3/4) * D[2][i] for i = 1:length(ζ[2])] * sqrt(2) * 2 * sqrt(pi)

Dn = STO_NG((ζ, D))
bRnl = AtomicOrbitalsRadials(Pn, Dn, spec) 
bYlm = RYlmBasis(Ylmdegree)

ord = 1
wf, spec, spec1p = BFwf_chain, spec, spec1p  = BFwf_lux(Nel, bRnl, bYlm, nuclei; totdeg = totdegree, ν = ord)

ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)
p, = destructure(ps)
length(p)
wf(X, ps, st)
@profview begin for i = 1:100000 wf(X, ps, st) end end

ham = SumH(nuclei)
sam = MHSampler(wf, Nel, nuclei, Δt = 0.5, burnin = 1000, nchains = 2000)

#using BenchmarkTools # N = 10, nuc = 10  # 50, 6325, 408425
#@btime $wf($X, $ps, $st)               # ord = 1: 22.764 μs (17 allocations: 19.67 KiB)
                                        # ord = 2: 119.143 μs (17 allocations: 19.67 KiB)
                                        # ord = 3: 9.800 ms (17 allocations: 19.67 KiB)

#@btime $gradient($wf, $X, $ps, $st)    # ord = 1: 81.785 μs (165 allocations: 153.27 KiB)
                                        # ord = 2: 742.233 μs (167 allocations: 1.11 MiB)
                                        # ord = 3: 60.314 ms (167 allocations: 62.47 MiB)

#@btime $grad_params($wf, $X, $ps, $st) # ord = 1: 74.807 μs (142 allocations: 94.42 KiB)
                                        # ord = 2: 732.846 μs (144 allocations: 1.05 MiB)
                                        # ord = 3: 59.758 ms (144 allocations: 62.41 MiB)

#@btime $laplacian($wf, $X, $ps, $st)   # ord = 1: 1.215 ms (611 allocations: 2.20 MiB)
                                        # ord = 2: 25.875 ms (611 allocations: 2.20 MiB)
                                        # ord = 3: 2.648 s (611 allocations: 2.20 MiB)

opt_vmc = VMC(5000, 0.015, ACEpsi.vmc.adamW(); lr_dc = 300.0)
#wf, err_opt, ps = gd_GradientByVMC(opt_vmc, sam, ham, wf, ps, st)

## MRCI+Q: -23.5092
## UHF:    -23.2997

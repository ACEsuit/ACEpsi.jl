using LinearAlgebra
using BenchmarkTools
using ACEpsi, Polynomials4ML
using ACEpsi: setupBFState, Nuc, BFwf_lux
using ACEpsi.AtomicOrbitals: make_nlms_spec
using Polynomials4ML.Utils: gensparse
using Lux
using Random
using StaticArrays

function BFwf_lux_AA(Nel::Integer, bRnl, bYlm, nuclei; totdeg = 15, 
   ν = 3, T = Float64, 
   sd_admissible = bb -> (true),
   envelope = x -> x) # enveolpe to be replaced by SJ-factor

   spec1p = make_nlms_spec(bRnl, bYlm; 
                          totaldegree = totdeg)

   # size(X) = (nX, 3); length(Σ) = nX
   # aobasis = AtomicOrbitalsBasis(bRnl, bYlm; totaldegree = totdeg, nuclei = nuclei, )

   # define sparse for n-correlations
   tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
   default_admissible = bb -> (length(bb) == 0) || (sum(b[1] - 1 for b in bb ) <= totdeg)

   specAA = gensparse(; NU = ν, tup2b = tup2b, admissible = default_admissible,
                        minvv = fill(0, ν), 
                        maxvv = fill(length(spec1p), ν), 
                        ordered = true)
   spec = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]

   # further restrict
   spec = [t for t in spec if sd_admissible([spec1p[t[j]] for j = 1:length(t)])]

   # define n-correlation
   corr1 = Polynomials4ML.SparseSymmProd(spec)

   # ----------- Lux connections ---------
   # AtomicOrbitalsBasis: (X, Σ) -> (length(nuclei), nX, length(spec1))
   prodbasis_layer = ACEpsi.AtomicOrbitals.ProductBasisLayer(spec1p, bRnl, bYlm)
   aobasis_layer = ACEpsi.AtomicOrbitals.AtomicOrbitalsBasisLayer(prodbasis_layer, nuclei)

   # BackFlowPooling: (length(nuclei), nX, length(spec1 from totaldegree)) -> (nX, 3, length(nuclei), length(spec1))
   pooling = ACEpsi.BackflowPooling(aobasis_layer)
   pooling_layer = ACEpsi.lux(pooling)

   reshape_func = x -> reshape(x, (size(x, 1), prod(size(x)[2:end])))

   # (nX, 3, length(nuclei), length(spec1 from totaldegree)) -> (nX, length(spec))
   corr_layer = Polynomials4ML.lux(corr1)
   return Chain(; ϕnlm = aobasis_layer, bA = pooling_layer, reshape = WrappedFunction(reshape_func), 
   bAA = corr_layer, transpose_layer = WrappedFunction(transpose))
end

Rnldegree = 4
Ylmdegree = 4
totdegree = 8
Nel = 5
X = randn(SVector{3, Float64}, Nel)
Σ = rand(spins(), Nel)
nuclei = [ Nuc(3 * rand(SVector{3, Float64}), 1.0) for _=1:3 ]

##

# Defining AtomicOrbitalsBasis
bRnl = ACEpsi.AtomicOrbitals.RnlExample(Rnldegree)
bYlm = RYlmBasis(Ylmdegree)

# setup state
BFwf_chain = BFwf_lux(Nel, bRnl, bYlm, nuclei; totdeg = totdegree, ν = 2)
ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)
F(X) = BFwf_chain(X, ps, st)[1]

# Define a shorter Chain up to AA
BFwf_AA_chain = BFwf_lux_AA(Nel, bRnl, bYlm, nuclei; totdeg = totdegree, ν = 2)
ps2, st2 = setupBFState(MersenneTwister(1234), BFwf_AA_chain, Σ)
F2(X) = BFwf_AA_chain(X, ps2, st2)[1]

AA = F2(X)

# getting weight from dense_layer
W = ps.branch.bf.hidden1.W
typeof(AA) # Transpose{Float64, ObjectPools.FlexCachedArray{Float64,....}}
typeof(W)
AA_mat = Matrix(AA) # Matrix{Float64}
@assert W * AA ≈ W * AA_mat

@btime F(X) # 46.671 μs
@btime F2(X) # 22.180 μs
@btime W * AA # 37.171 μs
@btime W * AA_mat # 9.400 μs
@btime W * Matrix(AA) # 18.500 μs
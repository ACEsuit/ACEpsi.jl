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

_get_ord = bb -> sum([bb[i].n .!= 1 for i = 1:length(bb)]) == 0 ? 1 : sum([bb[i].n .!= 1 for i = 1:length(bb)])
sd_admissible_func(ord,Deg) = bb -> (all([length(bb) == ord]) # must be of order ord (this truncation is allowed only because 1 is included in the basis expanding body-order 1), 
    && all([sum([ceil(((bb[i].n)-1)/2) for i = 2:length(bb)]) <= ceil((Deg[_get_ord(bb)]-1)/2)])#) # if the wave# of the basis is less than the max wave#
    && (bb[1].s == '∅') # ensure b-order=1 are of empty spin (can be deleted because I have enforced it in BFwfTrig_lux)
    && all([b.s != '∅' for b in bb[2:end] if b.n != 1]) # ensure (non-cst) b-order>1 term are of of non-empty spin
    && all([b.s == '∅' for b in bb[2:end] if b.n == 1])) # ensure cst is of ∅ spin to avoid repeats, e.g [(2,∅),(1,↑)] == [(2,∅),(1,↓)], but notice δ_σ↑ + δ_σ↓ == 1
sd_admissible = sd_admissible_func(ord, totdegree)

ν = 1
T = Float64
    
totdeg = totdegree[1]
spec1p = [(n = n) for n = 1:totdeg]

l_trans = Lux.WrappedFunction(x -> trans.(x))
l_Pn = Polynomials4ML.lux(Pn)
# BackFlowPooling: (length(nuclei), nX, length(spec1 from totaldegree)) -> (nX, 3, length(nuclei), length(spec1))
pooling = ACEpsi.BackflowPooling1dps()
pooling_layer = ACEpsi.lux(pooling)

spec1p = ACEpsi.get_spec(spec1p)
spec = [[i] for i in eachindex(spec1p)] # spec of order 1
spec = [t for t in spec if spec1p[t[1]].s == '∅'] # body order 1 term should be ∅

if ν > 1
    # define sparse for (n-1)-correlations for order ≥ 2 terms
    tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
    default_admissible = bb -> (length(bb) == 0) || (sum(ceil((b.n - 1)/2) for b in bb ) <= ceil((totdeg+ν)/2)) # totdeg>1 is P4ML (unnatural) index for Rtrig

    specAA = gensparse(; NU = ν-1, tup2b = tup2b, admissible = default_admissible,
                        minvv = fill(0, ν-1), 
                        maxvv = fill(length(spec1p), ν-1), 
                        ordered = true) ## gensparse automatically order the spec (it assumes S_N symmetry)
    spec_bf = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]

    # combining with order 1 orital
    spec_bf = [cat(b, t; dims=1) for b in spec for t in spec_bf]
    spec = cat(spec, spec_bf; dims=1) # this is the old spec format, except now each b∈spec only has b[2]≤…≤b[B] ordered

    # further restrict
    spec = [t for t in spec if sd_admissible([spec1p[t[j]] for j = 1:length(t)])]
end

# define n-correlation
corr1 = Polynomials4ML.SparseSymmProd(spec)

# (nX, 3, length(nuclei), length(spec1 from totaldegree)) -> (nX, length(spec))
corr_layer = Polynomials4ML.lux(corr1; use_cache = false)

reshape_func = x -> reshape(x, (size(x, 1), prod(size(x)[2:end])))

embed_layers = Tuple(collect(Lux.WrappedFunction(x -> ACEpsi.embed_diff_func(x, i)) for i = 1:Nel))
l_Pns = Tuple(collect(l_Pn for _ = 1:Nel))

_det = x -> size(x) == (1, 1) ? x[1,1] : det(Matrix(x))

BFwf_chain = Chain(; trans = l_trans, diff = Lux.BranchLayer(embed_layers...), Pn = Lux.Parallel(nothing, l_Pns...), bA = pooling_layer, reshape = WrappedFunction(reshape_func), bAA = corr_layer, hidden1 = LinearLayer(length(corr1), Nel), # hidden1 = ACEpsi.DenseLayer(Nel, length(corr1)), 
                     Mask = ACEpsi.MaskLayer(Nel), det = WrappedFunction(x -> _det(x)), logabs = WrappedFunction(x -> 2 * log(abs(x))))

X = rand(4)
BFwf_chain = Chain(; trans = l_trans, diff = Lux.BranchLayer(embed_layers...))
ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)
A = BFwf_chain(X, ps, st)[1]
BFwf_chain = Chain(; Pn = Lux.Parallel(nothing, l_Pns...))
ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)
P = BFwf_chain(A, ps, st)[1]

BFwf_chain = Chain(; bA = pooling_layer)
ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)
A = BFwf_chain(P, ps, st)[1]

BFwf_chain = Chain(; trans = l_trans, diff = Lux.BranchLayer(embed_layers...), Pn = Lux.Parallel(nothing, l_Pns...), bA = pooling_layer, reshape = WrappedFunction(reshape_func))
ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)
B = BFwf_chain(X, ps, st)[1]

BFwf_chain = Chain(; trans = l_trans, diff = Lux.BranchLayer(embed_layers...), Pn = Lux.Parallel(nothing, l_Pns...), bA = pooling_layer, reshape = WrappedFunction(reshape_func), bAA = corr_layer)
ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)
C = BFwf_chain(X, ps, st)[1]


BFwf_chain = Chain(; trans = l_trans, diff = Lux.BranchLayer(embed_layers...), Pn = Lux.Parallel(nothing, l_Pns...), bA = pooling_layer, reshape = WrappedFunction(reshape_func), bAA = corr_layer, hidden1 = LinearLayer(length(corr1), Nel))
ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)
D = BFwf_chain(X, ps, st)[1]

BFwf_chain = Chain(; trans = l_trans, diff = Lux.BranchLayer(embed_layers...), Pn = Lux.Parallel(nothing, l_Pns...), bA = pooling_layer, reshape = WrappedFunction(reshape_func), bAA = corr_layer, hidden1 = LinearLayer(length(corr1), Nel), # hidden1 = ACEpsi.DenseLayer(Nel, length(corr1)), 
            Mask = ACEpsi.MaskLayer(Nel))
ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)
E = BFwf_chain(X, ps, st)[1]

BFwf_chain = Chain(; trans = l_trans, diff = Lux.BranchLayer(embed_layers...), Pn = Lux.Parallel(nothing, l_Pns...), bA = pooling_layer, reshape = WrappedFunction(reshape_func), bAA = corr_layer, hidden1 = LinearLayer(length(corr1), Nel), # hidden1 = ACEpsi.DenseLayer(Nel, length(corr1)), 
                     Mask = ACEpsi.MaskLayer(Nel), det = WrappedFunction(x -> _det(x)), logabs = WrappedFunction(x -> 2 * log(abs(x))))
ps, st = setupBFState(MersenneTwister(1234), BFwf_chain, Σ)
F = BFwf_chain(X, ps, st)[1]


BFwf_chain = Chain(; Pn = Lux.Parallel(nothing, l_Pns...), bA = pooling_layer, reshape = WrappedFunction(reshape_func), bAA = corr_layer, hidden1 = LinearLayer(length(corr1), Nel),  
                         Mask = ACEpsi.MaskLayer(Nel), det = WrappedFunction(x -> _det(x)), logabs = WrappedFunction(x -> 2 * log(abs(x))))

deg_of_cos = 1
deg_of_mono = 1
J = ACEpsi.JCasino1dVb(0.05 * L , deg_of_mono, deg_of_cos, L)
Jastrow_chain = ACEpsi.JCasinoChain(J)

BFJwf = Chain(; trans = l_trans, diff = Lux.BranchLayer(embed_layers...), to_be_prod = Lux.BranchLayer(BFwf_chain, Jastrow_chain), Sum = WrappedFunction(x -> x[1] + 2*x[2][1]))
    
ps, st = setupBFState(MersenneTwister(1234), BFJwf, Σ)
F = BFJwf(X, ps, st)[1]


_getXineqj(Xs) = (Nel = length(Xs); [Xs[i][j] for i = 1:Nel for j = 1:Nel if i ≠ j])

# coordinate trans
L = J.L

# cos 
Np = J.Np

Np = 3
# cos(2πA/L xij) 
l_trig = Polynomials4ML.lux(RTrigBasis(Np))

#l_trigs = Tuple(collect(l_trig for _ = 2:2:2*Np+1)) 
CosChain = Chain(; getXineqj_cos = WrappedFunction(Xs -> _getXineqj(Xs)))
ps, st = setupBFState(MersenneTwister(1234), CosChain, Σ)
AA = CosChain(Xs, ps, st)[1]



CosChain = Chain(; abs = WrappedFunction(Xs -> norm.(Xs)))
ps, st = setupBFState(MersenneTwister(1234), CosChain, Σ)
AA = CosChain(AA, ps, st)[1]


CosChain = Chain(; SINCOS = l_trig)
ps, st = setupBFState(MersenneTwister(1234), CosChain, Σ)
BB = CosChain(AA, ps, st)[1]

CosChain = Chain(; getcos = WrappedFunction(x -> x[:, 2:2:2*Np+1]))
ps, st = setupBFState(MersenneTwister(1234), CosChain, Σ)
CC = CosChain(BB, ps, st)[1]

@assert length(2:2:2*Np+1) == Np

# cusp?
Nu = J.Nu
Nu = 4
l_mono = Polynomials4ML.lux(MonoBasis(Nu))

Lu = J.Lu

cusp_cut_Chain = Chain(; getXineqj_mono = WrappedFunction(Xs -> _getXineqj(Xs)))
ps, st = setupBFState(MersenneTwister(1234), cusp_cut_Chain, Σ)
AA = cusp_cut_Chain(Xs, ps, st)[1]

# x_ij
cusp_cut_Chain = Chain(; untrans = WrappedFunction(x -> norm.(x) .* L ./ (2pi)))
ps, st = setupBFState(MersenneTwister(1234), cusp_cut_Chain, Σ)
BB = cusp_cut_Chain(AA, ps, st)[1]

Θ(x) = x < zero(eltype(x)) ? zero(eltype(x)) : one(eltype(x)) # Heaviside
cut_layer = Lux.WrappedFunction(x -> Θ.(Lu .- x) .* ((x .- Lu) .^ 3))
    
## we need to "unstransform" coordinates
cusp_cut_Chain = Chain(; to_be_prod = Lux.BranchLayer(l_mono, cut_layer), prod = WrappedFunction(x -> x[1] .* x[2]))
ps, st = setupBFState(MersenneTwister(1234), cusp_cut_Chain, Σ)
CC = cusp_cut_Chain(BB, ps, st)[1]


CosChain = Chain(; getXineqj_cos = WrappedFunction(Xs -> _getXineqj(Xs)), abs = WrappedFunction(Xs -> norm.(Xs)), SINCOS = l_trig, getcos = WrappedFunction(x -> x[:, 2:2:2*Np+1])) # picking out only cosines
    
cusp_cut_Chain = Chain(; getXineqj_mono = WrappedFunction(Xs -> _getXineqj(Xs)) , untrans = WrappedFunction(x -> norm.(x) .* L ./ (2pi)), to_be_prod = Lux.BranchLayer(l_mono, cut_layer), prod = WrappedFunction(x -> x[1] .* x[2]))

l = Chain(; combine = Lux.BranchLayer(CosChain, cusp_cut_Chain), hiddenJS = Lux.Parallel(nothing, LinearLayer(Np, 1), LinearLayer(Nu + 1, 1)), poolJS = WrappedFunction(x -> sum(sum.(x))))
ps, st = setupBFState(MersenneTwister(1234), l, Σ)
CC = l(Xs, ps, st)[1]
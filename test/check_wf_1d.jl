using ACEpsi, StaticArrays
using Polynomials4ML
using Polynomials4ML: natural_indices, degree, SparseProduct, LinearLayer
using ACEpsi.vmc: d1_lattice
using ACEpsi: BackflowPooling1d,setupBFState, Jastrow, get_spec, embed_diff_func
using ACEpsi.vmc: gradient, laplacian, grad_params, SumH, MHSampler, VMC, gd_GradientByVMC, d1, adamW, sr
using LuxCore
using Lux
using Zygote
using Optimisers # mainly for the destrcuture(ps) function
using Random
using LinearAlgebra
using HyperDualNumbers: Hyper
using SpecialFunctions
using Polynomials4ML.Utils: gensparse

import ACEpsi: BFwfTrig_lux

Nel = 30
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
X = (rand(Nel).-1/2)*L

# One Body (Unrestricted Hartree-Fock) Wavefunction
totdegree = [21]
ord = length(totdegree)
Pn = Polynomials4ML.RTrigBasis(maximum(totdegree))
length(Pn)
trans = (x -> 2 * pi * x / L)

## sd_admissable
# sd_admissible = bb -> (bb[1].s == '∅') && all([b.s != '∅' for b in bb[2:end]]) # what is intended for BFwfTrig_lux

## sd_adm with total-degree-sparsification appears only starting body order 2
_get_ord = bb -> sum([bb[i].n .!= 1 for i = 1:length(bb)]) == 0 ? 1 : sum([bb[i].n .!= 1 for i = 1:length(bb)])
sd_admissible_func(ord,Deg) = bb -> (all([length(bb) == ord]) # must be of order ord (this truncation is allowed only because 1 is included in the basis expanding body-order 1), 
# && (all([sum([bb[i].n .!= 1 for i = 1:length(bb)]) == 0]) # all of the basis in bb must be (1, σ)
    # || 
    && all([sum([ceil(((bb[i].n)-1)/2) for i = 2:length(bb)]) <= ceil((Deg[_get_ord(bb)]-1)/2)])#) # if the wave# of the basis is less than the max wave#
    && (bb[1].s == '∅') # ensure b-order=1 are of empty spin (can be deleted because I have enforced it in BFwfTrig_lux)
    && all([b.s != '∅' for b in bb[2:end] if b.n != 1]) # ensure (non-cst) b-order>1 term are of of non-empty spin
    && all([b.s == '∅' for b in bb[2:end] if b.n == 1])) # ensure cst is of ∅ spin to avoid repeats, e.g [(2,∅),(1,↑)] == [(2,∅),(1,↓)], but notice δ_σ↑ + δ_σ↓ == 1
sd_admissible = sd_admissible_func(ord, totdegree)
wf, spec, spec1p = BFwfTrig_lux(Nel, Pn; ν = ord, trans = trans,  totdeg = totdegree[1], sd_admissible = sd_admissible)
ps, st = setupBFState(MersenneTwister(1234), wf, Σ)

sd_admissible = sd_admissible_func(ord, totdegree)


wf, spec, spec1p = BFwfTrig_lux(Nel, Pn; ν = ord, trans = trans,  totdeg = totdegree[1], sd_admissible = sd_admissible)
ps, st = setupBFState(MersenneTwister(1234), wf, Σ)

@show ps.hidden1.W

wf(X, ps, st)
# gradient(wf, X, ps, st)
# @profview let wf=wf, X=X
#     for nrun = 1:50
#        #laplacian(wf, X)
#        gradient(wf, X, ps, st) 
#     end
#  end
 

function getnicespec(spec::Vector, spec1p::Vector)
    return [[spec1p[i] for i = spec[j]] for j = eachindex(spec)]
end
@show getnicespec(spec, spec1p);

## 2-body model
totdegree = [21, 3]
ord = 2
Pn = Polynomials4ML.RTrigBasis(maximum(totdegree))
length(Pn)
trans = (x -> 2 * pi * x / L)

## sd_admissable (we now include total degree sparsification)
_get_ord = bb -> sum([bb[i].n .!= 1 for i = 1:length(bb)]) == 0 ? 1 : sum([bb[i].n .!= 1 for i = 1:length(bb)])
sd_admissible_func(ord,Deg) = bb -> (all([length(bb) == ord]) # must be of order ord (this truncation is allowed only because 1 is included in the basis expanding body-order 1), 
                                    # && (all([sum([bb[i].n .!= 1 for i = 1:length(bb)]) == 0]) # all of the basis in bb must be (1, σ)
                                        # || 
                                        && all([sum([bb[i].n for i = 2:length(bb)]) <= Deg[_get_ord(bb)]])#) # if the degree of the basis is less then the maxdegree
                                        && (bb[1].s == '∅') # ensure b-order=1 are of empty spin (can be deleted because I have enforced it in BFwfTrig_lux)
                                        && all([b.s != '∅' for b in bb[2:end] if b.n != 1]) # ensure (non-cst) b-order>1 term  are of of non-empty spin
                                        && all([b.s == '∅' for b in bb[2:end] if b.n == 1])) # to avoid repeats, e.g [(2,∅),(1,↑)] == [(2,∅),(1,↓)], but notice δ_σ↑ + δ_σ↓ == 1

sd_admissible = sd_admissible_func(ord, totdegree)
wf, spec, spec1p = BFwfTrig_lux(Nel, Pn; ν = ord, trans = trans,  totdeg = totdegree[1], sd_admissible = sd_admissible)
ps, st = setupBFState(MersenneTwister(1234), wf, Σ)
@show getnicespec(spec, spec1p);

## check by chain
totdeg = totdegree[1]
ν = ord
spec1p = [(n = n) for n = 1:totdeg]


l_trans = Lux.WrappedFunction(x -> trans.(x))
l_Pn = Polynomials4ML.lux(Pn)
# ----------- Lux connections ---------
# BackFlowPooling: (length(nuclei), nX, length(spec1 from totaldegree)) -> (nX, 3, length(nuclei), length(spec1))
pooling = ACEpsi.BackflowPooling1dps()
pooling_layer = ACEpsi.lux(pooling)

spec1p = get_spec(spec1p)
spec = [[i] for i in eachindex(spec1p)] # spec of order 1
# restrict (body order 1 term should be ∅)
spec = [t for t in spec if spec1p[t[1]].s == '∅']
@show spec;
if ν > 1
    # define sparse for (n-1)-correlations for indices representing order ≥ 2 
    tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
    default_admissible = bb -> (length(bb) == 0) || (sum(b.n - 1 for b in bb ) <= totdeg)

    specAA = gensparse(; NU = ν-1, tup2b = tup2b, admissible = default_admissible,
                        minvv = fill(0, ν-1), 
                        maxvv = fill(length(spec1p), ν-1), 
                        ordered = true)
    spec_bf = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]
    @show spec_bf;

    # combining with order 1 orbital
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

embed_layers = Tuple(collect(Lux.WrappedFunction(x -> embed_diff_func(x, i)) for i = 1:Nel))
l_Pns = Tuple(collect(l_Pn for _ = 1:Nel))


_det = x -> size(x) == (1, 1) ? x[1,1] : det(Matrix(x))

partial_chain = Chain(; trans = l_trans, diff = Lux.BranchLayer(embed_layers...), Pn = Lux.Parallel(nothing, l_Pns...), bA = pooling_layer, reshape = WrappedFunction(reshape_func), bAA = corr_layer, hidden1 = LinearLayer(length(corr1), Nel))
ps0, st = setupBFState(MersenneTwister(1234), partial_chain, Σ)

out, ps = partial_chain(X, ps0, st)

basis_mat_chain = Chain(; trans = l_trans, diff = Lux.BranchLayer(embed_layers...), Pn = Lux.Parallel(nothing, l_Pns...), bA = pooling_layer, reshape = WrappedFunction(reshape_func), bAA = corr_layer)
ps1, st1 = setupBFState(MersenneTwister(1234), basis_mat_chain, Σ)

out1, ps1 = basis_mat_chain(X, ps1, st1)
out1*(ps0.hidden1.W') == out

## customized inital state
for i = 1:Nel # Nel
    for j = eachindex(spec) # basis size
        if j > Int(Nel/2)
            ps0.hidden1.W[i,j] = 0.0
        end
    end
end
ps0.hidden1.W[1:30,1:16]
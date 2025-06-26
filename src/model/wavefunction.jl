using Polynomials4ML
using Polynomials4ML.Utils: gensparse
using Lux
using Random
using EquivariantTensors
using LinearAlgebra
export build_wavefunction, evalx, gradx, gradp, laplacian
using Polynomials4ML
import Polynomials4ML: _valtype

_valtype(::Polynomials4ML.RadialDecay, T::Type{<: Number}) = T

function build_wavefunction(mol::Molecule, basis_set::String, totdeg, ν; filename = "basis.json")
    basis = auto_load_basis(mol, basis_set; filename = filename)
    A_spec = get_spec1p(basis; spin = false)
    spec1p = get_spec1p(basis; spin = true)
    _totdegn = parseTotdegToInt(totdeg, spec1p) 
    tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
    filter = bb -> ((length(bb) == 0) ||  sum(b.s == '∅' for b in bb) == 1)
    specAA = gensparse(; NU = ν, tup2b = tup2b, admissible = bb -> (true), filter = filter, minvv = fill(0, ν), maxvv = fill(length(spec1p), ν), ordered = true)
    spec = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]
    admissible = bb -> ((length(bb) == 0) || (sum(b.n2 > _totdegn[b.I][length(bb)][b.l + 1] for b in bb)) == 0) 
    spec = [t for t in spec if admissible([spec1p[t[j]] for j = 1:length(t)])]
    
    branches = (; (Symbol("l", i) => b for (i, b) in enumerate(basis))...)
    AAbasis = SparseSymmProd(spec)

    l = Chain(; l_embed = Diff_layer(mol.nuclei), branch = Lux.Experimental.freeze(Parallel(hcat; branches...)), 
            pooling = BackflowPoolingLayer(A_spec, mol.Σ), corr = WrappedFunction(x -> Matrix(AAbasis(x)')), 
            linear = Dense(length(AAbasis), mol.Nel; use_bias=false), mask = MaskLayer(mol.Nel, mol.Σ), l_det = WrappedFunction(x -> det(x)))
    model = Chain(; branch = BranchLayer(; js = JastrowLayer(mol.Σ), bf = l, ), prod = WrappedFunction(x -> x[1] * x[2]), logabs = WrappedFunction(x -> 2 * log(abs(x))) )
    ps, st = Lux.setup(Random.default_rng(), model)
    return model, ps, st, spec, spec1p
end

evalx(wf, X::Vector{SVector{3, T}}, ps, st) where {T} = wf(X, ps, st)[1] 

using Zygote
using Optimisers: destructure
gradx(wf, X, ps, st) = Zygote.gradient(X -> wf(X, ps, st)[1], X)[1]
gradp(wf, X, ps, st) = destructure(Zygote.gradient(ps -> wf(X, ps, st)[1], ps)[1])[1]

using HyperDualNumbers: Hyper, Hyper256

function laphX(i, j, Nel, x)
    x2dualwrtj(x, j) = SVector{3}([Hyper(x[i], i == j, i == j, 0) for i = 1:3])
    hX = Vector{SVector{3, Hyper256}}(undef, Nel)
    for l = 1:Nel
        if l != j
            hX[l] = x2dualwrtj(x[l], 0)
        else
            hX[l] = x2dualwrtj(x[l], i)
        end 
    end
    return hX
end

function laplacian(wf, X::Vector{SVector{3, T}}, ps, st) where {T}
    Nel = length(X)
    ΔΨij = zeros(3, Nel)

    Threads.@threads for (i,j) in collect(Base.product(1:3, 1:Nel))
        ΔΨij[i, j] = wf(laphX(i, j, Nel, X), ps, st)[1].epsilon12
    end
    return sum(ΔΨij)
end

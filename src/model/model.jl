using StaticArrays, Random
using Lux, LuxCore, ChainRulesCore
using CUDA, KernelAbstractions
using GPUArrays
import KernelAbstractions as KA
using Polynomials4ML, EquivariantTensors
using ChemBasisSets: save_all_bases_to_json, load_basis_from_json
using KernelAbstractions: @atomic
using Polynomials4ML.Utils: gensparse
export auto_load_basis, get_spec1p, parseTotdegToInt
export build_wavefunction, laplacian

include("builder/difflayer.jl")
include("builder/fit_gaussian.jl")
include("builder/basis.jl")
include("builder/backflow.jl")
include("builder/sparsesymmprod.jl")
include("builder/dense.jl")
include("builder/mask.jl")
include("builder/det.jl")
include("builder/jastrow.jl")
include("builder/jnlayer.jl")
include("builder/utils.jl")

function build_wavefunction(
    mol::Molecule;
    family::Type{F} = Slater,      # Gaussian or Slater
    freeze_branches::Bool = true,  # true: freeze AO parameter；false: learnable
    spec_admissible = nothing, 
    spec1p_admissible = nothing
) where {F<:OrbitalFamily}
    basis = auto_load_basis(mol, "cc-pvtz", family; filename = "basis.json", spec1p_admissible = spec1p_admissible)
    A_spec = get_spec1p(basis; spin = false)
    spec1p = get_spec1p(basis; spin = true)
    AAbasis = SparseSymmProd(spec_admissible)
    AAbasis = sparsesymmprod(AAbasis.specs, AAbasis.ranges, AAbasis.hasconst)
    L = length(AAbasis)

    branches = (; (Symbol("l", i) => b for (i, b) in enumerate(basis))...)
    conn = (xs...) -> cat(xs...; dims = 3)

    branch_blk = Parallel(conn; branches...)
    branch_blk = freeze_branches ? Lux.Experimental.freeze(branch_blk) : branch_blk

    js_ee_layer = JastrowLayer(mol.spin)
    lbf = Chain(; branch   = branch_blk,
        pooling  = pooling(A_spec, mol.spin),
        corr     = AAbasis,
        reshape_ = WrappedFunction(AA -> reshape(permutedims(AA, (3, 1, 2)), L, :)),
        dense    = Dense(L, mol.Nel; use_bias = false),
        mask     = FusedMaskPermuteLayer(mol.spin))

    l = Chain(;l_embed  = DiffLayer(mol.nuclei),
        branchs = BranchLayer(; jsen = JNEnvelope(length(mol.nuclei), mol.spin, mol.Nel), bfs = lbf),
        prods   = WrappedFunction(x -> x[1] .* x[2]), 
        det_ = detlayer())

    model = Chain(;
            branch = BranchLayer(; js = js_ee_layer, bf = l),
            prod   = WrappedFunction(x -> 2 .* (x[1] .+ x[2])))

    readable_spec = displayspec(spec_admissible, spec1p)
    basis_spec1p = [b.Dn.spec for b in basis]
    ps, st = Lux.setup(Random.default_rng(), model)
    return model, ps, st, readable_spec, basis_spec1p
end

n1_ranges = Dict(
    0 => 1:4,   # s
    1 => 1:3,   # p
    2 => 1:2,   # d
    3 => 1:1    # f
)

alphas = Dict(
    :H   => Dict(0 => [8.0, 3.0], 1 => [0.80]),          
    :He  => Dict(0=>[8.0,3.0,1.2], 1=>[0.80,0.30], 2=>[0.80]),
    :Li  => Dict(0=>[12.0,5.0,2.0], 1=>[1.00,0.40], 2=>[0.80]),
    :Be  => Dict(0=>[20.0,8.0,3.0], 1=>[3.50,1.40], 2=>[0.90]),
    :B   => Dict(0=>[25.0,10.0,4.0], 1=>[6.00,2.50], 2=>[1.20,0.45]),
    :C   => Dict(0=>[30.0,12.0,5.0], 1=>[8.00,3.20], 2=>[1.60,0.60]),
    :N   => Dict(0=>[36.0,14.0,6.0], 1=>[10.0,4.00], 2=>[2.00,0.70]),
    :O   => Dict(0=>[42.0,16.0,7.0], 1=>[12.0,4.80], 2=>[2.40,0.85]),
    :F   => Dict(0=>[50.0,19.0,8.0], 1=>[14.0,5.60], 2=>[2.80,1.00]),
    :Ne  => Dict(0=>[58.0,22.0,9.0], 1=>[16.0,6.40], 2=>[3.20,1.10]),
    )

function build_spec(sym)
    specs = NamedTuple{(:n1,:n2,:l),Tuple{Int,Int,Int}}[]
    for (l, arr) in sort(collect(alphas[sym]); by=first)
        n1rng = n1_ranges[l]
        for (n2, _) in enumerate(arr), n1 in n1rng
            push!(specs, (n1=n1, n2=n2, l=l))
        end
    end
    return specs
end


default_spec1p(sym) = build_spec(sym)
# -------------------------------------------------------------------------
# Basis auto loader (unchanged semantics; supports custom spec for Slater)
# -------------------------------------------------------------------------
function auto_load_basis(mol::Molecule,
                         basis_set::AbstractString,
                         ::Type{Slater};
                         filename::AbstractString = "basis.json",
                         spec1p_admissible = nothing)
    atoms = [nuc.name for nuc in mol.nuclei]
    bases = [ACEpsi.load_basis_from_json(filename, atom, basis_set) for atom in atoms]

    aos = []
    for i in 1:length(bases)
        atom = atoms[i]
        atom_name = Symbol(mol.nuclei[i].name)
        b = bases[i]
        if spec1p_admissible == nothing
            spec1p_atom = build_spec(atom_name)
        else
            spec1p_atom = spec1p_admissible[i]
        end
        alphas_atom = rand(eltype(b.Dn.ζ), length(spec1p_atom), 1)
        ζ =  SMatrix{length(spec1p_atom), 1}(alphas_atom)
        D =  SMatrix{length(spec1p_atom), 1}(rand(eltype(b.Dn.D), length(spec1p_atom), 1))
        decay = b.Dn.decay
        Dfunc = Polynomials4ML.RadialDecay(ζ, D, decay, SVector{length(spec1p_atom)}(spec1p_atom))
        Ylm = Polynomials4ML.real_solidharmonics(maximum(x -> x.l, spec1p_atom); static=true)
        spec = []
        for ii in eachindex(spec1p_atom)
            n1 = spec1p_atom[ii].n1
            n2 = spec1p_atom[ii].n2
            l  = spec1p_atom[ii].l
            if l == 1
                for m in [1, -1, 0]
                    push!(spec, (n1 = n1, n2 = n2, l = l, m = m))
                end
            else
                for m = -l:l
                    push!(spec, (n1 = n1, n2 = n2, l = l, m = m))
                end
            end
        end
        spec = SVector{length(spec), typeof(spec[1])}(spec...)
        specidx = Vector{Tuple{Int, Int, Int}}(undef, length(spec))
        spec_Ylm = Polynomials4ML.natural_indices(Ylm); inv_Ylm = Polynomials4ML._invmap(spec_Ylm)
        spec_Dn = Polynomials4ML.natural_indices(Dfunc); inv_Dn = Polynomials4ML._invmap(spec_Dn)
        for (z, b_) in enumerate(spec)
            specidx[z] = (b_.n1, inv_Dn[(n1 = b_.n1, n2 = b_.n2, l = b_.l)], inv_Ylm[(l=b_.l, m=b_.m)])
        end
        push!(aos, AtomicOrbitals(Slater, Dfunc, Ylm, spec, specidx))
    end
    return [i for i in aos]
end


const default_n1_ranges_plain = Dict(
    0 => 1:5,   # s
    1 => 1:3,   # p
    2 => 1:1,   # d
)

const n1_ranges_plain = Dict(
    :H  => Dict(
        0 => 1:3,
        1 => 1:2,
        2 => 1:1,
    ),
    :default => default_n1_ranges_plain,
)


alphas_plain = Dict(
    :H   => Dict(0 => [8.0], 1 => [0.80], 2=>[0.80]),          
    :He  => Dict(0=>[8.0], 1=>[0.80], 2=>[0.80]),
    :Li  => Dict(0=>[12.0], 1=>[1.00], 2=>[0.80]),
    :Be  => Dict(0=>[20.0], 1=>[3.50], 2=>[0.90]),
    :B   => Dict(0=>[25.0], 1=>[6.00], 2=>[1.20]),
    :C   => Dict(0=>[30.0], 1=>[8.00], 2=>[1.60]),
    :N   => Dict(0=>[36.0], 1=>[10.0], 2=>[2.00]),
    :O   => Dict(0=>[42.0], 1=>[12.0], 2=>[2.40]),
    :F   => Dict(0=>[50.0], 1=>[14.0], 2=>[2.80]),
    :Ne  => Dict(0=>[58.0], 1=>[16.0], 2=>[3.20]),
    )

function build_spec_plain(sym)
    specs = NamedTuple{(:n1,:n2,:l),Tuple{Int,Int,Int}}[]
    n1ranges = get(n1_ranges_plain, sym, n1_ranges_plain[:default])

    for (l, arr) in sort(collect(alphas_plain[sym]); by=first)
        n1rng = n1ranges[l]
        for (n2, _) in enumerate(arr), n1 in n1rng
            push!(specs, (n1=n1, n2=n2, l=l))
        end
    end
    return specs
end


function auto_load_basis(mol::Molecule,
                         basis_set::AbstractString,
                         ::Type{Plain};
                         filename::AbstractString = "basis.json",
                         spec1p_admissible = nothing)
    atoms = [nuc.name for nuc in mol.nuclei]
    bases = [ACEpsi.load_basis_from_json(filename, atom, basis_set) for atom in atoms]

    aos = []
    for i in 1:length(bases)
        atom = atoms[i]
        atom_name = Symbol(mol.nuclei[i].name)
        b = bases[i]
        if spec1p_admissible == nothing
            spec1p_atom = build_spec_plain(atom_name)
        else
            spec1p_atom = spec1p_admissible[i]
        end
        alphas_atom = rand(eltype(b.Dn.ζ), length(spec1p_atom), 1)
        ζ =  SMatrix{length(spec1p_atom), 1}(alphas_atom)
        D =  SMatrix{length(spec1p_atom), 1}(rand(eltype(b.Dn.D), length(spec1p_atom), 1))
        decay = b.Dn.decay
        Dfunc = Polynomials4ML.RadialDecay(ζ, D, decay, SVector{length(spec1p_atom)}(spec1p_atom))
        Ylm = Polynomials4ML.real_solidharmonics(maximum(x -> x.l, spec1p_atom); static=true)
        spec = []
        for ii in eachindex(spec1p_atom)
            n1 = spec1p_atom[ii].n1
            n2 = spec1p_atom[ii].n2
            l  = spec1p_atom[ii].l
            if l == 1
                for m in [1, -1, 0]
                    push!(spec, (n1 = n1, n2 = n2, l = l, m = m))
                end
            else
                for m = -l:l
                    push!(spec, (n1 = n1, n2 = n2, l = l, m = m))
                end
            end
        end
        spec = SVector{length(spec), typeof(spec[1])}(spec...)
        specidx = Vector{Tuple{Int, Int, Int}}(undef, length(spec))
        spec_Ylm = Polynomials4ML.natural_indices(Ylm); inv_Ylm = Polynomials4ML._invmap(spec_Ylm)
        spec_Dn = Polynomials4ML.natural_indices(Dfunc); inv_Dn = Polynomials4ML._invmap(spec_Dn)
        for (z, b_) in enumerate(spec)
            specidx[z] = (b_.n1, inv_Dn[(n1 = b_.n1, n2 = b_.n2, l = b_.l)], inv_Ylm[(l=b_.l, m=b_.m)])
        end
        push!(aos, AtomicOrbitals(Plain, Dfunc, Ylm, spec, specidx))
    end
    return [i for i in aos]
end

function auto_load_basis(mol::Molecule,
                         basis_set::AbstractString,
                         ::Type{Gaussian};
                         filename::AbstractString = "basis.json",
                         spec1p_admissible = nothing)
    atoms = [nuc.name for nuc in mol.nuclei]
    bases = [ACEpsi.load_basis_from_json(filename, atom, basis_set) for atom in atoms]

    aos = []
    for (atom, b) in zip(atoms, bases)
        spec_str = displayspec1p(b.spec)
        println("=========== basis function for $(atom): ", spec_str, " =============")
    end
    aos = [AtomicOrbitals(Gaussian, bases[i].Dn,
                        Polynomials4ML.real_solidharmonics(first(typeof(bases[i].Ylm.scbasis).parameters); static=true),
                        bases[i].spec, bases[i].specidx) for i = 1:length(bases)]
    return [i for i in aos]
end

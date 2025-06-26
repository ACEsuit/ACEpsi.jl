using ChemBasisSets: save_all_bases_to_json, load_basis_from_json
export auto_load_basis

# Load basis set for all atoms in molecule and print shell info
function auto_load_basis(mol::Molecule, basis_set::String; filename = "basis.json", return_spec = false)
    atoms = [nuc.name for nuc in mol.nuclei]                                 # Get atom names
    #save_all_bases_to_json(unique(atoms), [basis_set], filename)            # Save basis data to JSON (if not already)
    basis = Vector([load_basis_from_json(filename, atom, basis_set) for atom in atoms])  # Load basis from JSON
    for (i, atom) in enumerate(atoms)
        spec = displayspec1p(basis[i].spec)                                  # Format basis info
        println("=========== basis function for $(atom):", spec, " =============")
    end
    if return_spec
        spec = [displayspec1p(basis[i].spec) for i = 1:length(atoms)]
        spec = [split.(s, ",") for s in spec]
        return basis, spec
    else
        return basis
    end
end

# Mapping between orbital types and their indices
orbitalIndexToType = Dict(0 => "s", 1 => "p", 2 => "d", 3 => "f", 4 => "g", 5 => "h")
orbitalTypeToIndex = Dict("s" => 0, "p" => 1, "d" => 2, "f" => 3, "g" => 4, "h" => 5)

# Parse totdeg into indexable polynomial degree array for each atom and orbital type
function parseTotdegToInt(
    totdeg::Vector{Vector{String}},
    spec1p::Vector{NamedTuple{(:s, :I, :n1, :n2, :l, :m), Tuple{Char, Vararg{Int64, 5}}}}
)
    ord = length(totdeg[1])
    M = length(totdeg)                           # Number of atoms
    maxn = [zeros(Int64, 6) for _ = 1:M]         # Max polynomial degree per angular shell

    for b in spec1p
        maxn[b.I][b.l + 1] = max(maxn[b.I][b.l + 1], b.n2)
    end

    _totdegn = [[zeros(Int64, 6) for _ = 1:ord] for _ = 1:M]  # Store totdeg per atom/component
    for z = 1:M
        for i = 1:ord
            i_const = totdeg[z][i]
            polynum = parse(Int64, i_const[1])                # e.g. "2s" → 2
            orbtype = string(i_const[2])                      # e.g. "2s" → "s"
            l = orbitalTypeToIndex[orbtype]
            _totdegn[z][i][l + 1] = polynum
            if l > 0
                for j = 1:l
                    _totdegn[z][i][j] = maxn[z][j]
                end
            end
        end
    end
    return _totdegn
end

function displayspec(spec, spec1p)
    nicespec = []
    for k = 1:length(spec)
        push!(nicespec, [spec1p[spec[k][j]] for j = 1:length(spec[k])])
    end
    return nicespec
end

# Convert 1p spec into string like "1s,2s,2p"
function displayspec1p(spec1p::AbstractArray{T}) where {T}
    shells = Vector{String}(undef, length(spec1p))
    for i = 1:length(spec1p)
        @assert spec1p[i].l ∈ keys(orbitalIndexToType) "only 0(s) - 5(h) shell are supported."
        shells[i] = "$(spec1p[i].n2)$(orbitalIndexToType[spec1p[i].l])"
    end
    return join(unique(shells), ",")
end

# Get spec1p from basis: optionally attach spin labels
function get_spec1p(basis::Vector{TS}; spin = false) where {TS}
    Nnlm = sum(length(b.spec) for b in basis)
    if spin
        spec = Array{NamedTuple{(:s, :I, :n1, :n2, :l, :m), Tuple{Char, Vararg{Int64, 5}}}}(undef, (3, Nnlm))
        for (is, s) in enumerate(extspins())
            t = 0
            for (i, b) in enumerate(basis), nlm in b.spec
                t += 1
                spec[is, t] = (s = s, I = i, nlm...)
            end
        end
    else
        spec = Array{NamedTuple{(:I, :n1, :n2, :l, :m), Tuple{Vararg{Int64, 5}}}}(undef, Nnlm)
        t = 0
        for (i, b) in enumerate(basis), nlm in b.spec
            t += 1
            spec[t] = (I = i, nlm...)
        end
    end
    return spec[:]
end

function _invmap(a)
    inva = Dict{eltype(a), Int}()
    for i = 1:length(a) 
        inva[a[i]] = i 
    end
   return inva 
end



function iteratespec1p(orbital, l)
    Bl = Vector{Vector{String}}()
    index = []
    for str in ("s", )
        ind = findall(x -> x == str, [string(orbital[i][2]) for i = 1:length(orbital)])
        index = [index..., ind...]
        if length(ind) > 0
            push!(Bl, orbital[ind])
        end
    end
    for str in ("p", "d", "f", "g", "h")
        ind = findall(x -> x == str, [string(orbital[i][2]) for i = 1:length(orbital)])
        if length(ind) > 0
            for i = 1:length(ind)
                push!(Bl, orbital[[index..., ind[1:i]...]])
            end
        end
        index = [index..., ind...]
    end
    for i = length(Bl)+1 : l
        push!(Bl, Bl[end])
    end
    return Bl
end

function orbital_for_iatom_jord(orbital, totdeg, i, j)
    return orbital[i][1:findall(x -> x == totdeg[i][j], orbital[i])[1]]
end

function sample_evenly(arr::AbstractVector; N = 10)
    len = length(arr)
    if len <= N
        return arr
    else
        idxs = round.(Int, range(1, len, length=N))
        return arr[idxs]
    end
end


function build_totdeglevels(mol, basis_set, totdeg, ν; ratio = 0.5, max_level::Union{Nothing, Int} = nothing)
    _, orbital = ACEpsi.auto_load_basis(mol, basis_set; return_spec = true)
    n_atom = length(orbital)

    lj = maximum(length.(ACEpsi.iteratespec1p.(ACEpsi.orbital_for_iatom_jord.(Ref(orbital), Ref(totdeg), 1:n_atom, 1), Ref(1))))
    _lj = max(Int(ceil(lj * ratio)), 1)
    spec1pl = [ACEpsi.iteratespec1p(ACEpsi.orbital_for_iatom_jord(orbital, totdeg, i, 1), lj) for i = 1:n_atom]

    totdeglevels = Vector{Vector{Vector{String}}}()
    νlevels = Int[]

    for l = 1:ν
        if l == 1
            for j = 1:_lj
                push!(totdeglevels, [[spec1pl[i][j][end]] for i = 1:n_atom])
                push!(νlevels, 1)
            end
        else
            prev = deepcopy(totdeglevels[end])
            for a = 1:n_atom
                push!(prev[a], spec1pl[a][1][end])
            end
            for j = 1:_lj
                new_level = deepcopy(prev)
                for a = 1:n_atom
                    new_level[a][end] = spec1pl[a][j][end]
                end
                push!(totdeglevels, new_level)
                push!(νlevels, l)
            end
        end
    end

    for i = _lj+1:lj
        for l = 1:ν
            last_level = deepcopy(totdeglevels[end])
            for a = 1:n_atom
                if l <= length(last_level[a])
                    last_level[a][l] = spec1pl[a][i][end]
                else
                    push!(last_level[a], spec1pl[a][i][end])
                end
            end
            push!(totdeglevels, last_level)
            push!(νlevels, l)
        end
    end
    for i = 1:length(νlevels)
        νlevels[i] = length(totdeglevels[i][1])
    end

    if max_level !== nothing && length(totdeglevels) > max_level
        totdeglevels = sample_evenly(totdeglevels; N = max_level)
        νlevels = sample_evenly(νlevels; N = max_level)
    end

    return totdeglevels, νlevels
end

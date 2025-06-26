export Molecule

# Immutable Molecule type storing nuclei, total electrons, species info, and spin config
struct Molecule{Nnuc, T, TT, TN <: Int64, TS <: Int64}
    nuclei::SVector{Nnuc, Nuc{T, TT}}   # Fixed-size array of nuclei
    Nel::TN                             # Total number of electrons
    spec::Dict{Symbol, TS}              # Element symbol => number of electrons
    Σ::Vector{Char}                     # Spin configuration (↑, ↓)
end

# Construct a Molecule from a list of nuclei
function create_molecule(nuclei::SVector{Nnuc, Nuc{T, TT}}) where {Nnuc, T, TT}
    element_names = [n.name for n in nuclei]                      # Element names
    unique_elements = unique(element_names)                      # Unique species
    total_electrons = sum(n.charge for n in nuclei)              # Total electron count

    # Count electrons per unique species
    electrons_per_element = [
        sum(name2idx[element_names[i]] for i in findall(==(el), element_names))
        for el in unique_elements
    ]
    element_electron_dict = Dict(Symbol.(unique_elements) .=> electrons_per_element)

    Σ = generate_spin_config(total_electrons)                    # Spin configuration
    return Molecule(nuclei, total_electrons, element_electron_dict, Σ)
end

# Generate spin configuration: ↑ first, then ↓
function generate_spin_config(Nel::Int)
    n_up = div(Nel + 1, 2)
    n_down = Nel - n_up
    return vcat(fill(↑, n_up), fill(↓, n_down))
end

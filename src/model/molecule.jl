export Molecule, create_molecule, Nuc

struct Nuc{T, TT}
    name::String
    rr::SVector{3,T}
    charge::TT
end

# Mappings between atomic numbers and element names
idx2name = Dict(
    1 => "H", 2 => "He", 3 => "Li", 4 => "Be", 5 => "B", 
    6 => "C", 7 => "N", 8 => "O", 9 => "F", 10 => "Ne"
)

name2idx = Dict(
    "H" => 1, "He" => 2, "Li" => 3, "Be" => 4, "B" => 5, 
    "C" => 6, "N" => 7, "O" => 8, "F" => 9, "Ne" => 10
)

# Construct Nuc from position and charge
function Nuc(rr::SVector{3, T}, charge::T) where {T <: Number}
    return Nuc(idx2name[Int(charge)], rr, charge)
end

# Construct Nuc from name and position vector
function Nuc(name::String, rr::Vector{T}) where {T <: Number}
    return Nuc(name, SVector{3}(rr), [k for (k,v) in idx2name if v == name]...)
end

struct Molecule{Nnuc, Nx, T,TT}
    nuclei::SVector{Nnuc, Nuc{T,TT}}
    Nel::Int
    spec::Dict{Symbol,Int}
    spin::NTuple{Nx,Char}
end

function _spin_config(Nel::Integer; target_ms::Union{Nothing,Tuple{Int,Int}}=nothing)
    Nel < 0 && throw(ArgumentError("Nel must be nonnegative, got $Nel"))
    if target_ms === nothing
        n_up = (Nel + 1) ÷ 2
        n_down = Nel - n_up
    else
        n_up, n_down = target_ms
        n_up < 0 && throw(ArgumentError("n_up must be ≥ 0"))
        n_down < 0 && throw(ArgumentError("n_down must be ≥ 0"))
        n_up + n_down == Nel ||
            throw(ArgumentError("n_up + n_down = $(n_up+n_down) must equal Nel = $Nel"))
    end
    return Tuple(vcat(fill('↑', n_up), fill('↓', n_down)))
end

function create_molecule(
    nuclei::SVector{Nnuc, Nuc{T,TT}};
    target_ms::Union{Nothing,Tuple{Int,Int}} = nothing) where {Nnuc,T,TT}
    names = getfield.(nuclei, :name)
    charges = getfield.(nuclei, :charge)
    Nel = Int(sum(charges))

    uniq = unique(names)
    spec = Dict(Symbol(el) => sum(n.charge for n in nuclei if n.name == el) for el in uniq)

    spin = _spin_config(Nel; target_ms=target_ms)

    return Molecule{Nnuc,Nel,T,TT}(nuclei, Nel, spec, spin)
end

const Spin = Char 
const ↑ = '↑'
const ↓ = '↓'
const ∅ = '∅'   # this is only for internal use
_spins = SA[↑, ↓]
_extspins = SA[↑, ↓, ∅]

spins() = _spins
extspins() = _extspins

function spin2idx(σ)
   if σ == ↑
      return 1
   elseif σ == ↓
      return 2
   elseif σ == ∅
      return 3
   end
   error("illegal spin char for spin2idx")
end

_rr(n) = getfield(n, :rr)
_charge(n) = Int(getfield(n, :charge))

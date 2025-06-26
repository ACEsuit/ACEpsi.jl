using StaticArrays  # For fixed-size vectors

# Struct for nucleus: name, position, and charge
struct Nuc{T, TT}
    name::String              # Element name (e.g., "H", "O")
    rr::SVector{3, T}         # 3D position
    charge::TT                # Nuclear charge
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

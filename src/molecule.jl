module molecules

using ACEpsi: Nuc, create_molecule
using StaticArrays

# Ground-state spin configurations: element symbol => (n_up, n_down)
const GROUND_SPIN = Dict{String,Tuple{Int,Int}}(
    "H"  => (1,0),
    "He" => (1,1),
    "Li" => (2,1),
    "Be" => (2,2),
    "B"  => (3,2),
    "C"  => (4,2),
    "N"  => (5,2),
    "O"  => (5,3),
    "F"  => (5,4),
    "Ne" => (5,5),
)

# === Single-atom molecules at the origin ===
H  = create_molecule(SVector(Nuc("H",  [0.0, 0.0, 0.0])); target_ms = GROUND_SPIN["H"])
He = create_molecule(SVector(Nuc("He", [0.0, 0.0, 0.0])); target_ms = GROUND_SPIN["He"])
Li = create_molecule(SVector(Nuc("Li", [0.0, 0.0, 0.0])); target_ms = GROUND_SPIN["Li"])   # -14.66733
Be = create_molecule(SVector(Nuc("Be", [0.0, 0.0, 0.0])); target_ms = GROUND_SPIN["Be"])   # -24.65370
B  = create_molecule(SVector(Nuc("B",  [0.0, 0.0, 0.0])); target_ms = GROUND_SPIN["B"])    # -37.84471
C  = create_molecule(SVector(Nuc("C",  [0.0, 0.0, 0.0])); target_ms = GROUND_SPIN["C"])    # -54.58882
N  = create_molecule(SVector(Nuc("N",  [0.0, 0.0, 0.0])); target_ms = GROUND_SPIN["N"])    # -75.06655
O  = create_molecule(SVector(Nuc("O",  [0.0, 0.0, 0.0])); target_ms = GROUND_SPIN["O"])    # -99.7329
F  = create_molecule(SVector(Nuc("F",  [0.0, 0.0, 0.0])); target_ms = GROUND_SPIN["F"])    # -128.9366
Ne = create_molecule(SVector(Nuc("Ne", [0.0, 0.0, 0.0])); target_ms = GROUND_SPIN["Ne"])   # -161.9 approx

# === Diatomic molecules ===
Li2 = create_molecule(SVector(
    Nuc("Li", [-5.051/2, 0.0, 0.0]),
    Nuc("Li", [ 5.051/2, 0.0, 0.0])
)) # -14.99475  (no explicit spin → default inside create_molecule)

LiH = create_molecule(SVector(
    Nuc("Li", [-3.015/2, 0.0, 0.0]),
    Nuc("H",  [ 3.015/2, 0.0, 0.0])
)) # -8.07050

N2 = create_molecule(SVector(
    Nuc("N", [-2.068/2, 0.0, 0.0]),
    Nuc("N", [ 2.068/2, 0.0, 0.0])
)) # -109.5388

CO = create_molecule(SVector(
    Nuc("C", [-2.173/2, 0.0, 0.0]),
    Nuc("O", [ 2.173/2, 0.0, 0.0])
)) # -113.3218

# === Polyatomic molecules ===
CH4 = create_molecule(SVector(
    Nuc("C", [0.0, 0.0, 0.0]),
    Nuc("H", [ 1.18886,  1.18886,  1.18886]),
    Nuc("H", [-1.18886, -1.18886,  1.18886]),
    Nuc("H", [ 1.18886, -1.18886, -1.18886]),
    Nuc("H", [-1.18886,  1.18886, -1.18886])
)) # -40.51400

# Water molecule
H2O = create_molecule(SVector(
    Nuc("O", [0.0, 0.0, 0.0]),
    Nuc("H", [-1.84345 * sin(110.6 / 360 * pi), -1.84345 * cos(110.6 / 360 * pi), 0.0]),
    Nuc("H", [ 1.84345 * sin(110.6 / 360 * pi), -1.84345 * cos(110.6 / 360 * pi), 0.0])
))

# === H4 square (on xy-plane) ===
H4(spacing) = create_molecule(SVector(
    Nuc("H", [-spacing/2, -spacing/2, 0.0]),
    Nuc("H", [-spacing/2,  spacing/2, 0.0]),
    Nuc("H", [ spacing/2,  spacing/2, 0.0]),
    Nuc("H", [ spacing/2, -spacing/2, 0.0])
))

# === Linear hydrogen chain of length N ===
Hchain(N, spacing) = create_molecule(SVector{N}([
    Nuc("H", [0.0, 0.0, (i - 1/2 - N/2) * spacing]) for i = 1:N
]))

end

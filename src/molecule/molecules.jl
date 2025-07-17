module molecules

using ACEpsi: Nuc, create_molecule
using StaticArrays

# === Single atom molecules at origin ===
H  = create_molecule(SVector(Nuc("H", [0.0, 0.0, 0.0])))
He = create_molecule(SVector(Nuc("He", [0.0, 0.0, 0.0])))
Li = create_molecule(SVector(Nuc("Li", [0.0, 0.0, 0.0])))
Be = create_molecule(SVector(Nuc("Be", [0.0, 0.0, 0.0])))
B  = create_molecule(SVector(Nuc("B", [0.0, 0.0, 0.0])))
C  = create_molecule(SVector(Nuc("C", [0.0, 0.0, 0.0])))
N  = create_molecule(SVector(Nuc("N", [0.0, 0.0, 0.0])))
O  = create_molecule(SVector(Nuc("O", [0.0, 0.0, 0.0])))
F  = create_molecule(SVector(Nuc("F", [0.0, 0.0, 0.0])))
Ne = create_molecule(SVector(Nuc("Ne", [0.0, 0.0, 0.0])))

# === Methane (CH₄) ===
CH4 = create_molecule(SVector(
    Nuc("C", [0.0, 0.0, 0.0]),
    Nuc("H", [ 1.18886,  1.18886,  1.18886]),
    Nuc("H", [-1.18886, -1.18886,  1.18886]),
    Nuc("H", [ 1.18886, -1.18886, -1.18886]),
    Nuc("H", [-1.18886,  1.18886, -1.18886])
))

Li2 = create_molecule(SVector(
    Nuc("Li", [-5.051/2, 0.0, 0.0]),
    Nuc("Li", [ 5.051/2, 0.0, 0.0])
))
LiH = create_molecule(SVector(
    Nuc("Li", [-3.015/2, 0.0, 0.0]),
    Nuc("H",  [ 3.015/2, 0.0, 0.0])
))

# === H₂O molecule ===
H2O = create_molecule(SVector(
    Nuc("O", [0.0, 0.0, 0.0]),
    Nuc("H", [-1.84345 * sin(110.6 / 360 * pi), -1.84345 * cos(110.6 / 360 * pi), 0.0]),
    Nuc("H", [ 1.84345 * sin(110.6 / 360 * pi), -1.84345 * cos(110.6 / 360 * pi), 0.0])
))

# === Diatomic and small molecules with spacing parameter ===
N2(spacing) = create_molecule(SVector(
    Nuc("N", [-spacing/2, 0.0, 0.0]),
    Nuc("N", [ spacing/2, 0.0, 0.0])
))

# === Square H₄ molecule on xy-plane ===
H4(spacing) = create_molecule(SVector(
    Nuc("H", [-spacing/2, -spacing/2, 0.0]),
    Nuc("H", [-spacing/2,  spacing/2, 0.0]),
    Nuc("H", [ spacing/2,  spacing/2, 0.0]),
    Nuc("H", [ spacing/2, -spacing/2, 0.0])
))

# === Linear H chain of length N ===
Hchain(N, spacing) = create_molecule(SVector{N}([
    Nuc("H", [0.0, 0.0, (i - 1/2 - N/2) * spacing]) for i = 1:N
]))

# === NH₂ molecule with bond angle ===
NH2(spacing, angle) = create_molecule(SVector(
    Nuc("N", [0.0, 0.0, 0.0]),
    Nuc("H", [-spacing * sin(angle / 360 * pi), -spacing * cos(angle / 360 * pi), 0.0]),
    Nuc("H", [ spacing * sin(angle / 360 * pi), -spacing * cos(angle / 360 * pi), 0.0])
))



end

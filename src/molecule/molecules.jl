module molecules

using ACEpsi: Nuc, create_molecule
using StaticArrays

# === Single atom molecules at origin ===
H  = create_molecule(SVector(Nuc("H", [0.0, 0.0, 0.0])))
He = create_molecule(SVector(Nuc("He", [0.0, 0.0, 0.0])))
# -7.47798
Li = create_molecule(SVector(Nuc("Li", [0.0, 0.0, 0.0])))
# -14.66733
Be = create_molecule(SVector(Nuc("Be", [0.0, 0.0, 0.0])))
# -24.65370
B  = create_molecule(SVector(Nuc("B", [0.0, 0.0, 0.0])))
# -37.84471
C  = create_molecule(SVector(Nuc("C", [0.0, 0.0, 0.0])))
# -54.58882
N  = create_molecule(SVector(Nuc("N", [0.0, 0.0, 0.0])))
# -75.06655
O  = create_molecule(SVector(Nuc("O", [0.0, 0.0, 0.0])))
# -99.7329
F  = create_molecule(SVector(Nuc("F", [0.0, 0.0, 0.0])))
# -128.9366
Ne = create_molecule(SVector(Nuc("Ne", [0.0, 0.0, 0.0])))

# -14.99475
Li2 = create_molecule(SVector(
    Nuc("Li", [-5.051/2, 0.0, 0.0]),
    Nuc("Li", [ 5.051/2, 0.0, 0.0])
))
# -8.07050
LiH = create_molecule(SVector(
    Nuc("Li", [-3.015/2, 0.0, 0.0]),
    Nuc("H",  [ 3.015/2, 0.0, 0.0])
))

# -40.51400
CH4 = create_molecule(SVector(
    Nuc("C", [0.0, 0.0, 0.0]),
    Nuc("H", [ 1.18886,  1.18886,  1.18886]),
    Nuc("H", [-1.18886, -1.18886,  1.18886]),
    Nuc("H", [ 1.18886, -1.18886, -1.18886]),
    Nuc("H", [-1.18886,  1.18886, -1.18886])
))

# -109.5388
N2 = create_molecule(SVector(
    Nuc("N", [-2.068/2, 0.0, 0.0]),
    Nuc("N", [ 2.068/2, 0.0, 0.0])
))

# -113.3218
CO = create_molecule(SVector(
    Nuc("C", [-2.173/2, 0.0, 0.0]),
    Nuc("O", [ 2.173/2, 0.0, 0.0])
))

# === H₂O molecule ===
H2O = create_molecule(SVector(
    Nuc("O", [0.0, 0.0, 0.0]),
    Nuc("H", [-1.84345 * sin(110.6 / 360 * pi), -1.84345 * cos(110.6 / 360 * pi), 0.0]),
    Nuc("H", [ 1.84345 * sin(110.6 / 360 * pi), -1.84345 * cos(110.6 / 360 * pi), 0.0])
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

end

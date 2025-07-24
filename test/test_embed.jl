using StaticArrays
using Test
using ACEpsi
using Optimisers: destructure
using LinearAlgebra
using ForwardDiff

# Define a water molecule (H2O) with O-H bond length 1.0 and bond angle 105°
mol = ACEpsi.molecules.H2O
Nel = mol.Nel                 # Get number of electrons in the molecule

# Build the wavefunction object and get its initial parameters and state
basis_set = "cc-pvtz"
totdeg = [["2d", "2d"], ["3p", "3p"], ["3p", "3p"]]
ν = 2
wf1, ps1, st1, spec1, spec1p1 = build_wavefunction(mol, basis_set, totdeg, ν, No_Decomposition())

totdeg = [["1f", "1f", "1f"], ["1d", "1d", "1d"], ["1d", "1d", "1d"]]
ν = 3
wf2, ps2, st2, spec2, spec1p2 = build_wavefunction(mol, basis_set, totdeg, ν, No_Decomposition())

ps2 = transfer_weights!(ps1, ps2, spec1, spec2, spec1p1, spec1p2)

# Generate random 3D coordinates for all electrons as input configuration X
X = [SVector{3}(rand(3)) for i = 1:Nel]

# Evaluate the wavefunction at given positions and parameters, get log|ψ|
y1 = wf1(X, ps1, st1)[1]
y2 = wf2(X, ps2, st2)[1]
@test norm(y1 - y2) < 1e-5

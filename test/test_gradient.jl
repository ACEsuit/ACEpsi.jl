using StaticArrays
using Test
using ACEpsi
using Optimisers: destructure
using LinearAlgebra
using ForwardDiff

# Define a water molecule (H2O) with O-H bond length 1.0 and bond angle 105°
mol = ACEpsi.molecules.H2O(1.0, 105)
Nel = mol.Nel                 # Get number of electrons in the molecule

# Build the wavefunction object and get its initial parameters and state
basis_set = "cc-pvtz"
totdeg = [["2d", "2d"], ["3p", "3p"], ["3p", "3p"]]
ν = 2
wf, ps, st, spec, spec1p = build_wavefunction(mol, basis_set, totdeg, ν, No_Decomposition())

# Generate random 3D coordinates for all electrons as input configuration X
X = [SVector{3}(rand(3)) for i = 1:Nel]

# Evaluate the wavefunction at given positions and parameters, get log|ψ|
y = wf(X, ps, st)[1]

# ======= Derivative w.r.t. positions (∇_x log|ψ|) =======
p, s = destructure(X)
d = ForwardDiff.gradient(x -> wf(s(x), ps, st)[1], p)  # Compute ∇_x log|ψ| using ForwardDiff
@test norm(s(d) - gradx(wf, X, ps, st)) < 1e-8

# ======= Derivative w.r.t. parameters (∇_θ log|ψ|) =======
p, s = destructure(ps)
d = ForwardDiff.gradient(p -> wf(X, s(p), st)[1], p)   # Compute ∇_θ log|ψ| using ForwardDiff
@test norm(d - ACEpsi.gradp(wf, X, ps, st)) < 1e-3

# ======= Laplacian check: ∇²_x log|ψ| =======
p, s = destructure(X)
d = ForwardDiff.hessian(x -> wf(s(x), ps, st)[1], p)   # Compute full Hessian w.r.t. positions
@test norm(sum([d[i,i] for i=1:size(d, 1)]) - laplacian(wf, X, ps, st)) < 1e-8  # Compare trace of Hessian with custom Laplacian

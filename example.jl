# add https://github.com/ACEsuit/ChemBasisSets.jl.git
# add ACEpsi #lux
using Distributed
N_procs = 8

import Pkg
Pkg.activate(Base.current_project())

if nprocs() == 1
    addprocs(N_procs - 1, exeflags="--project=$(Base.current_project())")
end
 
using ACEpsi

mol = ACEpsi.molecules.Be

basis_set = "cc-pvtz"
totdeg = [["1f", "1f"]]
ν = 2

model_list, ps_list, st_list, spec_list, spec1p_list, totdeg_list, ν_list = model_generator(mol, basis_set, totdeg, ν; ratio = 0.5);
test_wavefunction(model_list, ps_list, st_list, spec_list, spec1p_list, mol)

solver = (SPRINGSolver(), SketchSolver(800, 50, 50, 1.4), SVDSolver(800, 20, 20, 1.4))

iterations = 1000 * ones(Int64, length(spec_list))
iterations[end] = 30000
optimizer = OPTSETTING(solver[1], iterations = iterations, burnin = 1000, lag = 10, nchains = 2^8, 
                                 Δt = 0.08, acc_step = 10, acc_range = [0.45, 0.80],  
                                 clip = 5.0, lr = 0.02, lr_dc = 10000,  m = 0.99, 
                                 damping = 0.001, damping_decay = 100, damping_min = 0.001, 
                                 norm_constrain = 0.001, η = 0.95, res_path = "Be/SPRING/")

@everywhere begin 
    using ACEpsi
    using Statistics
    using LinearAlgebra
    using Optimisers: destructure
end
model_list, ps_list, st_list, val_list, var_list, rank_list = train(mol, model_list, ps_list, st_list, spec_list, spec1p_list, totdeg_list, ν_list, optimizer);
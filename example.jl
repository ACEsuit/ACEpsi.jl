# add https://github.com/ACEsuit/ChemBasisSets.jl.git
# add https://github.com/ACEsuit/ACEpsi.jl.git #lux
using ACEpsi

method = 1
mol = ACEpsi.molecules.Be
mol_name = "Be"
TD = No_Decomposition()
worldsize = N_procs = 16
setup(mol, mol_name, method, TD, worldsize; ν = 3)

x0, optimizer, model_list, ps_list, st_list, spec_list, spec1p_list, totdeg_list, ν_list = load_setup(mol_name, TD);
solver = (SPRINGSolver(), SketchSolver(800, 50, 50, 1.4), SVDSolver(800, 50, 50, 1.4))
optimizer.sr_method = solver[method]
string = method == 1 ? "SPRING" :
             method == 2 ? "SKETCH" :
             method == 3 ? "WSSR"   : error("Invalid method")
clean_TD = replace("$(TD)", r"[^A-Za-z0-9]" => "")
optimizer.res_path = "$mol_name/$string/$clean_TD/"

using Pkg
using Distributed
Pkg.activate(Base.current_project())

if nprocs() == 1
    addprocs(N_procs - 1, exeflags="--project=$(Base.current_project())")
end

@everywhere begin 
    using ACEpsi
    using Statistics
    using LinearAlgebra
    using Optimisers: destructure
end

model_list, ps_list, st_list, val_list, var_list, rank_list = train(x0, mol, model_list, ps_list, st_list, spec_list, spec1p_list, totdeg_list, ν_list, optimizer);


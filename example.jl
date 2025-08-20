# add https://github.com/ACEsuit/ChemBasisSets.jl.git
# add https://github.com/ACEsuit/ACEpsi.jl.git #lux
using ACEpsi

method = 1
mol = ACEpsi.molecules.O
mol_name = "O"
TD = No_Decomposition()
worldsize = N_procs = 8
basis_set = "cc-pvtz"

setup(mol, mol_name, method, TD, worldsize; nchains = 2^8, ν = 1)

x0, optimizer, model_list, ps_list, st_list, spec_list, spec1p_list, totdeg_list, ν_list = load_setup(mol_name, TD);
solver = (SPRINGSolver(), SketchSolver(800, 50, 50, 1.4), SVDSolver(800, 100, 100, 1.4))
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










atoms = [nuc.name for nuc in mol.nuclei]
    basis = Vector([ACEpsi.load_basis_from_json("basis.json", atom, basis_set) for atom in atoms])  # Load basis from JSON
    A = []
    for (i, atom) in enumerate(atoms)
        spec = ACEpsi.displayspec1p(basis[i].spec)
        push!(A, strip(last(split(spec, ','))))
    end
totdeg = map(x -> fill(x, 1), A)
model_list, ps_list, st_list, spec_list, spec1p_list, totdeg_list, ν_list =
        ACEpsi.model_generator(mol, basis_set, totdeg, 1; TD = TD, ratio = 1.0);

model, ps, st = model_list[end], ps_list[end], st_list[end]
ps.branch.bf.linear.weight .= C'
x0, _theta, _acc = init_walkers(mol, model, ps, st, 1000, 1000, 0.08)
ham = SumH(mol.nuclei)
_x = x0
dx = gradx.(Ref(model), _x, Ref(ps), Ref(st)) 
using LinearAlgebra
_elocs = ACEpsi.Vext.(Ref(model), _x, Ref(ham.nuclei), Ref(ps), Ref(st)) + 
         ACEpsi.Vee.(Ref(model), _x, Ref(ps), Ref(st)) - 
         1/4 * laplacian.(Ref(model), _x, Ref(ps), Ref(st)) - 
         1/8 * dot.(dx, dx)
v̄al = sum(_elocs) / length(_elocs)


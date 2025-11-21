using ACEpsi
using Lux, LuxCore, StaticArrays, LinearAlgebra, Random
using CUDA
using ForwardDiff, Zygote
using Test
using EquivariantTensors
using Optimisers: destructure

# Molecule setup
mol = ACEpsi.molecules.Be

# System size: Nel electrons, nbatch
Nel, nchains = mol.Nel, 4
# Basis expansion specs
ν = 2  # correlation degree

# Initialize electron positions around nuclei on CPU and GPU
Rcpu = initialize_around_nuclei(mol, nchains; Δt=0.08, σ = 4.0, device=:cpu)
Rgpu = initialize_around_nuclei(mol, nchains; Δt=0.08, σ = 4.0, device=:gpu)

# Build the wavefunction model with a given Gaussian basis
basis_set = "cc-pvtz"
filename = "basis.json"
model = model_generator(
    mol, ν;
    hf         = true,
    ratio      = 1.0,
    multilevel = true,
    family     = Plain,         # or Gaussian
    freeze_branches = true,      # false to let AO learnable
    spec1p_admissible = nothing
);
model_list, ps_list, st_list, spec_list, spec1p_list, totdeg_list, ν_list = model;
model = model_list[end]
ps = ps_list[end]
st = st_list[end]
dev = CUDADevice()

x_gpu = Rgpu
x_cpu = Array(Rgpu)
ps_dev, st_dev = dev(ps), dev(st);

# ----- CPU forward/grad checks -----
# Forward pass on CPU
out_cpu, st_eval = model(x_cpu, ps, st);
# Forward pass on GPU
out_gpu, st_eval_dev = model(x_gpu, ps_dev, st_dev);
norm(out_cpu - Array(out_gpu))
# ----- Finite-difference sanity checks -----

using ACEbase.Testing: print_tf, fdtest

# Test derivative w.r.t. coordinates
p, s = destructure(x_cpu)
bu = rand(length(p))
_BB(t) = s(p + t * bu)

F1(t) = sum(sum, model(_BB(t), ps, st_eval)[1]) 
dF(t) = Zygote.gradient(t -> F1(t), t)[1]       # AD derivative
print_tf(@test fdtest(F1, dF, 0.0; verbose=true))  # FD vs AD check

# Test derivative w.r.t. parameters 
p, s = destructure(ps) 
bu = rand(length(p))
_BB(t) = s(p + t * bu)

_, st = Lux.setup(Random.default_rng(), model)
F1(t) = sum(sum, model(x_cpu, _BB(t), st)[1])
FZ = Zygote.gradient(t -> F1(t), 0.0)[1]
F0 = ForwardDiff.derivative(t -> F1(t), 0.0)
FZ - F0

# ----- Laplacian check via Hessian trace (CPU) -----
Rcpu = initialize_around_nuclei(mol, 1; Δt=0.08, σ = 1.0, device=:cpu)
Rgpu = initialize_around_nuclei(mol, 1; Δt=0.08, σ = 1.0, device=:gpu)

x_cpu = Array(Rgpu)
x_gpu = Rgpu
p, s = destructure(x_cpu)
# Build Hessian of ψ w.r.t. flattened coordinates
ps, st = Lux.setup(Random.default_rng(), model)
ps_dev, st_dev = dev(ps), dev(st)
d = ForwardDiff.hessian(x -> model(s(x), ps, st)[1][1], p)

# Compare trace(H) with a Laplacian implementation
lap_cpu, st_lap = laplacian(model, x_cpu, ps, st);
lap_gpu, st_lap_dev = laplacian(model, x_gpu, ps_dev, st_dev);
@test norm(sum(d[i,i] for i in 1:size(d,1)) - lap_cpu[1]) < 1e-8

# 
function gradx_batch_xp(model, R, ps, st)
    N = size(R, 2); B = size(R, 3)
    G = similar(R)
    p, s = destructure(ps)
    O = similar(G, length(p), B)
    @inbounds for b in 1:B
        R1 = @view R[:, :, b:b]
        gR, gp = Zygote.gradient((Rflat, pflat) -> begin
            y = model(Rflat, s(pflat), st)[1]
            return sum(y)
        end, R1, p)
        @views copyto!(G[:, :, b], gR)
        @views copyto!(view(O, :, b), gp)
    end
    return G, O
end

Rcpu = initialize_around_nuclei(mol, 10; Δt=0.08, σ = 1.0, device=:cpu)
Rgpu = initialize_around_nuclei(mol, 10; Δt=0.08, σ = 1.0, device=:gpu)

x_cpu = Array(Rgpu)
x_gpu = Rgpu
ps, st = Lux.setup(Random.default_rng(), model)
ps_dev, st_dev = dev(ps), dev(st)
G1, O1 = gradx_batch_xp(model, x_cpu, ps, st);
G2 = ACEpsi.gradx_batch(model, x_cpu, ps, st);
O2 = ACEpsi.gradp_batch(model, x_cpu, ps, st);
@assert norm(G1 - G2) < 1e-3
@assert norm(O1 - O2) < 1e-3

G2_gpu = ACEpsi.gradx_batch(model, x_gpu, ps_dev, st_dev);
O2_gpu = ACEpsi.gradp_batch(model, x_gpu, ps_dev, st_dev);

norm(Array(O2_gpu) - O2) 
norm(Array(G2_gpu) - G2)
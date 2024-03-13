using ACEpsi, Polynomials4ML, StaticArrays, Test 
using Polynomials4ML: natural_indices, degree, SparseProduct
using ACEpsi.AtomicOrbitals: Nuc, make_nlms_spec, evaluate
using ACEpsi: BackflowPooling, BFwf_lux, setupBFState, Jastrow, displayspec
using ACEpsi.vmc: gradx, laplacian, grad_params, EmbeddingW!, _invmap, VMC_multilevel, wf_multilevel, VMC, gd_GradientByVMC, gd_GradientByVMC_multilevel, AdamW, SR, SumH, MHSampler
using ACEbase.Testing: print_tf, fdtest
using LuxCore
using Lux
using Zygote
using Optimisers
using Random
using Printf
using LinearAlgebra
using BenchmarkTools
using HyperDualNumbers: Hyper
using Distributions

Nel = 2
X = randn(SVector{3, Float64}, Nel)
Σ = [↑,↓]
nuclei = [ Nuc(SVector(0.0,0.0,0.0), Nel * 1.0)]
##

spec = [(n1 = 1, n2 = 1, l = 0), (n1 = 2, n2 = 1, l = 0), (n1 = 2, n2 = 1, l = 1), (n1 = 3, n2 = 1, l = 0), (n1 = 3, n2 = 1, l = 1)]
n1 = 3
Pn = Polynomials4ML.legendre_basis(n1+1)
Ylmdegree = 2
totdegree = 20
ζ = 10 * rand(length(spec))
Dn = SlaterBasis(ζ)
bYlm = RYlmBasis(Ylmdegree)

totdegree = [30,30,30]
ν = [1,1,2]
MaxIters = [100,100,300]
_spec = [spec[1:3], spec[1:4], spec]
#_spec = [spec[1:i] for i = 4:length(spec)]
#_spec = length(ν)>length(spec) ? reduce(vcat, [_spec, [spec[1:end] for i = 1:length(ν) - length(spec)]]) : _spec
wf_list, spec_list, spec1p_list, specAO_list, ps_list, st_list = wf_multilevel(Nel, Σ, nuclei, Dn, Pn, bYlm, _spec, totdegree, ν)

ham = SumH(nuclei)
sam = MHSampler(wf_list[1], Nel, nuclei, Δt = 0.5, burnin = 1000, nchains = 2000)
opt_vmc = VMC_multilevel(MaxIters, 0.5, ACEpsi.vmc.adamW(); lr_dc = 50.0)
#wf, err_opt, ps = gd_GradientByVMC_multilevel(opt_vmc, sam, ham, wf_list, ps_list, 
#                    st_list, spec_list, spec1p_list, specAO_list, batch_size = 500)


BFwf_chain = wf_list[end]
ps = ps_list[end]
st = st_list[end]

function rhō(r1; nchains = 10000000)
    c = 0
    for i = 1:nchains 
        x = (rand() - 0.5) * 10 # [-5,5]
        φ = (rand() - 0.5) * 2 * pi # [-pi, pi] sample
        θ = rand() * pi # [0, pi] sample
        S2 = Polynomials4ML.SphericalCoords(x, φ, θ) # spherical coords
        X2 = Polynomials4ML.spher2cart(S2) 
        φ = (rand() - 0.5) * 2 * pi # [-pi, pi] sample
        θ = rand() * pi # [0, pi] sample
        S1 = Polynomials4ML.SphericalCoords(r1, φ, θ) # spherical coords
        X1 = Polynomials4ML.spher2cart(S1) 
        X = [X1, X2]
        Y = BFwf_chain(X, ps, st)[1]
        c += exp(Y) * 2 * pi * pi * 2 * pi * pi * 10
    end
    return c/nchains
end

function testcusp_en(rhō::Function;verbose=true)
    errors = Float64[]
    # loop through finite-difference step-lengths
    verbose && @printf("---------|---------|----------|----------- \n")
    verbose && @printf("    h    |   ~Z    |  rho(h)  |  rho(h/2)  \n")
    verbose && @printf("---------|---------|----------|----------- \n")
    dr = 0.0
    for p = 0:5:70
        h = 0.9^p
        ρ̄h = rhō(h)
        ρ̄h2 = rhō(h/2)
        dr = ((ρ̄h2 - ρ̄h)/h)/ ρ̄h
        push!(errors, dr)
        verbose && @printf(" %1.1e | %.3f  | %.6f | %.6f\n", h, errors[end], ρ̄h, ρ̄h2)
    end
end

testcusp_en(rhō)

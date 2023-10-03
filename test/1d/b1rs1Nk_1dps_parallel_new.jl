using Distributed
using BenchmarkTools

if nprocs() == 1
    addprocs(5, exeflags="--project=$(Base.active_project())")
end

@everywhere begin|
    using ACEpsi, StaticArrays, Test
    using Polynomials4ML
    using Polynomials4ML: natural_indices, degree, SparseProduct
    using ACEpsi.vmc: d1_lattice, EmbeddingW!
    using ACEpsi: BackflowPooling1d, BFwf1dps_lux, BFwf1dps_lux2,setupBFState, Jastrow
    using ACEpsi.vmc: gradient, laplacian, grad_params, SumH, MHSampler, VMC, gd_GradientByVMC, d1, adamW, sr
    using LuxCore
    using Lux
    using Zygote
    using Optimisers # mainly for the destrcuture(ps) function
    using Random
    using LinearAlgebra
    using BenchmarkTools
    using HyperDualNumbers: Hyper
    using SpecialFunctions
    using Dates, JLD
end
@info("Running 1dLuxCode")
@everywhere begin
    Nel = 6
    rs = 1 # Wigner-Seitz radius r_s for 1D = 1/(2ρ); where ρ = N/L
    ρ = 1 / (2 * rs) # (average density)
    L = Nel / ρ # supercell size

    b = 1 # harmonic trap strength (the larger the "flatter") or simply "width"
    Σ = Array{Char}(undef, Nel)
    # paramagnetic 
    for i = 1:Int(Nel / 2)
        Σ[i] = ↑
        Σ[Int(Nel / 2)+i] = ↓
    end

    # Defining OrbitalsBasis
    totdegree = [34]
    ord = length(totdegree)
    Pn = Polynomials4ML.RTrigBasis(maximum(totdegree))
    length(Pn)
    trans = (x -> 2 * pi * x / L)

    # @info("setting up old wf in new code")
    _get_ord = bb -> sum([bb[i].n .!= 1 for i = 1:length(bb)]) == 0 ? 1 : sum([bb[i].n .!= 1 for i = 1:length(bb)])
    sd_admissible_func(ord,Deg) = bb -> (all([length(bb) == ord]) # must be of order ord, and 
                                        && (all([sum([bb[i].n .!= 1 for i = 1:length(bb)]) == 0]) # all of the basis in bb must be (1, σ)
                                            || all([sum([bb[i].n for i = 1:length(bb)]) <= Deg[_get_ord(bb)] + ord])) # if the degree of the basis is less then the maxdegree, "+ ord" since we denote degree 0 = 1
                                            && (bb[1].s == '∅') # ensure b=1 are of empty spin
                                            && all([b.s != '∅' for b in bb[2:end]])) # ensure b≠1 are of of non-empty spin
    sd_admissible = sd_admissible_func(ord,totdegree[1])

    wf, spec, spec1p = BFwf1dps_lux(Nel, Pn; ν = ord, trans = trans,  totdeg = totdegree[1], sd_admissible = sd_admissible)
    ps, st = setupBFState(MersenneTwister(1234), wf, Σ)
    
    ## check spec if needed
    # function getnicespec(spec::Vector, spec1p::Vector)
    #     return [[spec1p[i] for i = spec[j]] for j = eachindex(spec)]
    # end
    # @show getnicespec(spec, spec1p);

    ## customized initial parameters
    # use UHF calculation as initial guess
    # Dic = load("/zfs/users/berniehsu/berniehsu/OneD/ACEpsi.jl/test/1d/UHF_Trig_Data_K35.jld")
    # C_up = Dic["C_up"]
    # C_down = Dic["C_down"]
    # W1 = zeros(size(C_up)[1],Nel)
    # W1[:,1:Int(Nel/2)] = C_up[:,1:Int(Nel/2)]
    # W1[:,Int(Nel/2)+1:end] = C_down[:,1:Int(Nel/2)]

    # for i = axes(W1, 1)
    #     for j = axes(W1, 2)
    #         ps.hidden1.W[i,j] = W1[i,j]
    #     end
    # end

    # use old code good data
    # Dic = load("/zfs/users/berniehsu/berniehsu/OneD/ACEpsi.jl/test/1d/b1rs1maxnu3N30.jld")
    # c = Dic["params"]
    # Pold = c.p.P
    # for i = eachindex(Pold)
    #     for j = eachindex(Pold[i])
    #         ps.hidden1.W[j,i] = Pold[i][j]
    #     end
    # end
    
    # det structure
    p, = destructure(ps)

    # pair potential
    function v_ewald(x::AbstractFloat, b::Real, L::Real, M::Integer, K::Integer)

        erfox(y) = (erf(y) - 2 / sqrt(pi) * y) / (y + eps(y)) + 2 / sqrt(pi)
        f1(m) = (y = abs(x - m * L) / (2 * b); (sqrt(π) * erfcx(y) - erfox(y)) / (2 * b))
        f2(n) = (G = 2 * π / L; expint((b * G * n)^2) * cos(G * n * x))

        return sum(f1, -M:M) + sum(f2, 1:K) * 2 / L
    end
    M = 500
    K = 50
    vb(x) = v_ewald(x, b, L, M, K)
    V(X::AbstractVector) = sum(vb(X[i]-X[j]) for i = 1:length(X)-1 for j = i+1:length(X));
    # Mdelung energy
    Mad = (Nel / 2) * (vb(0.0) - sqrt(pi) / (2 * b))

    ACEpsi.Kin(wf, X::AbstractVector, ps, st) = -0.5 * laplacian(wf, X, ps, st)
    ACEpsi.Vext(wf, X::AbstractVector, ps, st) = 0.0
    ACEpsi.Vee(wf, X::AbstractVector, ps, st) = V(X) + Mad

    # # define lattice pts for sampler_restart # considering deleting this altogether
    # spacing = L / Nel
    # x0 = -L / 2 + spacing / 2
    # Lattice = [x0 + (k - 1) * spacing for k = 1:Nel]
    # d = d1_lattice(Lattice)

    burnin = 2000
    N_chain = 600
    MaxIters = 200
    lr = 0.02
    lr_dc = 100
    batch_size = floor(Int, N_chain / nprocs())
    @assert batch_size * nprocs() == N_chain

    ham = SumH(Kin, Vext, Vee)
    sam = MHSampler(wf, Nel, Δt = 0.5, burnin = burnin, nchains = N_chain) # remove d = d for now

    opt_vmc = VMC(MaxIters, lr, adamW(), lr_dc = lr_dc) # apparently using a good a basis I only get 1 iteration with default tol = 1e-3
end
@info("Create path")
# probably better to open a directory here so that initialization is saved
# results_dir = @__DIR__() * "/jellium_data/b1rs1maxnu3N$Nel" * string(Dates.now()) * "/"
# mkpath(results_dir)

@info("Running b$(b)rs$(rs)N$(Nel) with $(nprocs()) processes")

@assert N_chain % nprocs() == 0 "N_chain must be divisible by nprocs()"
# error function derived from MATH607
err_recip(K; L::Real=1, b::Real=1) = (1 / (pi * b)) * (L / (2 * pi * K * b))^3 * exp(-(2 * pi * b * K / L)^2)
err_real(M; L::Real=1, b::Real=1) = ((2 * b)^2 / (sqrt(2) * L^3)) * 1 / (M - 1)^2

@info("checking that error for this choice of truncation < 10^-8")
@assert err_recip(K; L=L, b=b) < 1e-8
@assert err_real(M; L=L, b=b) < 1e-8

## save initial config
# @info("initial config saved at : ", results_dir)
# save(results_dir * "Config_b1rs1maxnu3N$Nel.jld", "Nel", Nel, "MaxDeg" , totdegree[1], "MaxIters" , MaxIters, "burnin" , burnin, "N_chain" , N_chain, "lr", lr, "lr_dc", lr_dc)
# @info("Set-up done. Into VMC")
# @assert nchains / nprocs() == batch_size
wf, err_opt, ps = gd_GradientByVMC(opt_vmc, sam, ham, wf, ps, st; batch_size = batch_size )

# @show ps.hidden1.W

# # some sanity check
# length(p)
# X =  rand(30)
# wf(X, ps, st)

# using Plots
# p = plot(err_opt/N, w = 3)
# hline!([E_HF], lw=3, label="RHF($E_RHF)"

# # save parameters
# save(results_dir * "b1rs1N$(Nel)_W1.jld", "W", ps.hidden1.W)

# ## manual multilevel using embedding
# # first define wf2
# totdegree = [3, 2]
# ord = length(totdegree)
# Pn = Polynomials4ML.RTrigBasis(maximum(totdegree)+ord)
# sd_admissible = sd_admissible_func(ord,totdegree)
# wf2, spec2, spec1p2 = BFwf1dps_lux(Nel, Pn; ν = ord, trans = trans, totdeg = length(Pn), sd_admissible = sd_admissible)
# ps2, st = setupBFState(MersenneTwister(1234), wf2, Σ)
# # @show getnicespec(spec2, spec1p2)

# # EmbeddingW!(ps, ps2, spec, spec2, spec1p, spec1p2)

# wf2, err_opt, ps = gd_GradientByVMC(opt_vmc, sam, ham, wf2, ps2, st)
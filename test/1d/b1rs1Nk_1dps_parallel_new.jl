using Distributed
using BenchmarkTools

if nprocs() == 1
    addprocs(49, exeflags="--project=$(Base.active_project())")
end

@everywhere begin|
    using ACEpsi, StaticArrays, Test
    using Polynomials4ML
    using Polynomials4ML: natural_indices, degree, SparseProduct
    using ACEpsi.vmc: d1_lattice, EmbeddingW!
    using ACEpsi: BackflowPooling1d, BFwf1dps_lux, BFwf1dps_lux2,setupBFState, Jastrow
    using ACEpsi.vmc: gradient, laplacian, grad_params, SumH, MHSampler, VMC, gd_GradientByVMC, d1, adamW, SR
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
    Nel = N = 30
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

    # ## customized initial parameters
    ## use UHF calculation as initial guess
    # Dic = load("/zfs/users/berniehsu/berniehsu/OneD/ACEpsi.jl/test/1d/UHF_Trig_Data_K35.jld")
    # C_up = Dic["C_up"]
    # C_down = Dic["C_down"]
    # W1 = zeros(size(C_up)[1],Nel)
    # W1[:,1:Int(Nel/2)] = C_up[:,1:Int(Nel/2)]
    # W1[:,Int(Nel/2)+1:end] = C_down[:,1:Int(Nel/2)]

    # # normalization assumed in UHF code
    # W1[1,:] = W1[1,:]*sqrt(1/L)
    # W1[2:end,:] = W1[2:end,:]*sqrt(2/L)

    # for i = axes(W1, 1)
    #     for j = axes(W1, 2)
    #         ps.hidden1.W[j,i] = W1[i,j]
    #     end
    # end

    # use ACESchrodinger code good data
    # Dic = load("/zfs/users/berniehsu/berniehsu/OneD/ACEpsi.jl/test/1d/DataDeg35Ord1_140.jld")
    # c = Dic["params"]
    # Pold = c.p.P
    # for i = eachindex(Pold)
    #     for j = eachindex(Pold[i])
    #         ps.hidden1.W[i,j] = Pold[i][j]
    #     end
    # end
    
    # use good data from previous run
    Dic = load("/zfs/users/berniehsu/berniehsu/OneD/ACEpsi.jl/test/1d/jellium_data/b1rs1maxnu3N302023-10-19T10:52:54.994/Data_110.jld")
    c = Dic["params"]
    for i = axes(c, 1)
        for j = axes(c, 2)
            ps.hidden1.W[i,j] = c[i,j]
        end
    end
    
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


    Kin(wf, X::AbstractVector, ps, st) = -0.5 * laplacian(wf, X, ps, st)
    Vext(wf, X::AbstractVector, ps, st) = 0.0
    Vee(wf, X::AbstractVector, ps, st) = V(X) + Mad

    # define lattice pts for sampler_restart # considering deleting this altogether
    spacing = L / Nel
    x0 = -L / 2 + spacing / 2
    Lattice = [x0 + (k - 1) * spacing for k = 1:Nel]
    d = d1_lattice(Lattice)

    burnin = 2000
    N_chain = 2000
    MaxIters = 200
    lr = 0.01
    lr_dc = 999999
    Δt = 0.5*L
    batch_size = floor(Int, N_chain / nprocs())
    @assert batch_size * nprocs() == N_chain

    ham = SumH(Kin, Vext, Vee)
    sam = MHSampler(wf, Nel, Δt=Δt , burnin = burnin, nchains = N_chain, d = d) # d = d for now

    opt_vmc = VMC(MaxIters, lr, adamW(), lr_dc = lr_dc)
end
@info("Running b$(b)rs$(rs)N$(Nel) with $(nprocs()) processes")
@assert N_chain % nprocs() == 0 "N_chain must be divisible by nprocs()"
# error function derived from MATH607
err_recip(K; L::Real=1, b::Real=1) = (1 / (pi * b)) * (L / (2 * pi * K * b))^3 * exp(-(2 * pi * b * K / L)^2)
err_real(M; L::Real=1, b::Real=1) = ((2 * b)^2 / (sqrt(2) * L^3)) * 1 / (M - 1)^2

@info("checking that error for this choice of truncation < 10^-8")
@assert err_recip(K; L=L, b=b) < 1e-8
@assert err_real(M; L=L, b=b) < 1e-8
# save initial config
results_dir = @__DIR__() * "/jellium_data/b1rs1maxnu3N$Nel" * string(Dates.now()) * "/"
@info("saving initial config at : ", results_dir)
mkpath(results_dir)
save(results_dir * "Config_b1rs1maxnu3N$N.jld", "N", N, "totdegree" , totdegree, "MaxIters" , MaxIters, "burnin" , burnin, "N_chain" , N_chain, "lr", lr, "lr_dc", lr_dc, "Δt", Δt)


@info("Set-up done. Into VMC")
wf, err_opt, ps = gd_GradientByVMC(opt_vmc, sam, ham, wf, ps, st; batch_size = batch_size )

## post-processing (plots, data, etc)
# using LaTeXStrings, Plots, JLD
# Dic2 = load("/zfs/users/berniehsu/berniehsu/OneD/ACEpsi.jl/test/1d/jellium_data/b1rs1maxnu3N302023-10-19T10:52:54.994/Config_b1rs1maxnu3N30.jld")
# N, N_chain, burnin, MaxIters, totdegree, lr, lr_dc = Dic2["N"], Dic2["N_chain"], Dic2["burnin"], Dic2["MaxIters"], Dic2["totdegree"], Dic2["lr"], Dic2["lr_dc"]

# Dic3 = load("/zfs/users/berniehsu/berniehsu/OneD/ACEpsi.jl/test/1d/tmp_wf_data/Data_110.jld")
# Eavg = Dic3["err_opt"]
# Eavg = Eavg[1:110]/N

# # DeXuan averaged energy code -- (of past 20 steps) in the iteration
# using Statistics
# per = 0.2
# err_avg = zero(Eavg)
# for i = 1:length(Eavg)
#     err_avg[i] = mean(Eavg[Int(ceil(i-per  * i)):i])
# end

# p = plot(lw=2, title="b=1, rs=1, N=$N, burnin=$(burnin), N_chain=$(N_chain), lr=$(lr), lr_dc=$(lr_dc),\n Optimizer=AdamW", 
# xlabel="# Iterations", ylabel="Energy(Hartree)", 
# legend=:outerbottom, 
# size=(800, 800),
# minorgrid=true)
# plot!(1:length(Eavg), Eavg, linestyle=:dash, lw=2, c=:blue, label="$(totdegree)")
# plot!(1:length(Eavg), err_avg, lw=2, c=2, label="avg loss")

# UHF_minimalTrig = -0.15379210153763304
# UHF_minimalPW = -0.15943012791954408
# hline!([UHF_minimalPW], linestyle=:dash, lw=2, label="UHF minimal PW basis ($UHF_minimalPW)")
# hline!([UHF_minimalTrig], linestyle=:dash, lw=2, label="UHF minimal Trig basis ($UHF_minimalTrig)")
# savefig(p, "/zfs/users/berniehsu/berniehsu/OneD/ACEpsi.jl/test/1d/jellium_data/b1rs1maxnu3N302023-10-19T10:52:54.994/b1rs1N6.png")
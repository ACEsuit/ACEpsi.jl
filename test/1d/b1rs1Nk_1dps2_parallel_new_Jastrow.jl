using Distributed
using BenchmarkTools

if nprocs() == 1
    addprocs(79, exeflags="--project=$(Base.active_project())")
end

@everywhere begin
    using ACEpsi, StaticArrays, Test
    using Polynomials4ML
    using Polynomials4ML: natural_indices, degree, SparseProduct
    using ACEpsi.vmc: d1_lattice, EmbeddingW_J!
    using ACEpsi: BackflowPooling1d, setupBFState, BFJwfTrig_lux
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
    using ForwardDiff: Dual
    using Dates
    import JSON3
end
@info("Running 1dLuxCode")

@everywhere begin|

    Nel = N = 30
    rs = 1.0 # Wigner-Seitz radius r_s for 1D = 1/(2ρ); where ρ = N/L
    ρ = 1 / (2 * rs) # (average density)
    L = Nel / ρ # supercell size

    b = 1.0 # harmonic trap strength (the larger the "flatter") or simply "width"
    Σ = Array{Char}(undef, Nel)
    # paramagnetic 
    for i = 1:Int(Nel / 2)
        Σ[i] = ↑
        Σ[Int(Nel / 2)+i] = ↓
    end

    # Defining OrbitalsBasis
    totdegree = [15, 5]
    ord = length(totdegree)
    Pn = Polynomials4ML.RTrigBasis(maximum(totdegree))
    trans = (x -> (2 * pi * x / L)::Union{Float64, Dual{Nothing, Float64, 1}, Hyper{Float64}})#::typeof(x))# ::Union{Float64, Dual{Nothing, Float64, 1}, Hyper{Float64}})

    ## Jastrow Factor
    deg_of_cos = 3
    deg_of_mono = 3
    Lu = 0.03*L # cutoff for Jastrow factor
    J = ACEpsi.JCasino1dVb(Lu , deg_of_mono, deg_of_cos, L)
    JS = ACEpsi.JCasinoChain(J)

    # @info("setting up old wf in new code")
    _get_ord = bb -> sum([bb[i].n .!= 1 for i = 1:length(bb)]) == 0 ? 1 : sum([bb[i].n .!= 1 for i = 1:length(bb)])
    sd_admissible_func(ord,Deg) = bb -> (all([length(bb) == ord]) # must be of order ord (this truncation is allowed only because 1 is included in the basis expanding body-order 1), 
    # && (all([sum([bb[i].n .!= 1 for i = 1:length(bb)]) == 0]) # all of the basis in bb must be (1, σ)
        # || 
        && all([sum([ceil(((bb[i].n)-1)/2) for i = 2:length(bb)]) <= ceil((Deg[_get_ord(bb)]-1)/2)])#) # if the wave# of the basis is less than the max wave#
        && (bb[1].s == '∅') # ensure b-order=1 are of empty spin (can be deleted because I have enforced it in BFwfTrig_lux)
        && all([b.s != '∅' for b in bb[2:end] if b.n != 1]) # ensure (non-cst) b-order>1 term are of of non-empty spin
        && all([b.s == '∅' for b in bb[2:end] if b.n == 1])) # ensure cst is of ∅ spin to avoid repeats, e.g [(2,∅),(1,↑)] == [(2,∅),(1,↓)], but notice δ_σ↑ + δ_σ↓ == 1
    sd_admissible = sd_admissible_func(ord, totdegree)
    # remember to switch to (or not) embed_diff_func 
    wf, spec, spec1p = BFJwfTrig_lux(Nel, Pn, nothing; ν = ord, trans = trans,  totdeg = totdegree[1], sd_admissible = sd_admissible, Jastrow_chain = JS)
    Wigner = false # this is always false in this script
    diff_coord = true
    ps, st = setupBFState(MersenneTwister(1234), wf, Σ)
    
    ## customized initial parameters
    # minimal HF is actually a good initial guess
    for i = 1:Nel # Nel
        for j = eachindex(spec) # basis size
            if j > Int(Nel/2)
                ps.to_be_prod.layer_1.hidden1.W[i,j] = 0.0
            end
        end
    end
    
    # Embedding minimal HF
    # totdegree = [15, 5]
    # ord = length(totdegree)
    # sd_admissible = sd_admissible_func(ord, totdegree)
    # wf, spec2, spec1p2 = BFJwfTrig_lux(Nel, Pn, nothing; ν = ord, trans = trans,  totdeg = totdegree[1], sd_admissible = sd_admissible, Jastrow_chain = JS)
    # ps2, st = setupBFState(MersenneTwister(1234), wf, Σ)
    # EmbeddingW_J!(ps, ps2, spec, spec2, spec1p, spec1p2)
    ## intializr CASINO Jastrow so that we have HF
    for i = 1:deg_of_cos
        ps.to_be_prod.layer_2.hiddenJS.layer_1.W[i] = 0.0
    end
    for i = 1:deg_of_mono+1
        ps.to_be_prod.layer_2.hiddenJS.layer_2.W[i] = 0.0
    end

    # manual multi-level
    # data from previous run
    # Dic = JSON3.read("/zfs/users/berniehsu/berniehsu/OneD/ACEpsi.jl/test/1d/jellium_data/Jastrowb1rs1N302024-01-27T09:26:49.738/Data50.json")
    # c = Dic["W"]
    # basis_size = Int(length(c)/Nel)
    # c = reshape(c, (Nel, basis_size))
    # for i = axes(c, 1)
    #     for j = axes(c, 2)
    #         ps.to_be_prod.layer_1.hidden1.W[i,j] = c[i,j]
    #     end
    # end
    # r0 = Vector.(Dic["x0"]) # init sample

    ## PauliNet Jastrow
    # ps.to_be_prod.layer_2.α1 = 0.0
    # ps.to_be_prod.layer_2.α2 = 0.0    

    # pair potential
    function v_ewald(x::AbstractFloat, b::Real, L::Real, M::Integer, K::Integer)

        erfox(y) = (erf(y) - 2 / sqrt(pi) * y) / (y + eps(y)) + 2 / sqrt(pi)
        f1(m) = (y = abs(x - m * L) / (2 * b); (sqrt(π) * erfcx(y) - erfox(y)) / (2 * b))
        f2(n) = (G = 2 * π / L; expint((b * G * n)^2) * cos(G * n * x))

        return sum(f1, -M:M) + sum(f2, 1:K) * 2 / L
    end
    M = 80
    K = 40
    vb(x) = v_ewald(x, b, L, M, K)
    V(X::AbstractVector) = sum(vb(X[i]-X[j]) for i = 1:length(X)-1 for j = i+1:length(X));
    # Mdelung energy
    Mad = (Nel / 2) * (vb(0.0) - sqrt(pi) / (2 * b))

    Kin(wf, X::AbstractVector, ps, st) = -0.5 * laplacian(wf, X, ps, st)
    Vext(wf, X::AbstractVector, ps, st) = 0.0
    Vee(wf, X::AbstractVector, ps, st) = V(X) + Mad

    # # define lattice pts for sampler_restart # considering deleting this altogether
    spacing = L / Nel
    x0 = -L / 2 + spacing / 2
    Lattice = [x0 + (k - 1) * spacing for k = 1:Nel]
    d = d1_lattice(Lattice)

    ### to get over with NaN output in the ifirst run
    # burnin = 3
    # N_chain = nprocs()
    ### comment out for 1st run
    burnin = 2000
    N_chain = 1600
    ### 
    MaxIters = 200
    lr = 0.01
    lr_dc = 10000000000
    Δt = 0.3*L
    lag = 20 # increase to see if optimization helps
    batch_size = floor(Int, N_chain / nprocs())
    # @assert batch_size * nprocs() == N_chain

    ham = SumH(Kin, Vext, Vee)
    sam = MHSampler(wf, Nel, Δt=Δt , burnin = burnin, lag = lag, nchains = N_chain, d = d, L = L)

    opt_vmc = VMC(MaxIters, lr, adamW(), lr_dc = lr_dc) # Trying out to SR
end
@info("Running b=$(b) rs=$(rs) N=$(Nel) with $(nprocs()) processes")
@assert N_chain % nprocs() == 0 "N_chain must be divisible by nprocs()"
# error function derived from MATH607
err_recip(K; L::Real=1, b::Real=1) = (1 / (pi * b)) * (L / (2 * pi * K * b))^3 * exp(-(2 * pi * b * K / L)^2)
err_real(M; L::Real=1, b::Real=1) = ((2 * b)^2 / (sqrt(2) * L^3)) * 1 / (M - 1)^2

@info("checking that error for this choice of truncation < 10^-8")
@assert err_recip(K; L=L, b=b) < 1e-8
@assert err_real(M; L=L, b=b) < 1e-8
# save initial config ## replace the dictionary with json
config_name = "/Jastrowb1rs1N$Nel"
results_dir = @__DIR__() * "/jellium_data" * config_name * string(Dates.now()) * "/"
@info("saving initial config at : ", results_dir)
mkpath(results_dir)
# save(results_dir * "Config_b1rs1maxnu3N$N.jld", "N", N, "totdegree" , totdegree, "MaxIters" , MaxIters, "burnin" , burnin, "N_chain" , N_chain, "lr", lr, "lr_dc", lr_dc, "Δt", Δt)
## save data using json
json_config = """{"N": $(N), "totdegree" : $(totdegree), "MaxIters" : $(MaxIters), "burnin" : $(burnin), "N_chain" : $(N_chain), "lr": $(lr), "lr_dc": $(lr_dc), "Δt": $(Δt), "Wigner": $(Wigner), "diff_coord": $(diff_coord), "rs": $(rs), "b": $(b), "Lu": $(Lu), "deg_of_mono": $(deg_of_mono), "deg_of_cos": $(deg_of_cos)}"""
open(results_dir * "Config.json", "w") do io
    JSON3.write(io, JSON3.read(json_config))
end

@info("Wigner = $(Wigner)")

# @info("Running random evalutation to get rid of the NaN output from Hyperdual pkg")
# X = (rand(Nel).-1/2)*L
# hX = [Hyper(x, 0, 0, 0) for x in X]
# hX[1] = Hyper(X[1], 1, 1, 0)
# @show wf(hX, ps2, st)[1]

@info("Set-up done. Into VMC")
wf, err_opt, ps, x0 = gd_GradientByVMC(opt_vmc, sam, ham, wf, ps, st; batch_size = batch_size) # remember to change index whenever needed

## Post-processing
# open("test/1d/jellium_data/b1rs1maxnu3N302023-10-25T09:03:50.303/Config_b1rs1.json", "w") do io
#     JSON3.write(io, JSON3.read(json_config))
# end

# using JSON3
# config = JSON3.read("/zfs/users/berniehsu/berniehsu/OneD/ACEpsi.jl/test/1d/jellium_data/Jastrowb1rs1N302024-02-08T12:35:28.674/Config.json")
# N, totdegree, MaxIters, burnin, N_chain, lr, lr_dc, Δt, Wigner, diff_coord, b, rs = config["N"], config["totdegree"], config["MaxIters"], config["burnin"], config["N_chain"], config["lr"], config["lr_dc"], config["Δt"], config["Wigner"], config["diff_coord"], config["b"], config["rs"]
# N = 30
# ρ = 1 / (2 * rs) # (average density)
# L = N / ρ # supercell size
# Dic = JSON3.read("/zfs/users/berniehsu/berniehsu/OneD/ACEpsi.jl/test/1d/jellium_data/Jastrowb1rs1N302024-02-08T12:35:28.674/Data30.json")
# E = Dic["E"]
# σ = Dic["σ"]
# Eavg = E[1:200]/N
# σavg = σ[1:200]/N
# using Printf
# for i = 1:200
#     @printf(" %d | %.5f | %.5f \n", i, Eavg[i], σ[i])
# end
# alpha = Dic["α"]
# W = Dic["W"]
# W = reshape(W, (N, Int(length(W)/N)))

# Dic2 = JSON3.read("/zfs/users/berniehsu/berniehsu/OneD/ACEpsi.jl/test/1d/jellium_data/b1rs1maxnu3N302023-10-28T10:14:13.322/Data200ForPlot.json")
# E2 = Dic2["E"]
# σ2 = Dic2["σ"]
# Eavg2 = E2[1:200]/N
# σavg2 = σ2[1:200]/N

# Eavg = cat(E2[1:200], E[1:200], dims=1)/N
# σavg = cat(σ2[1:200], σ[1:200], dims=1)/N


# DicHF = JSON3.read("/zfs/users/berniehsu/berniehsu/OneD/ACEpsi.jl/test/1d/jellium_data/b1rs1maxnu3N302023-10-26T06:43:40.461/Data200.json")
# E_HF = DicHF["E"]
# σ_HF = DicHF["σ"]
# Eavg_HF = E_HF[1:200]/N
# σavg_HF = σ_HF[1:200]/N

# # DeXuan averaged energy code -- (of past 20 steps) in the iteration
# using Statistics
# per = 0.2
# err_avg = zero(Eavg)
# for i = 1:length(Eavg)
#     err_avg[i] = mean(Eavg[Int(ceil(i-per  * i)):i])
# end

# using Plots
# p = plot(lw=2, title="b=$(b), rs=$(rs), N=$N, burnin=$(burnin), N_chain=$(N_chain), lr=$(lr)",
# xlabel="# Iterations", ylabel="Energy(Hartree) per electron", 
# legend=:outerbottom, 
# size=(800, 800),
# minorgrid=true)
# plot!(1:length(Eavg), Eavg, ribbon=σavg,linestyle=:solid, lw=2, label="Degree = $(totdegree), diff_coord: $(diff_coord), Wigner: $(Wigner), w/ Jastrow")
# plot!(1:length(Eavg2), Eavg2, ribbon=σavg,linestyle=:solid, lw=2, label="Same degree, no Jastrow")
# # # plot!(length(Eavg2)+1:length(Eavg2)+length(Eavg), Eavg, ribbon=σavg,linestyle=:solid, lw=2, c=:blue, label="Degree = $(totdegree)")
# # # # plot!(1:length(Eavg), err_avg, lw=2, c=2, label="avg loss")

# ## benchmarks
# hline!([UHF_minimalPW], c=:black, linestyle=:dash, lw=1, label="HF minimal PW basis ($UHF_minimalPW)")
# hline!([CASINO_VMC], c=:black, linestyle=:dashdotdot, lw=1, label="CASINO ($CASINO_VMC)")
# savefig(p, "/zfs/users/berniehsu/berniehsu/OneD/ACEpsi.jl/test/1d/jellium_data/Jastrowb1rs1N302024-01-25T11:03:08.547/b1rs1.png")

# check particle density

# x0 = Dic["x0"]
# reduce(vcat,x0)
# x = reduce(vcat,reduce(vcat,x0))
# b_range = range(-L-100, L+100, length=42)
# hist = histogram(x, bins=b_range, normalize=:pdf)
# xlims!(-L,L)
# rp = (map(x -> mod.(x .+ L/2, L) .- L, x0))
## b=1 rs=1
UHF_minimalPW = -0.1594301 #2791954408
CASINO_VMC = -0.1710782

# ## b=1 rs=0.5
# UHF_minimalPW = 0.0891900
# CASINO_VMC = 0.0853158

# ## b=2 rs=1
# UHF_minimalPW = -0.05775642
# CASINO_VMC = -0.0610819

# ## b=1 rs=2
# UHF_minimalPW = -0.1716581
# CASINO_VMC = -0.2006410

# ## b=1 rs=5
# UHF_minimalPW = -0.1161756
# CASINO_VMC = -0.168710
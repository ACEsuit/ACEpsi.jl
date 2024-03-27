export MHSampler
using StatsBase, StaticArrays, Optimisers, Distributed 
using ACEpsi.AtomicOrbitals: Nuc
using Lux: Chain
using ParallelDataTransfer: @getfrom
using SharedArrays
using Optimisers

abstract type init_type end
struct gaussian <: init_type end
struct exponential <: init_type end

struct Physical_config{T, NNuc}
    nuclei::SVector{NNuc, Nuc{T}}
    inuc::Vector{Int}
    el_config::Vector{Vector{Int}}
end 

mutable struct MHSampler{T}
    Nel::Int64                  # Number of electrons
    Δt::Float64                 # Step size (of Gaussian proposal)
    burnin::Int64               # Burn-in iterations
    lag::Int64                  # Iterations between successive samples
    nchains::Int64              # Number of chains
    Ψ::Chain                    # Many-body wavefunction for sampling
    x0::Vector                  # Initial sampling
    init_method::init_type      # Initialization method
    physical_config::Physical_config{T} # Nuclear coordinates
end

MHSampler(Ψ, Nel, physical_config; Δt = 0.1, burnin = 100, lag = 10, nchains = 1000, init_method = gaussian()) = MHSampler(Nel, Δt, burnin, lag, nchains, Ψ, initialize_around_nuclei(nchains, physical_config, init_method, Nel), init_method, physical_config)

function initialize_around_nuclei(physical_config, init_method::gaussian, Nel::Int)
    r0 = randn(SVector{3, Float64}, Nel)
    for (i_el, i_nuc) in enumerate(physical_config.inuc)
        r0[i_el] += physical_config.nuclei[i_nuc].rr
    end
    return r0
end

function initialize_around_nuclei(physical_config, init_method::exponential, Nel::Int)
    r0 = []
    for m = 1:length(physical_config.el_config)
        for n = 1:length(physical_config.el_config[m])
            n_el_in_shell = physical_config.el_config[m][n]
            Z_eff = ACEpsi.vmc._get_effective_charge(physical_config.nuclei[m].charge, n, 1.0, 0.7)
            exponent = 2 * Z_eff / n
            push!(r0, ACEpsi.vmc.generate_exp_distributed(n_el_in_shell, exponent))
        end
    end
    r0 = reduce(vcat, r0)
    for (i_el, i_nuc) in enumerate(physical_config.inuc)
        r0[i_el] += physical_config.nuclei[i_nuc].rr
    end
    return r0
end

initialize_around_nuclei(nchains, physical_config, init_method, Nel::Int) = [initialize_around_nuclei(physical_config, init_method, Nel) for _ = 1:nchains]

eval_Ψ(Ψ, r, ps, st) = Ψ(r, ps, st)[1]

function sampler(sam::MHSampler, lag, ps, st; return_Ψx0 = true)
    @everywhere lag = $lag
    @everywhere ps = $ps
    @everywhere begin
        global r0, Ψx0, acc
        r0 = sam.x0
        Ψx0 = eval_Ψ.(Ref(sam.Ψ), r0, Ref(ps), Ref(st))
        acc = 0.0
        for _ = 1:lag
            global r0, Ψx0, acc
            local a
            r0, Ψx0, a = MHstep(r0, Ψx0, sam.Nel, sam, ps, st)
            acc += mean(a)
        end
        acc = acc / lag
    end
    r0_all = vcat([@getfrom k r0 for k in procs()]...)
    acc_all = [@getfrom k acc for k in procs()]
    if return_Ψx0
        Ψx0_all = vcat([@getfrom k Ψx0 for k in procs()]...)
        return r0_all, Ψx0_all, mean(acc_all)
    else
        return r0_all, nothing, mean(acc_all)
    end
end

function MHstep(r0::Vector{Vector{SVector{3, TT}}}, 
                Ψx0::Vector{T}, 
                Nels::Int64, 
                sam::MHSampler, ps::NamedTuple, st::NamedTuple) where {T, TT}
    rand_sample(X::Vector{SVector{3, TX}}, Nels::Int, Δt::Float64) where {TX}= X + Δt * randn(SVector{3, TX}, Nels)
    rp = rand_sample.(r0, Ref(Nels), Ref(sam.Δt))
    Ψxp = eval_Ψ.(Ref(sam.Ψ), rp, Ref(ps), Ref(st))
    accprob = exp.(Ψxp .- Ψx0)
    u = rand(sam.nchains)
    acc = u .<= accprob[:]
    r::Vector{Vector{SVector{3, TT}}} = acc .*  rp + (1.0 .- acc) .* r0
    Ψ = acc .*  Ψxp + (1.0 .- acc) .* Ψx0
    return r, Ψ, acc
end

##

params(a::NamedTuple) = destructure(a)[1]

function grad(wf, x, ps, st, E);
   dy = grad_params.(Ref(wf), x, Ref(ps), Ref(st));
   N = length(x)
   p = params.(dy)
   _, t = destructure(dy[1])
   g = 1/N * sum( p .* E) - 1/(N^2) * sum(E) * sum(p)
   g = t(g)
   return g;
end

# evaluate Elocal on each processor respectively and return - this reduces communication cost
function sampler_Elocal_grad_params(sam, lag, ps, st, ham)
    @everywhere lag = $lag
    @everywhere ps = $ps
    @everywhere begin
        global r0, Ψx0, acc
        r0 = sam.x0
        Ψx0 = eval_Ψ.(Ref(sam.Ψ), r0, Ref(ps), Ref(st)) # TODO: multithread
        acc = 0.0
        for _ = 1:lag
            global r0, Ψx0, acc
            local a
            r0, Ψx0, a = MHstep(r0, Ψx0, sam.Nel, sam, ps, st)
            acc += mean(a)
        end
        acc = acc / lag
        Eloc = Elocal.(Ref(ham), Ref(sam.Ψ), r0, Ref(ps), Ref(st)) # TODO: multithread
        # we set sam.x0 here so that we don't have to pass back the huge array
        # back to the call in mulitlevel VMC
        sam.x0 = r0
        dp = grad_params.(Ref(sam.Ψ), r0, Ref(ps), Ref(st))

        T = promote_type(eltype(r0[1]), eltype(destructure(ps)[1]))
        dp = Vector{Vector{T}}()
        Base.Threads.@threads for r in r0
            push!(dp, grad_params(sam.Ψ, r, ps, st)) # TODO: fix this
        end
    end

    Eloc_all = vcat([@getfrom k Eloc for k in procs()]...)
    acc_all = [@getfrom k acc for k in procs()]
    dps = vcat([@getfrom k dp for k in procs()]...)
    return Eloc_all, mean(acc_all), dps
end



function Eloc_Exp_TV_clip(wf, ps, st,
                sam::MHSampler, 
                ham::SumH;
                clip = 5.,)
    Eloc, acc, dps = sampler_Elocal_grad_params(sam, sam.lag, ps, st, ham)
    val = sum(Eloc) / length(Eloc)
    var = sqrt(sum((Eloc .-val).^2)/(length(Eloc)*(length(Eloc) -1)))
    ΔE = Eloc .- median( Eloc )
    a = clip * mean( abs.(ΔE) )
    ind = findall(_x -> abs(_x) > a, ΔE)
    ΔE[ind] = (a * sign.(ΔE) .* (1 .+ log.((1 .+(abs.(ΔE)/a).^2)/2)))[ind]
    E_clip = median(Eloc) .+ ΔE
    return val, var, E_clip, acc, dps
end

function _get_effective_charge(Z, n, s_lower_shell, s_same_shell)
    shielding = 0
    for n_shell in range(1, n + 1)
        n_electrons_in_shell = 2 * n_shell^2
        if n_shell == n
            n_el_in_lower_shells = sum([2 * k^2 for k in range(1, n_shell)])
            n_electrons_in_shell = min(Z - n_el_in_lower_shells, n_electrons_in_shell) - 1
        end
        if n_shell == n
            shielding += n_electrons_in_shell * s_same_shell
        elseif n_shell == (n - 1)
            shielding += n_electrons_in_shell * s_lower_shell
        else
            shielding += n_electrons_in_shell
        end
    end
    return max(Z - shielding, 1)
end

function generate_exp_distributed(n::Int, k = 1.0)
    """Sample 3D points which are distributed spherically symmetrically, with an exponentially decaying radial pdf

    p(r) = r^2 * exp(-k*r)
    """
    xp = [ 0.        ,  0.16646017,  0.2962203 ,  0.42016042,  0.54028054,
        0.6017006 ,  0.66318066,  0.78726079,  0.91426091,  1.04588105,
        1.15018115,  1.25946126,  1.37514138,  1.4998815 ,  1.73462173,
        2.22060222,  2.43890244,  2.66892267,  2.88428288,  3.004163  ,
        3.12246312,  3.23980324,  3.35662336,  3.47338347,  3.59030359,
        3.70764371,  3.82564383,  3.93746394,  4.05024405,  4.16416416,
        4.27938428,  4.3960844 ,  4.51440451,  4.63452463,  4.75662476,
        5.00734501,  5.26824527,  5.54106554,  5.82790583,  6.12646613,
        6.44380644,  6.78324678,  7.14934715,  7.54514755,  7.98038798,
        8.46468846,  9.01334901, 10.13523014, 11.71389171, 20.        ]
    Fp = [0.00000000e+00, 6.78872002e-04, 3.47483629e-03, 9.05128575e-03,
       1.76268412e-02, 2.32836169e-02, 2.98157678e-02, 4.56084288e-02,
       6.52255332e-02, 8.89313535e-02, 1.09892813e-01, 1.33656048e-01,
       1.60527321e-01, 1.91123424e-01, 2.51942878e-01, 3.82805274e-01,
       4.40421447e-01, 4.98732268e-01, 5.50391443e-01, 5.77741958e-01,
       6.03679841e-01, 6.28341184e-01, 6.51818158e-01, 6.74201746e-01,
       6.95532631e-01, 7.15858180e-01, 7.35220119e-01, 7.52589600e-01,
       7.69166627e-01, 7.84977683e-01, 8.00044955e-01, 8.14391869e-01,
       8.28035657e-01, 8.40997278e-01, 8.53296166e-01, 8.75965247e-01,
       8.96197668e-01, 9.14129002e-01, 9.29897296e-01, 9.43441707e-01,
       9.55144213e-01, 9.65128109e-01, 9.73528150e-01, 9.80434071e-01,
       9.86033946e-01, 9.90453671e-01, 9.93834179e-01, 9.97521537e-01,
       9.99334839e-01, 9.99999544e-01]
    r = [randn(SVector{3, Float64}) for _ = 1:n]
    u = rand(n)
    interp = LinearInterpolation(xp, Fp)
    x = interp(u)
    r = r ./ norm.(r) .* (x / k)
    return r
end




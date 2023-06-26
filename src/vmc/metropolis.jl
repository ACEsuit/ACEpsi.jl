using StatsBase
using StaticArrays
using Optimisers
export MHSampler
"""
`MHSampler`
Metropolis-Hastings sampling algorithm.
"""
mutable struct MHSampler
    Nel::Int
    Δt::Float64                 # step size (of Gaussian proposal)
    burnin::Int                 # burn-in iterations
    lag::Int                    # iterations between successive samples
    N_batch::Int                # batch size
    nchains::Int                # Number of chains
    Ψ                           # many-body wavefunction for sampling
    x0::Any                     # initial sampling 
    walkerType::String          # walker type: "unbiased", "Langevin"
    bc::String                  # boundary condition
    type::Int64                 # move how many electron one time 
end

MHSampler(Ψ, Nel; Δt = 0.1, 
            burnin = 100, 
            lag = 10, 
            N_batch = 1, 
            nchains = 1000,
            x0 = [],
            wT = "unbiased", 
            bc = "periodic", 
            type = 1) =
    MHSampler(Nel, Δt, burnin, lag, N_batch, nchains, Ψ, x0, wT, bc, type)


"""
unbiased random walk: R_n+1 = R_n + Δ⋅Wn
biased random walk:   R_n+1 = R_n + Δ⋅Wn + Δ⋅∇(log Ψ)(R_n)
"""

eval(wf, X::AbstractVector, ps, st) = wf(X, ps, st)[1]

function MHstep(r0, 
                Ψx0, 
                Nels::Int, 
                sam::MHSampler, ps, st)
    rand_sample(X::AbstractVector, Nels::Int, Δt::AbstractFloat) = begin
        return X + Δt * randn(SVector{3, Float64}, Nels)
    end
    rp = rand_sample.(r0, Ref(Nels), Ref(sam.Δt))
    Ψxp = eval.(Ref(sam.Ψ), rp, Ref(ps), Ref(st))
    accprob = accfcn(Ψx0, Ψxp)
    u = rand(sam.nchains)
    acc = u .<= accprob[:]
    r = acc .*  rp + (1.0 .- acc) .* r0
    Ψ = acc .*  Ψxp + (1.0 .- acc) .* Ψx0
    return r, Ψ, acc
end

"""
acceptance rate for log|Ψ|
ψₜ₊₁²/ψₜ² = exp((log|Ψₜ₊₁|^2-log |ψₜ|^2))
"""

function accfcn(Ψx0, Ψxp)  
    acc = exp.(Ψxp .- Ψx0)
    return acc
end

"""============== Metropolis sampling algorithm ============
type = "restart"
"""

function sampler_restart(sam::MHSampler, ps, st)
    r0 = [sam.Δt * randn(SVector{3, Float64}, sam.Nel) for _ = 1:sam.nchains]
    Ψx0 = eval.(Ref(sam.Ψ), r0, Ref(ps), Ref(st))
    acc = []
    for _ = 1 : sam.burnin
        r0, Ψx0, a = MHstep(r0, Ψx0, sam.Nel, sam, ps, st);
        push!(acc,a)
    end
    return r0, Ψx0, mean(mean(acc))
end

"""
type = "continue"
start from the previous sampling x0
"""
function sampler(sam::MHSampler, ps, st)
    if isempty(sam.x0)
        r0, Ψx0, = sampler_restart(sam, ps, st);
    else
        r0 = sam.x0
        Ψx0 = eval.(Ref(sam.Ψ), r0, Ref(ps), Ref(st))
    end
    acc = []
    for i = 1:sam.lag
        r0, Ψx0, a = MHstep(r0, Ψx0, sam.Nel, sam, ps, st);
        push!(acc, a)
    end
    return r0, Ψx0, mean(mean(acc))
end



"""
Rayleigh quotient by VMC using Metropolis sampling
"""
function rq_MC(Ψ, sam::MHSampler, ham::SumH, ps, st)
    r, ~, acc = sampler(sam, ps, st);
    Eloc = Elocal.(Ref(ham), Ref(Ψ), r, Ref(sam.Σ))
    val = sum(Eloc) / length(Eloc)
    var = sqrt(sum((Eloc .-val).^2)/(length(Eloc)*(length(Eloc)-1)))
    return val, var, acc
end

function Eloc_Exp_TV_clip(wf, ps, st,
                sam::MHSampler, 
                ham::SumH;
                clip = 5.)
    x, ~, acc = sampler(sam, ps, st)
    Eloc = Elocal.(Ref(ham), Ref(wf), x, Ref(ps), Ref(st))
    val = sum(Eloc) / length(Eloc)
    var = sqrt(sum((Eloc .-val).^2)/(length(Eloc)*(length(Eloc) -1)))
    ΔE = Eloc .- median( Eloc )
    a = clip * mean( abs.(ΔE) )
    ind = findall(x -> abs(x) > a, ΔE)
    ΔE[ind] = (a * sign.(ΔE) .* (1 .+ log.((1 .+(abs.(ΔE)/a).^2)/2)))[ind]
    E_clip = median(Eloc) .+ ΔE
    return val, var, E_clip, x, acc
end

function params(a::NamedTuple)
    p,= destructure(a)
    return p
end

function grad(wf, x, ps, st, E);
   dy = grad_params.(Ref(wf), x, Ref(ps), Ref(st));
   N = length(x)
   p = params.(dy)
   _,t = destructure(dy[1])
   g = 1/N * sum( p .* E) - 1/(N^2) * sum(E) * sum(p)
   g = t(g)
   return g;
end


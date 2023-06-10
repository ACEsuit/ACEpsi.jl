using StatsBase

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
            type = 3) =
    MHSampler(Nel, Δt, burnin, lag, N_batch, nchains, Ψ, x0, wT, bc, type)


"""
unbiased random walk: R_n+1 = R_n + Δ⋅Wn
biased random walk:   R_n+1 = R_n + Δ⋅Wn + Δ⋅∇(log Ψ)(R_n)
"""



function MHstep(r0, 
                Ψx0, 
                Nels::Int, 
                sam::MHSampler)
    rand_sample(X::AbstractVector, u::Int, Δt::AbstractFloat) = begin
        Y = copy(X)
        ind = sample(1:length(X), u, replace=false)
        Y[ind] = X[ind] + Δt * randn(SVector{3, Float64}, u)
        return Y
    end
    rp = rand_sample.(r0, Ref(sam.type), Ref(sam.Δt))
    Ψxp = ACEpsi.vmc.evaluate.(Ref(sam.Ψ), rp, Ref(ps), Ref(st))
    accprob = accfcn(r0, rp, Ψx0, Ψxp, sam)
    u = rand(sam.nchains)
    acc = u .<= accprob[:]
    r = acc .*  rp + (1 .- acc) .* r0
    Ψ = acc .*  Ψxp + (1 .- acc) .* Ψx0
    return r, Ψ, acc
end

"""
acceptance rate for log|Ψ|
ψₜ₊₁²/ψₜ² = exp((log|Ψₜ₊₁|^2-log |ψₜ|^2))
"""

function accfcn(r0, rp, Ψx0, Ψxp, sam::MHSampler)  
    acc = exp.(Ψxp .- Ψx0)
    return acc
end

"""============== Metropolis sampling algorithm ============
type = "restart"
"""
function sampler_restart(sam::MHSampler, ps, st)
    r0 = [randn(SVector{3, Float64}, Nel) for _ = 1:sam.nchains]
    Ψx0 = ACEpsi.vmc.evaluate.(Ref(sam.Ψ), r0, Ref(ps), Ref(st))
    acc = []
    for _ = 1 : sam.burnin
        r0, Ψx0, a = MHstep(r0, Ψx0, sam.Nel, sam);
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
        Ψx0 = ACEpsi.vmc.evaluate.(Ref(sam.Ψ), r0, Ref(ps), Ref(st))
    end
    acc = []
    for i = 1:sam.lag
        r0, Ψx0, a = MHstep(r0, Ψx0, sam.Nel, sam);
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
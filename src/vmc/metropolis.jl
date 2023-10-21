using StatsBase
using StaticArrays
using Optimisers
using Distributed: @spawnat, @fetch, @distributed
using SharedArrays
using Distributions

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
    d::sampler_dimension        # electron dimension
end

MHSampler(Ψ, Nel; Δt = 0.1, 
            burnin = 100, 
            lag = 10, 
            N_batch = 1, 
            nchains = 1000,
            x0 = [],
            wT = "unbiased", 
            bc = "periodic", 
            d = d3()) =
    MHSampler(Nel, Δt, burnin, lag, N_batch, nchains, Ψ, x0, wT, bc, d)


"""
unbiased random walk: R_n+1 = R_n + Δ⋅Wn
biased random walk:   R_n+1 = R_n + Δ⋅Wn + Δ⋅∇(log Ψ)(R_n)
"""

eval(wf, X::AbstractVector, ps, st) = wf(X, ps, st)[1]

function MHstep(r0, 
                Ψx0, 
                Nels::Int, 
                sam::MHSampler, ps, st; batch_size = 1)
    rp = rand_sample.(r0, Ref(Nels), Ref(sam.Δt), Ref(sam.d)) # might have found a mistake here. Once this line is executed, rp = r0 so... (see below)
    # rp = deepcopy(r0) # ACESchrodinger code
    # rand_sample.(rp, Ref(Nels), Ref(sam.Δt), Ref(sam.d))
    raw_data = pmap(rp; batch_size = batch_size) do d
        sam.Ψ(d, ps, st)[1]
    end
    Ψxp = vcat(raw_data)
    accprob = accfcn(Ψx0, Ψxp)
    u = rand(sam.nchains)
    acc = u .<= accprob[:]
    r = acc .*  rp + (1.0 .- acc) .* r0 # ctd: so r is the proposed position even if the move is rejected
    Ψ = acc .*  Ψxp + (1.0 .- acc) .* Ψx0
    return r, Ψ, acc
end

rand_sample(X::AbstractVector, Nels::Int, Δt::Number, d::d3) = begin
    @view(X[rand(1:Nels)]) .+= Δt * randn(SVector{3, eltype(X[1])}, 1)
    return X
end

rand_sample(X::AbstractVector, Nels::Int, Δt::Number, d::T) where T <: Union{d1, d1_lattice} = begin
    X1 = deepcopy(X)
    rand_index = rand(1:Nels)
    X1[rand_index] += Δt * rand(Normal(0.0, 1.0))
    # @view(X[rand(1:Nels)]) .+= Δt * rand(Normal(0.0, 1.0)) # need to ask what @view REALLY is
    return X1
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
rand_init(Δt::Number, Nel::Int, nchains::Int, d::d3) = [Δt * randn(SVector{3, Float64}, Nel) for _ = 1:nchains]

rand_init(Δt::Number, Nel::Int, nchains::Int, d::d1) = [Δt * randn(Nel) for _ = 1:nchains]

# same as d1 rand_init, except shifted by equally spaced lattice
rand_init(Δt::Number, Nel::Int, nchains::Int, d::d1_lattice) = [[x_e + randn() for x_e in d.L] for _ = 1:nchains]

function sampler_restart(sam::MHSampler, ps, st; batch_size = 1)
    r0 = rand_init(sam.Δt, sam.Nel, sam.nchains, sam.d)
    # println("Initial sample: ")
    # @show r0
    # Ψx0 = eval.(Ref(sam.Ψ), r0, Ref(ps), Ref(st))
    raw_data = pmap(r0; batch_size = batch_size) do d
        sam.Ψ(d, ps, st)[1]
    end
    Ψx0 = vcat(raw_data)
    acc = []
    for _ = 1 : sam.burnin
        r0, Ψx0, a = MHstep(r0, Ψx0, sam.Nel, sam, ps, st; batch_size = batch_size);
        push!(acc,a)
    end
    return r0, Ψx0, mean(mean(acc))
end

"""
type = "continue"
start from the previous sampling x0
"""
function sampler(sam::MHSampler, ps, st; batch_size = 1)
    if isempty(sam.x0)
        r0, Ψx0, = sampler_restart(sam, ps, st; batch_size = batch_size);
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
                clip = 20., batch_size = 1) # change clipping (5 -> 20) to see what happens
    x, ~, acc = sampler(sam, ps, st; batch_size = batch_size)
    raw_data = pmap(x; batch_size = batch_size) do d
        Elocal(ham, wf, d, ps, st)
    end
    Eloc = vcat(raw_data)
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
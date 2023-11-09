using StatsBase
using StaticArrays
using Optimisers
export MHSampler
using ACEpsi.AtomicOrbitals: Nuc
using Lux: Chain
using Distributed
"""
`MHSampler`
Metropolis-Hastings sampling algorithm.
"""
mutable struct MHSampler{T}
    Nel::Int64
    nuclei::Vector{Nuc{T}}
    Δt::Float64                 # step size (of Gaussian proposal)
    burnin::Int64               # burn-in iterations
    lag::Int64                  # iterations between successive samples
    N_batch::Int64              # batch size
    nchains::Int64              # Number of chains
    Ψ::Chain                    # many-body wavefunction for sampling
    x0::Vector                  # initial sampling 
end

MHSampler(Ψ, Nel, nuclei; Δt = 0.1, 
            burnin = 100, 
            lag = 10, 
            N_batch = 1, 
            nchains = 1000,
            x0 = Vector{Vector{SVector{3, Float64}}}(undef, nchains)) =
    MHSampler(Nel, nuclei, Δt, burnin, lag, N_batch, nchains, Ψ, x0)


"""
unbiased random walk: R_n+1 = R_n + Δ⋅Wn
biased random walk:   R_n+1 = R_n + Δ⋅Wn + Δ⋅∇(log Ψ)(R_n)
"""

eval(wf, X::AbstractVector, ps, st) = wf(X, ps, st)[1]

function MHstep(r0::Vector{Vector{SVector{3, TT}}}, 
                Ψx0::Vector{T}, 
                Nels::Int64, 
                sam::MHSampler, ps::NamedTuple, st::NamedTuple; batch_size = 1) where {T, TT}
    rand_sample(X::Vector{SVector{3, TX}}, Nels::Int, Δt::Float64) where {TX}= begin
        return X + Δt * randn(SVector{3, TX}, Nels)
    end
    rp = rand_sample.(r0, Ref(Nels), Ref(sam.Δt))
    raw_data = pmap(rp; batch_size = batch_size) do d
        sam.Ψ(d, ps, st)[1]
    end
    Ψxp = vcat(raw_data)
    accprob = accfcn(Ψx0, Ψxp)
    u = rand(sam.nchains)
    acc = u .<= accprob[:]
    r::Vector{Vector{SVector{3, TT}}} = acc .*  rp + (1.0 .- acc) .* r0
    Ψ = acc .*  Ψxp + (1.0 .- acc) .* Ψx0
    return r, Ψ, acc
end
 
"""
acceptance rate for log|Ψ|
ψₜ₊₁²/ψₜ² = exp((log|Ψₜ₊₁|^2-log |ψₜ|^2))
"""

function accfcn(Ψx0::Vector{T}, Ψxp::Vector{T}) where {T} 
    acc = exp.(Ψxp .- Ψx0)
    return acc
end

"""============== Metropolis sampling algorithm ============
type = "restart"
"""

function pos(sam::MHSampler)
    T = eltype(sam.nuclei[1].rr)
    M = length(sam.nuclei)
    rr = zeros(SVector{3, T}, sam.Nel)
    tt = zeros(Int, 1)
    @inbounds begin
        for i = 1:M
            @simd ivdep for j = Int(ceil(sam.nuclei[i].charge))
                tt[1] += 1
                rr[tt[1]] = sam.nuclei[i].rr
            end
        end
    end
    return rr
end

function sampler_restart(sam::MHSampler, ps, st; batch_size = 1)
    r = pos(sam)
    T = eltype(r[1])
    r0 = sam.x0
    r0 = [sam.Δt * randn(SVector{3, T}, sam.Nel) + r for _ = 1:sam.nchains]
    raw_data = pmap(r0; batch_size = batch_size) do d
        sam.Ψ(d, ps, st)[1]
    end
    Ψx0 = vcat(raw_data)
    #eval.(Ref(sam.Ψ), r0, Ref(ps), Ref(st))
    acc = zeros(T, sam.burnin)
    for i = 1 : sam.burnin
        r0, Ψx0, a = MHstep(r0, Ψx0, sam.Nel, sam, ps, st, batch_size = batch_size);
        acc[i] = mean(a)
    end
    return r0, Ψx0, mean(acc)
end

"""
type = "continue"
start from the previous sampling x0
"""
function sampler(sam::MHSampler, ps, st; batch_size = 1)
    r0 = sam.x0
    raw_data = pmap(r0; batch_size = batch_size) do d
        sam.Ψ(d, ps, st)[1]
    end
    Ψx0 = vcat(raw_data)
    T = eltype(r0[1][1])
    acc = zeros(T, sam.lag)
    for i = 1:sam.lag
        r0, Ψx0, a = MHstep(r0, Ψx0, sam.Nel, sam, ps, st, batch_size = batch_size);
        acc[i] = mean(a)
    end
    return r0, Ψx0, mean(acc)
end



"""
Rayleigh quotient by VMC using Metropolis sampling
"""
function rq_MC(Ψ, sam::MHSampler, ham::SumH, ps, st; batch_size = 1)
    r, ~, acc = sampler_restart(sam, ps, st, batch_size = batch_size);
    raw_data = pmap(r; batch_size = batch_size) do d
        Elocal(ham, Ψ, d, ps, st)
    end
    Eloc = vcat(raw_data)
    val = sum(Eloc) / length(Eloc)
    var = sqrt(sum((Eloc .-val).^2)/(length(Eloc)*(length(Eloc)-1)))
    return val, var, acc
end

function Eloc_Exp_TV_clip(wf, ps, st,
                sam::MHSampler, 
                ham::SumH;
                clip = 5., batch_size = 1)
    x, ~, acc = sampler(sam, ps, st, batch_size= batch_size)
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


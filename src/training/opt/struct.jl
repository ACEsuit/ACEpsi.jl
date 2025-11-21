using UnPack

export SR_method, MINSRSolver, SPRINGSolver, DirectSolver, SketchSolver, SVDSolver
export OPTSETTING, OPTPARAMS, OPTPARAMSKETCH, OPTPARAMSVD, OPTPARAMSPRING, OPTPARAMMINSR
export init_optim, embedding

# ----------------------------
# SR method tags
# ----------------------------
abstract type SR_method end
struct MINSRSolver   <: SR_method end
struct SPRINGSolver  <: SR_method end
struct DirectSolver  <: SR_method end

mutable struct SketchSolver <: SR_method
    sr_rank_max::Int
    sr_rank::Int
    sr_rank0::Int
    sr_scale::Float64
end
SketchSolver(sr_rank_max::Int; start::Int=min(16, sr_rank_max), scale::Real=1.1) =
    SketchSolver(sr_rank_max, min(start, sr_rank_max), min(start, sr_rank_max), Float64(scale))

mutable struct SVDSolver <: SR_method
    sr_rank_max::Int
    sr_rank::Int
    sr_rank0::Int
    sr_scale::Float64
    svd_iteration::Int
end

# ----------------------------
# Optimizer settings (parametric)
# ----------------------------
mutable struct OPTSETTING{T<:Real}
    sr_method::SR_method
    iterations::Vector{Int}
    burnin::Int
    lag::Int
    nchains::Int
    Δt::T
    acc_step::Int
    acc_range::Tuple{T,T}
    acc_opt::Vector{T}
    clip::T
    lr::T
    lr_dc::Int
    m::T
    damping::T
    damping_decay::Int
    damping_min::T
    norm_constrain::T
    η::T
    res_path::AbstractString
    checkpoints::Any
end

function OPTSETTING(sr_method::SR_method;
    iterations::Vector{<:Integer},
    burnin::Integer = 1000,
    lag::Integer = 10,
    nchains::Integer = 2048,
    Δt::Real = 0.2,
    acc_step::Integer = 10,
    acc_range::Tuple{<:Real,<:Real} = (0.45, 0.8),
    acc_opt::AbstractVector{<:Real} = fill(Float64(0), max(1, acc_step)),
    clip::Real = 5.0,
    lr::Real = 0.015,
    lr_dc::Integer = 10^4,
    m::Real = 0.99,
    damping::Real = 1e-3,
    damping_decay::Integer = 100,
    damping_min::Real = 1e-3,
    norm_constrain::Real = 0.01,
    η::Real = 0.95,
    res_path::AbstractString = "results/",
    checkpoints = nothing)

    T = promote_type(typeof(Δt), eltype(float.(acc_opt)), typeof(clip), typeof(lr),
                     typeof(m), typeof(damping), typeof(damping_min),
                     typeof(norm_constrain), typeof(η), eltype(float.(collect(acc_range))))

    iters = Int.(iterations)
    0 < burnin || error("burnin must be positive")
    0 < lag     || error("lag must be positive")
    0 < nchains || error("nchains must be positive")
    0 < acc_step || error("acc_step must be positive")
    lo, hi = T(acc_range[1]), T(acc_range[2])
    lo < hi || error("acc_range must satisfy lo < hi, got ($lo, $hi)")
    accbuf = Vector{T}(undef, max(1, acc_step))
    @inbounds for i in eachindex(accbuf)
        accbuf[i] = i <= length(acc_opt) && isfinite(acc_opt[i]) ? T(acc_opt[i]) : T(0)
    end

    return OPTSETTING{T}(sr_method, iters, Int(burnin), Int(lag), Int(nchains),
                         T(Δt), Int(acc_step), (lo, hi), accbuf,
                         T(clip), T(lr), Int(lr_dc), T(m),
                         T(damping), Int(damping_decay), T(damping_min),
                         T(norm_constrain), T(η),
                         res_path, checkpoints)
end

# ----------------------------
# Per-method optimizer state (parametric)
# ----------------------------
struct OPTPARAMSPRING <: AbstractLuxLayer
    dim_ps::Int
    nchains::Int
end

LuxCore.initialparameters(::AbstractRNG, l::OPTPARAMSPRING) = NamedTuple()

function LuxCore.initialstates(::AbstractRNG, l::OPTPARAMSPRING)
    dim_ps = l.dim_ps
    nchains = l.nchains
    s = zeros(nchains, nchains)
    f      = zeros(dim_ps)
    dw_tot = zeros(dim_ps)
    dow    = zeros(nchains)
    return (; f = f, dw_tot = dw_tot, dow = dow, s = s)
end

struct OPTPARAMS <: AbstractLuxLayer
    dim_ps::Int
    nchains::Int
end

LuxCore.initialparameters(::AbstractRNG, l::OPTPARAMS) = NamedTuple()

function LuxCore.initialstates(::AbstractRNG, l::OPTPARAMS)
    dim_ps = l.dim_ps
    nchains = l.nchains
    s_prev = zeros(dim_ps, dim_ps)
    invs   = zeros(dim_ps, dim_ps)
    return (; s_prev = s_prev, invs = invs, first = true)
end

struct OPTPARAMMINSR <: AbstractLuxLayer
    dim_ps::Int
    nchains::Int
end

LuxCore.initialparameters(::AbstractRNG, l::OPTPARAMMINSR) = NamedTuple()

function LuxCore.initialstates(::AbstractRNG, l::OPTPARAMMINSR)
    dim_ps = l.dim_ps
    nchains = l.nchains
    s = zeros(nchains, nchains)
    f      = zeros(dim_ps)
    dw_tot = zeros(dim_ps)
    dow    = zeros(nchains)
    return (; f = f, dw_tot = dw_tot, dow = dow, s = s)
end

struct OPTPARAMSKETCH <: AbstractLuxLayer
    dim_ps::Int
    nchains::Int
    sr_rank_max::Int
end

LuxCore.initialparameters(::AbstractRNG, l::OPTPARAMSKETCH) = NamedTuple()

function LuxCore.initialstates(::AbstractRNG, l::OPTPARAMSKETCH)
    sr_rank_max = l.sr_rank_max
    dim_ps = l.dim_ps
    nchains = l.nchains
    sr_o   = zeros(dim_ps, sr_rank_max)
    f      = zeros(dim_ps)
    dw_tot = zeros(dim_ps)
    uf     = zeros(sr_rank_max)
    ek     = zeros(sr_rank_max)
    return (; sr_o = sr_o, f = f, dw_tot = dw_tot, uf = uf, ek = ek, first = true)
end


struct OPTPARAMSVD <: AbstractLuxLayer
    dim_ps::Int
    nchains::Int
    sr_rank_max::Int
end

LuxCore.initialparameters(::AbstractRNG, l::OPTPARAMSVD) = NamedTuple()

function LuxCore.initialstates(::AbstractRNG, l::OPTPARAMSVD)
    sr_rank_max = l.sr_rank_max
    dim_ps = l.dim_ps
    nchains = l.nchains
    sr_o   = zeros(dim_ps, sr_rank_max)
    f      = zeros(dim_ps)
    dw_tot = zeros(dim_ps)
    uf     = zeros(sr_rank_max)
    ek     = zeros(sr_rank_max)
    u      = randn(dim_ps, sr_rank_max)
    return (; sr_o = sr_o, f = f, dw_tot = dw_tot, uf = uf, ek = ek, u = u, first = true)
end

init_optim(dim_ps::Integer, nchains::Integer, sr::DirectSolver) = OPTPARAMS(dim_ps, nchains)
init_optim(dim_ps::Integer, nchains::Integer, sr::SPRINGSolver) = OPTPARAMSPRING(dim_ps, nchains)
init_optim(dim_ps::Integer, nchains::Integer, sr::MINSRSolver) = OPTPARAMMINSR(dim_ps, nchains)
init_optim(dim_ps::Integer, nchains::Integer, sr::SketchSolver) = OPTPARAMSKETCH(dim_ps, nchains, sr.sr_rank_max)
init_optim(dim_ps::Integer, nchains::Integer, sr::SVDSolver) = OPTPARAMSVD(dim_ps, nchains, sr.sr_rank_max)

# ----------------------------
# embedding: carry state to expanded parameter space
# Direct
function embedding(damping::Real, nchains::Int, dim_ps::Int,
                   index_or_mask::AbstractVector,
                   prev, optset::DirectSolver)
    Op = init_optim(dim_ps, nchains, optset)
    rng = Random.default_rng()
    ps_opt_new, O = Lux.setup(rng, Op)
    s_prev = O.s_prev
    T = eltype(s_prev)
    @views s_prev[index_or_mask .> 0, index_or_mask .> 0] .= prev.s_prev
    newdiag = diagind(s_prev)
    oldn = size(prev.s_prev, 1)
    @inbounds for j in (oldn+1):dim_ps
        s_prev[newdiag[j]] = T(damping)
    end
    O = (; O..., s_prev = s_prev, first = false)
    return O
end

# Sketch
function embedding(::Real, nchains::Int, dim_ps::Int,
                   index_or_mask::AbstractVector,
                   prev, optset::SketchSolver)
    Op = init_optim(dim_ps, nchains, optset)
    rng = Random.default_rng()
    ps_opt_new, O = Lux.setup(rng, Op)
    sr_o = O.sr_o
    @views sr_o[index_or_mask .> 0, :] .= prev.sr_o
    O = (; O..., sr_o = sr_o, ek = prev.ek, first = false)
    return O
end

# SVD
function embedding(::Real, nchains::Int, dim_ps::Int,
                   index_or_mask::AbstractVector,
                   prev, optset::SVDSolver)
    Op = init_optim(dim_ps, nchains, optset)
    rng = Random.default_rng()
    ps_opt_new, O = Lux.setup(rng, Op)
    sr_o = O.sr_o
    u = O.u
    @views sr_o[index_or_mask .> 0, :] .= prev.sr_o
    @views u[index_or_mask .> 0, :]     .= prev.u
    O = (; O..., sr_o = sr_o, ek = prev.ek, u = u, first = false)
    return O
end

# Spring
function embedding(::Real, nchains::Int, dim_ps::Int,
                   index_or_mask::AbstractVector,
                   prev, optset::SPRINGSolver)
    Op = init_optim(dim_ps, nchains, optset)
    rng = Random.default_rng()
    ps_opt_new, O = Lux.setup(rng, Op)
    dw_tot = O.dw_tot
    @views dw_tot[index_or_mask .> 0] .= prev.dw_tot
    O = (; O..., dw_tot = dw_tot)
    return O
end

# MinSR
function embedding(::Real, nchains::Int, dim_ps::Int,
                   index_or_mask::AbstractVector,
                   prev, optset::MINSRSolver)
    Op = init_optim(dim_ps, nchains, optset)
    rng = Random.default_rng()
    ps_opt_new, O = Lux.setup(rng, Op)
    dw_tot = O.dw_tot
    @views dw_tot[index_or_mask .> 0] .= prev.dw_tot
    O = (; O..., dw_tot = dw_tot)
    return O
end


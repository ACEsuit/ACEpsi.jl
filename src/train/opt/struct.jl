using UnPack
export OPTSETTING
export MINSRSolver, SPRINGSolver, DirectSolver, SketchSolver, SVDSolver

abstract type SR_method end

mutable struct OPTSETTING
    sr_method::SR_method
    iterations::Vector{Int64}
    burnin::Int
    lag::Int
    nchains::Int
    Δt::Float64
    acc_step::Int
    acc_range::Vector{Float64}
    acc_opt
    clip::Float64
    lr::Float64
    lr_dc::Int
    m::Float64
    damping::Float64
    damping_decay::Int
    damping_min::Float64
    norm_constrain::Float64
    η::Float64
    res_path
    checkpoints
end

function OPTSETTING(sr_method::SR_method; iterations::Vector{Int}, burnin::Int, lag::Int, nchains::Int,
                    Δt::Float64, acc_step::Int, acc_range::Vector{Float64}, acc_opt, 
                    clip::Float64, lr::Float64, lr_dc::Int, m::Float64,
                    damping::Float64, damping_decay::Int, damping_min::Float64,
                    norm_constrain::Float64, η::Float64, res_path, checkpoints)
    return OPTSETTING(sr_method, iterations, burnin, lag, nchains, Δt, acc_step, acc_range, acc_opt, 
                      clip, lr, lr_dc, m, damping, damping_decay, damping_min,
                      norm_constrain, η, res_path, checkpoints)
end

struct MINSRSolver <: SR_method
end
struct SPRINGSolver <: SR_method
end

struct DirectSolver <: SR_method
end

mutable struct SketchSolver <: SR_method
    sr_rank_max::Int64
    sr_rank::Int64
    sr_rank0::Int64
    sr_scale::Float64
end

SketchSolver(sr_rank_max::Int64) = SketchSolver(sr_rank_max, min(10, sr_rank_max), min(10, sr_rank_max), 1.1)

mutable struct SVDSolver <: SR_method
    sr_rank_max::Int64
    sr_rank::Int64
    sr_rank0::Int64
    sr_scale::Float64
end

SVDSolver(sr_rank_max::Int64) = SVDSolver(sr_rank_max, min(10, sr_rank_max), min(10, sr_rank_max), 1.1)


mutable struct OPTPARAMS
    s_prev::Array{Float64, 2}
    invs::Array{Float64, 2}
    f::Array{Float64, 1}
    dw_tot::Array{Float64, 1}
end

function init_optim(dim_ps::Int64, nchains, sr::DirectSolver)
    s_prev = zeros(Float64, dim_ps, dim_ps)
    invs = zeros(Float64, dim_ps, dim_ps)
    f = zeros(Float64, dim_ps)
    dw_tot = zeros(Float64, dim_ps)
    return OPTPARAMS(s_prev, invs, f, dw_tot)
end

mutable struct OPTPARAMSKETCH
    sr_o::Array{Float64, 2}
    f::Array{Float64, 1}
    dw_tot::Array{Float64, 1}
    dk::Array{Float64, 1}
    ek::Array{Float64, 1}
    uf::Array{Float64, 1}
end

function init_optim(dim_ps::Int64, nchains, sr::SketchSolver)
    @unpack sr_rank_max = sr
    sr_o = zeros(Float64,dim_ps, sr_rank_max)
    f = zeros(Float64, dim_ps)
    dw_tot = zeros(Float64, dim_ps)
    uf = zeros(Float64, sr_rank_max)
    dk = zeros(Float64, dim_ps)
    ek = zeros(Float64, sr_rank_max)
    return OPTPARAMSKETCH(sr_o, f, dw_tot, dk, ek, uf)
end

mutable struct OPTPARAMSVD 
    sr_o::Array{Float64, 2}
    f::Array{Float64, 1}
    dw_tot::Array{Float64, 1}
    dk::Array{Float64, 1}
    ek::Array{Float64, 1}
    uf::Array{Float64, 1}
    u::Array{Float64, 2}
end

function init_optim(dim_ps::Int64, nchains, sr::SVDSolver)
    @unpack sr_rank_max = sr
    sr_o = zeros(Float64, dim_ps, sr_rank_max)
    f = zeros(Float64, dim_ps)
    dw_tot = zeros(Float64, dim_ps)
    dk = zeros(Float64, dim_ps)
    ek = zeros(Float64, sr_rank_max)
    uf = zeros(Float64, sr_rank_max)
    u = randn(Float64, dim_ps, sr_rank_max)
    return OPTPARAMSVD(sr_o, f, dw_tot, dk, ek, uf, u)
end

mutable struct OPTPARAMSPRING
    f::Array{Float64, 1}
    dw_tot::Array{Float64, 1}
    dow::Array{Float64, 1}
end

function init_optim(dim_ps::Int64, nchains, sr::SPRINGSolver)
    f = zeros(Float64, dim_ps)
    dw_tot = zeros(Float64, dim_ps) 
    dow = zeros(Float64, nchains) 
    return OPTPARAMSPRING(f, dw_tot, dow)
end

function Embedding(damping, nchains, dim_ps::Int64, index::Vector{T}, OptParams::OPTPARAMSPRING, optimizer) where {T} 
    O = init_optim(dim_ps, nchains, optimizer.sr_method)
    O.dw_tot[index .> 0] = OptParams.dw_tot
    return O
end

mutable struct OPTPARAMMINSR
    f::Array{Float64, 1}
    dw_tot::Array{Float64, 1}
    dow::Array{Float64, 1}
end

function init_optim(dim_ps::Int64, nchains, sr::MINSRSolver)
    f = zeros(Float64, dim_ps)
    dw_tot = zeros(Float64, dim_ps) 
    dow = zeros(Float64, nchains) 
    return OPTPARAMMINSR(f, dw_tot, dow)
end

function Embedding(damping, nchains, dim_ps::Int64, index::Vector{T}, OptParams::OPTPARAMMINSR, optimizer) where {T}
    O = init_optim(dim_ps, nchains, optimizer.sr_method)         
    O.dw_tot[index .> 0] .= OptParams.dw_tot
    return O
end

function Embedding(damping, nchains, dim_ps::Int64, index::Vector{T}, OptParams::OPTPARAMS, optimizer) where {T}
    O = init_optim(dim_ps, nchains, optimizer.sr_method)  
    O.s_prev[index .> 0, index .> 0] .= OptParams.s_prev
    O.s_prev[diagind(O.s_prev)[size(OptParams.s_prev, 1)+1:end]] .= damping
    return O
end

function Embedding(damping, nchains, dim_ps::Int64, index::Vector{T}, OptParams::OPTPARAMSKETCH, optimizer) where {T}
    O = init_optim(dim_ps, nchains, optimizer.sr_method)         
    O.sr_o[index .> 0, :] .= OptParams.sr_o
    O.dk[index .> 0] .= OptParams.dk
    O.ek .= OptParams.ek
    return O
end

function Embedding(damping, nchains, dim_ps::Int64, index::Vector{T}, OptParams::OPTPARAMSVD, optimizer) where {T}
    O = init_optim(dim_ps, nchains, optimizer.sr_method)         
    O.sr_o[index .> 0, :] .= OptParams.sr_o
    O.dk[index .> 0] .= OptParams.dk
    O.ek .= OptParams.ek
    O.u[index .> 0, :] .= OptParams.u
    return O
end


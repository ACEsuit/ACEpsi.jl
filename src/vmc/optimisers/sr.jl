using Optimisers
using LinearMaps, LinearAlgebra, IterativeSolvers
using Polynomials4ML: _make_reqfields, @reqfields, POOL, TMP, META, release!
using ObjectPools: acquire!

mutable struct SR <: opt
    ϵ₁::Number
    ϵ₂::Number
    β₁::Number
    β₂::Number
    _sr_type::sr_type
    st::Scalar_type
    nt::Norm_type
end

SR() = SR(0.0, 1e-4, 0.95, 0.0, QGT(), no_scale(), no_constraint())

function Optimization(type::SR, wf, ps, st, sam::MHSampler, ham::SumH, α, mₜ, vₜ, t; batch_size = 200)
    g, acc, λ₀, σ, x0, mₜ, vₜ, ϵ = grad_sr(type._sr_type, type, wf, ps, st, sam, ham, mₜ, vₜ, t, batch_size = batch_size)
    res = norm(g)
    p, s = destructure(ps)
    p = p - α * ϵ * mₜ
    return s(p), acc, λ₀, res, σ, x0, mₜ, vₜ
end

# O_kl = ∂ln ψθ(x_k)/∂θ_l : N_ps × N_sample
# Ō_k = 1/N_sample ∑_i=1^N_sample O_ki : N_ps × 1
# ΔO_ki = O_ki - Ō_k -> ΔO_ki/sqrt(N_sample)
function Jacobian_O(wf, ps, st, sam::MHSampler, ham::SumH; batch_size = 200)
    λ₀, σ, E, x0, acc = Eloc_Exp_TV_clip(wf, ps, st, sam, ham, batch_size = batch_size)
    dps = grad_params.(Ref(wf), x0, Ref(ps), Ref(st))
    _destructure(ps) = destructure(ps)[1]
    O = 1/2 * reshape(_destructure(dps), (length(_destructure(ps)), sam.nchains))
    Ō = mean(O, dims =2)
    ΔO = (O .- Ō)/sqrt(sam.nchains)
    return λ₀, σ, E, acc, ΔO, x0
end

function grad_sr(_sr_type::QGT, type::SR, wf, ps, st, sam::MHSampler, ham::SumH, mₜ, vₜ, t; batch_size = 200)
    λ₀, σ, E, acc, O, x0 = Jacobian_O(wf, ps, st, sam, ham, batch_size = batch_size)
    g0 = vec(2.0 * mean(O .* (E .- mean(E))', dims = 2))

    # S_ij = 1/N_sample ∑_k=1^N_sample ΔO_ik * ΔO_jk = ΔO * ΔO'/N_sample -> ΔO * ΔO': N_ps × N_ps
    S = O * O'/sam.nchains
    
    # Scale Regularization
    S, g0 = scale_regularization(S, g0, type.st)

    # momentum
    vₜ = momentum(vₜ, S, type.β₁)

    # damping: S_ij = S_ij + eps δ_ij
    vₜ[diagind(vₜ)] .*= (1 + type.ϵ₁)
    vₜ[diagind(vₜ)] .+= type.ϵ₂ #* max(0.001, exp(-t/2000))

    g = vₜ \ g0
    # momentum for g 
    mₜ = momentum(mₜ, g, type.β₂)
  
    # norm_constraint
    ϵ = norm_constraint(vₜ, g0, g, type.nt)
    return g, acc, λ₀, σ, x0, mₜ, vₜ, ϵ
end

norm_constraint(vₜ::AbstractMatrix, g::AbstractVector, g0::AbstractVector, nt::no_constraint) = 1.0
norm_constraint(vₜ::AbstractMatrix, g::AbstractVector, g0::AbstractVector, nt::norm_constraint) = min(1.0, sqrt(nt.c/ (g' * g0)))

momentum(vₜ::AbstractMatrix, S::AbstractMatrix, b::Number) = b * vₜ + (1-b) * S
momentum(m::AbstractVector, g::AbstractVector, b::Number) = b * m + (1-b) * g

function scale_regularization(vₜ::AbstractMatrix, g0::AbstractVector, st::scale_invariant)
    # S_ij = S_ij/sqrt(S_ii ⋅ S_jj)
    # g_i = g_i/sqrt(S_ii)
    diag_vₜ = sqrt.(diag(vₜ))
    vₜ = vₜ ./ diag_vₜ ./ diag_vₜ'
    g0 = g0 ./ diag_vₜ
    return vₜ, g0
end

scale_regularization(vₜ::AbstractMatrix, g0::AbstractVector, st::no_scale) = vₜ, g0

function initp(_opt::SR, ps::NamedTuple)
    _l = length(destructure(ps)[1])
    vₜ = 1.0 * Matrix(I(_l))
    mₜ = zeros(_l)
    return mₜ, vₜ
end

updatep(_opt::SR, _utype::_initial, ps, index, mₜ, vₜ) = initp(_opt, ps)

function updatep(_opt::SR, _utype::_continue, ps, index, mₜ, vₜ)   
    nmₜ, nvₜ = initp(_opt, ps)
    nmₜ[index .> 0] .= mₜ
    nvₜ[index .> 0, index .> 0] .= vₜ
    nvₜ[diagind(nvₜ)] .= vₜ[1,1]
    return nmₜ, nvₜ
end
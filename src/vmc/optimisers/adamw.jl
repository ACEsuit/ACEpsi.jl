using Optimisers

mutable struct adamW <: opt
    α::Number
    β₁::Number
    β₂::Number
    ϵ::Number
    θ₁::Number
    θ₂::Number
end

adamW() = adamW(0.1, 0.9, 0.9, eps(), 0.0, 0.025)

"""
All operations on vectors are element-wise.
t ← t+1
∇fₜ(θₜ₋₁) ← SelectBatch(θₜ₋₁)
gₜ ← ∇fₜ(θₜ₋₁) + λθₜ₋₁
mₜ ← β₁mₜ₋₁ + (1-β₁)gₜ
vₜ ← β₂vₜ₋₁ + (1-β₂)gₜ⊙gₜ 
m̂ₜ ← mₜ/(1-β₁ᵗ)
v̂ₜ ← vₜ/(1-β₂ᵗ)
ηₜ ← SetScheduleMultiplier(t) can be fixed, decay, or also be used for warm restarts
θₜ = θₜ₋₁ - ηₜ(αm̂ₜ/(√v̂ₜ+ϵ)+λθₜ₋₁)
"""
function Optimization(type::adamW, wf, ps, st, sam::MHSampler, ham::SumH, α, mₜ, vₜ, t)
    λ₀, σ, E, x0, acc = Eloc_Exp_TV_clip(wf, ps, st, sam, ham)
    g = grad(wf, x0, ps, st, E)

    p, s = destructure(ps)
    g, = destructure(g)
    res = norm(g)
    gₜ = g + type.θ₁ * p
    mₜ = type.β₁ * mₜ + (1-type.β₁) * gₜ
    vₜ = type.β₂ * vₜ + (1-type.β₂) * gₜ .* gₜ
    m̂ₜ = mₜ * (1/(1-type.β₁^t))
    v̂ₜ = vₜ * (1/(1-type.β₂^t))
    _p = p - α * (type.α * m̂ₜ ./(sqrt.(v̂ₜ) .+ Ref(type.ϵ)) .+ type.θ₂ * p)
    ps = s(_p)
    return ps, acc, λ₀, res, σ, x0, mₜ, vₜ
end

function initp(_opt::adamW, ps::NamedTuple)
    p, = destructure(ps)
    _l = length(p)
    vₜ = zeros(_l)
    mₜ = zeros(_l)
    return mₜ, vₜ
end

function updatep(_opt::adamW, _utype::_initial, ps, index, mₜ, vₜ)   
    nmₜ, nvₜ = initp(_opt, ps)
    return nmₜ, nvₜ
end

function updatep(_opt::adamW, _utype::_continue, ps, index, mₜ, vₜ)   
    nmₜ, nvₜ = initp(_opt, ps)
    nmₜ[index .> 0] .= mₜ
    nvₜ[index .> 0] .= vₜ
    return nmₜ, nvₜ
end


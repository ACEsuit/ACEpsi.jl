using Optimisers


mutable struct AdamW <: opt
    β::Tuple
    γ::Number
    ϵ::Number
end

AdamW() = AdamW((9f-1, 9.99f-1), 0.0, eps())

function Optimization(type::AdamW, wf, ps, st, sam::MHSampler, ham::SumH, α)
    λ₀, σ, E, x0, acc = Eloc_Exp_TV_clip(wf, ps, st, sam, ham)
    g = grad(wf, x0, ps, st, E)
    st_opt = Optimisers.setup(Optimisers.AdamW(α, type.β, type.γ, type.ϵ), ps)
    st_opt, ps = Optimisers.update(st_opt, ps, g)
    res = norm(destructure(g)[1])
    return ps, acc, λ₀, res, σ
end
using Optimisers


mutable struct SR <: opt
    ϵ1::Number
    ϵ2::Number
end

SR() = SR(0., 0.01)


_destructure(ps) = destructure(ps)[1]

function Optimization(type::SR, wf, ps, st, sam::MHSampler, ham::SumH, α)
    ϵ1 = type.ϵ1
    ϵ2 = type.ϵ2

    λ₀, σ, E, x0, acc = Eloc_Exp_TV_clip(wf, ps, st, sam, ham)
    dy_ps = grad_params.(Ref(wf), x0, Ref(ps), Ref(st))
    dy = _destructure.(dy_ps)
    O = 1/2 * dy 
    Ō = mean(O)
    ΔO = [O[i] - Ō for i = 1:length(O)]

    g0 = 2.0 * sum(ΔO .*E)/ length(x0)

    S = sum([ΔO[i] * ΔO[i]' for i = 1:length(O)])/length(x0)
    S = S + ϵ1 * Diagonal(diag(S)) + ϵ2 * Diagonal(diag(one(S)))
    
    g = S \ g0
    res = norm(g)

    p, s = destructure(ps)
    p = p - α * g
    ps = s(p)
    
    return ps, acc, λ₀, res, σ
end



    
    
    
    
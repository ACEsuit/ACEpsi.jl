using Optimisers
using Statistics
# stochastic reconfiguration

mutable struct SR <: opt
    ϵ1::Number
    ϵ2::Number
end

SR() = SR(0., 0.01)


_destructure(ps) = destructure(ps)[1]

function Optimization(type::SR, wf, ps, st, sam::MHSampler, ham::SumH, α)
    ϵ1 = type.ϵ1
    ϵ2 = type.ϵ2

    # E: N × 1
    λ₀, σ, E, x0, acc = Eloc_Exp_TV_clip(wf, ps, st, sam, ham)
    dy_ps = grad_params.(Ref(wf), x0, Ref(ps), Ref(st))
    # O: M × N
    O = 1/2 * reshape(_destructure(dy_ps), (length(_destructure(ps)),sam.nchains))
    # ⟨O⟩: M x 1
    Ō = mean(O,dims =2)
    ΔO = O .- Ō

    # g0: M x 1
    g0 = 2.0 * ΔO * E/sam.nchains

    # S: M x M
    S = cov(O',dims=1)
    S = S + ϵ1 * Diagonal(diag(S)) + ϵ2 * Diagonal(diag(one(S)))
    
    g = S \ g0
    res = norm(g)

    p, s = destructure(ps)
    p = p - α * g
    ps = s(p)
    
    return ps, acc, λ₀, res, σ
end
   


    
    
    
    
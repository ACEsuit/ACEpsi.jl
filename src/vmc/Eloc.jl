export SumH

# INTERFACE FOR HAMILTIANS   H ψ  ->  H(psi, X)
struct SumH{TK, TT, TE}
    K::TK
    Vext::TT
    Vee::TE
end

(H::SumH)(wf, X::AbstractVector, ps, st) =
        H.K(wf, X, ps, st) + (H.Vext(wf, X, ps, st) + H.Vee(wf, X, ps, st)) * evaluate(wf, X, ps, st)
   

# evaluate local energy with SumH

"""
E_loc = E_pot - 1/4 ∇²ᵣ ϕ(r) - 1/8 (∇ᵣ ϕ)²(r)
https://arxiv.org/abs/2105.08351
"""

function Elocal(H::SumH, wf, X::AbstractVector, ps, st)
    gra = gradient(wf, X, ps, st)
    val = H.Vext(wf, X, ps, st) + H.Vee(wf, X, ps, st) - 1/4 * laplacian(wf, X, ps, st) - 1/8 * gra' * gra
    return val
end
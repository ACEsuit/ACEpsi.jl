export SumH
using ACEpsi.AtomicOrbitals: Nuc
using StaticArrays

# INTERFACE FOR HAMILTIANS   H ψ  ->  H(psi, X)
struct SumH{T, NNuc}
    nuclei::SVector{NNuc, Nuc{T}}
end

function Vee(wf, X::Vector{SVector{3, T}}, ps, st) where {T}
    nX = length(X)
    v = zero(T)
    r = zero(T)
    @inbounds begin
        for i = 1:nX-1
            @simd ivdep for j = i+1:nX
                r = norm(X[i]-X[j])
                v = muladd(1, 1/r, v)
            end
        end
    end
    return v
end
 
function Vext(wf, X::Vector{SVector{3, T}}, nuclei::SVector{NNuc, Nuc{TT}}, ps, st) where {NNuc, T, TT}
    nX = length(X)
    v = zero(T)
    r = zero(T)
    @inbounds begin
        for i = 1:n
            @simd ivdep for j = 1:nX 
                r = norm(nuclei[i].rr - X[j])
                v = muladd(nuclei[i].charge, 1/r, v)
            end
        end
    end
    return -v
end
 
K(wf, X::Vector{SVector{3, T}}, ps, st) where {T} = -0.5 * laplacian(wf, X, ps, st)

(H::SumH)(wf, X::Vector{SVector{3, T}}, ps, st) where {T} =
        K(wf, X, ps, st) + (Vext(wf, X, H.nuclei, ps, st) + Vee(wf, X, ps, st)) * evaluate(wf, X, ps, st)
   

# evaluate local energy with SumH

"""
E_loc = E_pot - 1/4 ∇²ᵣ ϕ(r) - 1/8 (∇ᵣ ϕ)²(r)
https://arxiv.org/abs/2105.08351
"""

function Elocal(H::SumH, wf, X::AbstractVector, ps, st)
    gra = gradient(wf, X, ps, st)
    val = Vext(wf, X, H.nuclei, ps, st) + Vee(wf, X, ps, st) - 1/4 * laplacian(wf, X, ps, st) - 1/8 * gra' * gra
    return val
end


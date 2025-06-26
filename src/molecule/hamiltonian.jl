using LinearAlgebra: dot
export SumH

struct SumH{T, TT, Nnuc}
    nuclei::SVector{Nnuc, Nuc{T, TT}}  # Array of nuclei
end

# Electron–electron repulsion: ∑_{i<j} 1/|rᵢ - rⱼ|
function Vee(wf, X::Vector{SVector{3, T}}, ps, st) where {T}
    nX = length(X)
    v = zero(T)
    r = zero(T)
    @inbounds begin
        for i = 1:nX-1
            @simd ivdep for j = i+1:nX
                r = norm(X[i] - X[j])
                v = muladd(1, 1/r, v)
            end
        end
    end
    return v
end

# Electron–nucleus attraction: -∑_{i,j} Z_i / |R_i - r_j|
function Vext(wf, X::Vector{SVector{3, T}}, nuclei::SVector{NNuc, Nuc{TT, TN}}, ps, st) where {NNuc, T, TT, TN}
    nX = length(X)
    v = zero(T)
    r = zero(T)
    @inbounds begin
        for i = 1:NNuc
            @simd ivdep for j = 1:nX
                r = norm(nuclei[i].rr - X[j])
                v = muladd(nuclei[i].charge, 1/r, v)
            end
        end
    end
    return -v
end

# Nucleus–nucleus repulsion: ∑_{i<j} Z_i Z_j / |Rᵢ - Rⱼ|
function Vnn(mol::Molecule{Nnuc, T, TT, TN, TS}) where {Nnuc, T, TT, TN, TS}
    nuclei = mol.nuclei
    v = zero(T)
    for i = 1:Nnuc-1
        for j = i+1:Nnuc
            v += nuclei[i].charge * nuclei[j].charge / norm(nuclei[i].rr - nuclei[j].rr)
        end
    end
    return v
end

# Kinetic energy: -½ ∇² log|ψ|
K(wf, X::Vector{SVector{3, T}}, ps, st) where {T} = -0.5 * laplacian(wf, X, ps, st)

"""
Compute local energy using SumH Hamiltonian:
    E_loc = Vext + Vee - (1/4) ∇² log|ψ| - (1/8) |∇ log|ψ||²
Ref: https://arxiv.org/abs/2105.08351
"""
function Elocal(H::SumH, wf, X::Vector{SVector{3, T}}, ps, st) where {T}
    gra = gradx(wf, X, ps, st)
    val = Vext(wf, X, H.nuclei, ps, st) +
          Vee(wf, X, ps, st) -
          1/4 * laplacian(wf, X, ps, st) -
          1/8 * dot(gra, gra)
    return val
end

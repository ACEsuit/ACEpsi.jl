using Optimisers
using LinearMaps
using LinearAlgebra
using IterativeSolvers
using Polynomials4ML: _make_reqfields, @reqfields, POOL, TMP, META, release!
using ObjectPools: acquire!
# stochastic reconfiguration

mutable struct SR <: opt
    ϵ₁::Number
    ϵ₂::Number
    β₁::Number
    β₂::Number
    _sr_type::sr_type
    st::Scalar_type
    nt::Norm_type
end

#SR() = SR(0., 0.01, 0.0, 0.0, QGT(), no_scale(), no_constraint())
SR() = SR(0.0, 0.01, 0.95, 0.0, QGT(), no_scale(), norm_constraint(1.0))

_destructure(ps) = destructure(ps)[1]

function Optimization(type::SR, wf, ps, st, sam::MHSampler, ham::SumH, α, mₜ, vₜ, t; batch_size = 200)
    g, acc, λ₀, σ, x0, mₜ, vₜ, ϵ = grad_sr(type._sr_type, type, wf, ps, st, sam, ham, mₜ, vₜ, t, batch_size = batch_size)
    res = norm(g)

    p, s = destructure(ps)
    p = p - α * ϵ * mₜ
    ps = s(p)
    return ps, acc, λ₀, res, σ, x0, mₜ, vₜ
end

# O_kl = ∂ln ψθ(x_k)/∂θ_l : N_ps × N_sample
# Ō_k = 1/N_sample ∑_i=1^N_sample O_ki : N_ps × 1
# ΔO_ki = O_ki - Ō_k -> ΔO_ki/sqrt(N_sample)
function Jacobian_O(wf, ps, st, sam::MHSampler, ham::SumH; batch_size = 200)
    λ₀, σ, E, x0, acc = Eloc_Exp_TV_clip(wf, ps, st, sam, ham, batch_size = batch_size)
    dps = grad_params.(Ref(wf), x0, Ref(ps), Ref(st))
    O = 1/2 * reshape(_destructure(dps), (length(_destructure(ps)),sam.nchains))
    Ō = mean(O, dims =2)
    ΔO = (O .- Ō)/sqrt(sam.nchains)
    return λ₀, σ, E, acc, ΔO, x0
end

function grad_sr(_sr_type::QGT, type::SR, wf, ps, st, sam::MHSampler, ham::SumH, mₜ, vₜ, t; batch_size = 200)
    λ₀, σ, E, acc, ΔO, x0 = Jacobian_O(wf, ps, st, sam, ham, batch_size = batch_size)
    g0 = 2.0 * ΔO * E/sqrt(sam.nchains)

    # S_ij = 1/N_sample ∑_k=1^N_sample ΔO_ik * ΔO_jk = ΔO * ΔO'/N_sample -> ΔO * ΔO': N_ps × N_ps
    S = ΔO * ΔO'
    # momentum
    vₜ = momentum(vₜ, S, type.β₁)
    # Scale Regularization
    vₜ, g0 = scale_regularization(vₜ, g0, type.st)
    # damping: S_ij = S_ij + eps δ_ij
    vₜ[diagind(vₜ)] .*= (1+type.ϵ₁)
    vₜ[diagind(vₜ)] .+= type.ϵ₂
    #vₜ = vₜ + type.ϵ₁ * max(0.1, 100*0.9^t) * Diagonal(diag(vₜ)) + type.ϵ₂ * max(0.1, 100*0.9^t) * Diagonal(diag(one(S)))

    g = vₜ \ g0
    # momentum for g 
    mₜ = momentum(mₜ, g, type.β₂)
  
    # norm_constraint
    ϵ = norm_constraint(vₜ, g0, g, type.nt)
    return g, acc, λ₀, σ, x0, mₜ, vₜ, ϵ
end

function norm_constraint(vₜ::AbstractMatrix, g::AbstractVector, g0::AbstractVector, nt::no_constraint)
  return 1.0
end

function norm_constraint(vₜ::AbstractMatrix, g::AbstractVector, g0::AbstractVector, nt::norm_constraint)
  a = sqrt(nt.c/ (g' * g0))
  ϵ = min(1.0, a)
  return ϵ
end

function momentum(m::AbstractVector, g::AbstractVector, b::Number)
  return b * m + (1-b) * g
end

function momentum(vₜ::AbstractMatrix, S::AbstractMatrix, b::Number)
  return b * vₜ + (1-b) * S
end

function scale_regularization(vₜ::AbstractMatrix, g0::AbstractVector, st::scale_invariant)
    # S_ij = S_ij/sqrt(S_ii ⋅ S_jj)
    diag_vₜ = sqrt.(diag(vₜ))
    vₜ = vₜ ./ diag_vₜ ./ diag_vₜ'
  
    # g_i = g_i/sqrt(S_ii)
    g0 = g0 ./ diag_vₜ
    return vₜ, g0
end
  
function scale_regularization(vₜ::AbstractMatrix, g0::AbstractVector, st::no_scale)
    return vₜ, g0
end
  
function initp(_opt::SR, ps::NamedTuple)
    p, = destructure(ps)
    _l = length(p)
    vₜ = 1.0 * Matrix(I(_l))
    mₜ = zeros(_l)
    return mₜ, vₜ
end

function updatep(_opt::SR, _utype::_initial, ps, index, mₜ, vₜ)   
    nmₜ, nvₜ = initp(_opt, ps)
    return nmₜ, nvₜ
end

function updatep(_opt::SR, _utype::_continue, ps, index, mₜ, vₜ)   
    nmₜ, nvₜ = initp(_opt, ps)
    nmₜ[index .> 0] .= mₜ
    nvₜ[index .> 0, index .> 0] .= vₜ
    nvₜ[diagind(nvₜ)] .= vₜ[1,1]
    return nmₜ, nvₜ
end

"""
function grad_sr(_sr_type::QGTJacobian, wf, ps, st, sam::MHSampler, ham::SumH, ϵ1::Number, ϵ2::Number; batch_size = 200)
    λ₀, σ, E, acc, ΔO, x0 = Jacobian_O(wf, ps, st, sam, ham, batch_size = batch_size)
    g0 = 2.0 * ΔO * E/sqrt(sam.nchains)

    # S_ij = 1/N_sample ∑_k=1^N_sample ΔO_ik * ΔO_jk = ΔO * ΔO'/N_sample -> ΔO * ΔO': N_ps × N_ps
    # Sx = g0
    function Svp!(w, v)
        Δw = v' * ΔO
        for i = 1:length(v)
            w[i] = ϵ2 * v[i]
        end
        @inbounds begin 
            for i = 1:length(v)
                @simd ivdep for j = 1:length(Δw)
                    w[i] += ΔO[i,j] * Δw[j] + ϵ1 * ΔO[i,j]^2 * v[i] 
                end
            end
        end
        return w
    end
    LM_S = LinearMap(Svp!, size(ΔO)[1]; issymmetric=true, ismutating=true)
    g = gmres(LM_S, g0)
    return g, acc, λ₀, σ, x0
end

function grad_sr(_sr_type::QGTOnTheFly, wf, ps, st, sam::MHSampler, ham::SumH, ϵ1::Number, ϵ2::Number; batch_size = 200)
    λ₀, σ, E, x0, acc = Eloc_Exp_TV_clip(wf, ps, st, sam, ham, batch_size = batch_size)

    # w = O * v 
    function jvp(v::AbstractVector, wf, ps::NamedTuple, x0)
        _destructp, = destructure(ps)
        w = zero(_destructp)
        for i = 1:length(x0)
            _, back = Zygote.pullback(p -> wf(x0[i], p, st)[1], ps)
            w += 1/2 * destructure(back(v[i]))[1]
        end
        return w 
    end

    # w = v' * O
    function vjp(v::AbstractVector, wf, ps::NamedTuple, x0)
        _destructp, s = destructure(ps)
        w = zeros(length(x0))
        for i = 1:length(x0)
            f(t) = begin 
                p = s(_destructp + t * v)
                return wf(x0[i], p, st)[1]
            end
            w[i] = 1/2 * Zygote.gradient(f, 0.0)[1]
        end
        return w
    end
    
    g0 = 2 * jvp(E .- mean(E), wf, ps, x0)/sam.nchains # 2 * O * E/sam.nchains

    function Svp!(w, v)
        w̃ = 1/sam.nchains * vjp(v, wf, ps, x0)
        Δw = w̃ .- mean(w̃)
        ṽ = jvp(Δw, wf, ps, x0)
        for i = 1:length(v)
            w[i] = ṽ[i] + ϵ2 * v[i]
        end
        return w
    end
    LM_S = LinearMap(Svp!, length(g0); issymmetric=true, ismutating=true)
    g = gmres(LM_S, g0)
    return g, acc, λ₀, σ
end
"""
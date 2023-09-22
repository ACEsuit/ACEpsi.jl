using Optimisers
using LinearMaps
using LinearAlgebra
using IterativeSolvers
using Polynomials4ML: _make_reqfields, @reqfields, POOL, TMP, META, release!
using ObjectPools: acquire!
# stochastic reconfiguration

mutable struct SR <: opt
    ϵ1::Number
    ϵ2::Number
    _sr_type::sr_type
end

SR() = SR(0., 0.01, QGT())

SR(ϵ1::Number, ϵ2::Number) = SR(ϵ1, ϵ2, QGT())

_destructure(ps) = destructure(ps)[1]

function Optimization(type::SR, wf, ps, st, sam::MHSampler, ham::SumH, α)
    ϵ1 = type.ϵ1
    ϵ2 = type.ϵ2

    g, acc, λ₀, σ = grad_sr(type._sr_type, wf, ps, st, sam, ham, ϵ1, ϵ2)
    res = norm(g)

    p, s = destructure(ps)
    p = p - α * g
    ps = s(p)
    return ps, acc, λ₀, res, σ, x0
end
   

# O_kl = ∂ln ψθ(x_k)/∂θ_l : N_ps × N_sample
# Ō_k = 1/N_sample ∑_i=1^N_sample O_ki : N_ps × 1
# ΔO_ki = O_ki - Ō_k -> ΔO_ki/sqrt(N_sample)
function Jacobian_O(wf, ps, st, sam::MHSampler, ham::SumH)
    λ₀, σ, E, x0, acc = Eloc_Exp_TV_clip(wf, ps, st, sam, ham)
    dps = grad_params.(Ref(wf), x0, Ref(ps), Ref(st))
    O = 1/2 * reshape(_destructure(dps), (length(_destructure(ps)),sam.nchains))
    Ō = mean(O, dims =2)
    ΔO = (O .- Ō)/sqrt(sam.nchains)
    return λ₀, σ, E, acc, ΔO
end

function grad_sr(_sr_type::QGT, wf, ps, st, sam::MHSampler, ham::SumH, ϵ1::Number, ϵ2::Number)
    λ₀, σ, E, acc, ΔO = Jacobian_O(wf, ps, st, sam, ham)
    g0 = 2.0 * ΔO * E/sqrt(sam.nchains)

    # S_ij = 1/N_sample ∑_k=1^N_sample ΔO_ik * ΔO_jk = ΔO * ΔO'/N_sample -> ΔO * ΔO': N_ps × N_ps
    # Sx = g0
    S = ΔO * ΔO'
    S[diagind(S)] .*= (1+ϵ1)
    S[diagind(S)] .+= ϵ2
    g = S \ g0
    return g, acc, λ₀, σ
end

function grad_sr(_sr_type::QGTJacobian, wf, ps, st, sam::MHSampler, ham::SumH, ϵ1::Number, ϵ2::Number)
    λ₀, σ, E, acc, ΔO = Jacobian_O(wf, ps, st, sam, ham)
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
    return g, acc, λ₀, σ
end

function grad_sr(_sr_type::QGTOnTheFly, wf, ps, st, sam::MHSampler, ham::SumH, ϵ1::Number, ϵ2::Number)
    λ₀, σ, E, x0, acc = Eloc_Exp_TV_clip(wf, ps, st, sam, ham)

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
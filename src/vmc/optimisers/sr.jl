using Optimisers
using LinearAlgebra
using Polynomials4ML: _make_reqfields, @reqfields, POOL, TMP, META, release!
using ObjectPools: acquire!
using Optimisers

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

function Optimization(type::SR, wf, ps, st, sam::MHSampler, ham::SumH, α, mₜ, vₜ, t)
    g, acc, λ₀, σ, mₜ, vₜ, ϵ = grad_sr(type._sr_type, type, wf, ps, st, sam, ham, mₜ, vₜ, t)
    res = norm(g)
    p, s = destructure(ps)
    p = p - α * ϵ * mₜ
    return s(p), acc, λ₀, res, σ, mₜ, vₜ
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

# O_kl = ∂ln ψθ(x_k)/∂θ_l : N_ps × N_sample
# Ō_k = 1/N_sample ∑_i=1^N_sample O_ki : N_ps × 1
# ΔO_ki = O_ki - Ō_k -> ΔO_ki/sqrt(N_sample)
function Jacobian_O(wf, ps, st, sam::MHSampler, ham::SumH)
    λ₀, σ, E, acc, raw_dps = Eloc_Exp_TV_clip(wf, ps, st, sam, ham)
    dps = vcat(raw_dps...)
    O = 1/2 * reshape(dps, (length(destructure(ps)[1]), sam.nchains * nprocs()))
    Ō = mean(O, dims = 2)
    ΔO = (O .- Ō) / sqrt(sam.nchains * nprocs())
    return λ₀, σ, E, acc, ΔO
end

function grad_sr(_sr_type::QGT, type::SR, wf, ps, st, sam::MHSampler, ham::SumH, mₜ, vₜ, t)
    λ₀, σ, E, acc, ΔO = Jacobian_O(wf, ps, st, sam, ham)
    g0 = 2.0 * ΔO * E / sqrt(sam.nchains * nprocs())
    # S_ij = 1/N_sample ∑_k=1^N_sample ΔO_ik * ΔO_jk = ΔO * ΔO'/N_sample -> ΔO * ΔO': N_ps × N_ps
    S = ΔO * ΔO'
    
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
    return g, acc, λ₀, σ, mₜ, vₜ, ϵ
end


# === another implementation ===
function grad_sr_all(_sr_type::QGT, type::SR, wf, ps, st, sam::MHSampler, ham::SumH, mₜ, vₜ, t; clip = 5.)
    #λ₀, σ, E, acc, raw_dps = Eloc_Exp_TV_clip(wf, ps, st, sam, ham, batch_size = batch_size)

    # === sampler and Eloc ===
    @everywhere ps = $ps
    @everywhere begin
        global r0, Ψx0, acc
        r0 = sam.x0
        Ψx0 = eval_Ψ.(Ref(sam.Ψ), r0, Ref(ps), Ref(st))
        acc = 0.0
        for _ = 1:sam.lag
            global r0, Ψx0, acc
            local a
            r0, Ψx0, a = MHstep(r0, Ψx0, sam.Nel, sam, ps, st)
            acc += mean(a)
        end
        acc = acc / sam.lag
        Eloc = Elocal.(Ref(ham), Ref(sam.Ψ), r0, Ref(ps), Ref(st))
        # we set sam.x0 here so that we don't have to pass back the huge array
        # back to the call in mulitlevel VMC
        sam.x0 = r0
        dp = grad_params.(Ref(sam.Ψ), r0, Ref(ps), Ref(st))
        dp_vec = vcat(dp...)
        
    end

    Eloc_all = vcat([@getfrom k Eloc for k in procs()]...)
    acc_all = [@getfrom k acc for k in procs()]

    ##

    # === Energy cliping ===
    λ₀ = sum(Eloc_all) / length(Eloc_all)
    σ = sqrt(sum((Eloc_all .- λ₀).^2) / (length(Eloc_all)*(length(Eloc_all) -1)))
    ΔE = Eloc_all .- median(Eloc_all)
    a = clip * mean( abs.(ΔE) )
    ind = findall(_x -> abs(_x) > a, ΔE)
    ΔE[ind] = (a * sign.(ΔE) .* (1 .+ log.((1 .+(abs.(ΔE)/a).^2)/2)))[ind]
    E_clip = median(Eloc_all) .+ ΔE

    ##
    
    # === Jacobian_O ===
    # O_kl = ∂ln ψθ(x_k)/∂θ_l : N_ps × N_sample
    # Ō_k = 1/N_sample ∑_i=1^N_sample O_ki : N_ps × 1
    # ΔO_ki = O_ki - Ō_k -> ΔO_ki/sqrt(N_sample)
    @everywhere begin
        O = 1/2 * reshape(dp_vec, (length(destructure(ps)[1]), sam.nchains))
        Ō = mean(O, dims = 2)
    end

    # aseemble Ō here
    # passing Nprams × nprocs
    # procs() * nprocs
    Ō_all = mean([@getfrom k Ō for k in procs()])


    # === SR ===
    # and then send back to get ΔO, assemble g0 and S
    # mathematically, what we are doing is:
    # g0 = 2.0 * ΔO * E = 2.0 * ΔO * Σ_k (I ⊗ ... ⊗ Ok ⊗ ... ⊗ I)E, k ∈ procs()
    # but here we only grab the submatrix and do a BLAS2 operation on the k-th 
    # tensor product space in parallel
    @everywhere begin
        Ō_all = $Ō_all
        E_clip_trunk = $E_clip[(myid() - 1) * sam.nchains + 1 : myid() * sam.nchains]
        ΔO = (O .- Ō) / sqrt(sam.nchains * nprocs())
        S = ΔO * ΔO'
        g0 = 2.0 * ΔO * E_clip_trunk / sqrt(sam.nchains * nprocs())
    end

    # size of arrays that we are passing
    # Nprocs × Nparams^2 + Nprocs × Nprams
    # here we do sum since the normalization is take care w.r.t. nprocs for 
    # ΔO and g0
    S_all = sum([@getfrom k S for k in procs()])
    g0_all = sum([@getfrom k g0 for k in procs()])

    # Scale Regularization
    S_all, g0_all = scale_regularization(S_all, g0_all, type.st)

    # momentum
    vₜ = momentum(vₜ, S_all, type.β₁)

    # damping: S_ij = S_ij + eps δ_ij
    vₜ[diagind(vₜ)] .*= (1 + type.ϵ₁)
    vₜ[diagind(vₜ)] .+= type.ϵ₂ #* max(0.001, exp(-t/2000))

    g = vₜ \ g0_all
    
    # momentum for g 
    mₜ = momentum(mₜ, g, type.β₂)
  
    # norm_constraint
    ϵ = norm_constraint(vₜ, g0_all, g, type.nt)

    return g, mean(acc_all), λ₀, σ, mₜ, vₜ, ϵ
end
using Statistics: mean
export init_walkers, compute_Eloc_dp

function init_walkers(mol, model, ps, st, burnin, nchains, Δt)         
    x0 = initialize_around_nuclei(mol.nuclei, mol.Nel, nchains)
    x, θ, acc = burnin!(x0, model, ps, st, burnin, nchains; Δt = Δt)
    return x, θ, acc
end

function initialize_around_nuclei(nuclei::SVector{Nnuc, Nuc{T, TT}}, Nel::Int64, nchains::Int64) where {Nnuc, T <: Real, TT}
    r0 = Vector{SVector{3, T}}(undef, Nel)
    inuc = Vector{Vector{Int64}}(undef, Nnuc)
    @inbounds @simd for i = 1:Nnuc
        inuc[i] = i * ones(Int, Int(nuclei[i].charge))
    end
    inuc = vcat(inuc...)
    @inbounds for (i_el, i_nuc) in enumerate(inuc)
        r0[i_el] = nuclei[i_nuc].rr
    end

    r = map(_ -> randn(SVector{3, T}, Nel) + r0, 1:nchains)
    return r
end

function burnin!(_x, wf, ps, st, burnin::Int64, nchains::Int64; Δt = 0.08)
    _theta = evalx.(Ref(wf), _x, Ref(ps), Ref(st))
    _acc = zeros(burnin)
    _x, _theta, _acc = distributed_sampling!(wf, ps, st, _x, _theta, _acc, Δt, burnin)
    return _x, _theta, _acc 
end

function distributed_sampling!(wf, ps, st, _x::Vector{Vector{SVector{3, TX}}}, _theta::Vector{TT}, _acc::Vector{TT}, Δt::TN, T::Int64) where {TT, TN <: Float64, TX}
    Nel, nchains = length(_x[1]), length(_x)
    for i = 1:T
        _x, _theta, _acc = distributed_mhsteps!(wf, ps, st, _x, _theta, _acc, Nel, nchains, Δt, i)
    end
    return _x, _theta, _acc
end

function distributed_mhsteps!(wf, ps, st, _x::Vector{Vector{SVector{3, TX}}}, _theta::Vector{TT}, _acc::Vector{TT}, Nel::Int64, nchains::Int64, Δt::TN, i::Int64) where {TT, TN <: Float64, TX}
    xx = map(i -> _x[i] + Δt * randn(SVector{3, TX}, Nel), 1:nchains)
    theta_upd = evalx.(Ref(wf), xx, Ref(ps), Ref(st))
    logpsi_frac = theta_upd - _theta
    A = @fastmath exp.(logpsi_frac)
    acc = (rand(nchains) .<= A)
    _x[acc] .= xx[acc]
    _theta[acc] .= theta_upd[acc]
    _acc[i] = mean(acc)
    return _x, _theta, _acc
end

function compute_Eloc_dp(ham::SumH, wf, ps, st, _x, _theta::Vector{TT}, _acc::Vector{TA}, Δt::Float64, lag::Int64) where {TT, TA}
    _x, _theta, _acc = distributed_sampling!(wf, ps, st, _x, _theta, _acc, Δt, lag)
    dx = gradx.(Ref(wf), _x, Ref(ps), Ref(st)) 
    dp = gradp.(Ref(wf), _x, Ref(ps), Ref(st)) 
    _elocs = Vext.(Ref(wf), _x, Ref(ham.nuclei), Ref(ps), Ref(st)) + 
         Vee.(Ref(wf), _x, Ref(ps), Ref(st)) - 
         1/4 * laplacian.(Ref(wf), _x, Ref(ps), Ref(st)) - 
         1/8 * dot.(dx, dx)
    _o_tot = 1/2 * reshape(vcat(dp...), (length(dp[1]), length(dp)))
    return _x, _theta, _acc, _elocs, _o_tot
end
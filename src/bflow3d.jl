
using Polynomials4ML, Random 
using Polynomials4ML: OrthPolyBasis1D3T, LinearLayer, PooledSparseProduct, SparseSymmProdDAG, SparseSymmProd, release!
using Polynomials4ML.Utils: gensparse
using LinearAlgebra: qr, I, logabsdet, pinv, mul!, dot , tr, det
import ForwardDiff
using ACEpsi.AtomicOrbitals: make_nlms_spec
using ACEpsi: ↑, ↓, ∅, spins, extspins, Spin, spin2idx, idx2spin
using ACEpsi
using LuxCore: AbstractExplicitLayer
using LuxCore
using Lux
using Lux: Chain, WrappedFunction, BranchLayer
using ChainRulesCore
using ChainRulesCore: NoTangent
# ----------------------------------------
# some quick hacks that we should take care in P4ML later with careful thoughts
using ObjectPools: acquire!
using StrideArrays
using ObjectPools: unwrap

function ChainRulesCore.rrule(::typeof(LuxCore.apply), l::LinearLayer{false}, x::AbstractMatrix, ps, st)
   val = l(x, ps, st)
   function pb(A)
      return NoTangent(), NoTangent(), A[1] * ps.W, (W = transpose(PtrArray(A[1])) * unwrap(x),), NoTangent() #(W = transpose(A[1]) * x,), NoTangent()
   end
   return val, pb
end

# ----------------- custom layers ------------------
struct MaskLayer <: AbstractExplicitLayer 
   nX::Int64
end

(l::MaskLayer)(Φ, ps, st) = begin 
   T = eltype(Φ)
   A::Matrix{Bool} = [st.Σ[i] == st.Σ[j] for j = 1:l.nX, i = 1:l.nX] 
   val::Matrix{T} = Φ .* A
   return val, st
end

function ChainRulesCore.rrule(::typeof(LuxCore.apply), l::MaskLayer, Φ, ps, st) 
   T = eltype(Φ)
   A::Matrix{Bool} = [st.Σ[i] == st.Σ[j] for j = 1:l.nX, i = 1:l.nX]
   val::Matrix{T} = Φ .* A
   function pb(dΦ)
      return NoTangent(), NoTangent(), dΦ[1] .* A, NoTangent(), NoTangent()
   end
   return (val, st), pb
end

function get_spec(nuclei, spec1p) 
   spec = []
   Nnuc = length(nuclei)

   spec = Array{Any}(undef, (3, Nnuc, length(spec1p)))

   for (k, nlm) in enumerate(spec1p)
      for I = 1:Nnuc 
         for (is, s) in enumerate(extspins())
            spec[is, I, k] = (s=s, I = I, nlm...)
         end
      end
   end

   return spec[:]
end

# ----------------- custom layers ------------------
struct myReshapeLayer{N} <: AbstractExplicitLayer
   dims::NTuple{N, Int}
end

@inline function (r::myReshapeLayer)(x::AbstractArray, ps, st::NamedTuple)
   return reshape(unwrap(x), r.dims), st
end

function ChainRulesCore.rrule(::typeof(LuxCore.apply), l::myReshapeLayer{N}, X, ps, st) where {N}
   val = l(X, ps, st)
   function pb(dϕnlm) # dA is of a tuple (dAmat, st), dAmat is of size (Nnuc, Nel, Nnlm)
      A = reshape(unwrap(dϕnlm[1]), size(X))
      return NoTangent(), NoTangent(), A, NoTangent(), NoTangent()
   end
   return val, pb
end

function BFwf_lux(Nel::Integer, bRnl, bYlm, nuclei; totdeg = 15, 
   ν = 3, T = Float64, 
   sd_admissible = bb -> prod(b.s != '∅' for b in bb) == 0, js = JPauliNet(nuclei)) 

   spec1p = make_nlms_spec(bRnl, bYlm; 
                          totaldegree = totdeg)

   # ----------- Lux connections ---------
   # AtomicOrbitalsBasis: (X, Σ) -> (length(nuclei), nX, length(spec1))
   prodbasis_layer = ACEpsi.AtomicOrbitals.ProductBasisLayer(spec1p, bRnl, bYlm)
   aobasis_layer = ACEpsi.AtomicOrbitals.AtomicOrbitalsBasisLayer(prodbasis_layer, nuclei)

   # BackFlowPooling: (length(nuclei), nX, length(spec1 from totaldegree)) -> (nX, 3, length(nuclei), length(spec1))
   pooling = BackflowPooling(aobasis_layer)
   pooling_layer = ACEpsi.lux(pooling)

   spec1p = get_spec(nuclei, spec1p)
   # define sparse for n-correlations
   tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
   default_admissible = bb -> (length(bb) == 0) || (sum(b.n1 - 1 for b in bb ) <= totdeg)

   specAA = gensparse(; NU = ν, tup2b = tup2b, admissible = default_admissible,
                        minvv = fill(0, ν), 
                        maxvv = fill(length(spec1p), ν), 
                        ordered = true)
   spec = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]
   
   # further restrict
   spec = [t for t in spec if sd_admissible([spec1p[t[j]] for j = 1:length(t)])]
   
   # define n-correlation
   corr1 = Polynomials4ML.SparseSymmProd(spec)

   # (nX, 3, length(nuclei), length(spec1 from totaldegree)) -> (nX, length(spec))
   corr_layer = Polynomials4ML.lux(corr1; use_cache = false)

   jastrow_layer = ACEpsi.lux(js)

   BFwf_chain = Chain(; ϕnlm = aobasis_layer, bA = pooling_layer, reshape = myReshapeLayer((Nel, 3 * length(nuclei) * length(prodbasis_layer.sparsebasis))), 
                        bAA = corr_layer, hidden1 = LinearLayer(length(corr1), Nel), 
                        Mask = ACEpsi.MaskLayer(Nel), det = WrappedFunction(x -> det(x))) #, logabs = WrappedFunction(x -> 2 * log(abs(x))) )
   return Chain(; branch = BranchLayer(; js = jastrow_layer, bf = BFwf_chain, ), prod = WrappedFunction(x -> x[1] * x[2]), logabs = WrappedFunction(x -> 2 * log(abs(x))) ), spec, spec1p
   #return Chain(; js = jastrow_layer, logabs = WrappedFunction(x -> 2 * log(abs(x))) ), spec, spec1p
end


function displayspec(spec, spec1p)
   nicespec = []
   for k = 1:length(spec)
      push!(nicespec, ([spec1p[spec[k][j]] for j = 1:length(spec[k])]))
   end
   return nicespec
end

# """
# This function returns correct Si for pooling operation.
# """
# function onehot!(Si, i, Σ)
#    Si .= 0
#    for k = 1:length(Σ)
#       Si[k, spin2num(Σ[k])] = 1
#    end
#    # set current electron to ϕ, also remove their contribution in the sum of ↑ or ↓ basis
#    Si[i, 1] = 1 
#    Si[i, 2] = 0
#    Si[i, 3] = 0
# end


# """
# This function convert spin to corresponding integer value used in spec
# """
# function spin2num(σ)
#    if σ == '↑'
#       return 2
#    elseif σ == '↓'
#       return 3
#    elseif σ == '∅'
#       return 1
#    end
#    error("illegal spin char for spin2num")
# end

# """
# This function convert num to corresponding spin string.
# """
# function num2spin(σ)
#    if σ == 2
#       return '↑'
#    elseif σ == 3
#       return '↓'
#    elseif σ == 1
#       return '∅'
#    end
#    error("illegal integer value for num2spin")
# end

# function assemble_A(bA::RnlYlmEmbedding, X::AbstractVector, Σ)
      
#    nX = length(X)
#    # position embedding 
#    P = wf.P 
#    Xt = wf.trans.(X)
#    evaluate!(P, wf.polys, Xt)    # nX x dim(polys)
   
#    A = wf.A    # zeros(nX, length(wf.pooling)) 
#    Ai = wf.Ai  # zeros(length(wf.pooling))
#    Si = wf.Si  # zeros(Bool, nX, 2)

#    for i = 1:nX 
#       onehot!(Si, i, Σ)
#       ACEcore.evalpool!(Ai, wf.pooling, (unwrap(P), Si))
#       A[i, :] .= Ai
#    end
#    return A 
# end

# ---------- old implementation


# mutable struct BFwf{T, TT, TPOLY, TE}
#    embedding::TA
#    corr::SparseSymmProdDAG{T}
#    W::Matrix{T}
#    envelope::TE
#    spec::Vector{Vector{Int64}} # corr.spec TODO: this needs to be remove
#    # ---------------- Temporaries 
#    P::Matrix{T}
#    ∂P::Matrix{T}
#    dP::Matrix{T}
#    Φ::Matrix{T} 
#    ∂Φ::Matrix{T}
#    A::Matrix{T}
#    ∂A::Matrix{T}
#    Ai::Vector{T} 
#    ∂Ai::Vector{T}
#    Si::Matrix{Bool}
#    ∂AA::Matrix{T}
#    ∂Si::Matrix{T}
#    ∇AA::Array{T, 3}
# end

# (Φ::BFwf)(args...) = evaluate(Φ, args...)

# function BFwf(Nel::Integer, polys; totdeg = length(polys), 
#                      ν = 3, T = Float64, 
#                      trans = identity, 
#                      sd_admissible = bb -> (true),
#                      envelope = envelopefcn(x -> sqrt(1 + x^2), rand()))
#    # 1-particle spec 
#    K = length(polys)
#    spec1p = [ (k, σ) for σ in [1, 2, 3] for k in 1:K]  # (1, 2, 3) = (∅, ↑, ↓);
#    spec1p = sort(spec1p, by = b -> b[1]) # sorting to prevent gensparse being confused
   
#    pooling = PooledSparseProduct(spec1p)
#    # generate the many-particle spec 
#    tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
#    default_admissible = bb -> (length(bb) == 0) || (sum(b[1] - 1 for b in bb ) <= totdeg)
   
#    specAA = gensparse(; NU = ν, tup2b = tup2b, admissible = default_admissible,
#                         minvv = fill(0, ν), 
#                         maxvv = fill(length(spec1p), ν), 
#                         ordered = true)
   
   
#    spec = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]

#    # further restrict
#    spec = [t for t in spec if sd_admissible([spec1p[t[j]] for j = 1:length(t)])]

#    corr1 = SparseSymmProd(spec; T = Float64)
#    corr = corr1.dag   

#    # initial guess for weights 
#    Q, _ = qr(randn(T, length(corr), Nel))
#    W = Matrix(Q) 

#    return BFwf(trans, polys, pooling, corr, W, envelope, spec,
#                   zeros(T, Nel, length(polys)), 
#                   zeros(T, Nel, length(polys)), 
#                   zeros(T, Nel, length(polys)), 
#                   zeros(T, Nel, Nel), 
#                   zeros(T, Nel, Nel),
#                   zeros(T, Nel, length(pooling)), 
#                   zeros(T, Nel, length(pooling)), 
#                   zeros(T, length(pooling)), 
#                   zeros(T, length(pooling)), 
#                   zeros(Bool, Nel, 3),
#                   zeros(T, Nel, length(corr)), 
#                   zeros(T, Nel, 3), 
#                   zeros(T, Nel, Nel, length(corr)) )

# end


# """
# This function returns a nice version of spec.
# """
# function displayspec(wf::BFwf)
#    K = length(wf.polys)
#    spec1p = [ (k, σ) for σ in [1, 2, 3] for k in 1:K]
#    spec1p = sort(spec1p, by = b -> b[1])
#    _getnicespec = l -> (l[1], num2spin(l[2]))
#    nicespec = []
#    for k = 1:length(wf.spec)
#       push!(nicespec, _getnicespec.([spec1p[wf.spec[k][j]] for j = 1:length(wf.spec[k])]))
#    end
#    return nicespec
# end



# function evaluate(wf::BFwf, X::AbstractVector, Σ, Pnn=nothing)
#    nX = length(X)
#    A = assemble_A(wf, X, Σ)
#    AA = ACEcore.evaluate(wf.corr, A)  # nX x length(wf.corr)
   
#    # the only basis to be purified are those with same spin
#    # scan through all corr basis, if they comes from same spin, remove self interation by using basis 
#    # from same spin
#    # first we have to construct coefficent for basis coming from same spin, that is in other words the coefficent
#    # matrix of the original polynomial basis, this will be pass from the argument Pnn
#    # === purification goes here === #
   
#    # === #
#    Φ = wf.Φ
#    mul!(Φ, unwrap(AA), wf.W) # nX x nX
#    Φ = Φ .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX] # the resulting matrix should contains two block each comes from each spin
#    release!(AA)

#    env = wf.envelope(X)
#    return 2 * logabsdet(Φ)[1] + 2 * log(abs(env))
# end


# function gradp_evaluate(wf::BFwf, X::AbstractVector, Σ)
#    nX = length(X)
   
#    A = assemble_A(wf, X, Σ)
#    AA = ACEcore.evaluate(wf.corr, A)  # nX x length(wf.corr)
#    Φ = wf.Φ 
#    mul!(Φ, unwrap(AA), wf.W)
#    Φ = Φ .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX] # the resulting matrix should contains two block each comes from each spin


#    # ψ = log | det( Φ ) |
#    # ∂Φ = ∂ψ/∂Φ = Φ⁻ᵀ
#    ∂Φ = transpose(pinv(Φ))

#    # ∂W = ∂ψ / ∂W = ∂Φ * ∂_W( AA * W ) = ∂Φ * AA
#    # ∂Wij = ∑_ab ∂Φab * ∂_Wij( ∑_k AA_ak W_kb )
#    #      = ∑_ab ∂Φab * ∑_k δ_ik δ_bj  AA_ak
#    #      = ∑_a ∂Φaj AA_ai = ∂Φaj' * AA_ai
#    ∇p = transpose(unwrap(AA)) * ∂Φ

#    release!(AA)
#    ∇p = ∇p * 2


#    # ------ gradient of env (w.r.t. ξ) ----- 
#    # ∂ = ∂/∂ξ
#    # r = ||x||
#    # ∂(2 * logabs(env)) = ∂(2 * log(exp(-ξf(r)))) = ∂(-2ξf(r)) = -f(r)
#    ∇logabsenv = - 2 * wf.envelope.f(norm(X))

#    return (∇p = ∇p, ∇logabsenv = [∇logabsenv]) # TODO: return a named tuple (W = gradp, D = gradient w.r.t parameter of env)
# end



# # ----------------------- gradient 

# struct ZeroNoEffect end 
# Base.size(::ZeroNoEffect, ::Integer) = Inf
# Base.setindex!(A::ZeroNoEffect, args...) = nothing
# Base.getindex(A::ZeroNoEffect, args...) = Bool(0)


# function gradient(wf::BFwf, X, Σ)
#    nX = length(X)

#    # ------ forward pass  ----- 

#    # position embedding (forward-mode)
#    # here we evaluate and differentiate at the same time, which is cheap
#    P = wf.P 
#    dP = wf.dP
#    Xt = wf.trans.(X)
#    Polynomials4ML.evaluate_ed!(P, dP, wf.polys, Xt)
#    ∂Xt = ForwardDiff.derivative.(Ref(x -> wf.trans(x)), X)
#    @inbounds for k = 1:size(dP, 2)
#       @simd ivdep for i = 1:nX
#          dP[i, k] *= ∂Xt[i]
#       end
#    end
   
#    # no gradients here - need to somehow "constify" this vector 
#    # could use other packages for inspiration ... 

#    # pooling : need an elegant way to shift this loop into a kernel!
#    #           e.g. by passing output indices to the pooling function.

#    A = wf.A    # zeros(nX, length(wf.pooling)) 
#    Ai = wf.Ai  # zeros(length(wf.pooling))
#    Si = wf.Si  # zeros(Bool, nX, 3)
   
#    for i = 1:nX 
#       onehot!(Si, i, Σ)
#       ACEcore.evalpool!(Ai, wf.pooling, (unwrap(P), Si))
#       A[i, :] .= Ai
#    end
   
#    # n-correlations 
#    AA = ACEcore.evaluate(wf.corr, A)  # nX x length(wf.corr)

#    # generalized orbitals 
#    Φ = wf.Φ
#    mul!(Φ, unwrap(AA), wf.W)

#    # the resulting matrix should contains two block each comes from each spin
#    Φ = Φ .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX]
   
#    # envelope 
#    env = wf.envelope(X)

#    # and finally the wave function 
#    # ψ = logabsdet(Φ)[1] + log(abs(env))

#    # ------ backward pass ------
#    #∂Φ = ∂ψ / ∂Φ = Φ⁻ᵀ
#    ∂Φ = transpose(pinv(Φ))

#    # ∂AA = ∂ψ/∂AA = ∂ψ/∂Φ * ∂Φ/∂AA = ∂Φ * wf.W'
#    ∂AA = wf.∂AA 
#    mul!(∂AA, ∂Φ, transpose(wf.W))

#    # ∂A = ∂ψ/∂A = ∂ψ/∂AA * ∂AA/∂A -> use custom pullback
#    ∂A = wf.∂A   # zeros(size(A))
#    ACEcore.pullback_arg!(∂A, ∂AA, wf.corr, unwrap(AA))
#    release!(AA)

#    # ∂P = ∂ψ/∂P = ∂ψ/∂A * ∂A/∂P -> use custom pullback 
#    # but need to do some work here since multiple 
#    # pullbacks can be combined here into a single one maybe? 
#    ∂P = wf.∂P  # zeros(size(P))
#    fill!(∂P, 0)
#    ∂Si = wf.∂Si # zeros(size(Si))   # should use ZeroNoEffect here ?!??!
#    Si_ = zeros(nX, 3)
#    for i = 1:nX 
#       onehot!(Si_, i, Σ)
#       # note this line ADDS the pullback into ∂P, not overwrite the content!!
#       ∂Ai = @view ∂A[i, :]
#       ACEcore._pullback_evalpool!((∂P, ∂Si), ∂Ai, wf.pooling, (P, Si_))
#    end

#    # ∂X = ∂ψ/∂X = ∂ψ/∂P * ∂P/∂X 
#    #   here we can now finally employ the dP=∂P/∂X that we already know.
#    # ∂ψ/∂Xi = ∑_k ∂ψ/∂Pik * ∂Pik/∂Xi
#    #        = ∑_k ∂P[i, k] * dP[i, k]
#    g = zeros(nX)
#    @inbounds for k = 1:length(wf.polys)
#       @simd ivdep for i = 1:nX 
#          g[i] += ∂P[i, k] * dP[i, k]
#       end
#    end
#    # g = sum(∂P .* dP, dims = 2)[:]

#    # envelope 
#    ∇env = ForwardDiff.gradient(wf.envelope, X)
#    g += ∇env / env 
#    g = g * 2
#    return g
# end


# # ------------------ Laplacian implementation 

# function laplacian(wf::BFwf, X, Σ)

#    A, ∇A, ΔA = _assemble_A_∇A_ΔA(wf, X, Σ)
#    AA, ∇AA, ΔAA = _assemble_AA_∇AA_ΔAA(A, ∇A, ΔA, wf)

#    Δψ = _laplacian_inner(AA, ∇AA, ΔAA, wf, Σ)

#    # envelope 
#    env = wf.envelope(X)
#    ∇env = ForwardDiff.gradient(wf.envelope, X)
#    Δenv = tr(ForwardDiff.hessian(wf.envelope, X))

#    # Δ(ln(env)) = Δenv / env - ∇env ⋅ ∇env / env ^ 2
#    Δψ += Δenv / env - dot(∇env, ∇env) / env^2
#    Δψ = Δψ * 2
#    return Δψ
# end 

# function _assemble_A_∇A_ΔA(wf, X, Σ)
#    TX = eltype(X)
#    lenA = length(wf.pooling)
#    nX = length(X) 
#    A = zeros(TX, nX, lenA)
#    ∇A = zeros(TX, nX, nX, lenA)
#    ΔA = zeros(TX, nX, lenA)
#    spec_A = wf.pooling.spec

#    Xt = wf.trans.(X)
#    P, dP, ddP = Polynomials4ML.evaluate_ed2(wf.polys, Xt)
#    dtrans = x -> ForwardDiff.derivative(wf.trans, x)
#    ddtrans = x -> ForwardDiff.derivative(dtrans, x)
#    ∂Xt = dtrans.(X)
#    ∂∂Xt = ddtrans.(X)
#    @inbounds for k = 1:size(dP, 2)
#       @simd ivdep for i = 1:nX
#          dP[i, k], ddP[i, k] = ∂Xt[i] * dP[i, k], ∂∂Xt[i] * dP[i, k] + ∂Xt[i]^2 * ddP[i, k]
#       end
#    end

#    Si_ = zeros(nX, 3)
#    Ai = zeros(length(wf.pooling))
#    @inbounds for i = 1:nX # loop over orbital bases (which i becomes ∅)
#       fill!(Si_, 0)
#       onehot!(Si_, i, Σ)
#       ACEcore.evalpool!(Ai, wf.pooling, (P, Si_))
#       @. A[i, :] .= Ai
#       for (iA, (k, σ)) in enumerate(spec_A)
#          for a = 1:nX 
#             ∇A[a, i, iA] = dP[a, k] * Si_[a, σ]
#             ΔA[i, iA] += ddP[a, k] * Si_[a, σ]
#          end
#       end
#    end
#    return A, ∇A, ΔA 
# end

# function _assemble_AA_∇AA_ΔAA(A, ∇A, ΔA, wf)
#    nX = size(A, 1)
#    AA = zeros(nX, length(wf.corr))
#    ∇AA = wf.∇AA  
#    ΔAA = zeros(nX, length(wf.corr))

#    @inbounds for iAA = 1:wf.corr.num1 
#       @. AA[:, iAA] .= A[:, iAA] 
#       @. ∇AA[:, :, iAA] .= ∇A[:, :, iAA]
#       @. ΔAA[:, iAA] .= ΔA[:, iAA]
#    end

#    lenAA = length(wf.corr)
#    @inbounds for iAA = wf.corr.num1+1:lenAA 
#       k1, k2 = wf.corr.nodes[iAA]
#       for i = 1:nX 
#          AA_k1 = AA[i, k1]; AA_k2 = AA[i, k2]
#          AA[i, iAA] = AA_k1 * AA_k2 
#          L = ΔAA[i, k1] * AA_k2 
#          L = muladd(ΔAA[i, k2], AA_k1, L)
#          @simd ivdep for a = 1:nX         
#             ∇AA_k1 = ∇AA[a, i, k1]; ∇AA_k2 = ∇AA[a, i, k2]
#             L = muladd(2 * ∇AA_k1, ∇AA_k2, L)
#             g = ∇AA_k1 * AA_k2
#             ∇AA[a, i, iAA] = muladd(∇AA_k2, AA_k1, g)
#          end
#          ΔAA[i, iAA] = L         
#       end      
#    end
#    return AA, ∇AA, ΔAA
# end


# function _laplacian_inner(AA, ∇AA, ΔAA, wf, Σ)

#    # Δψ = Φ⁻ᵀ : ΔΦ - ∑ᵢ (Φ⁻ᵀ * Φᵢ)ᵀ : (Φ⁻ᵀ * Φᵢ)
#    # where Φᵢ = ∂_{xi} Φ

#    nX = size(AA, 1)
   
#    # the wf, and the first layer of derivatives 
#    Φ = wf.Φ 
#    mul!(Φ, unwrap(AA), wf.W)
#    Φ = Φ .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX] # the resulting matrix should contains two block each comes from each spin
#    Φ⁻ᵀ = transpose(pinv(Φ))
   
#    # first contribution to the laplacian
#    ΔΦ = ΔAA * wf.W
#    ΔΦ = ΔΦ .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX] # the resulting matrix should contains two block each comes from each spin

#    Δψ = dot(Φ⁻ᵀ, ΔΦ)
   
#    # the gradient contribution 
#    # TODO: we can rework this into a single BLAS3 call
#    # which will also give us a single back-propagation 
#    # ∇Φi = zeros(nX, nX)
#    ∇Φ_all = reshape(reshape(∇AA, nX*nX, :) * wf.W, nX, nX, nX)
#    Φ⁻¹∇Φi = zeros(nX, nX)
#    for i = 1:nX 
#       ∇Φi = ∇Φ_all[i, :, :] .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX]
#       mul!(Φ⁻¹∇Φi, transpose(Φ⁻ᵀ), ∇Φi)
#       Δψ -= dot(transpose(Φ⁻¹∇Φi), Φ⁻¹∇Φi)
#    end
   
#    return Δψ
# end


# # ------------------ gradp of Laplacian  


# function gradp_laplacian(wf::BFwf, X, Σ)


#    # ---- gradp of Laplacian of Ψ ----

#    nX = length(X) 

#    A, ∇A, ΔA = _assemble_A_∇A_ΔA(wf, X, Σ)
#    AA, ∇AA, ΔAA = _assemble_AA_∇AA_ΔAA(A, ∇A, ΔA, wf)

   
#    # the wf, and the first layer of derivatives 
#    Φ = wf.Φ 
#    mul!(Φ, unwrap(AA), wf.W)
#    Φ = Φ .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX] # the resulting matrix should contains two block each comes from each spin

#    Φ⁻¹ = pinv(Φ)
#    Φ⁻ᵀ = transpose(Φ⁻¹)

#    # first contribution to the laplacian
#    ΔΦ = ΔAA * wf.W
#    ΔΦ = ΔΦ .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX] # the resulting matrix should contains two block each comes from each spin
 
#    # Δψ += dot(Φ⁻ᵀ, ΔΦ) ... this leads to the next two terms 
#    ∂ΔΦ = Φ⁻ᵀ
#    ∇Δψ = transpose(ΔAA) * ∂ΔΦ
   
#    ∂Φ = - Φ⁻ᵀ * transpose(ΔΦ) * Φ⁻ᵀ
#    # ∇Δψ += transpose(AA) * ∂Φ


#    # the gradient contribution 
#    # TODO: we can rework this into a single BLAS3 call
#    # which will also give us a single back-propagation 
#    # ∇Φi = zeros(nX, nX)
#    # Φ⁻¹∇Φi = zeros(nX, nX)
#    ∇Φ_all = reshape(reshape(∇AA, nX*nX, :) * wf.W, nX, nX, nX)
#    ∂∇Φ_all = zeros(nX, nX, nX)

#    for i = 1:nX 
#       ∇Φi = ∇Φ_all[i, :, :] .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX]
#       ∇Φiᵀ = transpose(∇Φi)
#       # Δψ += - dot( [Φ⁻¹∇Φi]ᵀ, Φ⁻¹∇Φi )

#       # ∂∇Φi = - 2 * Φ⁻ᵀ * ∇Φiᵀ * Φ⁻ᵀ
#       ∂∇Φ_all[i, :, :] = - 2 * Φ⁻ᵀ * ∇Φiᵀ * Φ⁻ᵀ
#       # ∇Δψ += transpose(∇AA[i, :, :]) * ∂∇Φi

#       ∂Φ += 2 * Φ⁻ᵀ * ∇Φiᵀ * Φ⁻ᵀ * ∇Φiᵀ * Φ⁻ᵀ
#       # ∇Δψ += transpose(AA) * ∂Φ
#    end

#    ∇Δψ += transpose(AA) * ∂Φ

#    ∇Δψ += reshape( transpose(reshape(∇AA, nX*nX, :)) * reshape(∂∇Φ_all, nX*nX, nX), 
#                   size(∇Δψ) )


#    # ---- gradp of Laplacian of env ----
#    # ∂ = ∂/∂ξ  
#    # r = ||x||
#    # ∂(2 * Δ(logabs(env))) = ∂(2 * Δ(-ξf(r)) = -2 * Δf(r)
#    # Δf(r) = (n-1)/r f'(r) + f''(r)
#    r = norm(X)
#    f(r) = wf.envelope.f(r)
#    df(r) = ForwardDiff.derivative(f, r)
#    ddf(r) = ForwardDiff.derivative(r -> df(r), r)
#    Δf = ddf(r) + (length(X)-1) * df(r)/r

#    return (∇Δψ = 2 * ∇Δψ, ∇Δlogabsenv = [2 * -Δf])
# end 



# # ----------------- BFwf parameter wraging

# function get_params(U::BFwf)
#    return (U.W, U.envelope.ξ)
# end

# function set_params!(U::BFwf, para)
#    U.W = para[1]
#    set_params!(U.envelope, para[2])
#    return U
# end

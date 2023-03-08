using ACEcore, Polynomials4ML
using Polynomials4ML: OrthPolyBasis1D3T
using ACEcore: PooledSparseProduct, SparseSymmProdDAG, SparseSymmProd, release!
using ACEcore.Utils: gensparse
using LinearAlgebra: qr, I, logabsdet, pinv, mul!, dot , tr
import ForwardDiff
import Base.Experimental: @aliasscope

mutable struct BFwfs{T, TT, TPOLY, TE}
   pos::Vector{T}
   tpos::Vector{T}
   trans::TT
   polys::TPOLY
   pooling::PooledSparseProduct{3}
   corr::SparseSymmProdDAG{T}
   W::Matrix{T}
   envelope::TE
   spec::Vector{Vector{Int64}} # corr.spec TODO: this needs to be remove
   # ---------------- Temporaries
   P::Matrix{T}
   ∂P::Matrix{T}
   dP::Matrix{T}
   Φ::Matrix{T}
   ∂Φ::Matrix{T}
   A::Matrix{T}
   ∂A::Matrix{T}
   Ai::Vector{T}
   ∂Ai::Vector{T}
   Si::Matrix{Bool}
   ∂AA::Matrix{T}
   ∂Si::Matrix{T}
   ∇AA::Array{T, 3}
   T::Matrix{T}
   dT::Matrix{T}
   ∂T::Matrix{T}
   Tc::Vector{T}
   Ta::Matrix{T}
   Tb::Matrix{T}
end

(Φ::BFwfs)(args...) = evaluate(Φ, args...)

const ↑ = '↑'
const ↓ = '↓'
const Ø = 'Ø'

function BFwf(Nel::Integer, polys::OrthPolyBasis1D3T, envelope::Function; totdeg = length(polys),
                     ν = 3, T = Float64,
                     pos = [0.0],tpos = [0,0],
                     trans = identity,
                     sd_admissible = bb -> (true))
   
   @assert length(trans) == length(tpos)
   # 1-particle spec
   K = length(polys)
   M = length(trans)
   spec1p = [ (k, σ, m) for σ in [1, 2, 3] for k in 1:K for m in 1:M]  # (1, 2, 3) = (∅, ↑, ↓);
   spec1p = sort(spec1p, by = b -> b[1]) # sorting to prevent gensparse being confused
   pooling = PooledSparseProduct(spec1p)
   # generate the many-particle spec
   tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
   default_admissible = bb -> (length(bb) == 0) || (sum(b[1] - 1 for b in bb ) <= totdeg)

   specAA = gensparse(; NU = ν, tup2b = tup2b, admissible = default_admissible,
                        minvv = fill(0, ν),
                        maxvv = fill(length(spec1p), ν),
                        ordered = true)


   spec = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]

   # further restrict
   spec = [t for t in spec if sd_admissible([spec1p[t[j]] for j = 1:length(t)])]

   corr1 = SparseSymmProd(spec; T = Float64)
   corr = corr1.dag
   corr = SparseSymmProdDAG{Float64}(corr.nodes[corr1.proj],corr.num1,length(corr.nodes[corr1.proj]),corr.projection,corr.pool_AA,corr.bpool_AA)
   
   # initial guess for weights
   Q, _ = qr(randn(T, length(corr), Nel))
   W = Matrix(Q)

   return BFwfs(pos, tpos, trans, polys, pooling, corr, W, envelope, spec,
                  zeros(T, Nel, length(polys) * length(trans)),
                  zeros(T, Nel, length(polys) * length(trans)),
                  zeros(T, Nel, length(polys) * length(trans)),
                  zeros(T, Nel, Nel),
                  zeros(T, Nel, Nel),
                  zeros(T, Nel, length(pooling)),
                  zeros(T, Nel, length(pooling)),
                  zeros(T, length(pooling)),
                  zeros(T, length(pooling)),
                  zeros(Bool, Nel, 3),
                  zeros(T, Nel, length(corr)),
                  zeros(T, Nel, 3),
                  zeros(T, Nel, Nel, length(corr)),
                  ones(T, Nel, length(trans)),
                  zeros(T, Nel, length(trans)),
                  zeros(T, Nel, length(trans)),
                  zeros(T, length(trans)), # Tc: spec1p: exp(-Tc)
                  ones(T, Nel, length(pos)), # Ta: Φᵢ
                  ones(T, Nel, length(pos)) # Tb: Φᵢexp(-Tb)
                  )
end


function assemble_A(wf::BFwfs, X::AbstractVector, Σ::Vector{Char})
   nX = length(X)
   # position embedding
   P = wf.P # P = [P(trans[1]); P(trans[2]); ...; P(trans[M])]
   T = wf.T
   NP = length(wf.polys)
   for i = 1:length(wf.trans)
      Xt = wf.trans[i].(X)
      P[:,(i-1) * NP + 1: i * NP] = evaluate!(P[:,(i-1) * NP + 1: i * NP], wf.polys, Xt)
      T[:,i] = exp.(-wf.Tc[i] * sqrt.(Ref(1).+(X .- Ref(wf.tpos[i])).^2))
   end

   A = wf.A
   Ai = wf.Ai
   Si = wf.Si

   for i = 1:nX
      onehot!(Si, i, Σ)
      evalpool!(Ai, wf.pooling, (parent(P), Si, T), NP)
      A[i, :] .= Ai
   end
   return A
end


function evalpool!(A, basis::PooledSparseProduct{3}, BB, NP)
   nX = size(BB[1], 1)
   @assert all(B->size(B, 1) == nX, BB)
   BB = ACEcore.constify(BB) # Assumes that no B aliases A
   spec = ACEcore.constify(basis.spec)

   @aliasscope begin # No store to A aliases any read from any B
      @inbounds for (iA, ϕ) in enumerate(spec)
         ϕ1 = ((ϕ[3]-1) * NP + ϕ[1],ϕ[2],ϕ[3])
         a = zero(eltype(A))
         @simd ivdep for j = 1:nX
            a += BB_prod(ϕ1, BB, j) # Tk∘atan_m * decay_m
         end
         A[iA] = a
      end
   end
   return nothing
end


@inline function BB_prod(ϕ::NTuple{3}, BB, j)
   reduce(Base.FastMath.mul_fast, ntuple(Val(3)) do i
      @inline
      @inbounds BB[i][j, ϕ[i]]
   end)
end


function evaluate(wf::BFwfs, X::AbstractVector, Σ::Vector{Char})
   nX = length(X)
   A = assemble_A(wf, X, Σ)
   AA = ACEcore.evaluate(wf.corr, A)
   BB = _BB(wf.pos, wf.Ta, wf.Tb, wf.envelope, X)
   Φ = wf.Φ
   mul!(Φ, parent(AA), wf.W) # Φ = AA * W, nX x nX
   Φ =  Φ .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX] .* BB
   release!(AA)
   return 2 * logabsdet(Φ)[1]
end


function envelope(pos, Ta, Tb, f, X::Float64, i::Int)
   M = length(pos)
   a = zero(eltype(X))
   @simd ivdep for j = 1:M
         a += Ta[i,j] * exp(-Tb[i,j] * f(X - pos[j]))
   end
   return a
end


function _BB(pos, Ta, Tb, f, X)
   nX = length(X)
   return [envelope(pos, Ta, Tb, f, X[j], i) for j = 1:nX, i = 1:nX]
end


function Φ!(_Φ, pos, Ta, Tb, f, X)
   Φ = _Φ .* _BB(pos, Ta, Tb, f, X)
   return logabsdet(Φ)[1]
end


function gradp_evaluate(wf::BFwfs, X::AbstractVector, Σ::Vector{Char})
   #  =================== evaluate Φ=============================  #
   nX = length(X) # number of electrons
   BB = _BB(wf.pos, wf.Ta, wf.Tb, wf.envelope, X)
   A = assemble_A(wf, X, Σ)
   AA = ACEcore.evaluate(wf.corr, A)  # nX x length(wf.corr)
   Φ = wf.Φ
   mul!(Φ, parent(AA), wf.W)
   _Φ = Φ .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX]
   Φ =  _Φ .* BB
   #  ================================================  #

   # ψ = log | det( Φ ) |
   # ∂Φ = ∂ψ/∂Φ = Φ⁻ᵀ
   ∂Φ = transpose(pinv(Φ))

   #  =============== ∂ψ / ∂W =  ∂Φ * ∂Φ / ∂W=================================  #
   # ∂W = ∂ψ / ∂W = ∂Φ * ∂_W( AA * W ) = ∂Φ * AA
   # ∂Wij = ∑_ab ∂Φab * ∂_Wij( ∑_k AA_ak W_kb )
   #      = ∑_ab ∂Φab * ∑_k δ_ik δ_bj  AA_ak
   #      = ∑_a ∂Φaj AA_ai = ∂Φaj' * AA_ai
   ∇p = transpose(parent(AA)) * (∂Φ .* BB)

   release!(AA)
   ∇p = ∇p * 2

   ∇Ta = ForwardDiff.gradient(Ta -> Φ!(_Φ, wf.pos, Ta, wf.Tb, wf.envelope, X), wf.Ta)
   ∇Tb = ForwardDiff.gradient(Tb -> Φ!(_Φ, wf.pos, wf.Ta, Tb, wf.envelope, X), wf.Tb)
   ∇Ta = ∇Ta * 2
   ∇Tb = ∇Tb * 2
   return (∇p = ∇p, ∇Ta = ∇Ta, ∇Tb = ∇Tb)
end


# ----------------------- gradient

function gradient(wf::BFwfs, X::AbstractVector, Σ::Vector{Char})
   nX = length(X)
   # ------ forward pass  -----

   # position embedding (forward-mode)
   # here we evaluate and differentiate at the same time, which is cheap
   P = wf.P
   dP = wf.dP # ∂P/∂x
   T = wf.T
   dT = wf.dT # ∂T/∂x
   NP = length(wf.polys)

   for i = 1:length(wf.trans)
      #  =============== P, dP = ∂P/∂trans(x) by Polynomials4ML.evaluate_ed!=================================  #
      Xt = wf.trans[i].(X)
      P[:,(i-1) * NP + 1: i * NP],dP[:,(i-1) * NP + 1: i * NP] = Polynomials4ML.evaluate_ed!(P[:,(i-1) * NP + 1: i * NP], dP[:,(i-1) * NP + 1: i * NP], wf.polys, Xt)
      #  =============== ∂Xt =================================  #
      ∂Xt = ForwardDiff.derivative.(Ref(x -> wf.trans[i](x)), X)
      #  =============== dP = ∂P/∂trans(x) * ∂trans(x)/∂x =================================  #
      @inbounds for k = 1:Int(size(dP, 2)/length(wf.trans))
         @simd ivdep for j = 1:nX
            dP[j, (i-1)*Int(size(dP, 2)/length(wf.trans))+k] *= ∂Xt[j]
         end
      end
      #  =============== T ================================= #
      T[:,i] = exp.(-wf.Tc[i] * sqrt.(Ref(1).+(X .- Ref(wf.tpos[i])).^2))
      #  =============== dT = ∂T/∂x ================================= #
      dT[:,i] = ForwardDiff.derivative.(Ref(x -> exp(-wf.Tc[i] * sqrt(1+(x - wf.tpos[i])^2))), X)
   end

   # no gradients here - need to somehow "constify" this vector
   # could use other packages for inspiration ...

   # pooling : need an elegant way to shift this loop into a kernel!
   #           e.g. by passing output indices to the pooling function.

   A = wf.A    # zeros(nX, length(wf.pooling))
   Ai = wf.Ai  # zeros(length(wf.pooling))
   Si = wf.Si  # zeros(Bool, nX, 3)

   #  =============== Ai: ∑ᵢ T_k∘atan_m(x) * decay_m ================================= #
   for i = 1:nX
      onehot!(Si, i, Σ)
      evalpool!(Ai, wf.pooling, (parent(P), Si, T), NP)
      A[i, :] .= Ai
   end

   # n-correlations
   AA = ACEcore.evaluate(wf.corr, A)  # nX x length(wf.corr)
   #  =================== evaluate =============================== #
   # generalized orbitals
   BB = _BB(wf.pos, wf.Ta, wf.Tb, wf.envelope, X)
   Φ = wf.Φ
   mul!(Φ, parent(AA), wf.W)
   _Φ = Φ .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX]
   Φ =  _Φ .* BB
   #  ================================================== #

   # and finally the wave function
   # ψ = logabsdet(Φ)[1]

   # ------ backward pass ------
   #∂Φ = ∂ψ / ∂Φ = Φ⁻ᵀ
   ∂Φ = transpose(pinv(Φ))

   # ∂AA = ∂ψ/∂AA = ∂ψ/∂Φ * ∂Φ/∂AA = ∂Φ * wf.W'
   ∂AA = wf.∂AA
   mul!(∂AA, ∂Φ .* BB, transpose(wf.W))

   # ∂BB = ∂ψ/∂BB = ∂ψ/∂Φ * ∂Φ/∂BB = ∂Φ .* _Φ
   ∂BB = ∂Φ .* _Φ
   # dBBx = ∂BB/∂x
   dBB = [∇env(wf.pos, wf.Ta, wf.Tb, wf.envelope, X[j], i) for j = 1:nX, i = 1:nX]

   # ∂A = ∂ψ/∂A = ∂ψ/∂AA * ∂AA/∂A -> use custom pullback
   ∂A = wf.∂A   # zeros(size(A))
   ACEcore.pullback_arg!(∂A, ∂AA, wf.corr, parent(AA))
   release!(AA)

   # ∂P = ∂ψ/∂P = ∂ψ/∂A * ∂A/∂P -> use custom pullback
   # but need to do some work here since multiple
   # pullbacks can be combined here into a single one maybe?
   ∂P = wf.∂P  # zeros(size(P))
   fill!(∂P, 0)
   ∂Si = wf.∂Si # zeros(size(Si))   # should use ZeroNoEffect here ?!??!
   ∂T = wf.∂T   # zeros(size(T))
   fill!(∂T, 0)
   Si_ = zeros(nX, 3)
   for i = 1:nX
      onehot!(Si_, i, Σ)
      # note this line ADDS the pullback into ∂P, not overwrite the content!!
      ∂Ai = @view ∂A[i, :]
      _pullback_evalpool!((∂P, ∂Si, ∂T), ∂Ai, wf.pooling, (P, Si_, T), NP)
   end

   # ∂X = ∂ψ/∂X = ∂ψ/∂P * ∂P/∂X + ∂ψ/∂T * ∂T/∂X
   #   here we can now finally employ the dP=∂P/∂X that we already know.
   # ∂ψ/∂Xi = ∑_k ∂ψ/∂Pik * ∂Pik/∂Xi
   #        = ∑_k ∂P[i, k] * dP[i, k]
   g = zeros(nX)
   @inbounds for k = 1:size(dP,2)
      @simd ivdep for i = 1:nX
         g[i] += ∂P[i, k] * dP[i, k]
      end
   end
   @inbounds for k = 1:size(dT,2)
      @simd ivdep for i = 1:nX
         g[i] += ∂T[i, k] * dT[i, k]
      end
   end
   @inbounds for k = 1:size(dBB,2)
      @simd ivdep for i = 1:nX
         g[i] += ∂BB[i, k] * dBB[i, k]
      end
   end
   # g = sum(∂P .* dP, dims = 2)[:]
   g = g * 2
   return g
end

function ∇env(pos, Ta, Tb, f, X::Float64, i::Int)
   df = X -> ForwardDiff.derivative(x -> f(x), X)
   M = length(pos)
   a = zero(eltype(X))
   @simd ivdep for j = 1:M
      dx = df(X - pos[j])
      a += Ta[i,j] * exp(-Tb[i,j] * f(X - pos[j])) * -Tb[i,j] * dx
   end
   return a
end

function _pullback_evalpool!(∂BB, ∂A, basis::PooledSparseProduct{3}, BB::Tuple, NP)
   nX = size(BB[1], 1)
   NB = 3
   @assert all(nX <= size(BB[i], 1) for i = 1:NB)
   @assert all(nX <= size(∂BB[i], 1) for i = 1:NB)
   @assert all(size(∂BB[i], 2) >= size(BB[i], 2) for i = 1:NB)
   @assert length(∂A) == length(basis)
   @assert length(BB) == NB
   @assert length(∂BB) == NB

   @inbounds for (iA, ϕ) in enumerate(basis.spec)
      ϕ1 = ((ϕ[3]-1) * NP + ϕ[1],ϕ[2],ϕ[3])
      ∂A_iA = ∂A[iA]
      @simd ivdep for j = 1:nX
         b = ntuple(Val(NB)) do i
            @inbounds BB[i][j, ϕ1[i]]
         end
         g = ACEcore._prod_grad(b, Val(NB))
         for i = 1:NB
            ∂BB[i][j, ϕ1[i]] = muladd(∂A_iA, g[i], ∂BB[i][j, ϕ1[i]])
         end
      end
   end
   return nothing
end

# ------------------ Laplacian implementation

function laplacian(wf::BFwfs, X::AbstractVector, Σ::Vector{Char})
   A, ∇A, ΔA = _assemble_A_∇A_ΔA(wf, X, Σ)
   AA, ∇AA, ΔAA = _assemble_AA_∇AA_ΔAA(A, ∇A, ΔA, wf)

   Δψ = _laplacian_inner(AA, ∇AA, ΔAA, wf, X, Σ)
   Δψ = Δψ * 2
   return Δψ
end

function _assemble_A_∇A_ΔA(wf::BFwfs, X::AbstractVector, Σ::Vector{Char})
   TX = eltype(X)
   lenA = length(wf.pooling)
   nX = length(X)
   A = zeros(TX, nX, lenA)
   ∇A = zeros(TX, nX, nX, lenA)
   ΔA = zeros(TX, nX, lenA)
   spec_A = wf.pooling.spec
   NP = length(wf.polys)

   P = zero(wf.P)
   dP = zero(wf.dP)
   ddP = zero(wf.P)
   T = zero(wf.T)
   dT = zero(wf.dT)
   ddT = zero(wf.T)
   for i = 1:length(wf.trans)
      #  =============== P, dP=================================  #
      Xt = wf.trans[i].(X)
      P[:,(i-1) * NP + 1: i * NP], dP[:,(i-1) * NP + 1: i * NP], ddP[:,(i-1) * NP + 1: i * NP] = Polynomials4ML.evaluate_ed2(wf.polys, Xt)
      #  =============== ∂Xt/∂X, ∂Xt^2/∂^2XX=================================  #
      dtrans = x -> ForwardDiff.derivative(wf.trans[i], x)
      ddtrans = x -> ForwardDiff.derivative(dtrans, x)
      ∂Xt = dtrans.(X)
      ∂∂Xt = ddtrans.(X)
      #  =============== dP = ∂P/∂trans(x) * ∂trans(x)/∂x, ddP = ∂^P/∂trans(x) * (∂trans(x)/∂x)^2=================================  #
      @inbounds for k = 1:Int(size(dP, 2)/length(wf.trans))
         @simd ivdep for j = 1:nX
            dP[j, (i-1) * NP + k], ddP[j, (i-1) * NP + k] = ∂Xt[j] * dP[j, (i-1) * NP + k], ∂∂Xt[j] * dP[j, (i-1) * NP + k] + ∂Xt[j]^2 * ddP[j, (i-1) * NP + k]
         end
      end
      #  =============== T ================================= #
      T[:,i] = exp.(-wf.Tc[i] * sqrt.(Ref(1).+(X .- Ref(wf.tpos[i])).^2))
      #  =============== dT = ∂T/∂x, ddT ================================= #
      dtrans = X -> ForwardDiff.derivative.(x -> exp(-wf.Tc[i] * sqrt(1+(x - wf.tpos[i])^2)), X)
      ddtrans = x -> ForwardDiff.derivative(dtrans, x)
      dT[:,i] = dtrans.(X)
      ddT[:,i] = ddtrans.(X)
   end

   Si_ = zeros(nX, 3)
   Ai = zeros(length(wf.pooling))

   @inbounds for i = 1:nX # loop over orbital bases (which i becomes ∅)
      fill!(Si_, 0)
      onehot!(Si_, i, Σ)
      evalpool!(Ai, wf.pooling, (parent(P), Si_, T), NP)
      @. A[i, :] .= Ai
      for (iA, (k, σ, m)) in enumerate(spec_A)
         for a = 1:nX
            ∇A[a, i, iA] = Si_[a, σ] * (dP[a, (m-1) * NP + k] * T[a,m] + P[a, (m-1) * NP + k] * dT[a, m])
            ΔA[i, iA] += Si_[a, σ] * (ddP[a, (m-1) * NP + k] * T[a,m] + ddT[a, m] * P[a, (m-1) * NP + k] + 2 * dP[a, (m-1) * NP + k] * dT[a, m])
         end
      end
   end
   return A, ∇A, ΔA
end

function _assemble_AA_∇AA_ΔAA(A, ∇A, ΔA, wf::BFwfs)
   nX = size(A, 1)
   AA = zeros(nX, length(wf.corr))
   ∇AA = wf.∇AA
   ΔAA = zeros(nX, length(wf.corr))

   @inbounds for iAA = 1:wf.corr.num1
      @. AA[:, iAA] .= A[:, iAA]
      @. ∇AA[:, :, iAA] .= ∇A[:, :, iAA]
      @. ΔAA[:, iAA] .= ΔA[:, iAA]
   end

   lenAA = length(wf.corr)
   @inbounds for iAA = wf.corr.num1+1:lenAA
      k1, k2 = wf.corr.nodes[iAA]
      for i = 1:nX
         AA_k1 = AA[i, k1]; AA_k2 = AA[i, k2]
         AA[i, iAA] = AA_k1 * AA_k2
         L = ΔAA[i, k1] * AA_k2
         L = muladd(ΔAA[i, k2], AA_k1, L)
         @simd ivdep for a = 1:nX
            ∇AA_k1 = ∇AA[a, i, k1]; ∇AA_k2 = ∇AA[a, i, k2]
            L = muladd(2 * ∇AA_k1, ∇AA_k2, L)
            g = ∇AA_k1 * AA_k2
            ∇AA[a, i, iAA] = muladd(∇AA_k2, AA_k1, g)
         end
         ΔAA[i, iAA] = L
      end
   end
   return AA, ∇AA, ΔAA
end

function Δenv(pos, Ta, Tb, f, X::Float64, i::Int)
   df = X -> ForwardDiff.derivative(x -> f(x), X)
   ddf = X -> ForwardDiff.derivative(df, X)
   M = length(pos)
   a = zero(eltype(X))
   @simd ivdep for j = 1:M
      dx = df(X - pos[j])
      ddx = ddf(X-pos[j])
      a += Ta[i,j] * exp(-Tb[i,j] * f(X - pos[j])) * -Tb[i,j] * dx * -Tb[i,j] * dx
      a += Ta[i,j] * exp(-Tb[i,j] * f(X - pos[j])) * -Tb[i,j] * ddx
   end
   return a
end

function _laplacian_inner(AA, ∇AA, ΔAA, wf::BFwfs, X::AbstractVector, Σ::Vector{Char})

   # Δψ = Φ⁻ᵀ : ΔΦ - ∑ᵢ (Φ⁻ᵀ * Φᵢ)ᵀ : (Φ⁻ᵀ * Φᵢ)
   # where Φᵢ = ∂_{xi} Φ

   nX = size(AA, 1)

   # the wf, and the first layer of derivatives
   BB = _BB(wf.pos, wf.Ta, wf.Tb, wf.envelope, X)
   Φ = wf.Φ
   mul!(Φ, parent(AA), wf.W)
   _Φ = Φ .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX]
   Φ =  _Φ .* BB

   Φ⁻ᵀ = transpose(pinv(Φ))

   # first contribution to the laplacian
   ∇Φ_all = reshape(reshape(∇AA, nX*nX, :) * wf.W, nX, nX, nX)

   ΔBB = [Δenv(wf.pos, wf.Ta, wf.Tb, wf.envelope, X[j], i) for j = 1:nX, i = 1:nX]
   ∇BB = zeros(nX,nX,nX)
   for j = 1:nX
      ∇BB[j,j,:] = [∇env(wf.pos, wf.Ta, wf.Tb, wf.envelope, X[j], i) for i = 1:nX]
   end


   ΔΦ = ΔAA * wf.W .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX] .* BB
   ΔΦ += _Φ .* ΔBB
   ΔΦ += 2 * sum(∇Φ_all[i,:,:] .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX] .* ∇BB[i,:,:] for i = 1:nX)

   Δψ = dot(Φ⁻ᵀ, ΔΦ)

   # the gradient contribution
   # TODO: we can rework this into a single BLAS3 call
   # which will also give us a single back-propagation
   # ∇Φi = zeros(nX, nX)
   Φ⁻¹∇Φi = zeros(nX, nX)
   for i = 1:nX
      ∇Φi = ∇Φ_all[i, :, :] .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX] .* BB
      ∇Φi += _Φ .* ∇BB[i, :, :]
      mul!(Φ⁻¹∇Φi, transpose(Φ⁻ᵀ), ∇Φi)
      Δψ -= dot(transpose(Φ⁻¹∇Φi), Φ⁻¹∇Φi)
   end

   return Δψ
end

# ----------------- BFwf parameter wraging

function get_params(U::BFwfs)
   return (U.W, U.Ta, U.Tb)
end

function set_params!(U::BFwfs, para::Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}})
   U.W = para[1]
   U.Ta = para[2]
   U.Tb = para[3]
   return U
end

function Scaling(U::BFwfs, γ::Float64)
   c = get_params(U)
   uu = []
   _spec = U.spec
   for i = 1:length(_spec)
      push!(u, sum(_spec[i] .^ 2))
   end
   return (uu = γ * uu .* c[1], d = zeros(length(c[2])))
end

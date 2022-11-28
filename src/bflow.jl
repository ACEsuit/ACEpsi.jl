
using ACEcore, Polynomials4ML
using Polynomials4ML: OrthPolyBasis1D3T
using ACEcore: PooledSparseProduct, SparseSymmProdDAG, SparseSymmProd, release!
using ACEcore.Utils: gensparse
using LinearAlgebra: qr, I, logabsdet, pinv, mul!, dot , tr 
import ForwardDiff

struct BFwf{T, TT, TPOLY, TE}
   trans::TT
   polys::TPOLY
   pooling::PooledSparseProduct{2}
   corr::SparseSymmProdDAG{T}
   W::Matrix{T}
   envelope::TE
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
end

(Φ::BFwf)(args...) = evaluate(Φ, args...)

function BFwf(Nel::Integer, polys; totdeg = length(polys), 
                     ν = 3, T = Float64, 
                     trans = identity, 
                     envelope = _ -> 1.0)
   # 1-particle spec 
   K = length(polys)
   spec1p = [ (k, σ) for σ in [1, 2] for k in 1:K ]
   pooling = PooledSparseProduct(spec1p)

   # generate the many-particle spec 
   tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
   admissible = bb -> (length(bb) == 0) || (sum( b[1]-1 for b in bb ) <= totdeg )
   
   specAA = gensparse(; NU = ν, tup2b = tup2b, admissible = admissible,
                        minvv = fill(0, ν), 
                        maxvv = fill(length(spec1p), ν), 
                        ordered = true )
   
   spec = [ vv[vv .> 0] for vv in specAA ][2:end]                     
   corr1 = SparseSymmProd(spec; T = Float64)
   corr = corr1.dag

   # initial guess for weights 
   Q, _ = qr(randn(T, length(corr), Nel))
   W = Matrix(Q) 

   return BFwf(trans, polys, pooling, corr, W, envelope, 
                  zeros(T, Nel, length(polys)), 
                  zeros(T, Nel, length(polys)), 
                  zeros(T, Nel, length(polys)), 
                  zeros(T, Nel, Nel), 
                  zeros(T, Nel, Nel),
                  zeros(T, Nel, length(pooling)), 
                  zeros(T, Nel, length(pooling)), 
                  zeros(T, length(pooling)), 
                  zeros(T, length(pooling)), 
                  zeros(Bool, Nel, 2),
                  zeros(T, Nel, length(corr)), 
                  zeros(T, Nel, 2), 
                  zeros(T, Nel, Nel, length(corr)) )

end


function onehot!(Si, i)
   Si[:, 1] .= 0 
   Si[:, 2] .= 1 
   Si[i, 1] = 1 
   Si[i, 2] = 0
end


# ------------------- evaluate 

function assemble_A(wf::BFwf, X::AbstractVector)
   nX = length(X)

   # position embedding 
   P = wf.P 
   Xt = wf.trans.(X)
   evaluate!(P, wf.polys, Xt)    # nX x dim(polys)
   
   # one-hot embedding - generalize to ∅, ↑, ↓
   A = wf.A    # zeros(nX, length(wf.pooling)) 
   Ai = wf.Ai  # zeros(length(wf.pooling))
   Si = wf.Si  # zeros(Bool, nX, 2)
   for i = 1:nX 
      onehot!(Si, i)
      ACEcore.evalpool!(Ai, wf.pooling, (parent(P), Si))
      A[i, :] .= Ai
   end
   return A 
end

function evaluate(wf::BFwf, X::AbstractVector)
   
   A = assemble_A(wf, X)
   AA = ACEcore.evaluate(wf.corr, A)  # nX x length(wf.corr)
   Φ = wf.Φ 
   mul!(Φ, parent(AA), wf.W)
   release!(AA)

   env = wf.envelope(X)

   return logabsdet(Φ)[1] + log(abs(env))
end


function gradp_evaluate(wf::BFwf, X::AbstractVector)
   nX = length(X)
   
   A = assemble_A(wf, X)
   AA = ACEcore.evaluate(wf.corr, A)  # nX x length(wf.corr)
   Φ = wf.Φ 
   mul!(Φ, parent(AA), wf.W)

   # ψ = log | det( Φ ) |
   # ∂Φ = ∂ψ/∂Φ = Φ⁻ᵀ
   ∂Φ = transpose(pinv(Φ))

   # ∂W = ∂ψ / ∂W = ∂Φ * ∂_W( AA * W ) = ∂Φ * AA
   # ∂Wij = ∑_ab ∂Φab * ∂_Wij( ∑_k AA_ak W_kb )
   #      = ∑_ab ∂Φab * ∑_k δ_ik δ_bj  AA_ak
   #      = ∑_a ∂Φaj AA_ai = ∂Φaj' * AA_ai
   ∇p = transpose(parent(AA)) * ∂Φ

   release!(AA)
   return ∇p 
end



# ----------------------- gradient 

struct ZeroNoEffect end 
Base.size(::ZeroNoEffect, ::Integer) = Inf
Base.setindex!(A::ZeroNoEffect, args...) = nothing
Base.getindex(A::ZeroNoEffect, args...) = Bool(0)


function gradient(wf::BFwf, X)
   nX = length(X)

   # ------ forward pass  ----- 

   # position embedding (forward-mode)
   # here we evaluate and differentiate at the same time, which is cheap
   P = wf.P 
   dP = wf.dP
   Xt = wf.trans.(X) 
   Polynomials4ML.evaluate_ed!(P, dP, wf.polys, Xt)
   ∂Xt = ForwardDiff.derivative.(Ref(x -> wf.trans(x)), X)
   @inbounds for k = 1:size(dP, 2)
      @simd ivdep for i = 1:nX
         dP[i, k] *= ∂Xt[i]
      end
   end
   
   # one-hot embedding - TODO: generalize to ∅, ↑, ↓
   # no gradients here - need to somehow "constify" this vector 
   # could use other packages for inspiration ... 

   # pooling : need an elegant way to shift this loop into a kernel!
   #           e.g. by passing output indices to the pooling function.
   A = wf.A     # zeros(nX, length(wf.pooling)) 
   Ai = wf.Ai   # zeros(length(wf.pooling))
   Si = wf.Si   # zeros(Bool, nX, 2)
   for i = 1:nX 
      onehot!(Si, i)
      ACEcore.evalpool!(Ai, wf.pooling, (parent(P), Si))
      A[i, :] .= Ai
   end

   # n-correlations 
   AA = ACEcore.evaluate(wf.corr, A)  # nX x length(wf.corr)

   # generalized orbitals 
   Φ = wf.Φ
   mul!(Φ, parent(AA), wf.W)

   # envelope 
   env = wf.envelope(X)

   # and finally the wave function 
   # ψ = logabsdet(Φ)[1] + log(abs(env))

   # ------ backward pass ------
   #∂Φ = ∂ψ / ∂Φ = Φ⁻ᵀ
   ∂Φ = transpose(pinv(Φ))

   # ∂AA = ∂ψ/∂AA = ∂ψ/∂Φ * ∂Φ/∂AA = ∂Φ * wf.W'
   ∂AA = wf.∂AA 
   mul!(∂AA, ∂Φ, transpose(wf.W))

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
   Si_ = zeros(nX, 2)
   for i = 1:nX 
      onehot!(Si_, i)
      # note this line ADDS the pullback into ∂P, not overwrite the content!!
      ∂Ai = @view ∂A[i, :]
      ACEcore._pullback_evalpool!((∂P, ∂Si), ∂Ai, wf.pooling, (P, Si_))
   end

   # ∂X = ∂ψ/∂X = ∂ψ/∂P * ∂P/∂X 
   #   here we can now finally employ the dP=∂P/∂X that we already know.
   # ∂ψ/∂Xi = ∑_k ∂ψ/∂Pik * ∂Pik/∂Xi
   #        = ∑_k ∂P[i, k] * dP[i, k]
   g = zeros(nX)
   @inbounds for k = 1:length(wf.polys)
      @simd ivdep for i = 1:nX 
         g[i] += ∂P[i, k] * dP[i, k]
      end
   end
   # g = sum(∂P .* dP, dims = 2)[:]

   # envelope 
   ∇env = ForwardDiff.gradient(wf.envelope, X)
   g += ∇env / env 

   return g
end


# ------------------ Laplacian implementation 

function laplacian(wf::BFwf, X)

   A, ∇A, ΔA = _assemble_A_∇A_ΔA(wf, X)
   AA, ∇AA, ΔAA = _assemble_AA_∇AA_ΔAA(A, ∇A, ΔA, wf)

   Δψ = _laplacian_inner(AA, ∇AA, ΔAA, wf)

   # envelope 
   env = wf.envelope(X)
   ∇env = ForwardDiff.gradient(wf.envelope, X)
   Δenv = tr(ForwardDiff.hessian(wf.envelope, X))
   Δψ += Δenv / env - dot(∇env, ∇env) / env^2
   
   return Δψ
end 

function _assemble_A_∇A_ΔA(wf, X)
   TX = eltype(X)
   lenA = length(wf.pooling)
   nX = length(X) 
   A = zeros(TX, nX, lenA)
   ∇A = zeros(TX, nX, nX, lenA)
   ΔA = zeros(TX, nX, lenA)
   spec_A = wf.pooling.spec

   Xt = wf.trans.(X)
   P, dP, ddP = Polynomials4ML.evaluate_ed2(wf.polys, Xt)
   dtrans = x -> ForwardDiff.derivative(wf.trans, x)
   ddtrans = x -> ForwardDiff.derivative(dtrans, x)
   ∂Xt = dtrans.(X)
   ∂∂Xt = ddtrans.(X)
   @inbounds for k = 1:size(dP, 2)
      @simd ivdep for i = 1:nX
         dP[i, k], ddP[i, k] = ∂Xt[i] * dP[i, k], ∂∂Xt[i] * dP[i, k] + ∂Xt[i]^2 * ddP[i, k]
      end
   end

   Si_ = zeros(nX, 2)
   Ai = zeros(length(wf.pooling))
   @inbounds for i = 1:nX # loop over orbital bases (which i becomes ∅)
      fill!(Si_, 0)
      onehot!(Si_, i)
      ACEcore.evalpool!(Ai, wf.pooling, (P, Si_))
      @. A[i, :] .= Ai
      for (iA, (k, σ)) in enumerate(spec_A)
         for a = 1:nX 
            ∇A[a, i, iA] = dP[a, k] * Si_[a, σ]
            ΔA[i, iA] += ddP[a, k] * Si_[a, σ]
         end
      end
   end
   return A, ∇A, ΔA 
end

function _assemble_AA_∇AA_ΔAA(A, ∇A, ΔA, wf)
   nX = size(A, 1)
   AA = zeros(nX, length(wf.corr))
   ∇AA = zeros(nX, nX, length(wf.corr))   # wf.∇AA  
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


function _laplacian_inner(AA, ∇AA, ΔAA, wf)

   # Δψ = Φ⁻ᵀ : ΔΦ - ∑ᵢ (Φ⁻ᵀ * Φᵢ)ᵀ : (Φ⁻ᵀ * Φᵢ)
   # where Φᵢ = ∂_{xi} Φ

   nX = size(AA, 1)
   
   # the wf, and the first layer of derivatives 
   Φ = wf.Φ 
   mul!(Φ, parent(AA), wf.W)
   Φ⁻ᵀ = transpose(pinv(Φ))

   # first contribution to the laplacian
   ΔΦ = ΔAA * wf.W 
   Δψ = dot(Φ⁻ᵀ, ΔΦ)

   # the gradient contribution 
   # TODO: we can rework this into a single BLAS3 call
   # which will also give us a single back-propagation 
   ∇Φi = zeros(nX, nX)
   Φ⁻¹∇Φi = zeros(nX, nX)
   for i = 1:nX 
      mul!(∇Φi, (@view ∇AA[i, :, :]), wf.W)
      mul!(Φ⁻¹∇Φi, transpose(Φ⁻ᵀ), ∇Φi)
      Δψ -= dot(transpose(Φ⁻¹∇Φi), Φ⁻¹∇Φi)
   end

   return Δψ
end



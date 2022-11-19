
using ACEcore, Polynomials4ML
using Polynomials4ML: OrthPolyBasis1D3T
using ACEcore: PooledSparseProduct, SparseSymmProdDAG, SparseSymmProd, release!
using ACEcore.Utils: gensparse
using LinearAlgebra: qr, I, logabsdet, pinv, mul!


struct BFwf{T, TPOLY}
   polys::TPOLY
   pooling::PooledSparseProduct{2}
   corr::SparseSymmProdDAG{T}
   W::Matrix{T}
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
end

(Φ::BFwf)(args...) = evaluate(Φ, args...)

function BFwf(Nel::Integer, polys; totdeg = length(polys), 
                     ν = 3, T = Float64)
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

   return BFwf(polys, pooling, corr, W, 
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
                  zeros(T, Nel, 2) )

end


function onehot!(Si, i)
   Si[:, 1] .= 0 
   Si[:, 2] .= 1 
   Si[i, 1] = 1 
   Si[i, 2] = 0
end


function evaluate(wf::BFwf, X::AbstractVector)
   nX = length(X)
   
   # position embedding 
   P = wf.P 
   evaluate!(P, wf.polys, X)    # nX x dim(polys)
   
   # one-hot embedding - generalize to ∅, ↑, ↓
   A = wf.A    # zeros(nX, length(wf.pooling)) 
   Ai = wf.Ai  # zeros(length(wf.pooling))
   Si = wf.Si  # zeros(Bool, nX, 2)
   for i = 1:nX 
      onehot!(Si, i)
      ACEcore.evalpool!(Ai, wf.pooling, (parent(P), Si))
      A[i, :] .= Ai
   end

   AA = ACEcore.evaluate(wf.corr, A)  # nX x length(wf.corr)
   Φ = wf.Φ 
   mul!(Φ, parent(AA), wf.W)
   release!(AA)
   return logabsdet(Φ)[1] 
end

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
   Polynomials4ML.evaluate_ed!(P, dP, wf.polys, X)
   
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

   # and finally the wave function 
   ψ = logabsdet(Φ)[1]

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
   return g
end


# ------------------ Laplacian implementation 

function laplacian(wf::BFwf, X)
   
   spec_AA = ACEcore.reconstruct_spec(wf.corr)
   spec_A = wf.pooling.spec 

end 

"""
This will compute the following: 
* `A` just the normal pooling operation, `A[k] = ∑_i P_k(x_i)`
* `dA[k, k'] = ∑_i ∂_i P_k * ∂_i P_k'` 
* `ddA[k] = ∑_i P_k''(x_i) = ΔA[k]`.
"""
function _assemble_A_dA_ddA(wf, X)
   TX = eltype(X)
   A = zeros(TX, length(X), length(wf.pooling))
   dA = zeros(TX, length(X), length(wf.pooling), length(wf.pooling))
   ddA = zeros(TX, length(X), length(wf.pooling))
   _assemble_A_dA_ddA!(A, dA, ddA, wf, X)
   return A, dA, ddA
end

# TODO: to do this more elegantly we really need 
#       the jacobian of the pooling w.r.t. X

import ForwardDiff

function _assemble_A_dA_ddA!(A, dA, ddA, wf, X)
   nX = length(X) 
   spec_A = wf.pooling.spec
   P, dP, ddP = Polynomials4ML.evaluate_ed2(wf.polys, X)
   Si_ = zeros(nX, 2)
   Ai = zeros(length(wf.pooling))
   ∂Ai = zeros(eltype(P), length(wf.pooling), nX)
   for i = 1:nX # loop over orbital bases (which i becomes ∅)
      fill!(Si_, 0)
      onehot!(Si_, i)
      # ACEcore.evalpool!(Ai, wf.pooling, (P, Si_))
      Ai = ACEcore.evalpool(wf.pooling, (P, Si_))
      A[i, :] = Ai 
   
      for (iA, (k, σ)) in enumerate(spec_A)
         # jacobian ∂Ai
         # and laplacian ddA[i, :]
         ddA[i, iA] = sum(ddP[:, k] .* Si_[:, σ])
         ∂Ai[iA, :] = dP[:, k] .* Si_[:, σ]
      end

      # # a little test for debugging -> this passes now 
      # _Ai = ACEcore.evalpool(wf.pooling, (P, Si_))
      # _∂Ai = ForwardDiff.jacobian(X -> ACEcore.evalpool(wf.pooling, (wf.polys(X), Si_)), X)
      # @show Ai ≈ _Ai
      # @show ∂Ai ≈ _∂Ai

      # now the `dA` array actually contains something else: 
      #   dA[k, k'] = ∑_i ∂_i A_k * ∂_i A_k'
      dA[i, :, :] = ∂Ai * transpose(∂Ai)
   end
   return nothing       
end


function _laplacian_inner()
   # Δψ = Φ⁻ᵀ : ΔΦ - ∑ᵢ (Φ⁻ᵀ * Φᵢ)ᵀ : (Φ⁻ᵀ * Φᵢ)
   # where Φᵢ = ∂_{xi} Φ
   nX = length(X)
   
   # compute the n-correlations, the wf, and the first layer of derivatives 
   AA = ACEcore.evaluate(wf.corr, A)  # nX x length(wf.corr)
   Φ = wf.Φ 
   mul!(Φ, parent(AA), wf.W)
   Φ⁻ᵀ = transpose(pinv(Φ))

   # this is now the key component 
   ΔAA = zeros(nX, length(wf.corr))

   for iAA = 1:length(spec)
      



   end
end


# # ------------------ old codes 

# function _assemble_A_dA_ddA(pibasis, cfg)
#    B1p = pibasis.basis1p 
#    TX = typeof(cfg[1].x)
#    A = zeros(TX, length(B1p))
#    dA = zeros(TX, length(B1p), length(B1p))
#    ddA = zeros(TX, length(B1p))
#    _assemble_A_dA_ddA!(A, dA, ddA, B1p, cfg) 
#    return A, dA, ddA 
# end

# function _assemble_A_dA_ddA!(A, dA, ddA, B1p, cfg)
#    # this is a hack because we know a priori what the basis is.  .... 
#    # P = B1p["Pn"].basis
#    _deriv(y) = getproperty.(evaluate_d(B1p, y), :x)
#    _deriv(y, δx) = _deriv(El1dState(y.x+δx, y.σ))

#    for y in cfg
#       A[:] += evaluate(B1p, y)
#       dϕ = _deriv(y)
#       dA[:,:] += dϕ * dϕ'
#       dϕ_p = _deriv(y,  1e-4)
#       dϕ_m = _deriv(y, -1e-4)
#       ddA[:] += (dϕ_p - dϕ_m) / (2*1e-4)
#    end
#    return dA, ddA 
# end

# function _At(A, spec, iAA, t)
#    iAt = spec.iAA2iA[iAA, t]
#    return A[iAt], iAt
# end 

# _prodA(A, spec, iAA) = 
#       prod( _At(A, spec, iAA, t)[1] for t = 1:spec.orders[iAA];
#             init = one(eltype(A)) )

# function _laplacian_inner(spec, c, A, dA, ddA)
#    Δ = 0.0
#    for iAA = 1:length(spec)
#       ord = spec.orders[iAA]

#       if ord == 1
#          ddA1 = ddA[ spec.iAA2iA[iAA, 1] ]
#          Δ += c[iAA] * ddA1
#          continue
#       end

#       if ord == 2 
#          A1, iA1 = _At(A, spec, iAA, 1)     
#          A2, iA2 = _At(A, spec, iAA, 2)
#          Δ += c[iAA] * ( A1 * ddA[iA2] + ddA[iA1] * A2 + 2 * dA[iA1, iA2] )
#          continue 
#       end

#       # compute the product basis function 
#       aa = _prodA(A, spec, iAA)

#       # now compute the back-prop       
#       for t = 1:ord
#          At, iAt = _At(A, spec, iAA, t)
#          AA_t = sign(At) * aa / (abs(At) + eps())
#          Δ += c[iAA] * AA_t * ddA[iAt]

#          for s = t+1:ord 
#             As, iAs = _At(A, spec, iAA, s)
#             AA_ts = sign(As) * AA_t / (abs(As) + eps())
#             Δ += 2 * c[iAA] * AA_ts * dA[iAt, iAs]
#          end
#       end
#    end

#    return Δ
# end

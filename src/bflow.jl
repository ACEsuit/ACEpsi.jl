
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
   spec::Vector{Vector{Int64}} # TODO: this needs to be remove
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
   spec1p = [ (k, σ) for σ in [1, 2, 3] for k in 1:K]  # (1, 2, 3) = (∅, ↑, ↓);
   spec1p = sort(spec1p, by = b -> b[1]) # sorting to prevent gensparse being confused
   
   pooling = PooledSparseProduct(spec1p)
   # generate the many-particle spec 
   tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
   admissible = bb -> (length(bb) == 0) || (sum(b[1] - 1 for b in bb ) <= totdeg)
   
   specAA = gensparse(; NU = ν, tup2b = tup2b, admissible = admissible,
                        minvv = fill(0, ν), 
                        maxvv = fill(length(spec1p), ν), 
                        ordered = true)
   
   spec = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]
   
   corr1 = SparseSymmProd(spec; T = Float64)
   @show length(corr1)
   corr = corr1.dag   

   # initial guess for weights 
   Q, _ = qr(randn(T, length(corr), Nel))
   W = Matrix(Q) 
   return BFwf(polys, pooling, corr, W, spec,
                zeros(T, Nel, length(polys)), 
                zeros(T, Nel, length(polys)), 
                zeros(T, Nel, length(polys)), 
                zeros(T, Nel, Nel), 
                zeros(T, Nel, Nel),
                zeros(T, Nel, length(pooling)), 
                zeros(T, Nel, length(pooling)), 
                zeros(T, length(pooling)), 
                zeros(T, length(pooling)), 
                zeros(Bool, Nel, 3),
                zeros(T, Nel, length(corr)), 
                zeros(T, Nel, 3) )

end

"""
This function return correct Si for pooling operation.
"""
function onehot!(Si, i, Σ)
   Si .= 0
   for k = 1:length(Σ)
      Si[k, spin2num(Σ[k])] = 1
   end
   # each current electron to ϕ, also remove their contribution in the sum of ↑ or ↓ basis
   Si[i, 1] = 1 
   Si[i, 2] = 0
   Si[i, 3] = 0
end

"""
This function convert spin to corresponding integer value used in spec
"""
function spin2num(σ)
   if σ == '↑'
      return 2
   elseif σ == '↓'
      return 3
   elseif σ == '∅'
      return 1
   end
   error("illegal spin char")
end

"""
This function convert num to corresponding spin string.
"""
function num2spin(σ)
   if σ == 2
      return '↑'
   elseif σ == 3
      return '↓'
   elseif σ == 1
      return '∅'
   end
   error("illegal integer value for num2spin")
end

"""
This function return a nice version of spec.
"""
function displayspec(spec, spec1p)
   _getnicespec = l -> (l[1], num2spin(l[2]))
   nicespec = []
   for k = 1:length(spec)
      push!(nicespec, _getnicespec.([spec1p[spec[k][j]] for j in length(spec[k])]))
   end
   return nicespec
end

function evaluate(wf::BFwf, X::AbstractVector, Σ, Pnn=nothing)
      
   nX = length(X)
   # position embedding 
   P = wf.P 
   evaluate!(P, wf.polys, X)    # nX x dim(polys)
   
   
   A = wf.A    # zeros(nX, length(wf.pooling)) 
   Ai = wf.Ai  # zeros(length(wf.pooling))
   Si = wf.Si  # zeros(Bool, nX, 2)

   for i = 1:nX 
      onehot!(Si, i, Σ)
      ACEcore.evalpool!(Ai, wf.pooling, (parent(P), Si))
      A[i, :] .= Ai
   end

   AA = ACEcore.evaluate(wf.corr, A)  # nX x length(wf.corr)
   
   # the only basis to be purified are those with same spin
   # scan through all corr basis, if they comes from same spin, remove self interation by using basis 
   # from same spin
   # first we have to construct coefficent for basis coming from same spin, that is in other words the coefficent
   # matrix of the original polynomial basis, this will be pass from the argument Pnn
   # === purification goes here === #
   
   # === #

   Φ = wf.Φ
   mul!(Φ, parent(AA), wf.W)
   release!(AA)
   return logabsdet(Φ)[1]
end

struct ZeroNoEffect end 
Base.size(::ZeroNoEffect, ::Integer) = Inf
Base.setindex!(A::ZeroNoEffect, args...) = nothing
Base.getindex(A::ZeroNoEffect, args...) = Bool(0)


function gradient(wf::BFwf, X, Σ)
   nX = length(X)

   # ------ forward pass  ----- 

   # position embedding (forward-mode)
   # here we evaluate and differentiate at the same time, which is cheap
   P = wf.P 
   dP = wf.dP
   Polynomials4ML.evaluate_ed!(P, dP, wf.polys, X)
   
   # no gradients here - need to somehow "constify" this vector 
   # could use other packages for inspiration ... 

   # pooling : need an elegant way to shift this loop into a kernel!
   #           e.g. by passing output indices to the pooling function.

   A = wf.A    # zeros(nX, length(wf.pooling)) 
   Ai = wf.Ai  # zeros(length(wf.pooling))
   Si = wf.Si  # zeros(Bool, nX, 2)
   
   for i = 1:nX 
      onehot!(Si, i, Σ)
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
   Si_ = zeros(nX, 3)
   for i = 1:nX 
      onehot!(Si_, i, Σ)
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
   
end 

"""
This will compute the following: 
* `A` just the normal pooling operation, `A[k] = ∑_i P_k(x_i)`
* `dA[k, k'] = ∑_i ∂_i P_k * ∂_i P_k'` 
* `ddA[k] = ∑_i P_k''(x_i) = ΔA[k]`.
"""
function _assemble_A_dA_ddA(polys, X)
   TX = eltype(X)
   A = zeros(TX, length(polys))
   dA = zeros(TX, length(polys), length(polys))
   ddA = zeros(TX, length(polys))
   _assemble_A_dA_ddA!(A, dA, ddA, polys, X)
   return A, dA, ddA
end

function _assemble_A_dA_ddA!(A, dA, ddA, polys, X)
   P, dP, ddP = Polynomials4ML.evaluate_ed(polys, X)
   for i = 1:length(X) 
      A[:] += P[i, :]
      dA[:,:] += dP[i,:] * dP[i,:]'
      ddA[:] += ddP[i,:]
   end
   return nothing 
end


function _laplacian_inner(spec, c, A, dA, ddA)
   # Δψ = Φ⁻ᵀ : ΔΦ - ∑ᵢ (Φ⁻ᵀ * Φᵢ)ᵀ : (Φ⁻ᵀ * Φᵢ)
   # where Φᵢ = ∂_{xi} Φ
   Δ = 0.0


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

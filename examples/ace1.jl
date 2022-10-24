
using Polynomials4ML
using ACEcore 

##

module M

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

   function BFwf(Nel::Integer, polys; 
                        ν = 3, T = Float64)
      # 1-particle spec 
      K = length(polys)
      spec1p = [ (k, σ) for σ in [1, 2] for k in 1:K ]
      pooling = PooledSparseProduct(spec1p)

      # generate the many-particle spec 
      tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
      admissible = bb -> (length(bb) == 0) || (sum( b[1] for b in bb ) <= 7 )
      
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


end


##

Nel = 5
polys = legendre_basis(15)
wf = M.BFwf(Nel, polys; ν=4)

X = 2 * rand(Nel) .- 1
wf(X)
g = M.gradient(wf, X)

##

using ACEbase.Testing: fdtest 

fdtest(wf, X -> M.gradient(wf, X), X)


##
using BenchmarkTools
@btime M.evaluate($wf, $X)
@btime M.gradient($wf, $X)

##

@profview let wf=wf, X=X 
   for n = 1:50_000 
      M.gradient(wf, X)
   end
end

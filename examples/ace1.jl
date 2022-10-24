
using Polynomials4ML
using ACEcore 

##

module M

   using ACEcore, Polynomials4ML
   using Polynomials4ML: OrthPolyBasis1D3T
   using ACEcore: PooledSparseProduct, SparseSymmProdDAG, SparseSymmProd
   using ACEcore.Utils: gensparse
   using LinearAlgebra: qr, I, logabsdet, pinv


   struct BFwf{T, TPOLY}
      polys::TPOLY
      pooling::PooledSparseProduct{2}
      corr::SparseSymmProdDAG{T}
      W::Matrix{T}
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

      return BFwf(polys, pooling, corr, W)
   end


   function onehot!(Si, i)
      Si[:, 1] .= 0 
      Si[:, 2] .= 1 
      Si[i, 1] = 1 
      Si[i, 2] = 0
   end

   function orbitals(wf::BFwf, X::AbstractVector)
      nX = length(X)
      
      # position embedding 
      P = wf.polys(X) # nX x dim(polys)
      
      # one-hot embedding - generalize to ∅, ↑, ↓
      A = zeros(nX, length(wf.pooling)) 
      Ai = zeros(length(wf.pooling))
      Si = zeros(Bool, nX, 2)
      for i = 1:nX 
         onehot!(Si, i)
         ACEcore.evalpool!(Ai, wf.pooling, (parent(P), Si))
         A[i, :] .= Ai
      end

      AA = ACEcore.evaluate(wf.corr, A)  # nX x length(wf.corr)
      return parent(AA) * wf.W
   end


   evaluate(wf::BFwf, X) = logabsdet(orbitals(wf, X))[1]

   struct ZeroNoEffect end 
   Base.size(::ZeroNoEffect, ::Integer) = Inf
   Base.setindex!(A::ZeroNoEffect, args...) = nothing
   Base.getindex(A::ZeroNoEffect, args...) = Bool(0)


   function gradient(wf::BFwf, X)
      nX = length(X)

      # ------ forward pass  ----- 

      # position embedding (forward-mode)
      # here we evaluate and differentiate at the same time, which is cheap
      P, dP = Polynomials4ML.evaluate_ed(wf.polys, X)
      
      # one-hot embedding - TODO: generalize to ∅, ↑, ↓
      # no gradients here - need to somehow "constify" this vector 
      # could use other packages for inspiration ... 
      S = zeros(Bool, nX, 2)

      # pooling : need an elegant way to shift this loop into a kernel!
      #           e.g. by passing output indices to the pooling function.
      A = zeros(nX, length(wf.pooling)) 
      Ai = zeros(length(wf.pooling))
      Si = zeros(Bool, nX, 2)
      for i = 1:nX 
         onehot!(Si, i)
         ACEcore.evalpool!(Ai, wf.pooling, (parent(P), Si))
         A[i, :] .= Ai
      end

      # n-correlations 
      AA = ACEcore.evaluate(wf.corr, A)  # nX x length(wf.corr)

      # generalized orbitals 
      Φ = parent(AA) * wf.W

      # and finally the wave function 
      ψ = logabsdet(Φ)[1]

      # ------ backward pass ------
      #∂Φ = ∂ψ / ∂Φ = Φ⁻ᵀ
      ∂Φ = pinv(Φ)'

      # ∂AA = ∂ψ/∂AA = ∂ψ/∂Φ * ∂Φ/∂AA = ∂Φ * wf.W'
      ∂AA =  ∂Φ * wf.W'

      # ∂A = ∂ψ/∂A = ∂ψ/∂AA * ∂AA/∂A -> use custom pullback
      ∂A = zeros(size(A))
      ACEcore.pullback_arg!(∂A, ∂AA, wf.corr, AA)

      # ∂P = ∂ψ/∂P = ∂ψ/∂A * ∂A/∂P -> use custom pullback 
      # but need to do some work here since multiple 
      # pullbacks can be combined here into a single one maybe? 
      ∂P = zeros(size(P))
      ∂Pi = zeros(size(P))
      ∂Si = zeros(size(Si))   # should use ZeroNoEffect here ?!??!
      for i = 1:nX 
         onehot!(Si, i)
         fill!(∂Pi, 0)
         ACEcore._pullback_evalpool!((∂Pi, ∂Si), ∂A[i, :], wf.pooling, (P, Si))
         ∂P += ∂Pi
      end

      # ∂X = ∂ψ/∂X = ∂ψ/∂P * ∂P/∂X 
      #   here we can now finally employ the dP=∂P/∂X that we already know.
      # ∂ψ/∂Xi = ∑_k ∂ψ/∂Pik * ∂Pik/∂Xi
      #        = ∑_k ∂P[i, k] * dP[i, k]
      g = sum(∂P .* dP, dims = 2)[:]
      return g
   end


end


##

Nel = 5
polys = legendre_basis(8)
wf = M.BFwf(Nel, polys)

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

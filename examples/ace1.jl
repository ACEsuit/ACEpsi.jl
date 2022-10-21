
using Polynomials4ML
using ACEcore 

##

module M

   using ACEcore
   using Polynomials4ML: OrthPolyBasis1D3T
   using ACEcore: PooledSparseProduct, SparseSymmProdDAG, SparseSymmProd
   using ACEcore.Utils: gensparse
   using LinearAlgebra: qr, I


   struct BFOrbs{T, TPOLY}
      polys::TPOLY
      pooling::PooledSparseProduct{2}
      corr::SparseSymmProdDAG{T}
      W::Matrix{T}
   end

   (Φ::BFOrbs)(args...) = evaluate(Φ, args...)

   function BFOrbs(Nel::Integer, polys; 
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

      return BFOrbs(polys, pooling, corr, W)
   end


   function onehot!(Si, i)
      Si[:, 1] .= 0 
      Si[:, 2] .= 1 
      Si[i, 1] = 1 
      Si[i, 2] = 0
   end

   function evaluate(orb::BFOrbs, X::AbstractVector)
      nX = length(X)
      
      # position embedding 
      P = orb.polys(X) # nX x dim(polys)
      
      # one-hot embedding - generalize to ∅, ↑, ↓
      S = Matrix{Bool}(I, (nX, nX))

      A = zeros(nX, length(orb.pooling)) 
      Si = zeros(Bool, nX, 2)
      for i = 1:nX 
         onehot!(Si, i)
         Ai = ACEcore.evaluate(orb.pooling, (parent(P), Si))
         A[i, :] .= parent(Ai)
      end

      AA = ACEcore.evaluate(orb.corr, A)  # nX x length(orb.corr)
      return parent(AA) * orb.W
   end
end


Nel = 5
polys = legendre_basis(8)
orb = M.BFOrbs(Nel, polys)

X = 2 * rand(Nel) .- 1
orb(X)


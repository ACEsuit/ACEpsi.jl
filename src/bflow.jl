using ACEcore, Polynomials4ML
using Polynomials4ML: OrthPolyBasis1D3T
using ACEcore: PooledSparseProduct, SparseSymmProdDAG, SparseSymmProd, release!
using ACEcore.Utils: gensparse
using LinearAlgebra: qr, I, logabsdet, pinv, mul!, dot , tr 
using Distributions: Uniform
import ForwardDiff
using SparseArrays: SparseVector, sparse, spzeros, SparseMatrixCSC


mutable struct BFwf{T, TT, TPOLY, TE}
   trans::TT
   polys::TPOLY
   pooling::PooledSparseProduct{2}
   corr::SparseSymmProdDAG{T}
   W::Matrix{T}
   envelope::TE
   spec::Vector{Vector{Int64}} # corr.spec TODO: this needs to be remove
   C::SparseMatrixCSC{T} # purification operator
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
                     sd_admissible = bb -> (true),
                     envelope = envelopefcn(x -> sqrt(1 + x^2), rand()),
                     purify = false)
   # # 1-particle spec
   # if sd_admissible != (bb -> (true)) && purify == true
   #    @info("adding extra admissible requirement, take care of purification")
   # end

   K = length(polys)
   spec1p = [ (k, σ) for σ in [1, 2, 3] for k in 1:K]  # (1, 2, 3) = (∅, ↑, ↓);
   spec1p = sort(spec1p, by = b -> b[1]) # sorting to prevent gensparse being confused
   
   pooling = PooledSparseProduct(spec1p)
   # generate the many-particle spec 
   tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
   # default_admissible = bb -> (length(bb) == 0) || (sum(b[1] - 1 for b in bb ) <= totdeg)
   default_admissible = bb -> (length(bb) == 0) || (sum(b[1] - 1 for b in bb ) < totdeg)
   
   specAA = gensparse(; NU = ν, tup2b = tup2b, admissible = default_admissible,
                        minvv = fill(0, ν), 
                        maxvv = fill(length(spec1p), ν), 
                        ordered = true)
   
   
   spec = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]

   # further restrict
   spec = [t for t in spec if sd_admissible([spec1p[t[j]] for j = 1:length(t)])]

   corr1 = SparseSymmProd(spec; T = Float64)
   corr = corr1.dag   

   # initial guess for weights 
   Q, _ = qr(randn(T, length(corr), Nel))
   W = Matrix(Q) 
   
   C = zeros(length(spec), length(spec))

   if purify == false
      C = sparse(Matrix(1.0I, length(spec), length(spec)))
   else
      C = generalImpure2PureMap(spec, spec1p, polys, ν)
   end

   return BFwf(trans, polys, pooling, corr, W, envelope, spec, C,
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
                  zeros(T, Nel, 3), 
                  zeros(T, Nel, Nel, length(corr)) )

end

"""
This function return correct Si for pooling operation.
"""
function onehot!(Si, i, Σ)
   Si .= 0
   for k = 1:length(Σ)
      Si[k, spin2num(Σ[k])] = 1
   end
   # set current electron to ϕ, also remove their contribution in the sum of ↑ or ↓ basis
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
   error("illegal spin char for spin2num")
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
This function returns a nice version of spec.
"""
function displayspec(wf::BFwf)
   K = length(wf.polys)
   spec1p = [ (k, σ) for σ in [1, 2, 3] for k in 1:K]
   spec1p = sort(spec1p, by = b -> b[1])
   _getnicespec = l -> (l[1], num2spin(l[2]))
   nicespec = []
   for k = 1:length(wf.spec)
      push!(nicespec, _getnicespec.([spec1p[wf.spec[k][j]] for j = 1:length(wf.spec[k])]))
   end
   return nicespec
end


function assemble_A(wf::BFwf, X::AbstractVector, Σ)
      
   nX = length(X)
   # position embedding 
   P = wf.P 
   Xt = wf.trans.(X)
   evaluate!(P, wf.polys, Xt)    # nX x dim(polys)
   
   A = wf.A    # zeros(nX, length(wf.pooling)) 
   Ai = wf.Ai  # zeros(length(wf.pooling))
   Si = wf.Si  # zeros(Bool, nX, 2)

   for i = 1:nX 
      onehot!(Si, i, Σ)
      ACEcore.evalpool!(Ai, wf.pooling, (parent(P), Si))
      A[i, :] .= Ai
   end
   return A
end

function evaluate(wf::BFwf, X::AbstractVector, Σ, Pnn=nothing)
   nX = length(X)
   A = assemble_A(wf, X, Σ)
   AA = ACEcore.evaluate(wf.corr, A)  # nX x length(wf.corr)
   
   # purify the product basis
   # AA = AA * (wf.C)'
   Φ = wf.Φ
   mul!(Φ, parent(AA), (wf.C)' * wf.W) # nX x nX
   Φ = Φ .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX] # the resulting matrix should contains two block each comes from each spin
   release!(AA)

   env = wf.envelope(X)
   return 2 * logabsdet(Φ)[1] + 2 * log(abs(env))
end


function gradp_evaluate(wf::BFwf, X::AbstractVector, Σ)
   nX = length(X)
   
   A = assemble_A(wf, X, Σ)
   AA = ACEcore.evaluate(wf.corr, A)  # nX x length(wf.corr)
   
   Φ = wf.Φ 

   mul!(Φ, parent(AA) * (wf.C)', wf.W)
   # mul!(Φ, parent(AA), (wf.C)' * wf.W)
   Φ = Φ .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX] # the resulting matrix should contains two block each comes from each spin


   # ψ = log | det( Φ ) |
   # ∂Φ = ∂ψ/∂Φ = Φ⁻ᵀ
   ∂Φ = transpose(pinv(Φ))

   # ∂W = ∂ψ / ∂W = ∂Φ * ∂_W( AA * W ) = ∂Φ * AA
   # ∂Wij = ∑_ab ∂Φab * ∂_Wij( ∑_k AA_ak W_kb )
   #      = ∑_ab ∂Φab * ∑_k δ_ik δ_bj  AA_ak
   #      = ∑_a ∂Φaj AA_ai = ∂Φaj' * AA_ai
   ∇p = transpose(parent(AA) * (wf.C)') * ∂Φ
   
   # ∇p = wf.C * (transpose(parent(AA)) * ∂Φ)

   release!(AA)
   ∇p = ∇p * 2


   # ------ gradient of env (w.r.t. ξ) ----- 
   # ∂ = ∂/∂ξ
   # r = ||x||
   # ∂(2 * logabs(env)) = ∂(2 * log(exp(-ξf(r)))) = ∂(-2ξf(r)) = -f(r)
   ∇logabsenv = - 2 * wf.envelope.f(norm(X))

   return (∇p = ∇p, ∇logabsenv = [∇logabsenv]) # TODO: return a named tuple (W = gradp, D = gradient w.r.t parameter of env)
end



# ----------------------- gradient 

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
   Xt = wf.trans.(X)
   Polynomials4ML.evaluate_ed!(P, dP, wf.polys, Xt)
   ∂Xt = ForwardDiff.derivative.(Ref(x -> wf.trans(x)), X)
   @inbounds for k = 1:size(dP, 2)
      @simd ivdep for i = 1:nX
         dP[i, k] *= ∂Xt[i]
      end
   end
   
   # no gradients here - need to somehow "constify" this vector 
   # could use other packages for inspiration ... 

   # pooling : need an elegant way to shift this loop into a kernel!
   #           e.g. by passing output indices to the pooling function.

   A = wf.A    # zeros(nX, length(wf.pooling)) 
   Ai = wf.Ai  # zeros(length(wf.pooling))
   Si = wf.Si  # zeros(Bool, nX, 3)
   
   for i = 1:nX 
      onehot!(Si, i, Σ)
      ACEcore.evalpool!(Ai, wf.pooling, (parent(P), Si))
      A[i, :] .= Ai
   end
   
   # n-correlations 
   AA = ACEcore.evaluate(wf.corr, A)  # nX x length(wf.corr)
   AA = AA * (wf.C)'

   # generalized orbitals 
   Φ = wf.Φ
   @assert (parent(AA) == AA)

   mul!(Φ, parent(AA), wf.W)

   # the resulting matrix should contains two block each comes from each spin
   Φ = Φ .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX]
   
   # envelope 
   env = wf.envelope(X)

   # and finally the wave function 
   # ψ = logabsdet(Φ)[1] + log(abs(env))

   # ------ backward pass ------
   #∂Φ = ∂ψ / ∂Φ = Φ⁻ᵀ
   ∂Φ = transpose(pinv(Φ))

   # ∂AA = ∂ψ/∂AA = ∂ψ/∂Φ * ∂Φ/∂AA = ∂Φ * wf.W'
   ∂AA = wf.∂AA
   mul!(∂AA, ∂Φ, transpose(transpose(wf.C) * wf.W))

   # ∂A = ∂ψ/∂A = ∂ψ/∂AA * ∂AA/∂A -> use custom pullback
   ∂A = wf.∂A   # zeros(size(A))
   ACEcore.pullback_arg!(∂A, ∂AA, wf.corr, parent(AA) * inv(Matrix(wf.C)'))
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

   # envelope 
   ∇env = ForwardDiff.gradient(wf.envelope, X)
   g += ∇env / env 
   g = g * 2
   return g
end


# ------------------ Laplacian implementation 

function laplacian(wf::BFwf, X, Σ)

   A, ∇A, ΔA = _assemble_A_∇A_ΔA(wf, X, Σ)
   AA, ∇AA, ΔAA = _assemble_AA_∇AA_ΔAA(A, ∇A, ΔA, wf)

   Δψ = _laplacian_inner(AA, ∇AA, ΔAA, wf, Σ)

   # envelope 
   env = wf.envelope(X)
   ∇env = ForwardDiff.gradient(wf.envelope, X)
   Δenv = tr(ForwardDiff.hessian(wf.envelope, X))

   # Δ(ln(env)) = Δenv / env - ∇env ⋅ ∇env / env ^ 2
   Δψ += Δenv / env - dot(∇env, ∇env) / env^2
   Δψ = Δψ * 2
   return Δψ
end 

function _assemble_A_∇A_ΔA(wf::BFwf, X, Σ)
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

   Si_ = zeros(nX, 3)
   Ai = zeros(length(wf.pooling))
   @inbounds for i = 1:nX # loop over orbital bases (which i becomes ∅)
      fill!(Si_, 0)
      onehot!(Si_, i, Σ)
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

function _assemble_AA_∇AA_ΔAA(A, ∇A, ΔA, wf::BFwf)
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


function _laplacian_inner(AA, ∇AA, ΔAA, wf::BFwf, Σ)

   # Δψ = Φ⁻ᵀ : ΔΦ - ∑ᵢ (Φ⁻ᵀ * Φᵢ)ᵀ : (Φ⁻ᵀ * Φᵢ)
   # where Φᵢ = ∂_{xi} Φ

   nX = size(AA, 1)
   
   # the wf, and the first layer of derivatives 
   Φ = wf.Φ 
   mul!(Φ, parent(AA) * (wf.C)', wf.W)
   Φ = Φ .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX] # the resulting matrix should contains two block each comes from each spin
   Φ⁻ᵀ = transpose(pinv(Φ))
   
   # first contribution to the laplacian
   ΔΦ = ΔAA * (wf.C)' * wf.W
   ΔΦ = ΔΦ .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX] # the resulting matrix should contains two block each comes from each spin

   Δψ = dot(Φ⁻ᵀ, ΔΦ)
   
   # the gradient contribution 
   # TODO: we can rework this into a single BLAS3 call
   # which will also give us a single back-propagation 
   # ∇Φi = zeros(nX, nX)
   ∇Φ_all = reshape(reshape(∇AA, nX*nX, :) * (wf.C)' * wf.W, nX, nX, nX)
   Φ⁻¹∇Φi = zeros(nX, nX)
   for i = 1:nX 
      ∇Φi = ∇Φ_all[i, :, :] .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX]
      mul!(Φ⁻¹∇Φi, transpose(Φ⁻ᵀ), ∇Φi)
      Δψ -= dot(transpose(Φ⁻¹∇Φi), Φ⁻¹∇Φi)
   end
   
   return Δψ
end


# ------------------ gradp of Laplacian  


function gradp_laplacian(wf::BFwf, X, Σ)


   # ---- gradp of Laplacian of Ψ ----

   nX = length(X) 

   A, ∇A, ΔA = _assemble_A_∇A_ΔA(wf, X, Σ)
   AA, ∇AA, ΔAA = _assemble_AA_∇AA_ΔAA(A, ∇A, ΔA, wf)

   
   # the wf, and the first layer of derivatives 
   Φ = wf.Φ 
   mul!(Φ, parent(AA) * (wf.C)', wf.W)
   Φ = Φ .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX] # the resulting matrix should contains two block each comes from each spin

   Φ⁻¹ = pinv(Φ)
   Φ⁻ᵀ = transpose(Φ⁻¹)

   # first contribution to the laplacian
   ΔΦ = ΔAA * (wf.C)' * wf.W
   ΔΦ = ΔΦ .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX] # the resulting matrix should contains two block each comes from each spin
 
   # Δψ += dot(Φ⁻ᵀ, ΔΦ) ... this leads to the next two terms 
   ∂ΔΦ = Φ⁻ᵀ
   ∇Δψ = transpose(ΔAA * (wf.C)') * ∂ΔΦ
   
   ∂Φ = - Φ⁻ᵀ * transpose(ΔΦ) * Φ⁻ᵀ
   # ∇Δψ += transpose(AA) * ∂Φ


   # the gradient contribution 
   # TODO: we can rework this into a single BLAS3 call
   # which will also give us a single back-propagation 
   # ∇Φi = zeros(nX, nX)
   # Φ⁻¹∇Φi = zeros(nX, nX)
   ∇Φ_all = reshape(reshape(∇AA, nX*nX, :) * (wf.C)' * wf.W, nX, nX, nX)
   ∂∇Φ_all = zeros(nX, nX, nX)

   for i = 1:nX 
      ∇Φi = ∇Φ_all[i, :, :] .* [Σ[i] == Σ[j] for j = 1:nX, i = 1:nX]
      ∇Φiᵀ = transpose(∇Φi)
      # Δψ += - dot( [Φ⁻¹∇Φi]ᵀ, Φ⁻¹∇Φi )

      # ∂∇Φi = - 2 * Φ⁻ᵀ * ∇Φiᵀ * Φ⁻ᵀ
      ∂∇Φ_all[i, :, :] = - 2 * Φ⁻ᵀ * ∇Φiᵀ * Φ⁻ᵀ
      # ∇Δψ += transpose(∇AA[i, :, :]) * ∂∇Φi

      ∂Φ += 2 * Φ⁻ᵀ * ∇Φiᵀ * Φ⁻ᵀ * ∇Φiᵀ * Φ⁻ᵀ
      # ∇Δψ += transpose(AA) * ∂Φ
   end

   ∇Δψ += transpose(AA * (wf.C)') * ∂Φ

   ∇Δψ += reshape( transpose(reshape(∇AA, nX*nX, :) * (wf.C)') * reshape(∂∇Φ_all, nX*nX, nX), 
                  size(∇Δψ) )


   # ---- gradp of Laplacian of env ----
   # ∂ = ∂/∂ξ  
   # r = ||x||
   # ∂(2 * Δ(logabs(env))) = ∂(2 * Δ(-ξf(r)) = -2 * Δf(r)
   # Δf(r) = (n-1)/r f'(r) + f''(r)
   r = norm(X)
   f(r) = wf.envelope.f(r)
   df(r) = ForwardDiff.derivative(f, r)
   ddf(r) = ForwardDiff.derivative(r -> df(r), r)
   Δf = ddf(r) + (length(X)-1) * df(r)/r

   return (∇Δψ = 2 * ∇Δψ, ∇Δlogabsenv = [2 * -Δf])
end 



# ----------------- BFwf parameter wraging

function get_params(U::BFwf)
   return (U.W, U.envelope.ξ)
end

function set_params!(U::BFwf, para)
   U.W = para[1]
   set_params!(U.envelope, para[2])
   return U
end

function Scaling(U::BFwf, γ::Float64)
   c = get_params(U)
   uu = []
   _spec = U.spec
   for i = 1:length(_spec)
      push!(u, sum(_spec[i] .^ 2))
   end
   return (uu = γ * uu .* c[1], d = zeros(length(c[2])))
end

# ----------------- purification
function _getκσIdx(spec1p, κ, σ)
   for (i, bb) in enumerate(spec1p)
      if (κ, σ) == bb
         return i
      end
   end
   @error("such (κ, σ) not found in spec1p")
end

function spec2col(NNi, NN)
   for k in eachindex(NN)
       if NN[k] == NNi
           return k
       end
   end
   # @error("such NNi not found in NN")
   return nothing
end

function P_kappa_prod_coeffs(poly, NN, tol = 1e-10)
   L = 5000
#  sample_points = chev_nodes(L)
   NN23b = NN[length.(NN) .<= 2]
   sample_points = rand(Uniform(-1, 1), L)

   RR = poly(sample_points)
   
   qrF = qr(RR)
   
   Pnn = Dict{Vector{Int64}, SparseVector{Float64, Int64}}() # key: the index of correpsonding tuple in the NN list; value: SparseVector
   
   # solve RR * x - Rnn = 0 for each basis
   for nn in NN23b
      Rnn = RR[:, nn[1]]
      for t = 2:length(nn)
         Rnn = Rnn .* RR[:, nn[t]] # do product on the basis according to the tuple nn, would be the ground truth target in the least square problem
      end
      p_nn = map(p -> (abs(p) < tol ? 0.0 : p), qrF \ Rnn) # for each element p in qrF\Rnn, if p is < tol, set it to 0 for converting to sparse matrix
      @assert norm(RR * p_nn - Rnn, Inf) < tol
      Pnn[nn] = sparse(p_nn)
   end
   return Pnn
end

function C_kappa_prod_coeffs(spec, spec1p, poly)
   # construct Pκ coefficients according to maximum degree of spec1p
   #spec = wf.spec
   #spec1p = wf.pooling.spec
   #poly = wf.polys

   poly_max = maximum([bb[1] for bb in spec1p])
   spec1p_poly = [i for i = 1:poly_max]
   tup2b = vv -> [ spec1p_poly[v] for v in vv[vv .> 0]]
   admissible = bb -> (length(bb) == 0) || (sum(b[1] - 1 for b in bb ) < poly_max)

   # only product basis up to 2b is needed, use to get Pnn_all
   NN2b = gensparse(; NU = 2, tup2b = tup2b, admissible = admissible, minvv = fill(0, 2), maxvv = fill(length(spec1p_poly), 2), ordered = true)
   NN2b = [ vv[vv .> 0] for vv in NN2b if !(isempty(vv[vv .> 0]))]

   Pnn_all = P_kappa_prod_coeffs(poly, NN2b)

   # get coefficient for ϕ according to spec2b
   spec2b = spec[length.(spec) .== 2]
   Cnn_all = Dict{Vector{Int64}, SparseVector{Float64, Int64}}()
   for nσ in spec2b
      # get index in terms of spec1p
      idx1_1p, idx2_1p = spec1p[nσ[1]], spec1p[nσ[2]]
      # get the coefficient Pκ from Pnn_all
      Pκk1k2 = Pnn_all[[idx1_1p[1], idx2_1p[1]]] 
      # expand as new coefficients, C_nn should be a vector of length = number of spec1p
      C_nn = zeros(length(spec1p))
      
      # C_nn is non-zero only if σ1 == σ2
      if idx1_1p[2] == idx2_1p[2]
         for κ = 1:length(Pκk1k2.nzind)
            C_nn[_getκσIdx(spec1p, κ, idx1_1p[2])] = Pκk1k2.nzval[κ]
         end
      end
      Cnn_all[nσ] = sparse(C_nn)
      
   end
   return Cnn_all
end


function generalImpure2PureMap(spec, spec1p, poly, Remove)

   Pnn_all = C_kappa_prod_coeffs(spec, spec1p, poly)
   S = length(spec)
   C = spzeros(S, S) # transformation matrix
   max_ord = maximum(length.(spec))
   spec_len_list = zeros(Int64, max_ord) # a list for storing number of basis of ord <= i, where i is the index of the array
   for t in spec
      spec_len_list[length(t)] += 1
   end
   

   # corresponding to 2 and 3 body basis
   for i = 1:sum(spec_len_list[1:2])
      C[i, i] = 1.0
   end


   # Base case, adjust coefficient for 3 body (ν = 2)
   for i = spec_len_list[1] + 1:sum(spec_len_list[1:2])
      pnn = Pnn_all[spec[i]]
      for k = 1:length(pnn.nzind)
         C[i, pnn.nzind[k]] -= pnn.nzval[k]
      end
   end


   # for each order
   for ν = 3:Remove
      # for each of basis of order ν
      for i = sum(spec_len_list[1:ν-1]) + 1:sum(spec_len_list[1:ν])
         # adjusting coefficent for the term \matcal{A}_{k1...kN} * A_{N+1}
         # first we get the coefficient corresponding to purified basis of order ν - 1
         last_ip2pmap = C[spec2col(spec[i][1:end - 1], spec), :]
         for k = 1:length(last_ip2pmap.nzind) # for each of the coefficient in last_ip2pmap
            # first we get the corresponing specification correpsonding to (spec of last_ip2pmap[i], spec[end])
            target_spec = [t for t in spec[last_ip2pmap.nzind[k]]]
            push!(target_spec, spec[i][end])
            C[i, spec2col(sort(target_spec), spec)] += last_ip2pmap.nzval[k]
         end

         # adjusting coefficent for terms Σ^{ν -1}_{β = 1} P^{κ} \mathcal{A}
         P_κ_list  = [Pnn_all[[spec[i][j], spec[i][ν]]] for j = 1:ν - 1]
         for (idx, P_κ) in enumerate(P_κ_list)
            for k = 1:length(P_κ.nzind) # for each kappa, P_κ.nzind[k] == κ in the sum
               # first we get the coefficient corresponding to the \mathcal{A}
               pureA_spec = [spec[i][r] for r = 1:ν-1 if r != idx] # r!=idx since κ runs through the 'idx' the coordinate
               push!(pureA_spec, P_κ.nzind[k]) # add κ into the sum
               last_ip2pmap = C[spec2col(sort(pureA_spec), spec), :]
               C[i, :] -= P_κ.nzval[k] .* last_ip2pmap
            end
         end
      end
   end

   return C
end
using Polynomials4ML, ForwardDiff

const NLM{T} = NamedTuple{(:n, :l, :m), Tuple{T, T, T}}
const NL{T} = NamedTuple{(:n, :l), Tuple{T, T}}

struct RnlExample{TP, TI}
   Pn::TP
   spec::Vector{NL{TI}}
end

Base.length(basis::RnlExample) = length(basis.spec)


# -------- Evaluation Code 

_alloc(basis::RnlExample, r::T) where {T <: Real} = 
      zeros(T, length(Rnl))

_alloc(basis::RnlExample, rr::Vector{T}) where {T <: Real} = 
      zeros(T, length(rr), length(basis))

      
evaluate(basis::RnlExample, r::Number) = evaluate(basis, [r,])[:]

function evaluate(basis::RnlExample, R::AbstractVector)
   nR = length(R) 
   Pn = Polynomials4ML.evaluate(basis.Pn, R)
   Rnl = _alloc(basis, R)

   maxL = maximum(b.l for b in basis.spec)
   rL = ones(eltype(R), length(R), maxL+1)

   # @inbounds begin 

   for l = 1:maxL 
      # @simd ivdep 
      for j = 1:nR 
         rL[j, l+1] = R[j] * rL[j, l] 
      end
   end

   for (i, b) in enumerate(basis.spec)
      # @simd ivdep 
      for j = 1:nR
         Rnl[j, i] = Pn[j, b.n] * rL[j, b.l+1]
      end
   end

   # end

   return Rnl 
end


function evaluate_ed(basis::RnlExample, R)
   nR = length(R) 
   Pn, dPn = Polynomials4ML.evaluate_ed(basis.Pn, R)
   Rnl = _alloc(basis, R)
   dRnl = _alloc(basis, R)

   maxL = maximum(b.l for b in basis.spec)
   rL = ones(eltype(R), length(R), maxL+1)
   drL = zeros(eltype(R), length(R), maxL+1)
   for l = 1:maxL 
      # @simd ivdep 
      for j = 1:nR 
         rL[j, l+1] = R[j] * rL[j, l] 
         drL[j, l+1] = l * rL[j, l] 
      end
   end

   for (i, b) in enumerate(basis.spec)
      # @simd ivdep 
      for j = 1:nR
         Rnl[j, i] = Pn[j, b.n] * rL[j, b.l+1]
         dRnl[j, i] = dPn[j, b.n] * rL[j, b.l+1] + Pn[j, b.n] * drL[j, b.l+1]
      end
   end

   return Rnl, dRnl 
end


function evaluate_ed2(basis::RnlExample, R)
   nR = length(R) 
   Pn, dPn, ddPn = Polynomials4ML.evaluate_ed2(basis.Pn, R)
   Rnl = _alloc(basis, R)
   dRnl = _alloc(basis, R)
   ddRnl = _alloc(basis, R)

   maxL = maximum(b.l for b in basis.spec)
   rL = ones(eltype(R), length(R), maxL+1)
   drL = zeros(eltype(R), length(R), maxL+1)
   ddrL = zeros(eltype(R), length(R), maxL+1)
   for l = 1:maxL 
      # @simd ivdep 
      for j = 1:nR 
         rL[j, l+1] = R[j] * rL[j, l]  # r^l
         drL[j, l+1] = l * rL[j, l]    # l * r^(l-1)
         ddrL[j, l+1] = l * drL[j, l] # (l-1) * drL[j, l]   # l * (l-1) * r^(l-2)
      end
   end

   for (i, b) in enumerate(basis.spec)
      # @simd ivdep 
      for j = 1:nR
         Rnl[j, i] = Pn[j, b.n] * rL[j, b.l+1]
         dRnl[j, i] = dPn[j, b.n] * rL[j, b.l+1] + Pn[j, b.n] * drL[j, b.l+1]
         ddRnl[j, i] = (      ddPn[j, b.n] *   rL[j, b.l+1]  
                         + 2 * dPn[j, b.n] *  drL[j, b.l+1] 
                         +      Pn[j, b.n] * ddrL[j, b.l+1] )
      end
   end

   return Rnl, dRnl, ddRnl
end

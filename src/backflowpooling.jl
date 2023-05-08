using ACEpsi.AtomicOrbitals: AtomicOrbitalsBasis

mutable struct BackflowPooling
   basis::AtomicOrbitalsBasis
end

function evaluate(pooling::BackflowPooling, ϕnlm, Σ)
   basis = pooling.basis
   nuc = basis.nuclei 
   Nnuc = length(nuc)
   Nnlm = length(basis.prodbasis.sparsebasis.spec) 
   Nel = length(Σ)
   T = promote_type(eltype(ϕnlm))

    # evaluate the pooling operation
   #                spin  I    k = (nlm) 

   Aall = zeros(T, (2, Nnuc, Nnlm))
   for k = 1:Nnlm
      for i = 1:Nel 
         iσ = spin2idx(Σ[i])
         for I = 1:Nnuc 
            Aall[iσ, I, k] += ϕnlm[I, i, k]
         end
      end
   end

   # now correct the pooling Aall and write into A^(i)
   # with do it with i leading so that the N-correlations can 
   # be parallelized over i 
   #
   # A[i, :] = A-basis for electron i, with channels, s, I, k=nlm 
   # A[i, ∅, I, k] = ϕnlm[I, i, k]
   # for σ = ↑ or ↓ we have
   # A[i, σ, I, k] = ∑_{j ≂̸ i : Σ[j] == σ}  ϕnlm[I, j, k]
   #               = ∑_{j : Σ[j] == σ}  ϕnlm[I, j, k] - (Σ[i] == σ) * ϕnlm[I, i, k]
   #
   #
   # TODO: discuss - this could be stored much more efficiently as a 
   #       lazy array. Could that have advantages? 
   #

   @assert spin2idx(↑) == 1
   @assert spin2idx(↓) == 2
   @assert spin2idx(∅) == 3
   A = zeros(T, ((Nel, 3, Nnuc, Nnlm)))
   for k = 1:Nnlm 
      for I = 1:Nnuc 
         for i = 1:Nel             
            A[i, 3, I, k] = ϕnlm[I, i, k]
         end
         for iσ = 1:2 
            σ = idx2spin(iσ)
            for i = 1:Nel 
               A[i, iσ, I, k] = Aall[iσ, I, k] - (Σ[i] == σ) * ϕnlm[I, i, k]
            end
         end
      end
   end

   return A 
end








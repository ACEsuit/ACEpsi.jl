
# EMBED : NamedTuple
#
#

mutable struct EmbedAndPool{ENV, NAMES, EMBED} <: AbstractContainerLayer
   environ::ENV
   embeddings::NamedTuple{NAMES, EMBED}
   pooling::BackflowPooling
   #---
   @reqfields
end

function EmbedAndPool(embeddings...)
   # construct the pooling
   return EmbedAndPool(basis, 
                  _make_reqfields()...)
end

(pooling::BackflowPooling)(args...) = evaluate(pooling, args...)


@generated function evaluate(embedpool::EmbedAndPool{ENV, NAMES, EMBED}, 
                     X::AbstractVector, ps, st) where {ENV, NAMES, EMBED}
    
   names = NAMES.parameters
   NE = length(names)
 
   quote
      # needs extra layer
      # X -> XX stores the environments of the electons 
      XX = evaluate(embedpool.environ, X, ps.environ, st.environ)
      # XX = Nin x Nel 

      @nexprs $NE a -> begin
         ps_a = ps[$(names[a])]
         st_a = st[$(names[a])]
         # E_a = Nin x Nfeat_a x Nel
         E_a = evaluate(embedpool.embeddings[a], XX, ps_a, st_a)
      end

      # pooling 
      # EE = tuple(E_1,...., )
      A = evaluate(embedpool.pooling, EE, ps.pooling, st.pooling) 
      # A = Nfeat x Nel 

      return A 
   end
end


# @generated function pullback(embedpool::EmbedAndPool{NE, EMBED}, 
#                              X::AbstractVector, ps, st, ∂A)   where {NE, EMBED}
#    quote
#       @nexprs $NE i -> begin
#          ps_i = ps[$(NAMES[i])]
#          st_i = st[$(NAMES[i])]
#          E_i, dE_i = evaluate_ed(embedpool.embeddings[i], X, ps_i, st_i)
#       end

#       # pooling 
#       EE = tuple(E_1,...., )
#       # A = evaluate(embedpool.pooling, EE, ps.pooling, st.pooling) 
#       ∂EE = pullback(embedpool.pooling, EE, ps, st, ∂A)

#       @nexprs $NE i -> begin
#          ∂ps_i, ∂X_i = pullback(embedpool.embeddings[i], X, ps, st, ∂EE[i])
#       end

#       ∂ps = ∂ps_$NE 
#       ∂X = ∂X_$NE
#       @nexprs = $(NE-1) i -> begin
#          ∂ps += ∂ps_i
#          ∂X += ∂X_i
#       end

#       return ∂X, ∂ps
#    end
# end

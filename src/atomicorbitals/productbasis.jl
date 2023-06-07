using Lux: WrappedFunction
using Lux
using Polynomials4ML: SparseProduct, AbstractPoly4MLBasis

function _invmap(a::AbstractVector)
   inva = Dict{eltype(a), Int}()
   for i = 1:length(a) 
      inva[a[i]] = i 
   end
   return inva 
end

function dropnames(namedtuple::NamedTuple, names::Tuple{Vararg{Symbol}}) 
   keepnames = Base.diff_names(Base._nt_names(namedtuple), names)
   return NamedTuple{keepnames}(namedtuple)
end

function ProductBasisLayer(spec1::Vector, bRnl::AbstractPoly4MLBasis, bYlm::AbstractPoly4MLBasis)
    spec1idx = Vector{Tuple{Int, Int}}(undef, length(spec1))
    spec_Rnl = natural_indices(bRnl); inv_Rnl = _invmap(spec_Rnl)
    spec_Ylm = natural_indices(bYlm); inv_Ylm = _invmap(spec_Ylm)
 
    spec1idx = Vector{Tuple{Int, Int}}(undef, length(spec1))
    for (i, b) in enumerate(spec1)
       spec1idx[i] = (inv_Rnl[dropnames(b,(:m,))], inv_Ylm[(l=b.l, m=b.m)])
    end
    sparsebasis = SparseProduct(spec1idx)
 
    # wrap into lux layers
    l_Rn = Polynomials4ML.lux(bRnl)
    l_Ylm = Polynomials4ML.lux(bYlm)
    l_ϕnlm = Polynomials4ML.lux(sparsebasis)
    
    # formming model with Lux Chain
    _norm(x) = norm.(x)
 
    l_xnx = Lux.Parallel(nothing; normx = WrappedFunction(_norm), x = WrappedFunction(identity))
    l_embed = Lux.Parallel(nothing; Rn = l_Rn, Ylm = l_Ylm)
    return Chain(; xnx = l_xnx, embed = l_embed, ϕnlms = l_ϕnlm)
 end
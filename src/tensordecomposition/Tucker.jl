import ChainRulesCore: rrule, NoTangent

using ChainRulesCore: ignore_derivatives
using LuxCore
using LuxCore: AbstractExplicitLayer
using Random
using StrideArrays
using LinearAlgebra: mul!
using ObjectPools: acquire!
using Polynomials4ML: release!
using Polynomials4ML: _make_reqfields, @reqfields, POOL, TMP, META
using Tullio

export TuckerLayer


mutable struct Tucker <: Tensor_Decomposition
    P::Integer
end

"""
`struct Tucker` : 

* `x::AbstractMatrix` of size `(N, 3, M, spec1p)`, where `N = electron number`, `M = nuclei number`
* `W::AbstractMatrix` of size `(N, 3, P, M, spec1p)` `P = feature dimension`, 
* `B::AbstractMatrix` of size `(N, 3, P,)`
```

"""
struct TuckerLayer <: AbstractExplicitLayer 
   P::Integer # reduced dimension
   K::Integer # spec1p
   Nel::Integer # number of electrons
   @reqfields()
end

TuckerLayer(P::Integer, K::Integer, Nel::Integer) = TuckerLayer(P, K, Nel, _make_reqfields()...)

_valtype(l::TuckerLayer, x::AbstractArray, ps)  = promote_type(eltype(x), eltype(ps.W))

function (l::TuckerLayer)(x::AbstractArray, ps, st)
    A = @tullio out[a, i, j, p] := ps.W[a, j, p, k] * x[i, j, k] (k in 1:l.K)
    out = ntuple(a -> reshape(A[a,:,:,:], l.Nel, :), l.Nel)
    ignore_derivatives() do
        release!(x)
    end
    return out, st
end

LuxCore.initialparameters(rng::AbstractRNG, l::TuckerLayer) = ( W = randn(rng, l.Nel, 3, l.P, l.K), )
LuxCore.initialstates(rng::AbstractRNG, l::TuckerLayer) = NamedTuple()



import ChainRulesCore: rrule, NoTangent

using ChainRulesCore: ignore_derivatives
using LuxCore
using LuxCore: AbstractExplicitLayer
using Lux: softmax
using Random
using StrideArrays
using LinearAlgebra: mul!
using ObjectPools: acquire!
using Polynomials4ML: release!
using Polynomials4ML: _make_reqfields, @reqfields, POOL, TMP, META
using Tullio

export AttentionLayer


mutable struct self_attention <: attention
    D::Integer
end

"""
`struct self_attention` : 

* `x::PtrArray` of size `(N, spec)`, where `N = electron number`, `spec = number of B-order basis`
* `Wq::AbstractMatrix` of size `(N, D)` `D = feature dimension`, 
* `Wk::AbstractMatrix` of size `(N, D)` `D = feature dimension`, 
* `B::AbstractMatrix` of size `(N, 3, P)`
```

"""
struct AttentionLayer <: AbstractExplicitLayer 
   N::Integer # electron number
   D::Integer # feature dimension
   @reqfields()
end

AttentionLayer(N::Integer, D::Integer) = AttentionLayer(N, D, _make_reqfields()...)

_valtype(l::AttentionLayer, x::AbstractArray, ps)  = promote_type(eltype(x), eltype(ps.Wk))

function (l::AttentionLayer)(x::AbstractArray, ps, st)
    # i = 1, ..., N, x[i,:] * ps.Wq[i,:]' * ps.Wk[i,:] * x[i,:]'
    @tullio out[i, j, p] := x[i,j] * ps.Wq[i,k] * ps.Wk[i,k] * x[i,p]
    out = softmax(out, dims = 2)
    @tullio outx[i, p] := out[i, j, p] * x[i, j]
    ignore_derivatives() do
        release!(x)
    end
    return outx, st
end

LuxCore.initialparameters(rng::AbstractRNG, l::AttentionLayer) = ( Wq = randn(rng, l.N, l.D),  Wk = randn(rng, l.N, l.D), )
LuxCore.initialstates(rng::AbstractRNG, l::AttentionLayer) = NamedTuple()
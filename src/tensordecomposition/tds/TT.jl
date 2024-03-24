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

export TTLayer

mutable struct TT <: TDs
    P::Integer
end

"""
`struct TT` : 

* `x::AbstractMatrix` of size `(N, spec)`, where `N = electron number`
* `W::AbstractMatrix` of size `(NBF, N, B, spec1p, P, P)`, where `P = feature dimension`, `B = body order` 
* `out::AbstractMatrix` of size `(NBF, N, N)`
```
"""

struct TTLayer{T, TT} <: AbstractExplicitLayer 
   NBF::Integer # number of BF det
   P::Integer   # reduced dimension
   K::Integer   # spec1p
   M::Integer   # spec
   Nel::Integer # number of electrons/orbitals
   B::Integer   # body-order
   spec1p::Vector{T}
   spec::Matrix{TT}
   @reqfields()
end

# c = spec1p * ... * spec1p * spec1p [B prod]
# c[v1, ..., vB] = ∑_p1∑_p2...∑_pB W[1, v1, 1, p2] * W[2, v2, p2, p3] * ... * W[B-1, vB-1, pB-1, pB] * W[B, vB, pB, 1]

TTLayer(NBF::Integer, P::Integer, Nel::Integer, spec1p, spec) = TTLayer(NBF, P, length(spec1p), size(spec, 1), Nel, length(spec[end]), spec1p, spec, _make_reqfields()...)

_valtype(l::TTLayer, x::AbstractArray, ps)  = promote_type(eltype(x), eltype(ps.W))

function (l::TTLayer)(x::AbstractArray, ps, st)
    # BF[a, b] = ∑_{m\in spec} x[a, m] * c[b][spec[m][1], spec[m][2], ..., spec[m][B]]
    #          = ∑_{m\in spec} x[a, m] * ∑_p1∑_p2...∑_pB W[b][1, spec[m][1], 1, p2] * W[b][2, spec[m][2], p2, p3] * ... * W[b][B-1, spec[m][B-1], pB-1, pB] * W[b][B, spec[m][B], pB, 1]
    A = @tullio out[c, a, b] := ps.W[c, b, 1, l.spec[m, 1], 1, p2] * ps.W[c, b, 2, l.spec[m, 2], p2, 1] * x[a, m] (m in 1:l.M, p2 in 1:l.P)
    out = ntuple(a -> A[a,:,:], l.NBF)
    ignore_derivatives() do
        release!(x)
    end
    return out, st
end

LuxCore.initialparameters(rng::AbstractRNG, l::TTLayer) = ( W = randn(rng, l.NBF, l.Nel, l.B, l.K, l.P, l.P), )
LuxCore.initialstates(rng::AbstractRNG, l::TTLayer) = NamedTuple()



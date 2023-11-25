import ChainRulesCore: rrule, NoTangent
using LuxCore
using LuxCore: AbstractExplicitLayer
using Random
using StrideArrays
using LinearAlgebra: mul!
using ObjectPools: acquire!
using Polynomials4ML: release!
using Polynomials4ML: _make_reqfields, @reqfields, POOL, TMP, META

export TuckerLayer


mutable struct Tucker <: Tensor_Decomposition
    P::Integer
end

"""
`struct Tucker` : 

* `x::AbstractMatrix` of size `(N, 3, M, spec1p)`, where `N = electron number`, `M = nuclei number`
* `W::AbstractMatrix` of size `(N, 3, P, M * spec1p)` `P = feature dimension`, 
* `B::AbstractMatrix` of size `(N, 3, P,)`
```

"""
struct TuckerLayer <: AbstractExplicitLayer 
   P::Integer # reduced dimension
   N::Integer # number of electron
   M::Integer # number of nuclei
   K::Integer # spec1p
   @reqfields()
end

TuckerLayer(P::Integer, N::Integer, M::Integer, K::Integer) = TuckerLayer(P, N, M, K, _make_reqfields()...)

_valtype(l::TuckerLayer, x::AbstractArray, ps)  = promote_type(eltype(x), eltype(ps.W))

function (l::TuckerLayer)(x::AbstractArray, ps, st)
    out = acquire!(l.pool, :AT, (l.N, 3, l.P), _valtype(l, x, ps))
    @inbounds for i = 1:l.N
        @simd for j = 1:3
            out[i,j,:] = ps.W[i,j,:,:] * x[i,j,:,:][:]
        end
    end
    release!(x); 
    return out, st
end
 
# Jerry: Maybe we should use Glorot Uniform if we have no idea about what we should use?
LuxCore.initialparameters(rng::AbstractRNG, l::TuckerLayer) = ( W = randn(rng, l.N, 3, l.P, l.K * l.M), )

LuxCore.initialstates(rng::AbstractRNG, l::TuckerLayer) = NamedTuple()
 
# TODO: check whether we can do this without multiple dispatch on vec/mat without loss of performance
function rrule(::typeof(LuxCore.apply), l::TuckerLayer, x, ps, st)
    val = l(x, ps, st)
    function pb(A)
        out_x = acquire!(l.pool, :Tx, (l.N, 3, l.M, l.K), _valtype(l, x, ps))
        out_ps = acquire!(l.pool, :Tp, (l.N, 3, l.P, l.M * l.K), _valtype(l, x, ps))
        @inbounds for i = 1:l.N
            @simd for j = 1:3
                out_x[i,j,:,:] = reshape(ps.W[i,j,:,:]' * A[1][i,j,:], l.M, l.K)
                out_ps[i,j,:,:] =  A[1][i,j,:] * transpose(x[i,j,:,:][:])
            end
        end
        return NoTangent(), NoTangent(), out_x, (W = out_ps,), NoTangent()
    end
    return val, pb
end

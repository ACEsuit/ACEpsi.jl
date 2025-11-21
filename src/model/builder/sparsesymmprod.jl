import KernelAbstractions as KA
using ChainRulesCore
using CUDA
using EquivariantTensors: _static_prod_ed

struct sparsesymmprod{ORD,TS} <: AbstractLuxLayer
    specs::TS                        # Tuple{Vector{NTuple{1,Int}}, Vector{NTuple{2,Int}}, ...}
    ranges::NTuple{ORD, UnitRange{Int}}
    hasconst::Bool 
end

Base.length(basis::sparsesymmprod) = sum(length, basis.specs) + basis.hasconst

LuxCore.initialparameters(::AbstractRNG, ::sparsesymmprod) = NamedTuple()

function LuxCore.initialstates(::AbstractRNG, l::sparsesymmprod)
    return (; backend = nothing,
             out_buf = nothing,
             dA_buf = nothing,
             specs_dev = nothing)  # will be filled as Tuple matching l.specs on first use
end
(l::sparsesymmprod)(A, ps, st) = evaluate(l, A, ps, st)

@inline @generated function _mul_tuple_N(::Val{N}, aa) where {N}
    ex = :(one(eltype(aa)))
    for j in 1:N
        ex = :($ex * aa[$j])
    end
    return :( $ex )
end

@kernel function _ssp_forward!(AA, @Const(A), @Const(spec), ::Val{N}, offset::Int) where {N}
    z, i, iAA = @index(Global, NTuple)
    ϕ = @inbounds spec[iAA]
    aa = ntuple(t -> @inbounds(A[z,i,ϕ[t]]), Val(N))
    acc = _mul_tuple_N(Val(N), aa)
    @inbounds AA[z, i, offset + iAA - 1] = acc
end


@kernel function _ssp_backward!(
    dA,                   # (Nb, Nel, K)
    @Const(dAA),          # (Nb, Nel, Ltot)
    @Const(A),            # (Nb, Nel, K)
    @Const(spec),         # AbstractVector{NTuple{N,Int}}
    ::Val{N},
    offset::Int) where {N}

    z, i, iAA = @index(Global, NTuple)
    ϕ = spec[iAA]
    ∂cur = dAA[z, i, offset + iAA - 1]

    aa = ntuple(t -> A[z, i, ϕ[t]], N)
    _, ∇prod = _static_prod_ed(aa)   # ∇prod[j] = ∂(∏ aa)/∂aa[j]

    @inbounds for j in 1:N
        @atomic dA[z, i, ϕ[j]] += ∂cur * ∇prod[j]
    end
end

function _to_device_specs(backend, specs_host, specs_dev_cached)
    if KernelAbstractions.isgpu(backend)
        if specs_dev_cached === nothing
            return Tuple(cu(s) for s in specs_host)
        else
            return specs_dev_cached
        end
    else
        return specs_host
    end
end

function evaluate(l::sparsesymmprod{ORD,TS},
                  A::AbstractArray{T,3},
                  ps,
                  st) where {ORD,TS,T}

    Nb, Nel, K = size(A)
    Lspec = sum(length, l.specs)
    Ltot  = Lspec + (l.hasconst ? 1 : 0)

    backend = st.backend === nothing ? KA.get_backend(A) : st.backend
    groupsize = KernelAbstractions.isgpu(backend) ? 128 : 128

    out = st.out_buf
    need_new = (out === nothing) ||
               !(eltype(out) === T && ndims(out) == 3 &&
                 size(out,1) == Nb && size(out,2) == Nel && size(out,3) == Ltot)

    specs_dev = need_new ? _to_device_specs(backend, l.specs, get(st, :specs_dev, nothing)) : st.specs_dev
    
    out = need_new ? similar(A, T, Nb, Nel, Ltot) : out
    st2 = need_new ? merge(st, (; backend=backend, out_buf=out, specs_dev=specs_dev)) : st
    kernel! = _ssp_forward!(backend, groupsize)
    for n in 1:ORD
        r   = l.ranges[n]
        off = first(r)
        specn = specs_dev[n]
        ndr = (Nb, Nel, length(specn))
        kernel!(out, A, specn, Val{n}(), off; ndrange = ndr)  
    end
    KA.synchronize(backend)
    return out, st2
end

function ChainRulesCore.rrule(::typeof(evaluate),
                              l::sparsesymmprod{ORD,TS},
                              A::AbstractArray{T,3},
                              ps,
                              st) where {ORD,TS,T}

    Y, st2 = evaluate(l, A, ps, st)
    Nb, Nel, _ = size(Y)
    backend = st2.backend === nothing ? KA.get_backend(A) : st2.backend
    groupsize = KernelAbstractions.isgpu(backend) ? 128 : 128
    specs_dev = st2.specs_dev
    Lspec = sum(length, l.specs)
    Ltot  = Lspec + (l.hasconst ? 1 : 0)

    dA = st2.dA_buf
    need_new = (dA === nothing) ||
               !(eltype(dA) === T && ndims(dA) == 3 &&
                 size(dA,1) == Nb && size(dA,2) == Nel && size(dA,3) == size(A,3))
    dA = need_new ? similar(A) : dA
    st3 = need_new ? merge(st, (; dA_buf = dA)) : st2
    fill!(dA, zero(T))

    function pullback(Ȳ_raw)
        Ȳ, st̄ = Ȳ_raw
        dY = unthunk(Ȳ)
        kernel! = _ssp_backward!(backend, groupsize)
        for n in 1:ORD
            r   = l.ranges[n]; off = first(r)
            specn = specs_dev[n]
            ndr = (Nb, Nel, length(specn))
            kernel!(dA, dY, A, specn, Val{n}(), off; ndrange = ndr)
        end
        KA.synchronize(backend)
        return NoTangent(), NoTangent(), dA, NoTangent(), NoTangent()
    end

    return (Y, st3), pullback
end

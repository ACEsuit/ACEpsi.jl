import KernelAbstractions as KA
using StaticArrays
using ChainRulesCore

struct DiffLayer{T, Nnuc} <: AbstractLuxLayer
    rr::SMatrix{3, Nnuc, T}
end

function DiffLayer(nuclei::AbstractVector{<:Nuc{T,TT}}) where {T,TT}
    Nnuc = length(nuclei)
    M = Matrix{T}(undef, 3, Nnuc)
    @inbounds for j in 1:Nnuc
        M[:, j] = nuclei[j].rr
    end
    return DiffLayer{T, Nnuc}(SMatrix{3, Nnuc, T}(M))
end

LuxCore.initialparameters(::AbstractRNG, ::DiffLayer) = NamedTuple()
function LuxCore.initialstates(::AbstractRNG, l::DiffLayer{Tn, Nnuc}) where {Tn, Nnuc}
    rrX = SMatrix{3, Nnuc, Float64}(l.rr)
    rr = reshape(rrX, 3, 1, 1, Nnuc)
    (; rr = collect(rr), out_buf=nothing)
end

(l::DiffLayer{Tn, Nnuc})(X, ps, st) where {Tn, Nnuc} = begin
    Y, st2 = evaluate(l, X, ps, st)
    Yt = slices(Y, Val(Nnuc))
    return Yt, st2
end

@generated function slices(Y::AbstractArray{T,4}, ::Val{Nnuc}) where {T,Nnuc}
    exprs = [:(@view(Y[:,:,:, $(i)])) for i in 1:Nnuc]
    return Expr(:tuple, exprs...)
end

function evaluate(l::DiffLayer{Tn, Nnuc},
                  X::AbstractArray{Tx,3},
                  ps,
                  st) where {Tn, Nnuc, Tx}
    Nel = size(X,2); B = size(X,3)
    Y = st.out_buf
    rr = st.rr
    need_new = (Y === nothing) ||!(eltype(Y)===Tx && ndims(Y)==4 && size(Y,1)==3 && size(Y,2)==Nel && size(Y,3)==B && size(Y,4)==Nnuc)
    need_new_r = !(eltype(rr) === Tx)

    Y = need_new ? similar(X, Tx, 3, Nel, B, Nnuc) : Y  
    rr = need_new_r ? Tx.(rr) : rr

    st2 = need_new ? merge(st, (; out_buf=Y)) : st 
    st3 = need_new_r ? merge(st2, (; rr=rr)) : st2

    broadcast!(-, Y, X, rr)
    return Y, st3
end

function ChainRulesCore.rrule(::typeof(evaluate),
                              l::DiffLayer{Tn,Nnuc},
                              X::AbstractArray{Tx,3},
                              ps,
                              st) where {Tn,Nnuc,Tx}

    Y, st2 = evaluate(l, X, ps, st)

    function pullback(Ȳ_raw)
        Ȳ, st̄ = Ȳ_raw
        dY = unthunk(Ȳ)                          # (3, Nel, B, Nnuc)
        # dX = sum_i dY[:,:,:,i]
        dX4 = sum(dY; dims=4)                    # (3, Nel, B, 1)
        dX  = dropdims(dX4; dims=4)              # (3, Nel, B)
        return NoTangent(), NoTangent(), dX, NoTangent(), NoTangent()
    end

    return (Y, st2), pullback
end
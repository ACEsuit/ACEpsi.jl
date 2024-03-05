using HyperDualNumbers: Hyper

Base.real(x::Hyper{<:Number}) = Hyper(real(x.value), real(x.epsilon1), real(x.epsilon2), real(x.epsilon12))  

struct NTarr{NTT}
    nt::NTT
end

export array

array(nt::NamedTuple) = NTarr(nt)

# ------------------------------
#  0 

zero!(a::AbstractArray{ <: Number}) = fill!(a, zero(eltype(a)))

zero!(a::Vector{Vector{Float64}}) = begin
   for aa in a
      fill!(aa, zero(eltype(aa)))
   end
end

zero!(a::Vector{Vector{Hyper}}) = begin
   for aa in a
      fill!(aa, zero(eltype(aa)))
   end
end

zero!(a::Nothing) = nothing 

function zero!(nt::NamedTuple)
   for k in keys(nt)
      zero!(nt[k])
   end
   return nt
end 

Base.zero(nt::NamedTuple) = zero!(deepcopy(nt))

Base.zero(nt::NTarr) = NTarr(zero(nt.nt))

# ------------------------------
#  + 


function _add!(a1::AbstractArray, a2::AbstractArray) 
    a1[:] .= a1[:] .+ a2[:]
    return nothing 
end

_add!(at::Nothing, args...) = nothing 

function _add!(nt1::NamedTuple, nt2)
    for k in keys(nt1)
       _add!(nt1[k], nt2[k])
    end
    return nothing 
end

function _add(nt1::NamedTuple, nt2::NamedTuple)
    nt = deepcopy(nt1)
    _add!(nt, nt2)
    return nt
end

Base.:+(nt1::NTarr, nt2::NTarr) = NTarr(_add(nt1.nt, nt2.nt))

# ------------------------------
#  * 

_mul!(::Nothing, args... ) = nothing 

function _mul!(a::AbstractArray, λ::Number)
    a[:] .= a[:] .* λ
    return nothing 
end

function _mul!(nt::NamedTuple, λ::Number)
    for k in keys(nt)
       _mul!(nt[k], λ)
    end
    return nothing 
end

function _mul(nt::NamedTuple, λ::Number)
    nt = deepcopy(nt)
    _mul!(nt, λ)
    return nt
end

Base.:*(λ::Number, nt::NTarr) = NTarr(_mul(nt.nt, λ))
Base.:*(nt::NTarr, λ::Number) = NTarr(_mul(nt.nt, λ))

# ------------------------------
#   map 

_map!(f, a::AbstractArray) = map!(f, a, a) 

_map!(f, ::Nothing) = nothing 

function _map!(f, nt::NamedTuple)
    for k in keys(nt)
       _map!(f, nt[k])
    end
    return nothing 
end

function Base.map!(f, dest::NTarr, src::NTarr)
    _map!(f, nt.nt)
    return nt
end

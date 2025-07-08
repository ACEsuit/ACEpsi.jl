using Tullio
export No_Decomposition, SCPMultipleW, STKMultipleW, SCPCommonW, STKCommonW

abstract type Tensor_Decomposition end

abstract type TDs <: Tensor_Decomposition end

abstract type STDs <: TDs end

struct No_Decomposition <: Tensor_Decomposition
end

mutable struct SCPMultipleW <: STDs
    P::Integer
end

struct SCPMultipleLayer <: AbstractLuxLayer 
    P::Integer # reduced dimension
    K::Integer # spec1p
    Nel::Integer # number of electrons/orbitals
end

function (l::SCPMultipleLayer)(x::AbstractArray, ps, st)
    # k = spec1p, j = spin(3), p = CP(p), i = Nel(highlighted), a = orbital
    A = @tullio out[a, i, j, p] := ps.W[a, j, p, k] * x[i, j, k] (k in 1:l.K)
    out = ntuple(a -> reshape(A[a,:,:,:], l.Nel, :), l.Nel) # orbital, Nel, spin(3) * CP(p)
    return out, st
end

LuxCore.initialparameters(rng::AbstractRNG, l::SCPMultipleLayer) = ( W = randn(rng, l.Nel, 3, l.P, l.K), )
LuxCore.initialstates(rng::AbstractRNG, l::SCPMultipleLayer) = NamedTuple()

mutable struct STKMultipleW <: STDs
    P::Integer
end

struct STKMultipleLayer <: AbstractLuxLayer 
    P::Integer # reduced dimension
    K::Integer # spec1p
    Nel::Integer # number of electrons/orbitals
end

function (l::STKMultipleLayer)(x::AbstractArray, ps, st)
    # k = spec1p, j = spin(3), p = Tucker(p), i = Nel(highlighted), a = orbital
    A = @tullio out[a, i, j, p] := ps.W[a, j, p, k] * x[i, j, k] (k in 1:l.K)
    out = ntuple(a -> reshape(A[a,:,:,:], l.Nel, :), l.Nel) # orbital, Nel, spin(3) * Tucker(p)
    return out, st
end

LuxCore.initialparameters(rng::AbstractRNG, l::STKMultipleLayer) = ( W = randn(rng, l.Nel, 3, l.P, l.K), )
LuxCore.initialstates(rng::AbstractRNG, l::STKMultipleLayer) = NamedTuple()


# all orbitals are the same W
mutable struct SCPCommonW <: STDs
    P::Integer
end

struct SCPCommonLayer <: AbstractLuxLayer 
    P::Integer # reduced dimension
    K::Integer # spec1p
    Nel::Integer # number of electrons/orbitals
end

function (l::SCPCommonLayer)(x::AbstractArray, ps, st)
    # k = spec1p, j = spin(3), p = CP(p), i = Nel(highlighted), a = orbital
    A = @tullio out[i, j, p] := ps.W[j, p, k] * x[i, j, k] (k in 1:l.K)
    out = reshape(A, l.Nel, :)
    return out, st
end 

LuxCore.initialparameters(rng::AbstractRNG, l::SCPCommonLayer) = ( W = randn(rng, 3, l.P, l.K), )
LuxCore.initialstates(rng::AbstractRNG, l::SCPCommonLayer) = NamedTuple()


# all orbitals are the same W
mutable struct STKCommonW <: STDs
    P::Integer
end

struct STKCommonLayer <: AbstractLuxLayer 
    P::Integer # reduced dimension
    K::Integer # spec1p
    Nel::Integer # number of electrons/orbitals
end


function (l::STKCommonLayer)(x::AbstractArray, ps, st)
    # k = spec1p, j = spin(3), p = Tucker(p), i = Nel(highlighted), a = orbital
    A = @tullio out[i, j, p] := ps.W[j, p, k] * x[i, j, k] (k in 1:l.K)
    out = reshape(A, l.Nel, :)
    return out, st
end 

LuxCore.initialparameters(rng::AbstractRNG, l::STKCommonLayer) = ( W = randn(rng, 3, l.P, l.K), )
LuxCore.initialstates(rng::AbstractRNG, l::STKCommonLayer) = NamedTuple()

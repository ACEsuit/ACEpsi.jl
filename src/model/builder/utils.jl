using HyperDualNumbers: Hyper, Hyper256
using HyperDualNumbers: epsilon12

@inline make_dual_on(x) = Hyper256(x, true,  true,  0.0)
@inline make_dual_off(x) = Hyper256(x, false, false, 0.0)

@inline function seed_on!(Y, X, i::Int, j::Int)
    @views Yi = Y[i,j,:]; Xi = X[i,j,:]
    @. Yi = make_dual_on(Xi)
    return nothing
end

@inline function seed_off!(Y, X, i::Int, j::Int)
    @views Yi = Y[i,j,:]; Xi = X[i,j,:]
    @. Yi = make_dual_off(Xi)
    return nothing
end

function laplacian(model, X::AbstractArray{T,3}, ps, st) where {T}
    @assert size(X,1) == 3
    N   = 3
    Nel = size(X,2)
    Nb  = size(X,3)

    Δ = similar(X, Nb)
    fill!(Δ, 0.0)
        
    Y = similar(X, Hyper256, N, Nel, Nb)
    @. Y = Hyper256(X, false, false, zero(T))
    for j in 1:Nel, i in 1:3
        seed_on!(Y, X, i, j)
        ψ, st = model(Y, ps, st)
        Δ .+= epsilon12.(ψ)
        seed_off!(Y, X, i, j)
    end
    return T.(Δ), st
end

function displayspec1pn1(spec1p::AbstractArray{T}) where {T}
    shells = Vector{String}(undef, length(spec1p))
    for i = 1:length(spec1p)
        @assert spec1p[i].l ∈ keys(orbitalIndexToType) "only 0(s) - 5(h) shell are supported."
        shells[i] = "$(spec1p[i].n2)-th-$(spec1p[i].n1)$(orbitalIndexToType[spec1p[i].l])"
    end
    return join(unique(shells), ",")
end

function displayspec1p(spec1p::AbstractArray{T}) where {T}
    shells = Vector{String}(undef, length(spec1p))
    for i = 1:length(spec1p)
        @assert spec1p[i].l ∈ keys(orbitalIndexToType) "only 0(s) - 5(h) shell are supported."
        shells[i] = "$(spec1p[i].n2)$(orbitalIndexToType[spec1p[i].l])"
    end
    return join(unique(shells), ",")
end

orbitalIndexToType = Dict(0 => "s", 1 => "p", 2 => "d", 3 => "f", 4 => "g", 5 => "h")
orbitalTypeToIndex = Dict("s" => 0, "p" => 1, "d" => 2, "f" => 3, "g" => 4, "h" => 5)

function get_spec1p(basis::Vector{TS}; spin = false) where {TS}
    Nnlm = sum(length(b.spec) for b in basis)
    if spin
        spec = Array{NamedTuple{(:s, :I, :n1, :n2, :l, :m), Tuple{Char, Vararg{Int64, 5}}}}(undef, (3, Nnlm))
        for (is, s) in enumerate(extspins())
            t = 0
            for (i, b) in enumerate(basis), nlm in b.spec
                t += 1
                spec[is, t] = (s = s, I = i, n1 = nlm[1], n2= nlm[2], l = nlm[3], m = nlm[4])
            end
        end
    else
        spec = Array{NamedTuple{(:I, :n1, :n2, :l, :m), Tuple{Vararg{Int64, 5}}}}(undef, Nnlm)
        t = 0
        for (i, b) in enumerate(basis), nlm in b.spec
            t += 1
            spec[t] = (I = i, n1 = nlm[1], n2 = nlm[2], l = nlm[3], m = nlm[4])
        end
    end
    return Tuple(spec[:])
end
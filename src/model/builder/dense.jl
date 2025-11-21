using NNlib

function dense_blocks(G::AbstractMatrix{T}, X::AbstractMatrix{T}, Nel::Integer;
                   Nb::Integer = size(G,2) ÷ Nel) where {T<:Real}
    m, tot  = size(G)     # m == Nel
    n, totX = size(X)     # n == L

    G3 = reshape(G, m, Nb, Nel)                 # m × Nb × Nel
    X3 = reshape(X, n, Nb, Nel)                 # n × Nb × Nel

    Gp = permutedims(G3, (1,3,2))               # m × Nel × Nb
    Xp = permutedims(X3, (1,3,2))               # n × Nel × Nb
    # 2 * (G_b * X_b')
    Y = 2 .* NNlib.batched_mul(Gp, permutedims(Xp, (2,1,3)))  # m × n × Nb

    return reshape(Y, m*n, Nb)                     # (m*n) × Nb
end

function LuxCore.initialparameters(rng::AbstractRNG, d::Dense)
    weight = if d.init_weight === nothing
        Lux.kaiming_uniform(
            rng,
            Float64,
            d.out_dims,
            d.in_dims;
            gain= Lux.Utils.calculate_gain(d.activation, √5.0f0),
        )
    else
        d.init_weight(rng, d.out_dims, d.in_dims)
    end
    Lux.has_bias(d) || return (; weight)
    return (; weight, bias=Lux.init_linear_bias(rng, d.init_bias, d.in_dims, d.out_dims))
end

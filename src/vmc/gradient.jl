using Zygote

x2dualwrtj(x, j) = SVector{3}([Hyper(x[i], i == j, i == j, 0) for i = 1:3])

function _gradlap(g_bchain, x)
    function _mapadd!(f, dest::NamedTuple, src::NamedTuple) 
       for k in keys(dest)
          _mapadd!(f, dest[k], src[k])
       end
       return nothing 
    end
    _mapadd!(f, dest::Nothing, src) = nothing
    _mapadd!(f, dest::AbstractArray, src::AbstractArray) = 
             map!((s, d) -> d + f(s), dest, src, dest)
 
    Δ = zero!(g_bchain(x))
    hX = [x2dualwrtj(xx, 0) for xx in x]
    for i = 1:3
       for j = 1:length(x)
          hX[j] = x2dualwrtj(x[j], i)
          _mapadd!(ε₁ε₂part, Δ, g_bchain(hX))
          hX[j] = x2dualwrtj(x[j], 0)
       end
    end
    return Δ
end

evaluate(wf, X::AbstractVector, ps, st) = wf(X, ps, st)[1]

gradient(wf, x, ps, st) = Zygote.gradient(x -> wf(x, ps, st)[1], x)[1]

grad_params(wf, x, ps, st) = Zygote.gradient(p -> wf(x, p, st)[1], ps)[1]

function laplacian(wf, x, ps, st)
    ΔΨ = 0.0
    hX = [x2dualwrtj(xx, 0) for xx in x]
    Nel = length(x)
    for i = 1:3
       for j = 1:Nel
          hX[j] = x2dualwrtj(x[j], i) # ∂Φ/∂xj_{i}
          ΔΨ += wf(hX, ps, st)[1].epsilon12
          hX[j] = x2dualwrtj(x[j], 0)
       end
    end
    return ΔΨ
end

function grad_lap(wf, x, ps, st)
    g_bchain = xx -> Zygote.gradient(p -> wf(xx, p, st)[1], ps)[1]
    p, = destructure(_gradlap(g_bchain, x))
    return p
end
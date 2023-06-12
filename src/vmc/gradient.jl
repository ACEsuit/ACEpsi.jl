using Zygote
using HyperDualNumbers: Hyper

x2dualwrtj(x, j) = SVector{3}([Hyper(x[i], i == j, i == j, 0) for i = 1:3])

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

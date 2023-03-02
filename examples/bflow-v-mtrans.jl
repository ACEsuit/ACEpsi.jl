N = 8
Σ = vcat(rand([↑],Int(ceil(N/2))),rand([↓],N - Int(ceil(N/2))))
x0 = -Inf
x1 = Inf
pin = 0
pcut = 0

pos = [-70.,-50.,-30.,-10.,10.,30.,50.,70.]
trans = [λ("r -> atan(r+70.0)"),λ("r -> atan(r+50.0)"),λ("r -> atan(r+30.0)"),λ("r -> atan(r+10.0)"),λ("r -> atan(r-10.0)"),λ("r -> atan(r-30.0)"),λ("r -> atan(r-50.0)"),λ("r -> atan(r-70.0)")]
tpos = reduce(vcat,pos)
pos = reduce(vcat,pos)
M = length(pos)
MaxDeg = [4]

P = ACE.scal1pbasis(:x, :n, maximum(MaxDeg), trans[1], x1, x0; pin = pin, pcut = pcut)
ww, xx = P.basis.F[2].ww, P.basis.F[2].tdf
WW = Polynomials4ML.DiscreteWeights(xx, ww)
polys = Polynomials4ML.orthpolybasis(maximum(MaxDeg), WW)
U = BFwf(N, polys, x -> sqrt(1+x^2); pos = pos, tpos = tpos, ν=length(MaxDeg[1]), totdeg = maximum(MaxDeg), trans = trans,sd_admissible = bb -> (length(bb) == 0 || all([bb[i][1] <= MaxDeg[length(bb)] for i = 1:length(bb)])))

X = rand(N)
evaluate(U,X,Σ)
gradient(U,X,Σ)

import ACEbase.Testing: fdtest, println_slim
corr = fdtest(X -> evaluate(U, X, Σ), X -> gradient(U, X, Σ), X)


using BenchmarkTools
@btime evaluate($U, $X, $Σ)
@btime gradient($U, $X, $Σ)
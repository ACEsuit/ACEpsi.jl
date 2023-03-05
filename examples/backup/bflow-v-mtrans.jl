using ACE
using Polynomials4ML
using ACEpsi

const ↑, ↓, ∅ = '↑','↓','∅'

Nel = 8
Σ = vcat(rand([↑],Int(ceil(Nel/2))),rand([↓],Nel - Int(ceil(Nel/2))))

pos = [-70.,-50.,-30.,-10.,10.,30.,50.,70.]
trans = [λ("r -> atan(r+70.0)"),λ("r -> atan(r+50.0)"),λ("r -> atan(r+30.0)"),λ("r -> atan(r+10.0)"),λ("r -> atan(r-10.0)"),λ("r -> atan(r-30.0)"),λ("r -> atan(r-50.0)"),λ("r -> atan(r-70.0)")]
tpos = reduce(vcat,pos)
pos = reduce(vcat,pos)
M = length(pos)
MaxDeg = [6, 6, 6]

polys = Polynomials4ML.legendre_basis(maximum(MaxDeg))
wf = ACEpsi.BFwf(Nel, polys, x -> sqrt(1+x^2); pos = pos, tpos = tpos, ν=length(MaxDeg[1]), totdeg = maximum(MaxDeg), trans = trans,sd_admissible = bb -> (length(bb) == 0 || all([bb[i][1] <= MaxDeg[length(bb)] for i = 1:length(bb)])))

X = rand(Nel)
ACEpsi.evaluate(U,X,Σ)
ACEpsi.gradient(U,X,Σ)
using ACE
using Polynomials4ML
using ACEpsi

const ↑, ↓, ∅ = '↑','↓','∅'

Nel = 8
Σ = vcat(rand([↑],Int(ceil(Nel/2))),rand([↓],Nel - Int(ceil(Nel/2))))
trans = λ("r -> 2/pi * atan(r)")
MaxDeg = [4, 4]
polys = Polynomials4ML.legendre_basis(maximum(MaxDeg))
U = ACEpsi.BFwf(Nel, polys; ν=length(MaxDeg), totdeg = maximum(MaxDeg), trans = trans,sd_admissible = bb -> (length(bb) == 0 || all([bb[i][1] <= MaxDeg[length(bb)] for i = 1:length(bb)])))

X = rand(Nel)
ACEpsi.evaluate(U,X,Σ)
ACEpsi.gradient(U,X,Σ)





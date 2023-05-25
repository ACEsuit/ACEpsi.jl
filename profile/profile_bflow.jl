using Polynomials4ML, ACEcore, ACEpsi, ACEbase, Printf
using ACEpsi: BFwf, gradient, evaluate, laplacian 
using LinearAlgebra
using BenchmarkTools

##

N = 8
Σ = vcat(rand([↑],Int(ceil(N/2))),rand([↓],N - Int(ceil(N/2))))

pos = [-70.,-50.,-30.,-10.,10.,30.,50.,70.]
trans = [λ("r -> atan(r+70.0)"),λ("r -> atan(r+50.0)"),λ("r -> atan(r+30.0)"),λ("r -> atan(r+10.0)"),λ("r -> atan(r-10.0)"),λ("r -> atan(r-30.0)"),λ("r -> atan(r-50.0)"),λ("r -> atan(r-70.0)")]
tpos = reduce(vcat,pos)
pos = reduce(vcat,pos)
M = length(pos)
MaxDeg = [6, 6, 6]

polys = Polynomials4ML.legendre_basis(maximum(MaxDeg))
wf = BFwf(N, polys, x -> sqrt(1+x^2); pos = pos, tpos = tpos, ν=length(MaxDeg[1]), totdeg = maximum(MaxDeg), trans = trans,sd_admissible = bb -> (length(bb) == 0 || all([bb[i][1] <= MaxDeg[length(bb)] for i = 1:length(bb)])))

##

X = 2 * rand(Nel) .- 1
Σ = rand([↑, ↓], Nel)
wf(X, Σ)
gradient(wf, X, Σ)
laplacian(wf, X, Σ)

## 

@info("evaluate")
@btime wf($X, Σ)

@info("gradient")
@btime gradient($wf, $X, $Σ)

@info("laplacian")
@btime laplacian($wf, $X, $Σ)


##

# @profview let wf=wf, X=X
#    for nrun = 1:50_000 
#       laplacian(wf, X)
#    end
# end

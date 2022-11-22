

using Polynomials4ML, ACEcore, ACEpsi, ACEbase, Printf
using ACEpsi: BFwf, gradient, evaluate, laplacian 
using LinearAlgebra
using BenchmarkTools

##

Nel = 10
polys = legendre_basis(15)
wf = BFwf(Nel, polys; ν=3)


##

X = 2 * rand(Nel) .- 1
wf(X)
gradient(wf, X)
laplacian(wf, X)

## 

@info("evaluate")
@btime $wf($X)

@info("gradient")
@btime gradient($wf, $X)

@info("laplacian")
@btime laplacian($wf, $X)

##

# @profview let wf=wf, X=X
#    for nrun = 1:50_000 
#       laplacian(wf, X)
#    end
# end

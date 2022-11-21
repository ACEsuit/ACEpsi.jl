
using Polynomials4ML, ACEcore, ACEpsi, ACEbase
using ACEpsi: BFwf, gradient, evaluate

##

Nel = 10
polys = legendre_basis(10)
wf = BFwf(Nel, polys; ν=4)

X = 2 * rand(Nel) .- 1
wf(X)
g = gradient(wf, X)

##

using ACEbase.Testing: fdtest 

fdtest(wf, X -> gradient(wf, X), X)



##

ACEpsi.laplacian(wf, X)


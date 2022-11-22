
using Polynomials4ML, ACEcore, ACEpsi, ACEbase
using ACEpsi: BFwf, gradient, evaluate
using Printf
using LinearAlgebra

"This function should be removed later to test in a nicer way..."
function fdtest(F, Σ, dF, x::AbstractVector; h0 = 1.0, verbose=true)
    errors = Float64[]
    E = F(x, Σ)
    dE = dF
    # loop through finite-difference step-lengths
    verbose && @printf("---------|----------- \n")
    verbose && @printf("    h    | error \n")
    verbose && @printf("---------|----------- \n")
    for p = 2:11
       h = 0.1^p
       dEh = copy(dE)
       for n = 1:length(dE)
          x[n] += h
          dEh[n] = (F(x, Σ) - E) / h
          x[n] -= h
       end
       push!(errors, norm(dE - dEh, Inf))
       verbose && @printf(" %1.1e | %4.2e  \n", h, errors[end])
    end
    verbose && @printf("---------|----------- \n")
    if minimum(errors) <= 1e-3 * maximum(errors)
       verbose && println("passed")
       return true
    else
       @warn("""It seems the finite-difference test has failed, which indicates
       that there is an inconsistency between the function and gradient
       evaluation. Please double-check this manually / visually. (It is
       also possible that the function being tested is poorly scaled.)""")
       return false
    end
 end
##
const ↑, ↓, ∅ = '↑','↓','∅'

Σ = [↑, ↑, ↑, ↓, ↓];
Nel = 10
polys = legendre_basis(10)
wf = BFwf(Nel, polys; ν=4)

X = 2 * rand(Nel) .- 1
wf(X, Σ)
g = gradient(wf, X, Σ)

##

using LinearAlgebra
using Printf
#using ACEbase.Testing: fdtest 

fdtest(wf, Σ, g, X)

# ##

# todo: move this to a performance benchmark script 
# using BenchmarkTools
# @btime $wf($X)
# @btime gradient($wf, $X)

##



using Polynomials4ML, ACEcore, ACEbase
# using ACEpsi: BFwf, gradient, evaluate
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


Σ = [↑, ↑, ↓, ↓, ↓];
Nel = 5
polys = chebyshev_basis(10, normalize=false)
wf = BFwf(Nel, polys; ν=2, totdeg = 10)

using JSON


# @show dict2

data = JSON.parse(open("/home/jerryho/julia_ws/ACEpsi.jl/test/bftest.json"))
X = data[1]["X"]
PP = data[1]["P"]
@show X
X = atan.(X)
@show X
for i = 1:5
   wf.W[:, i] = PP[i][2:end]
end
@show size(wf.W)
@show wf(X, Σ)
@show wf.spec
spec1p = [ (k, σ) for σ in [1, 2, 3] for k in 1:length(polys) ]  # (1, 2, 3) = (∅, ↑, ↓);

@show displayspec(wf.spec, spec1p)


# =============
# g = gradient(wf, X, Σ)
# @show length(wf.spec)
# @show length(wf.W)
# ##

# using LinearAlgebra
# using Printf

# fdtest(wf, Σ, g, X)
# @show "test"
# ##

# todo: move this to a performance benchmark script 
# using BenchmarkTools
# @btime $wf($X)
# @btime gradient($wf, $X)



# check spec

# Σ = [↑, ↑, ↓, ↓, ↓];
# Nel = 5
# polys = chebyshev_basis(5, normalize = false)
# wf = BFwf(Nel, polys; ν=2, totdeg = 5)
# data = JSON.parse(open("/home/jerryho/julia_ws/ACEpsi.jl/test/check_spec.json"))
# read_check_spec = data[1]["check_spec"]
# check_spec = zeros(106, 2)
# check_spec[:, 1] = read_check_spec[1]
# check_spec[:, 2] = read_check_spec[2]

# spec1p = [ (k, σ) for σ in [1, 2, 3] for k in 1:length(polys) ]  # (1, 2, 3) = (∅, ↑, ↓);

# spec = wf.spec
# for i = 1:length(spec)
#    @show spec[i], check_spec[i+1, :]
# end
# # generate the many-particle spec 
# spec1p = sort(spec1p)
# @show spec1p

# for i = 1:length(polys) * 3
#    @show spec1p[spec[i][1]]
# end

# for i = length(polys) * 3 + 1 :length(spec)
#     @show spec1p[spec[i][1]], spec1p[spec[i][2]]
# end

using Polynomials4ML, ACEcore, ACEpsi, ACEbase, Printf
using ACEpsi: BFwf, gradient, evaluate, laplacian
using LinearAlgebra

##


function grad_test(f, df, X)
   F = f(X) 
   ∇F = df(X)
   nX, nF = size(F)
   U = randn(nX)
   V = randn(nF) ./ (1:nF).^2
   f0 = U' * F * V
   ∇f0 = [ U' * ∇F[i, :, :] * V for i = 1:nX ]
   EE = Matrix(I, (Nel, Nel))
   for h in 0.1.^(2:10)
      gh = [ (U'*f(X + h * EE[:, i])*V - f0) / h for i = 1:Nel ]
      @printf(" %.1e | %.2e \n", h, norm(gh - ∇f0, Inf))
   end
end

using Polynomials4ML, ACEcore, ACEpsi, ACEbase
using ACEpsi: BFwf, gradient, evaluate

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

Nel = 10
polys = legendre_basis(10)
wf = BFwf(Nel, polys; ν=4)

X = 2 * rand(Nel) .- 1
wf(X)
g = gradient(wf, X)

##

using LinearAlgebra
using Printf
#using ACEbase.Testing: fdtest 

fdtest(wf, X -> gradient(wf, X), X)


# ##

# todo: move this to a performance benchmark script 
# using BenchmarkTools
# @btime $wf($X)
# @btime gradient($wf, $X)

##

A, ∇A, ΔA = ACEpsi._assemble_A_∇A_ΔA(wf, X)

@info("test ∇A")
grad_test(X -> ACEpsi._assemble_A_∇A_ΔA(wf, X)[1], 
          X -> ACEpsi._assemble_A_∇A_ΔA(wf, X)[2], 
          X)

@info("test ΔA")          
lap_test(X -> ACEpsi._assemble_A_∇A_ΔA(wf, X)[1], 
         X -> ACEpsi._assemble_A_∇A_ΔA(wf, X)[3], 
         X)

 
##

function f_AA(X)
   A, ∇A, ΔA = ACEpsi._assemble_A_∇A_ΔA(wf, X)
   AA, ∇AA, ΔAA = ACEpsi._assemble_AA_∇AA_ΔAA(A, ∇A, ΔA, wf)
   return AA, ∇AA, ΔAA
end

@info("test ∇A")
grad_test(X -> f_AA(X)[1], X -> f_AA(X)[2], X)

@info("test ΔA")          
lap_test(X -> f_AA(X)[1], X -> f_AA(X)[3], X)


##

@info("test Δψ")
lap_test(X -> [wf(X);;], X -> [ACEpsi.laplacian(wf, X);;], X)


##

@info("Test ∇ψ w.r.t. parameters")

ACEpsi.gradp_evaluate(wf, X)


W0 = copy(wf.W)
sz0 = size(W0) 
w0 = W0[:]

Fp = w -> ( wf.W[:] .= w[:]; wf(X) )
dFp = w -> ( wf.W[:] .= w[:]; ACEpsi.gradp_evaluate(wf, X)[:] )

grad_test2(Fp, dFp, w0)

##

@info("Test ∇Δψ w.r.t. parameters")

Fp = w -> ( wf.W[:] .= w[:]; ACEpsi.laplacian(wf, X) )
dFp = w -> ( wf.W[:] .= w[:]; ACEpsi.gradp_laplacian(wf, X)[:] )

grad_test2(Fp, dFp, w0)

# ##

# using BenchmarkTools

# @info("ψ")
# @btime ACEpsi.evaluate($wf, $X)
# @info("∇ψ")
# @btime ACEpsi.gradp_evaluate($wf, $X)
# @info("Δψ")
# @btime ACEpsi.laplacian($wf, $X)
# @info("∇Δψ")
# @btime ACEpsi.gradp_laplacian($wf, $X)


##


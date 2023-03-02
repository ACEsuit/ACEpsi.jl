using Polynomials4ML, ACEcore, ACEbase, Printf, ACEpsi
using ACEpsi: BFwf, gradient, evaluate, laplacian
using LinearAlgebra
#using Random
#Random.seed!(123)
##
function lap_test(f, Δf, X)
   F = f(X) 
   ΔF = Δf(X)
   nX = length(X) 
   n1, nF = size(F)
   U = randn(n1)
   V = randn(nF) ./ (1:nF).^2
   f0 = U' * F * V
   Δf0 = U' * ΔF * V
   EE = Matrix(I, (nX, nX))
   for h in sqrt(0.1).^((5:12))
      Δfh = 0.0
      for i = 1:nX
         Δfh += (U'*f(X + h * EE[:, i])*V - f0) / h^2
         Δfh += (U'*f(X - h * EE[:, i])*V - f0) / h^2
      end
      @printf(" %.1e | %.2e \n", h, abs(Δfh - Δf0))
   end
end

function grad_test2(f, df, X::AbstractVector)
   F = f(X) 
   ∇F = df(X)
   nX = length(X)
   EE = Matrix(I, (nX, nX))
   
   for h in 0.1.^(3:12)
      gh = [ (f(X + h * EE[:, i]) - F) / h for i = 1:nX ]
      @printf(" %.1e | %.2e \n", h, norm(gh - ∇F, Inf))
   end
end

function grad_test3(f, df, X::Float64)
   F = f(X) 
   ∇F = df(X) # return as vector of size 1
   for h in 0.1.^(3:10)
      gh = [ (f(X + h) - F) / h]
      @printf(" %.1e | %.2e \n", h, norm(gh - ∇F, Inf))
   end
end


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
Nel = 5
polys = legendre_basis(8)
wf = BFwf(Nel, polys; ν=3)

X = 2 * rand(Nel) .- 1
Σ = rand([↑, ↓], Nel)

wf(X, Σ)
g = gradient(wf, X, Σ)

##

using LinearAlgebra
using Printf
#using ACEbase.Testing: fdtest 

@info("Fd test of gradient w.r.t. X")
fdtest(wf, Σ, g, X)


# ##

# todo: move this to a performance benchmark script 
# using BenchmarkTools
# @btime $wf($X)
# @btime gradient($wf, $X)

##

A, ∇A, ΔA = ACEpsi._assemble_A_∇A_ΔA(wf, X, Σ)

@info("test ∇A")
grad_test(X -> ACEpsi._assemble_A_∇A_ΔA(wf, X, Σ)[1], 
          X -> ACEpsi._assemble_A_∇A_ΔA(wf, X, Σ)[2], 
          X)

@info("test ΔA")          
lap_test(X -> ACEpsi._assemble_A_∇A_ΔA(wf, X, Σ)[1], 
         X -> ACEpsi._assemble_A_∇A_ΔA(wf, X, Σ)[3], 
         X)

 
##

function f_AA(X, Σ)
   A, ∇A, ΔA = ACEpsi._assemble_A_∇A_ΔA(wf, X, Σ)
   AA, ∇AA, ΔAA = ACEpsi._assemble_AA_∇AA_ΔAA(A, ∇A, ΔA, wf)
   return AA, ∇AA, ΔAA
end

@info("test ∇A")
grad_test(X -> f_AA(X, Σ)[1], X -> f_AA(X, Σ)[2], X)

@info("test ΔA")          
lap_test(X -> f_AA(X, Σ)[1], X -> f_AA(X, Σ)[3], X)


##

@info("test Δψ")
lap_test(X -> [wf(X, Σ);;], X -> [ACEpsi.laplacian(wf, X, Σ);;], X)


##

@info("Test ∇ψ w.r.t. parameters")

ACEpsi.gradp_evaluate(wf, X, Σ)


W0 = copy(wf.W)
w0 = W0[:]
Fp = w -> ( wf.W[:] .= w[:]; wf(X, Σ))
dFp = w -> ( wf.W[:] .= w[:]; ACEpsi.gradp_evaluate(wf, X, Σ)[1][:] )

grad_test2(Fp, dFp, w0)

##

@info("test ∇env w.r.t. parameter")
Ξ0 =  copy(wf.envelope.ξ)
ξ0 =  Ξ0

Envp = w -> (wf.envelope.ξ = w; wf(X, Σ))
dEnvp = w -> (wf.envelope.ξ = w;  ACEpsi.gradp_evaluate(wf, X, Σ)[2])

grad_test3(Envp, dEnvp, ξ0)

##

@info("Test ∇Δψ w.r.t. parameters")

Fp = w -> ( wf.W[:] .= w[:]; ACEpsi.laplacian(wf, X, Σ))
dFp = w -> ( wf.W[:] .= w[:]; ACEpsi.gradp_laplacian(wf, X, Σ)[1][:] )

grad_test2(Fp, dFp, w0)

##

@info("Test ∇Δenv w.r.t. parameters")

Ξ0 =  copy(wf.envelope.ξ)
ξ0 =  Ξ0


Fp = w -> ( wf.envelope.ξ = w; ACEpsi.laplacian(wf, X, Σ))
dFp = w -> (wf.envelope.ξ = w; ACEpsi.gradp_laplacian(wf, X, Σ)[2][:] )

grad_test3(Fp, dFp, ξ0)

##

@info("Test getting/setting parameters")

wf1 = BFwf(Nel, polys; ν=3)
wf2 = BFwf(Nel, polys; ν=3)
@printf(" wf1 - wf2: %f \n", abs(wf1(X, Σ) - wf2(X, Σ)))
param1 = ACEpsi.get_params(wf1)
wf2 = ACEpsi.set_params!(wf2, param1)
@printf(" wf1 - wf2 with parameter wf1: %f \n", abs(wf1(X, Σ) - wf2(X, Σ)))

##

# @warn("removed compac test since json file is missing")
# @info("Test compatibility with ACESchrodinger") # Jerry: Not sure if this should be kept in the same file
# include("compare_bflow.jl")

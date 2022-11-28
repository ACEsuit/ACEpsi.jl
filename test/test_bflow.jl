
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
   for h in sqrt(0.1).^((2:10))
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
   for h in 0.1.^(2:10)
      gh = [ (f(X + h * EE[:, i]) - F) / h for i = 1:nX ]
      @printf(" %.1e | %.2e \n", h, norm(gh - ∇F, Inf))
   end
end

##

Nel = 4

trans = x -> atan(x)/π
envelope = X -> exp(- 0.1 * sum(X.^2))

polys = legendre_basis(6)
wf = BFwf(Nel, polys; ν=3, trans=trans, envelope=envelope)

X = 2 * rand(Nel) .- 1
wf(X)
g = gradient(wf, X)
laplacian(wf, X)

##

using ACEbase.Testing: fdtest 

fdtest(wf, X -> gradient(wf, X), X)



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
using BenchmarkTools
@btime ACEpsi.gradp_evaluate($wf, $X)

##



using Polynomials4ML, ACEcore, ACEpsi, ACEbase, Printf
using ACEpsi: BFwf, gradient, evaluate
using LinearAlgebra

##

Nel = 4
polys = legendre_basis(6)
wf = BFwf(Nel, polys; ν=3)

X = 2 * rand(Nel) .- 1
wf(X)
g = gradient(wf, X)

##

using ACEbase.Testing: fdtest 

fdtest(wf, X -> gradient(wf, X), X)



##

X = 2 * rand(Nel) .- 1
ACEpsi.laplacian(wf, X)

## 


A, ∇A, ΔA = ACEpsi._assemble_A_dA_ddA(wf, X)

@info("test ∇A")
grad_test(X -> ACEpsi._assemble_A_dA_ddA(wf, X)[1], 
          X -> ACEpsi._assemble_A_dA_ddA(wf, X)[2], 
          X)

@info("test ΔA")          
lap_test(X -> ACEpsi._assemble_A_dA_ddA(wf, X)[1], 
         X -> ACEpsi._assemble_A_dA_ddA(wf, X)[3], 
         X)

 
##

function f_AA(X)
   A, ∇A, ΔA = ACEpsi._assemble_A_dA_ddA(wf, X)
   AA, ∇AA, ΔAA = ACEpsi._assemble_AA_∇AA_ΔAA(A, ∇A, ΔA, wf)
   return AA, ∇AA, ΔAA
end

@info("test ∇A")
grad_test(X -> f_AA(X)[1], X -> f_AA(X)[2], X)

@info("test ΔA")          
lap_test(X -> f_AA(X)[1], X -> f_AA(X)[3], X)


##

lap_test(X -> [wf(X);;], X -> [ACEpsi.laplacian(wf, X);;], X)

## 

function grad_test(f, df, X)
   F = f(X) 
   ∇F = df(X)
   nX, nF = size(F)
   U = randn(nX)
   V = randn(nF) ./ (1:nF).^2
   f0 = U' * F * V
   ∇f0 = [ U' * ∇F[:, i, :] * V for i = 1:nX ]
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

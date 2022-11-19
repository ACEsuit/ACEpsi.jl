
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


# ##

# todo: move this to a performance benchmark script 
# using BenchmarkTools
# @btime $wf($X)
# @btime gradient($wf, $X)

##

X = 2 * rand(Nel) .- 1
_A1(X) = ACEpsi._assemble_A_dA_ddA(wf, X)[1][1,:]
_ΔA1(X) = ACEpsi._assemble_A_dA_ddA(wf, X)[3][1,:]
_∂A1(X) = ForwardDiff.jacobian(_A1, X)
_∂2A1(X) = ForwardDiff.jacobian(_∂A1, X)
function _adΔA1(X)
   A = _A1(X) 
   H = reshape(_∂2A1(X), length(A), length(X), length(X))

   Δ = zeros(size(A))
   for i = 1:length(X) 
      Δ += H[:, i,  i]
   end
   return Δ
end

_A1(X)
_∂A1(X)
_ΔA1(X) ≈ _adΔA1(X)


##

A, dA, ddA = ACEpsi._assemble_A_dA_ddA(wf, X)

##

using ForwardDiff

function f(X)
   nX = length(X)
   P = wf.polys(X)
   Si = zeros(Bool, nX, 2)
   ACEpsi.onehot!(Si, 1)
   A = ACEcore.evalpool(wf.pooling, (P, Si))
   AA = ACEcore.evaluate(wf.corr, A)
   return AA
end

f(X)

ForwardDiff.jacobian(f, X)

dAA = randn(274, 10)

nX = length(X)
Si = zeros(Bool, nX, 2)
ACEpsi.onehot!(Si, 1)
using Polynomials4ML, ACEcore, ACEbase
using ACEpsi: BFwf, gradient, evaluate, envelopefcn
using JSON
using Printf
using LinearAlgebra

const ↑, ↓, ∅ = '↑','↓','∅'
using JSON

# == test configs == 
data_ortho = JSON.parse(open("/home/jerryho/julia_ws/ACEpsi.jl/test/orthopolyweights.json")) # import weights from ACESchrodinger.jl 
data = JSON.parse(open("/home/jerryho/julia_ws/ACEpsi.jl/test/bftest.json")) # import input electron position and parameter of model

ww = data_ortho[1]["ww"]
xx = data_ortho[1]["tdf"]
X = data[1]["X"]
PP = data[1]["P"]

ww = Float64.(ww)
xx = Float64.(xx)
Σ = [↑, ↑, ↓, ↓, ↓];
Nel = 5
WW = DiscreteWeights(xx, ww)
polys = orthpolybasis(10, WW)

wf = BFwf(Nel, polys; ν=2, totdeg = 10, trans = atan, envelope = envelopefcn(x -> sqrt(x^2 + 1), 0.5)) # using default
# == 

for i = 1:5
   wf.W[:, i] = PP[i][2:end] # the first entry of PP[i] is the extra constant
end

@show wf(X, Σ)

spec1p = [ (k, σ) for σ in [1, 2, 3] for k in 1:length(polys) ]  # (1, 2, 3) = (∅, ↑, ↓);

# @show displayspec(wf.spec, spec1p)

using ACEpsi, Polynomials4ML, StaticArrays
using Polynomials4ML: natural_indices, degree
using ACEpsi.AtomicOrbitals: AtomicOrbitalsBasis, Nuc, proto_evaluate


##

totdeg = 3
bRnl = ACEpsi.AtomicOrbitals.RnlExample(totdeg)
bYlm = RYlmBasis(totdeg)
nuclei = [ Nuc(3 * rand(SVector{3, Float64}), 1.0) for _=1:3 ]

bAnlm = AtomicOrbitalsBasis(bRnl, bYlm; 
                            totaldegree = totdeg, 
                            nuclei = nuclei )

spec = ACEpsi.AtomicOrbitals.get_spec(bAnlm)

##

Nel = 5
X = randn(SVector{3, Float64}, Nel)
Σ = rand(spins(), Nel)

proto_evaluate(bAnlm, X, Σ)
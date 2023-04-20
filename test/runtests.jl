using ACEpsi
using Test

@testset "ACEpsi.jl" begin
    @testset "BFwf" begin include("test_bflow.jl") end 
    @testset "AtomicOrbitals" begin include("test_atomicorbitals.jl") end
end

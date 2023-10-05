using ACEpsi
using Test

@testset "ACEpsi.jl" begin
    @testset "BFwf" begin include("test_bflow.jl") end 
    @testset "BFwf_lux" begin include("test_bflow_lux.jl") end 
    @testset "AtomicOrbitals" begin include("test_atomicorbitals.jl") end
    @testset "AtomicOrbitalsBasis" begin include("test_atorbbasis.jl") end
end

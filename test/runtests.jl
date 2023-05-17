using ACEpsi
using Test

@testset "ACEpsi.jl" begin
    #@testset "BFwf" begin include("test_bflow1.jl") end 
    @testset "BFwf_lux" begin include("test_bflow_lux.jl") end 
    @testset "AtomicOrbitals" begin include("test_atomicorbitals.jl") end
end

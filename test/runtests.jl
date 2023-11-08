using ACEpsi
using Test

@testset "ACEpsi.jl" begin
    @testset "1D" begin
        @testset "Old BFwf" begin include("test_bflow.jl") end 
        @testset "BFwf1d_lux" begin include("test_bflow_lux1d.jl") end 
        @testset "BFwf1dps_lux" begin include("test_bflow_lux1dps.jl") end 
        @testset "Wigner BFwf1dps_lux" begin include("test_bflow_lux1dWigner.jl") end 
    end
    @testset "3D" begin
        @testset "AtomicOrbitals" begin include("test_atomicorbitals.jl") end
        @testset "AtomicOrbitalsBasis" begin include("test_atorbbasis.jl") end
        @testset "BFwf_lux" begin include("test_bflow_lux.jl") end 
    end
end

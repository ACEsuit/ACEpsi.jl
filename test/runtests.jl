using ACEpsi
using Test

@testset "ACEpsi.jl" begin
    @testset "BFwf" begin include("test_bflow.jl") end 
    @testset "BFwf" begin include("test_bflows.jl") end 
end

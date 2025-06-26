using ACEpsi
using Test

@testset "ACEpsi.jl" begin
    @testset "gradient" begin include("test_gradient.jl"); end
    @testset "embed" begin include("test_embed.jl"); end
end


module vmc

abstract type opt end

abstract type sr_type end

struct QGT <: sr_type
end

struct QGTJacobian <: sr_type
end
struct QGTOnTheFly <: sr_type
end


include("Eloc.jl")
include("gradient.jl")
include("metropolis.jl")

include("vmc_utils.jl")
include("vmc.jl")
include("optimisers/adamw.jl")
include("optimisers/sr.jl")

end
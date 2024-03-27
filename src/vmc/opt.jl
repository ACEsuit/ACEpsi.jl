module vmc

abstract type opt end

abstract type uptype end

abstract type sr_type end

abstract type Scalar_type end

abstract type Norm_type end

struct QGT <: sr_type
end
struct QGTJacobian <: sr_type
end
struct QGTOnTheFly <: sr_type
end

struct scale_invariant <: Scalar_type
end
struct no_scale <: Scalar_type
end

struct _continue <: uptype
end
struct _initial <: uptype
end

struct norm_constraint <: Norm_type
    c::Number
end
struct no_constraint <: Norm_type
end

include("Eloc.jl")
include("gradient.jl")
include("metropolis.jl")

include("vmc_utils.jl")
include("vmc.jl")
include("multilevel.jl")
include("optimisers/adamw.jl")
include("optimisers/sr.jl")

include("IOs.jl")

end
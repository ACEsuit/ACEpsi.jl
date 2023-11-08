module ACEpsi

using Distributed

# define operation on HyperDualNumbers
include("hyper.jl")

# define spin symbols and some basic functionality 
include("spins.jl")

# the old 1d backflow code, keep around for now...
include("envelope.jl")
include("bflow.jl")

# the new 3d backflow code 
include("atomicorbitals/atomicorbitals.jl")
include("jastrow.jl")
include("bflow3d.jl")
include("bflow1d.jl")
include("bflow1dps.jl")

include("backflowpooling.jl")
include("backflowpooling1d.jl")
include("backflowpooling1dps.jl")
# lux utils for bflow
include("lux_utils.jl")

# vmc
include("vmc/opt.jl")

end

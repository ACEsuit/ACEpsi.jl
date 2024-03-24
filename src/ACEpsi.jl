module ACEpsi

# define operation on HyperDualNumbers
include("hyper.jl")

# define spin symbols and some basic functionality 
include("spins.jl")

# the old 1d backflow code, keep around for now...
include("bflow.jl")
include("envelope.jl")

# the new 3d backflow code 
include("atomicorbitals/atomicorbitals.jl")
include("jastrow.jl")
include("tensordecomposition/TD.jl")
include("backflowpooling.jl")

include("wf/bflow3d.jl")
include("wf/bflow3dSTucker.jl")
include("wf/bflow3dSCP.jl")
include("wf/bflow3dTD.jl")
include("wf/embed.jl")


# lux utils for bflow
include("lux_utils.jl")

# vmc
include("vmc/opt.jl")

end

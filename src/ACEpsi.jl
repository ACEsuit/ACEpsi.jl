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
include("cluster/Cluster.jl")
include("bflow3d.jl")
include("bflow3dTucker.jl")
include("bflow3dCluster.jl")
include("backflowpooling.jl")

# lux utils for bflow
include("lux_utils.jl")

# vmc
include("vmc/opt.jl")

end

module TD

abstract type Tensor_Decomposition end

abstract type TDs <: Tensor_Decomposition end

struct No_Decomposition <: Tensor_Decomposition
end

include("Tucker.jl")
include("tds/TK.jl")
include("tds/CP.jl")
include("tds/TR.jl")
include("tds/TT.jl")
include("TD_utils.jl")

end
abstract type Tensor_Decomposition end

struct No_Decomposition <: Tensor_Decomposition
end

mutable struct SCP <: Tensor_Decomposition
    P::Integer
end
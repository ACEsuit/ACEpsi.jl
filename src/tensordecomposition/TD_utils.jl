mutable struct SCP <: Tensor_Decomposition
    P::Integer
end

TDLayer(NBF::Integer, P::Integer, Nel::Integer, spec1p, spec, TD::CP) = TCLayer(NBF, P, Nel, spec1p, spec)
TDLayer(NBF::Integer, P::Integer, Nel::Integer, spec1p, spec, TD::TK) = TKLayer(NBF, P, Nel, spec1p, spec)
TDLayer(NBF::Integer, P::Integer, Nel::Integer, spec1p, spec, TD::TT) = TTLayer(NBF, P, Nel, spec1p, spec)
TDLayer(NBF::Integer, P::Integer, Nel::Integer, spec1p, spec, TD::TR) = TRLayer(NBF, P, Nel, spec1p, spec)

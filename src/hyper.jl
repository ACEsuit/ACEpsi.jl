using HyperDualNumbers: Hyper

Base.real(x::Hyper{<:Number}) = Hyper(real(x.value), real(x.epsilon1), real(x.epsilon2), real(x.epsilon12))  

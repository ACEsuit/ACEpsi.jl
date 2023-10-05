using LinearAlgebra

function InverseLR(ν, lr, lr_dc)
    return lr / (1 + ν / lr_dc), ν+1
end

function acc_adjust(k::Int, Δt::Number, acc_opt::AbstractVector, acc_range::AbstractVector, acc_step::Int)
    if mod(k, acc_step) == 0
        if mean(acc_opt) < acc_range[1]
            Δt = Δt * exp(1/10 * (mean(acc_opt) - acc_range[1])/acc_range[1])
        elseif mean(acc_opt) > acc_range[2]
            Δt = Δt * exp(1/10 * (mean(acc_opt) - acc_range[2])/acc_range[2])
        end
    end
    return Δt
end






        
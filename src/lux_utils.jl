using LuxCore
using HyperDualNumbers

function replace_namedtuples(nt, rp_st, Σ)
    if :tmp in keys(nt) || :pool in keys(nt) || length(nt) == 0 || :ζ in keys(nt) 
        return (; nt...)
    else
        for i in 1:length(keys(nt))
            # @show nt[i]
            if length(nt[i]) == 0         
                rp_st = (; rp_st..., (; keys(nt)[i] => (Σ = Σ, ))...)
            else
                rp_st = (; rp_st..., keys(nt)[i] => replace_namedtuples(nt[i], (;), Σ))
            end
        end
        return rp_st
    end
end

function setupBFState(rng, bf, Σ)
    ps, st = LuxCore.setup(rng, bf)
    rp_st = replace_namedtuples(st, (;), Σ)
    return ps, rp_st
end

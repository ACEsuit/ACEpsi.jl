using Distributed
using ACEpsi: _classfy

export write_args

"""
write_args

vmc::VMC_multilevel
sam::MHSampler

return a dictionary for the configs of training
"""
function write_args(vmc::VMC_multilevel, sam::MHSampler, accMCMC, 
    spec1p_list, spec_list, specAO_list, Nlm_list, dist_list,
    wf_list, ps_list, st_list)
    
    
    # res holder
    res = Dict()

    # system settings
    res["system"] = Dict(
        "Nel" => sam.Nel,
        "nuclei" => sam.physical_config.nuclei,
        "Σ" => st_list[1].branch.js.Σ, 
    )

    # write basis configurations
    res["configs"] = Dict(
        "accMCMC" => string.(accMCMC),
        "tol" => vmc.tol, 
        "MaxIter" => string.(vmc.MaxIter),
        "lr" => vmc.lr,
        "lr_dc" => vmc.lr_dc,
        "utype" => string(vmc.utype),

        # optimizer
        "opt" => _write_dict(vmc.type),

        # type of basis
        "Pds" => string.(typeof(wf_list[1].layers.branch.layers.bf.layers.Pds)),
        "XYlm" => string.(typeof(wf_list[1].layers.branch.layers.bf.layers.Pds.bYlm)),

        # Number of BFs
        "NBFs" => _get_nbfs.(wf_list),

        # specs
        "spec1p_list" => spec1p_list, 
        "spec_list" => spec_list,
        "specAO_list" => specAO_list,
        "Nlm_list" => Nlm_list,
        "dist_list" => dist_list,       

        # type of tensor decomposition 
        "TD_list" => string.(_classfy.(ps_list)),
        )

    # sampler

    res["sampler"] = Dict(
        "nchains" => nprocs() * sam.nchains, 
        "burnin" => sam.burnin,
        "el_config" => sam.physical_config.el_config,
        "nuclei" => string.(sam.physical_config.nuclei),
        "inuc" => string.(sam.physical_config.inuc),    
    )

    # if there is a JS 
    try
        res["JS factor"] = _get_js.(wf_list)
    catch
        @warn("Error getting JS factor - please check")
        res["JS factor"] = ""
    end


    return res
end

_get_js(wf) = string(typeof(wf.layers.branch.layers.js))
_get_nbfs(wf) = wf.layers.branch.layers.bf.layers.sum.Ndet

function _write_dict(opt::SR)
    d = Dict(
        "ϵ1" => opt.ϵ₁,
        "ϵ2" => opt.ϵ₂,
        "β1" => opt.β₁,
        "β2" => opt.β₂,
        "sr_type" => string.(typeof(opt._sr_type)),
        "scalar_type" => string.(typeof(opt.st)),
        "Norm_type" => string.(typeof(opt.nt)),
    ) 
    return d
end
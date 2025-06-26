export transfer_weights!, model_generator

function transfer_weights!(ps1, ps2, spec1, spec2, spec1p1, spec1p2)
    readable_spec1 = displayspec(spec1, spec1p1)
    readable_spec2 = displayspec(spec2, spec1p2)
    _map  = _invmap(readable_spec2)
    ps2.branch.bf.linear.weight .= 0.0
    for (idx, t) in enumerate(readable_spec1)
        ps2.branch.bf.linear.weight[:, _map[t]] = ps1.branch.bf.linear.weight[:, idx]
    end
    return ps2
end

function transfer_weights_idx!(ps1, ps2, spec1, spec2, spec1p1, spec1p2)
    p, s = destructure(ps1)
    ips = s(collect(1:length(p)[1]))
    ips2 = transfer_weights!(ips, deepcopy(ps2), spec1, spec2, spec1p1, spec1p2)
    index, = destructure(ips2) 
    return index
end

function model_generator(mol, basis_set, totdeg, ν; ratio = 0.5, multilevel = true, filename = "basis.json")
    if multilevel
        totdeg_list, ν_list = build_totdeglevels(mol, basis_set, totdeg, ν; ratio = ratio)
        results = [build_wavefunction(mol, basis_set, totdeg_list[i], ν_list[i]; filename = filename) for i = 1:length(totdeg_list)]
        model_list = getindex.(results, 1)
        ps_list    = getindex.(results, 2)
        st_list    = getindex.(results, 3)
        spec_list = getindex.(results, 4)
        spec1p_list = getindex.(results, 5)
        return model_list, ps_list, st_list, spec_list, spec1p_list, totdeg_list, ν_list
    else
        model, ps, st, spec, spec1p = build_wavefunction(mol, basis_set, totdeg, ν; filename = fieldname)
        return [model], [ps], [st], [spec], [spec1p], [totdeg], [ν]
    end
end

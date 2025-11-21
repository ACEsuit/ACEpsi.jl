export model_generator

function displayspec(spec, spec1p)
    nicespec = []
    for k = 1:length(spec)
        push!(nicespec, [spec1p[spec[k][j]] for j = 1:length(spec[k])])
    end
    return nicespec
end

function _invmap(a)
    inva = Dict{eltype(a), Int}()
    for i = 1:length(a) 
        inva[a[i]] = i 
    end
   return inva 
end

function transfer_weights!(ps1, ps2, spec1, spec2, spec1p1, spec1p2)
    _map  = _invmap(spec2)
    if hasproperty(ps2.branch.bf, :dense)
        ps2.branch.bf.dense.weight .= 0.0
        for (idx, t) in enumerate(spec1)
            ps2.branch.bf.dense.weight[:, _map[t]] = ps1.branch.bf.dense.weight[:, idx]
        end
    end
    if hasproperty(ps2.branch.bf, :branchs)
        ps2.branch.bf.branchs.bfs.dense.weight .= 0.0
        for (idx, t) in enumerate(spec1)
            ps2.branch.bf.branchs.bfs.dense.weight[:, _map[t]] = ps1.branch.bf.branchs.bfs.dense.weight[:, idx]
        end
        ps2.branch.bf.branchs.jsen.P .= ps1.branch.bf.branchs.jsen.P
        ps2.branch.bf.branchs.jsen.L .= ps1.branch.bf.branchs.jsen.L
    end

    if hasproperty(ps1.branch.bf, :branch) && hasproperty(ps2.branch.bf, :branch)
        for i = 1:length(ps2.branch.bf.branch)
            b2 = ps2.branch.bf.branch[i]
            _mapAO = _invmap(spec1p2[i])
            if hasproperty(ps2.branch.bf.branch[i], :ζ) 
                ps2.branch.bf.branch[i].ζ .= 1.0
                for (idx, t) in enumerate(spec1p1[i])
                    ps2.branch.bf.branch[i].ζ[_mapAO[t], :] = ps1.branch.bf.branch[i].ζ[idx, :]
                end
            end
            if hasproperty(ps2.branch.bf.branch[i], :D)
                ps2.branch.bf.branch[i].D .= 0.001 
                for (idx, t) in enumerate(spec1p1[i])
                    ps2.branch.bf.branch[i].D[_mapAO[t], :] = ps1.branch.bf.branch[i].D[idx, :]
                end
            end
        end
    end
    return ps2
end

function transfer_weights_idx!(ps1, ps2, spec1, spec2, spec1p1, spec1p2)
    p, s = destructure(ps1)
    ips = s(collect(1:length(p)))
    ips2 = transfer_weights!(ips, deepcopy(ps2), spec1, spec2, spec1p1, spec1p2)
    index, = destructure(ips2) 
    return Int.(index)
end

function model_generator(mol, nu;
                         ratio::Real = 5.,
                         multilevel::Bool = true,
                         hf::Bool = true,
                         family::Type{F} = Slater,
                         freeze_branches::Bool = true, 
                         spec1p_admissible = nothing, 
                         basis_set = "cc-pvtz") where {F<:OrbitalFamily}

    _pack(r) = (model  = r[1], ps = r[2], st = r[3], spec = r[4], spec1p = r[5])

    basis = auto_load_basis(mol, basis_set, family; filename = "basis.json", spec1p_admissible = spec1p_admissible)
    A_spec = get_spec1p(basis; spin = false)
    spec1p = get_spec1p(basis; spin = true)
    tup2b = vv -> [spec1p[v] for v in vv[vv .> 0]]
    only_one_void = bb -> ((length(bb) == 0) || sum(b.s == '∅' for b in bb) == 1)
    specAA = gensparse(; NU = nu, tup2b = tup2b,
                       admissible = bb -> true,
                       filter = only_one_void,
                       minvv = fill(0, nu),
                       maxvv = fill(length(spec1p), nu),
                       ordered = true)

    spec = [vv[vv .> 0] for vv in specAA if !isempty(vv[vv .> 0])]
    readable_spec = displayspec(spec, spec1p)

    if multilevel
        spec_sub1 = ACEpsi.layer_specs(spec, readable_spec; hf = hf, ratio = ratio)
        spec_sub = []
        for i = 1:length(spec_sub1)
            spec1 = [vv[vv .> 0] for vv in spec if vv in spec_sub1[i]]
            push!(spec_sub, spec1)
        end
        spec_sub = [[j for j in i] for i in spec_sub]
        totdeg_list = length.(spec_sub)
        nu_list     = [maximum(length.(spec_sub[i])) for i = 1:length(spec_sub)]
        results = [ACEpsi.build_wavefunction(mol;
                                         family = family,
                                         freeze_branches = freeze_branches, 
                                         spec_admissible = i, 
                                         spec1p_admissible = spec1p_admissible) for i in spec_sub];

        recs    = map(_pack, results)
        models  = getfield.(recs, :model)
        ps_list = getfield.(recs, :ps)
        st_list = getfield.(recs, :st)
        spec    = getfield.(recs, :spec)
        spec1p  = getfield.(recs, :spec1p)
         
        indx = []
        for i = 1:length(spec)
            if length(spec[i]) >= mol.Nel
                push!(indx, i)
            end
        end
        models     = models[indx]
        ps_list    = ps_list[indx]      
        st_list    = st_list[indx]
        spec       = spec[indx]
        spec1p     = spec1p[indx]
        totdeg_list = totdeg_list[indx]
        nu_list     = nu_list[indx]

        idx = findlast(==(1), nu_list)
        idx === nothing && error("hf=true but no level with nu == 1 was produced by build_totdeglevels")

        slicer     = idx:length(nu_list)
        models     = models[slicer]
        ps_list    = ps_list[slicer]
        st_list    = st_list[slicer]
        spec       = spec[slicer]
        spec1p     = spec1p[slicer]
        totdeg_list = totdeg_list[slicer]
        nu_list     = nu_list[slicer]

        select_even_keep_last(v) = iseven(length(v)) ? [v[1:2:end]..., v[end]] : v[1:2:end]
        models      = select_even_keep_last(models)
        ps_list     = select_even_keep_last(ps_list)
        st_list     = select_even_keep_last(st_list)
        spec        = select_even_keep_last(spec)
        spec1p      = select_even_keep_last(spec1p)
        totdeg_list = select_even_keep_last(totdeg_list)
        nu_list     = select_even_keep_last(nu_list)

        if family == Gaussian()

            mol_map = Dict(
                ACEpsi.molecules.He => "He",
                ACEpsi.molecules.Li => "Li",
                ACEpsi.molecules.Be => "Be",
                ACEpsi.molecules.B  => "B",
                ACEpsi.molecules.C  => "C",
                ACEpsi.molecules.N  => "N",
                ACEpsi.molecules.O  => "O",
                ACEpsi.molecules.F  => "F",
                ACEpsi.molecules.Ne  => "Ne",
                ACEpsi.molecules.LiH  => "LiH",
                ACEpsi.molecules.Li2  => "Li2",
                ACEpsi.molecules.H2O  => "H2O",
            )
            ps_list[1] = init_hf(mol_map[mol], ps_list[1], basis_set)
        end
        return (model  = [i for i in models],
                ps     = [i for i in ps_list],
                st     = [i for i in st_list],
                spec   = [[j for j in i] for i in spec],
                spec1p = [i for i in spec1p],
                totdeg = [i for i in totdeg_list],
                nu     = [i for i in nu_list])
    else
        r  = build_wavefunction(mol;
                                family = family,
                                freeze_branches = freeze_branches,
                                spec1p_admissible = spec1p_admissible, 
                                spec_admissible = spec)
        pr = _pack(r)
        return (model  = [pr.model],
                ps     = [pr.ps],
                st     = [pr.st],
                spec   = [pr.spec],
                spec1p = [pr.spec1p],
                totdeg = [length(pr.spec)],
                nu     = [nu])
    end
end

function layer_specs(spec, readable_spec; hf::Bool=false, ratio::Float64=0.1, wr=0.6, wl=1.0)
    l = length.(spec)

    ItemT   = eltype(spec)
    LayersT = Vector{Vector{ItemT}}
    layers  = LayersT()

    rem           = copy(spec)
    rem_readable  = copy(readable_spec)
    curr          = ItemT[]

    if hf
        ord1_idx = findall(==(1), l)
        both_idx = findall(>(1), l)

        l_ord1 = Dict(i => (only(readable_spec[i]).l)::Int for i in ord1_idx)

        levels = sort!(unique(values(l_ord1)))
        empty!(layers)
        for L in levels
            idxL = [i for (i, li) in l_ord1 if li <= L]
            push!(layers, copy(spec[idxL]))
        end

        curr = copy(spec[ord1_idx])

        rem = ItemT[]
        append!(rem, spec[both_idx])

        rem_readable = typeof(readable_spec)()
        append!(rem_readable, readable_spec[both_idx])
    end

    if isempty(rem_readable)
        return copy.(layers)
    end

    radial_cost  = t -> max(0.0, float(t.n1))
    angular_cost = t -> float(t.l) * (float(t.l) + 1)

    rc_all = Float64[]; lc_all = Float64[]
    for x in rem_readable, t in x
        push!(rc_all, radial_cost(t))
        push!(lc_all, angular_cost(t))
    end

    normer = function(vec)
        v = sort(vec)
        med = v[cld(length(v),2)]
        q1  = v[clamp(Int(floor(0.25*length(v))), 1, length(v))]
        q3  = v[clamp(Int(ceil(0.75*length(v))), 1, length(v))]
        iqr = max(q3 - q1, 1e-8)
        f = x -> (x - med)/iqr
        return (f, med, iqr)
    end
    fr, _, _ = normer(rc_all)
    fl, _, _ = normer(lc_all)

    single = t -> (wr*fr(radial_cost(t)) + wl*fl(angular_cost(t)))

    scores = similar(rem_readable, Float64)
    for (i, x) in pairs(rem_readable)
        scores[i] = sum(map(single, x))
    end

    min_s, max_s = extrema(scores)
    if max_s == min_s
        scores .= 0.0
    else
        scores = (scores .- min_s) ./ (max_s - min_s)
    end

    keep_idx = findall(s -> s <= ratio, scores)

    bucket = x -> round(x; digits=2)
    pos_by_score = Dict{Float64, Vector{Int}}()
    for i in keep_idx
        sc = bucket(scores[i])
        push!(get!(pos_by_score, sc, Int[]), i)
    end

    selected = Int[]
    for sc in sort!(collect(keys(pos_by_score)))
        append!(selected, pos_by_score[sc])
        push!(layers, vcat(curr, rem[selected]))
    end

    return copy.(layers)
end

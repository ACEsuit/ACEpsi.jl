export EmbeddingW!, _invmap, VMC_multilevel, wf_multilevel, gd_GradientByVMC_multilevel
using Printf
using LinearAlgebra
using Optimisers
using Polynomials4ML
using Random
using ACEpsi: BackflowPooling, BFwf_lux, setupBFState, Jastrow, displayspec
using ACEpsi.AtomicOrbitals: _invmap, Nuc, make_nlms_spec
using ACEpsi.TD: Tensor_Decomposition, No_Decomposition, Tucker
using Plots

mutable struct VMC_multilevel
    tol::Float64
    MaxIter::Vector{Int}
    lr::Float64
    lr_dc::Float64
    type::opt
    utype::uptype
end

VMC_multilevel(MaxIter::Vector{Int}, lr::Float64, type; tol = 1.0e-3, lr_dc = 50.0) = VMC_multilevel(tol, MaxIter, lr, lr_dc, type, _initial());
     
# TODO: this should be implemented to recursively embed the wavefunction

function _invmapAO(a::AbstractVector)
    inva = Dict{eltype(a), Int}()
    for i = 1:length(a) 
       inva[a[i]] = i 
    end
    return inva 
end

function gd_GradientByVMC_multilevel(opt_vmc::VMC_multilevel, sam::MHSampler, ham::SumH, wf_list, ps_list, st_list, spec_list, spec1p_list, specAO_list, Nlm_list; 
                                        verbose = true, density = false, 
                                        accMCMC = [10, [0.45, 0.55]], 
                                        batch_size = 1)
    # first level
    wf = wf_list[1]
    ps = ps_list[1]
    st = st_list[1]
    spec = spec_list[1]
    spec1p = spec1p_list[1]
    specAO = specAO_list[1]
    Nlm = Nlm_list[1]
    mₜ, vₜ = initp(opt_vmc.type, ps_list[1])
    sam.Ψ = wf
    ν = 1
    # burnin 
    res, λ₀, α = 1.0, 0., opt_vmc.lr
    err_opt = [zeros(opt_vmc.MaxIter[i]) for i = 1:length(opt_vmc.MaxIter)]

    x0, ~, acc = sampler_restart(sam, ps, st, batch_size = batch_size)

    density && begin 
        x = reduce(vcat,reduce(vcat,x0))
        display(histogram(x, xlim = (-10,10), ylim = (0,1), normalize=:pdf))
    end

    acc_step, acc_range = accMCMC
    acc_opt = zeros(acc_step)
    
    verbose && @printf("Initialize MCMC: Δt = %.2f, accRate = %.4f \n", sam.Δt, acc)
    verbose && @printf("   k |  𝔼[E_L]  |  V[E_L] |   res   |   LR    |accRate|   Δt    \n")
    for l in 1:length(wf_list)
       # do embeddings
       if l > 1
            wf = wf_list[l]
            p, s = destructure(ps)
            # embed for mt and vt
            ips = s(collect(1:length(p)))
            ips = EmbeddingP!(ips, ps_list[l], spec, spec_list[l], spec1p, spec1p_list[l], specAO, specAO_list[l], Nlm, Nlm_list[l])
            index, = destructure(ips) 
            mₜ, vₜ = updatep(opt_vmc.type, opt_vmc.utype, ps_list[l], index, mₜ, vₜ )
            # embed for ps
            ps = EmbeddingW!(ps, ps_list[l], spec, spec_list[l], spec1p, spec1p_list[l], specAO, specAO_list[l], Nlm, Nlm_list[l])
            st = st_list[l]
            Nlm = Nlm_list[l]
            spec = spec_list[l]
            specAO = specAO_list[l]
            spec1p = spec1p_list[l]
            sam.Ψ = wf
       end
       v = maximum(length.(spec))
       _Nbf = length(keys(ps.branch.bf.hidden))
       if :hidden in keys(ps.branch.bf.hidden.layer_1)
            _basis_size = size(ps.branch.bf.hidden.layer_1.hidden.W, 2)
            @info("level = $l, order = $v, size of basis = $_basis_size, number of bfs = $_Nbf")
       else 
            @info("level = $l, order = $v, number of bfs = $_Nbf")
       end
       # optimization
       for k = 1 : opt_vmc.MaxIter[l]
          sam.x0 = x0
          
          # adjust Δt
          acc_opt[mod(k,acc_step)+1] = acc
          sam.Δt = acc_adjust(k, sam.Δt, acc_opt, acc_range, acc_step)
 
          # adjust learning rate
          α, ν = InverseLR(ν, opt_vmc.lr, opt_vmc.lr_dc)
 
          # optimization
          ps, acc, λ₀, res, σ, x0, mₜ, vₜ = Optimization(opt_vmc.type, wf, ps, st, sam, ham, α, mₜ, vₜ, ν, batch_size = batch_size)
          density && begin 
            if k % 10 == 0
                x = reduce(vcat,reduce(vcat,x0))
                display(histogram(x, xlim = (-10,10), ylim = (0,1), normalize=:pdf))
            end
          end 
          
          # err
          verbose && @printf(" %3.d | %.5f | %.5f | %.5f | %.5f | %.3f | %.3f \n", k, λ₀, σ, res, α, acc, sam.Δt)
          err_opt[l][k] = λ₀
 
          if res < opt_vmc.tol
                ps_list[l] = deepcopy(ps)
                break;
          end  
       end
       ps_list[l] = deepcopy(ps)
       opt_vmc.lr = α
    end
    
    return wf_list, err_opt, ps_list
end

function dropnames(namedtuple::NamedTuple, names::Tuple{Vararg{Symbol}}) 
    keepnames = Base.diff_names(Base._nt_names(namedtuple), names)
    return NamedTuple{keepnames}(namedtuple)
 end

# slaterbasis
function wf_multilevel(Nel::Int, Σ::Vector{Char}, nuclei::Vector{Nuc{T}}, 
    Dn::SlaterBasis, Pn::OrthPolyBasis1D3T, bYlm::Union{RYlmBasis, CYlmBasis, CRlmBasis, RRlmBasis},
    _spec, speclist::Vector{Int}, Nbf::Vector{Int},  
    totdegree::Vector{Int}, ν::Vector{Int}, TD::Vector{TT}) where {T, TT<:Tensor_Decomposition}
    level = length(ν)
    Nlm, wf, spec, spec1p, ps, st = [], [], [], [], [], []
    for i = 1:level
        bRnl = [AtomicOrbitalsRadials(Pn, SlaterBasis(10 * rand(length(_spec[i][j]))), _spec[i][speclist[j]]) for j = 1:length(_spec[i])]

        Nnuc = length(speclist)
        spec_Ylm = natural_indices(bYlm); inv_Ylm = _invmap(spec_Ylm)
        _spec1idx = []
        for j = 1:Nnuc
            spec1 = make_nlms_spec(bRnl[speclist[j]], bYlm, totaldegree = totdegree[i])
            spec1idx = Vector{Tuple{Int, Int}}(undef, length(spec1))
            spec_Rnl = natural_indices(bRnl[speclist[j]]); inv_Rnl = _invmap(spec_Rnl)
            for (z, b) in enumerate(spec1)
                spec1idx[z] = (inv_Rnl[dropnames(b,(:m,))], inv_Ylm[(l=b.l, m=b.m)])
            end
            push!(_spec1idx, spec1idx)
        end
        sparsebasis = [SparseProduct(_spec1idx[j]) for j = 1:Nnuc]
        push!(Nlm, [length(sparsebasis[speclist[z]].spec) for z = 1:Nnuc])
        _wf, _spec1, _spec1p = BFwf_lux(Nel, Nbf[i], speclist, bRnl, bYlm, nuclei, TD[i]; totdeg = totdegree[i], ν = ν[i])
        _ps, _st = setupBFState(MersenneTwister(1234), _wf, Σ)
        push!(wf, _wf)
        push!(spec, _spec1)
        push!(spec1p, _spec1p)
        push!(ps, _ps)
        push!(st, _st)
    end
    return wf, spec, spec1p, _spec, ps, st, Nlm
end

# gaussianbasis
function wf_multilevel(Nel::Int, Σ::Vector{Char}, nuclei::Vector{Nuc{T}}, 
    Dn::GaussianBasis, Pn::OrthPolyBasis1D3T, bYlm::Union{RYlmBasis, CYlmBasis, CRlmBasis, RRlmBasis},
    _spec, speclist::Vector{Int}, Nbf::Vector{Int},  
    totdegree::Vector{Int}, ν::Vector{Int}, TD::Vector{TT}) where {T, TT<:Tensor_Decomposition}
    level = length(ν)
    wf, spec, spec1p, ps, st = [], [], [], [], []
    for i = 1:level
        bRnl = [AtomicOrbitalsRadials(Pn, GaussianBasis(10 * rand(length(_spec[i][j]))), _spec[i][speclist[j]]) for j = 1:length(_spec[i])]
        _wf, _spec1, _spec1p = BFwf_lux(Nel, Nbf[i], speclist, bRnl, bYlm, nuclei, TD[i]; totdeg = totdegree[i], ν = ν[i])
        _ps, _st = setupBFState(MersenneTwister(1234), _wf, Σ)
        push!(wf, _wf)
        push!(spec, _spec1)
        push!(spec1p, _spec1p)
        push!(ps, _ps)
        push!(st, _st)
    end
    return wf, spec, spec1p, _spec, ps, st
end

function wf_multilevel(Nel::Int, Σ::Vector{Char}, nuclei::Vector{Nuc{T}}, 
    Dn::Vector{Vector{STO_NG}}, Pn::OrthPolyBasis1D3T, bYlm::Union{RYlmBasis, CYlmBasis, CRlmBasis, RRlmBasis},
    _spec, speclist::Vector{Int}, Nbf::Vector{Int},  
    totdegree::Vector{Int}, ν::Vector{Int}, TD::Vector{TT}) where {T, TT<:Tensor_Decomposition}
    level = length(ν)
    wf, spec, spec1p, ps, st = [], [], [], [], []
    for i = 1:level
        bRnl = [AtomicOrbitalsRadials(Pn, Dn[i][speclist[j]], _spec[i][speclist[j]]) for j = 1:length(_spec[i])]
        _wf, _spec1, _spec1p = BFwf_lux(Nel, Nbf[i], speclist, bRnl, bYlm, nuclei, TD[i]; totdeg = totdegree[i], ν = ν[i])
        _ps, _st = setupBFState(MersenneTwister(1234), _wf, Σ)
        push!(wf, _wf)
        push!(spec, _spec1)
        push!(spec1p, _spec1p)
        push!(ps, _ps)
        push!(st, _st)
    end
    return wf, spec, spec1p, _spec, ps, st
end

function EmbeddingW!(ps, ps2, spec, spec2, spec1p, spec1p2, specAO, specAO2, Nlm, Nlm2)
    readable_spec = displayspec(spec, spec1p)
    readable_spec2 = displayspec(spec2, spec1p2)
    @assert size(ps.branch.bf.hidden.layer_1.hidden.W, 1) == size(ps2.branch.bf.hidden.layer_1.hidden.W, 1)
    @assert size(ps.branch.bf.hidden.layer_1.hidden.W, 2) ≤ size(ps2.branch.bf.hidden.layer_1.hidden.W, 2)
    @assert all(t in readable_spec2 for t in readable_spec)
    @assert length(specAO) == length(specAO2)
    for i = 1:length(specAO)
        @assert all(t in specAO2[i] for t in specAO[i])
    end

    # set all parameters to zero
    for i in keys(ps2.branch.bf.hidden)
        ps2.branch.bf.hidden[i].hidden.W .= 0.0
    end

    # _map[spect] = index in readable_spec2
    _map  = _invmap(readable_spec2)
    # embed
    for i in keys(ps.branch.bf.hidden)
        for (idx, t) in enumerate(readable_spec)
            ps2.branch.bf.hidden[i].hidden.W[:, _map[t]] = ps.branch.bf.hidden[i].hidden.W[:, idx]
        end
    end
    if :Pds in keys(ps.branch.bf)
        for i = 1:length(specAO2)
            ps2.branch.bf.Pds.ζ[i] .= 1.0
            _mapAO = _invmapAO(specAO2[i])  
            for (idx, t) in enumerate(specAO[i])
                ps2.branch.bf.Pds.ζ[i][_mapAO[t]] = ps.branch.bf.Pds.ζ[i][idx]
            end
        end
    end

    if :TK in keys(ps2.branch.bf)
        ps2.branch.bf.TK.W .= 0.0
        W = ps.branch.bf.TK.W
        idx = []
        for ii = 1:length(Nlm2)
            for k = 1:Nlm[ii]
                push!(idx, _ind(ii, k, Nlm2))
            end
        end
        ps2.branch.bf.TK.W[:,:,1:size(W)[3],idx] .= W
    end
    return ps2
end

function _ind(ii::Integer, k::Integer, Nnlm::Vector{TI}) where {TI} 
    return sum(Nnlm[1:ii-1]) + k
end

function EmbeddingP!(ps, ps2, spec, spec2, spec1p, spec1p2, specAO, specAO2, Nlm, Nlm2)
    readable_spec = displayspec(spec, spec1p)
    readable_spec2 = displayspec(spec2, spec1p2)
    @assert size(ps.branch.bf.hidden.layer_1.hidden.W, 1) == size(ps2.branch.bf.hidden.layer_1.hidden.W, 1)
    @assert size(ps.branch.bf.hidden.layer_1.hidden.W, 2) ≤ size(ps2.branch.bf.hidden.layer_1.hidden.W, 2)
    @assert all(t in readable_spec2 for t in readable_spec)
    for i = 1:length(specAO)
        @assert all(t in specAO2[i] for t in specAO[i])
    end

    # set all parameters to zero
    for i in keys(ps2.branch.bf.hidden)
        ps2.branch.bf.hidden[i].hidden.W .= 0.0
    end

    # _map[spect] = index in readable_spec2
    _map  = _invmap(readable_spec2)
    # embed
    for i in keys(ps.branch.bf.hidden)
        for (idx, t) in enumerate(readable_spec)
            ps2.branch.bf.hidden[i].hidden.W[:, _map[t]] = ps.branch.bf.hidden[i].hidden.W[:, idx]
        end
    end
    if :Pds in keys(ps.branch.bf)
        for i = 1:length(specAO2)
            ps2.branch.bf.Pds.ζ[i] .= 0.0
            _mapAO = _invmapAO(specAO2[i])  
            for (idx, t) in enumerate(specAO[i])
                ps2.branch.bf.Pds.ζ[i][_mapAO[t]] = ps.branch.bf.Pds.ζ[i][idx]
            end
        end
    end

    if :TK in keys(ps.branch.bf)
        ps2.branch.bf.TK.W .= 0.0
        ps2.branch.bf.TK.W[:,:,1:size(ps.branch.bf.TK.W)[3],1:size(ps.branch.bf.TK.W)[4]] .= ps.branch.bf.TK.W
    end
    return ps2
end

export EmbeddingW!, _invmap, VMC_multilevel, wf_multilevel, gd_GradientByVMC_multilevel
using Printf
using LinearAlgebra
using Optimisers
using Polynomials4ML
using Random
using ACEpsi
using ACEpsi: BackflowPooling, BFwf_lux, setupBFState, Jastrow, displayspec
using ACEpsi.AtomicOrbitals: _invmap, Nuc, make_nlms_spec
using ACEpsi.TD: Tensor_Decomposition, No_Decomposition, Tucker
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


function gd_GradientByVMC_multilevel(opt_vmc::VMC_multilevel, sam::MHSampler, ham::SumH, wf_list, ps_list, st_list, spec_list, spec1p_list, specAO_list, Nlm_list, dist_list; 
                                        verbose = true, density = false, 
                                        accMCMC = [10, [0.45, 0.55]], 
                                        batch_size = 1)
    # first level
    wf, ps, st, spec, spec1p, specAO, Nlm, dispec = wf_list[1], ps_list[1], st_list[1], spec_list[1], spec1p_list[1], specAO_list[1], Nlm_list[1], dist_list[1]
    mₜ, vₜ = initp(opt_vmc.type, ps_list[1])
    sam.Ψ = wf

    # burnin 
    res, λ₀, α, ν = 1.0, 0., opt_vmc.lr, 1
    err_opt = [zeros(opt_vmc.MaxIter[i]) for i = 1:length(opt_vmc.MaxIter)]
    x0, ~, acc = sampler(sam, sam.burnin, ps, st; batch_size = batch_size)#, return_Ψx0 = false)

    density && begin 
        x = reduce(vcat,reduce(vcat,x0))
        display(histogram(x, xlim = (-10,10), ylim = (0,1), normalize=:pdf))
    end

    @everywhere begin
        acc_step, acc_range = $accMCMC
        acc_opt = zeros(acc_step)
    end
    
    verbose && @printf("Initialize MCMC: Δt = %.2f, accRate = %.4f \n", sam.Δt, acc)
    verbose && @printf("   k |  𝔼[E_L]  |  V[E_L] |   res   |   LR    |accRate|   Δt  |free_memory  \n")
    for l in 1:length(wf_list)
        # do embeddings
        if l > 1
            wf = wf_list[l]
            p, s = destructure(ps)
            # embed for mt and vt
            ips = s(collect(1:length(p)))
            ips = EmbeddingW!(ips, ps_list[l], spec, spec_list[l], spec1p, spec1p_list[l], specAO, specAO_list[l], Nlm, Nlm_list[l], dispec, dist_list[l]; c = 0.0)
            index, = destructure(ips) 
            mₜ, vₜ = updatep(opt_vmc.type, opt_vmc.utype, ps_list[l], index, mₜ, vₜ )
            # embed for ps
            ps = EmbeddingW!(ps, ps_list[l], spec, spec_list[l], spec1p, spec1p_list[l], specAO, specAO_list[l], Nlm, Nlm_list[l], dispec, dist_list[l])
            st, Nlm, spec, specAO, spec1p, dispec = st_list[l], Nlm_list[l], spec_list[l], specAO_list[l], spec1p_list[l], dist_list[l]

            # sync over different procs
            @everywhere begin
               sam.Ψ = $wf
               ps = $ps
               st = $st 
            end
        end
        v, _Nbf, _basis_size = maximum(length.(spec)), length(keys(ps.branch.bf.hidden)), ACEpsi._size(ps)
        @info("level = $l, order = $v, size of basis = $_basis_size, number of bfs = $_Nbf")
        # optimization
        for k = 1 : opt_vmc.MaxIter[l]

            ν += 1
            # TODO: This is a bug that has problem with GC and OOM?
            Sys.free_memory() / Sys.total_memory() < 0.2 && GC.gc()
            # we don't have to set x0 here - was done on each proc
            #@everywhere sam.x0 = $x0[(myid() -1) * sam.nchains + 1 : myid() * sam.nchains]
          
            # adjust Δt - this is not the same as serial - fix later!!!!
            @everywhere acc_opt[mod($k,acc_step)+1] = acc
            @everywhere sam.Δt = acc_adjust($k, sam.Δt, acc_opt, acc_range, acc_step)
 
            # adjust learning rate
            α, ~ = InverseLR(k, opt_vmc.lr, opt_vmc.lr_dc)
 
            # optimization
            #begin_time = time()
            ps, acc, λ₀, res, σ, mₜ, vₜ = Optimization(opt_vmc.type, wf, ps, st, sam, ham, α, mₜ, vₜ, ν, batch_size = batch_size)
            #println("Total time: ", time() - begin_time)
            # density && begin 
            #     if k % 10 == 0
            #         x = reduce(vcat,reduce(vcat,x0))
            #         display(histogram(x, xlim = (-10,10), ylim = (0,1), normalize=:pdf))
            #     end
            # end 
          
            # err
            verbose && @printf(" %3.d | %.5f | %.5f | %.5f | %.5f | %.3f | %.3f | %.3f \n", k, λ₀, σ, res, α, acc, sam.Δt, Sys.free_memory() / 2^30)
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
function wf_multilevel(Nel::Int, Σ::Vector{Char}, nuclei::SVector{NNuc, Nuc{T}}, 
    Dn::SlaterBasis, Pn::OrthPolyBasis1D3T, bYlm::Union{RYlmBasis, CYlmBasis, CRlmBasis, RRlmBasis},
    _spec, speclist::Vector{Int}, Nbf::Vector{Int},  
    totdegree::Vector{Int}, ν::Vector{Int}, TD::Vector{TT}; js = ACEpsi.JPauliNet(nuclei), cluster = Nel * ones(Int, length(ν))) where {NNuc, T, TT<:Tensor_Decomposition}
    level = length(ν)
    Nlm, wf, spec, spec1p, disspec, ps, st = [], [], [], [], [], [], []
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
        _wf, _spec1, _spec1p, _disspec = BFwf_lux(Nel, Nbf[i], speclist, bRnl, bYlm, nuclei, TD[i]; totdeg = totdegree[i], ν = ν[i], js = js, cluster = cluster[i])
        _ps, _st = setupBFState(MersenneTwister(1234), _wf, Σ)
        push!(wf, _wf)
        push!(spec, _spec1)
        push!(spec1p, _spec1p)
        push!(ps, _ps)
        push!(st, _st)
        push!(disspec, _disspec)
    end
    return wf, spec, spec1p, _spec, ps, st, Nlm, disspec
end

# gaussianbasis
function wf_multilevel(Nel::Int, Σ::Vector{Char}, nuclei::SVector{NNuc, Nuc{T}}, 
    Dn::GaussianBasis, Pn::OrthPolyBasis1D3T, bYlm::Union{RYlmBasis, CYlmBasis, CRlmBasis, RRlmBasis},
    _spec, speclist::Vector{Int}, Nbf::Vector{Int},  
    totdegree::Vector{Int}, ν::Vector{Int}, TD::Vector{TT}; js = ACEpsi.JPauliNet(nuclei), cluster = Nel * ones(Int, length(ν))) where {NNuc, T, TT<:Tensor_Decomposition}
    level = length(ν)
    Nlm, wf, spec, spec1p, disspec, ps, st = [], [], [], [], [], [], []
    for i = 1:level
        bRnl = [AtomicOrbitalsRadials(Pn, Gaussian(10 * rand(length(_spec[i][j]))), _spec[i][speclist[j]]) for j = 1:length(_spec[i])]

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
        _wf, _spec1, _spec1p, _disspec = BFwf_lux(Nel, Nbf[i], speclist, bRnl, bYlm, nuclei, TD[i]; totdeg = totdegree[i], ν = ν[i], js = js, cluster = cluster[i])
        _ps, _st = setupBFState(MersenneTwister(1234), _wf, Σ)
        push!(wf, _wf)
        push!(spec, _spec1)
        push!(spec1p, _spec1p)
        push!(ps, _ps)
        push!(st, _st)
        push!(disspec, _disspec)
    end
    return wf, spec, spec1p, _spec, ps, st, Nlm, disspec
end

function wf_multilevel(Nel::Int, Σ::Vector{Char}, nuclei::SVector{NNuc, Nuc{T}}, 
    Dn::STO_NG, Pn::OrthPolyBasis1D3T, bYlm::Union{RYlmBasis, CYlmBasis, CRlmBasis, RRlmBasis},
    _spec, speclist::Vector{Int}, Nbf::Vector{Int},  
    totdegree::Vector{Int}, ν::Vector{Int}, TD::Vector{TT}; js = ACEpsi.JPauliNet(nuclei), cluster = Nel * ones(Int, length(ν))) where {NNuc, T, TT<:Tensor_Decomposition}
    level = length(ν)
    Nlm, wf, spec, spec1p, disspec, ps, st = [], [], [], [], [], [], []
    for i = 1:level
        bRnl = [AtomicOrbitalsRadials(Pn, Dn, _spec[i][speclist[j]]) for j = 1:length(_spec[i])]
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
        _wf, _spec1, _spec1p, _disspec = BFwf_lux(Nel, Nbf[i], speclist, bRnl, bYlm, nuclei, TD[i]; totdeg = totdegree[i], ν = ν[i], js = js, cluster = cluster[i])
        _ps, _st = setupBFState(MersenneTwister(1234), _wf, Σ)
        push!(wf, _wf)
        push!(spec, _spec1)
        push!(spec1p, _spec1p)
        push!(ps, _ps)
        push!(st, _st)
        push!(disspec, _disspec)
    end
    return wf, spec, spec1p, _spec, ps, st, Nlm, disspec
end

# slaterbasis
function wf_multilevel(Nel::Int, Σ::Vector{Char}, nuclei::SVector{NNuc, Nuc{T}}, 
    Dn::SlaterBasis, Pn::OrthPolyBasis1D3T, bYlm::Union{RYlmBasis, CYlmBasis, CRlmBasis, RRlmBasis},
    _spec, speclist::Vector{Int}, Nbf::Vector{Int},  
    totdegree::Vector{Int}, ν::Vector{Int}, TD::Vector{TT}, c::ACEpsi.Cluster._bf_orbital; js = ACEpsi.JPauliNet(nuclei), cluster = Nel * ones(Int, length(ν))) where {NNuc, T, TT<:Tensor_Decomposition}
    level = length(ν)
    Nlm, wf, spec, spec1p, disspec, ps, st = [], [], [], [], [], [], []
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
        _wf, _spec1, _spec1p, _disspec = BFwf_lux(Nel, Nbf[i], speclist, bRnl, bYlm, nuclei, TD[i], c; totdeg = totdegree[i], ν = ν[i], js = js, cluster = cluster[i])
        _ps, _st = setupBFState(MersenneTwister(1234), _wf, Σ)
        push!(wf, _wf)
        push!(spec, _spec1)
        push!(spec1p, _spec1p)
        push!(ps, _ps)
        push!(st, _st)
        push!(disspec, _disspec)
    end
    return wf, spec, spec1p, _spec, ps, st, Nlm, disspec
end

# gaussianbasis
function wf_multilevel(Nel::Int, Σ::Vector{Char}, nuclei::SVector{NNuc, Nuc{T}}, 
    Dn::GaussianBasis, Pn::OrthPolyBasis1D3T, bYlm::Union{RYlmBasis, CYlmBasis, CRlmBasis, RRlmBasis},
    _spec, speclist::Vector{Int}, Nbf::Vector{Int},  
    totdegree::Vector{Int}, ν::Vector{Int}, TD::Vector{TT}, c::ACEpsi.Cluster._bf_orbital; js = ACEpsi.JPauliNet(nuclei), cluster = Nel * ones(Int, length(ν))) where {NNuc, T, TT<:Tensor_Decomposition}
    level = length(ν)
    Nlm, wf, spec, spec1p, disspec, ps, st = [], [], [], [], [], [], []
    for i = 1:level
        bRnl = [AtomicOrbitalsRadials(Pn, Gaussian(10 * rand(length(_spec[i][j]))), _spec[i][speclist[j]]) for j = 1:length(_spec[i])]

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
        _wf, _spec1, _spec1p, _disspec = BFwf_lux(Nel, Nbf[i], speclist, bRnl, bYlm, nuclei, TD[i], c; totdeg = totdegree[i], ν = ν[i], js = js, cluster = cluster[i])
        _ps, _st = setupBFState(MersenneTwister(1234), _wf, Σ)
        push!(wf, _wf)
        push!(spec, _spec1)
        push!(spec1p, _spec1p)
        push!(ps, _ps)
        push!(st, _st)
        push!(disspec, _disspec)
    end
    return wf, spec, spec1p, _spec, ps, st, Nlm, disspec
end

function wf_multilevel(Nel::Int, Σ::Vector{Char}, nuclei::SVector{NNuc, Nuc{T}}, 
    Dn::STO_NG, Pn::OrthPolyBasis1D3T, bYlm::Union{RYlmBasis, CYlmBasis, CRlmBasis, RRlmBasis},
    _spec, speclist::Vector{Int}, Nbf::Vector{Int},  
    totdegree::Vector{Int}, ν::Vector{Int}, TD::Vector{TT}, c::ACEpsi.Cluster._bf_orbital; js = ACEpsi.JPauliNet(nuclei), cluster = Nel * ones(Int, length(ν))) where {NNuc, T, TT<:Tensor_Decomposition}
    level = length(ν)
    Nlm, wf, spec, spec1p, disspec, ps, st = [], [], [], [], [], [], []
    for i = 1:level
        bRnl = [AtomicOrbitalsRadials(Pn, Dn, _spec[i][speclist[j]]) for j = 1:length(_spec[i])]
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
        _wf, _spec1, _spec1p, _disspec = BFwf_lux(Nel, Nbf[i], speclist, bRnl, bYlm, nuclei, TD[i], c; totdeg = totdegree[i], ν = ν[i], js = js, cluster = cluster[i])
        _ps, _st = setupBFState(MersenneTwister(1234), _wf, Σ)
        push!(wf, _wf)
        push!(spec, _spec1)
        push!(spec1p, _spec1p)
        push!(ps, _ps)
        push!(st, _st)
        push!(disspec, _disspec)
    end
    return wf, spec, spec1p, _spec, ps, st, Nlm, disspec
end

function EmbeddingW!(ps, ps2, spec, spec2, spec1p, spec1p2, specAO, specAO2, Nlm, Nlm2, dispec, dispec2; c = 1.0)
    readable_spec = displayspec(spec, spec1p)
    readable_spec2 = displayspec(spec2, spec1p2)
    # _map[spect] = index in readable_spec2
    _map, _tucker  = _invmap(readable_spec2), ACEpsi._classfy(ps)
    Nbf1, Nbf2 = length(keys(ps.branch.bf.hidden)), length(keys(ps2.branch.bf.hidden))
    ACEpsi.embed_W!(ps, ps2, readable_spec, Nbf1, Nbf2, _map, Nlm, Nlm2, dispec, dispec2, _tucker)
    ACEpsi.embed_ζ!(ps, ps2, specAO, specAO2, c)
    return ps2
end

function _ind(ii::Integer, k::Integer, Nnlm::Vector{TI}) where {TI} 
    return sum(Nnlm[1:ii-1]) + k
end
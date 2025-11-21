using Lux
using Zygote
using NNlib
using Optimisers: destructure
using ChainRulesCore: NoTangent

function gradx_batch(model, R::AbstractArray{T, 3}, ps, st) where {T}
    N = size(R, 2); B = size(R, 3)
    val, pb = Zygote.pullback(R -> model(R, ps, st)[1], R)
    dp = similar(val)
    fill!(dp, one(T))
    G  = pb(dp)[1]
    return G
end

getlayer(m, syms::Symbol...) = foldl((acc, s)->getproperty(getproperty(acc, :layers), s),
                    syms; init=m)

function gradp_batch(model, R, ps, st)
    Nel = size(R, 2)
    if hasproperty(model.layers.branch.layers.bf.layers, :branchs)
        l_embed    = getlayer(model, :branch, :bf, :l_embed)
        l_jsen     = getlayer(model, :branch, :bf, :branchs, :jsen)
        l_branch   = getlayer(model, :branch, :bf, :branchs, :bfs, :branch)
        l_pooling  = getlayer(model, :branch, :bf, :branchs, :bfs, :pooling)
        l_corr     = getlayer(model, :branch, :bf, :branchs, :bfs, :corr)
        l_reshape_ = getlayer(model, :branch, :bf, :branchs, :bfs, :reshape_)
        l_dense    = getlayer(model, :branch, :bf, :branchs, :bfs, :dense)
        l_mask     = getlayer(model, :branch, :bf, :branchs, :bfs, :mask)
        l_prod     = getlayer(model, :branch, :bf, :prods)
        l_det_     = getlayer(model, :branch, :bf, :det_)

        ps_embed    = ps.branch.bf.l_embed;                 st_embed    = st.branch.bf.l_embed
        ps_jsen     = ps.branch.bf.branchs.jsen;            st_jsen    = st.branch.bf.branchs.jsen
        ps_branch   = ps.branch.bf.branchs.bfs.branch;      st_branch   = st.branch.bf.branchs.bfs.branch      
        ps_pooling  = ps.branch.bf.branchs.bfs.pooling;     st_pooling  = st.branch.bf.branchs.bfs.pooling
        ps_corr     = ps.branch.bf.branchs.bfs.corr;        st_corr     = st.branch.bf.branchs.bfs.corr
        ps_reshape_ = ps.branch.bf.branchs.bfs.reshape_;    st_reshape_ = st.branch.bf.branchs.bfs.reshape_
        ps_dense    = ps.branch.bf.branchs.bfs.dense;       st_dense    = st.branch.bf.branchs.bfs.dense
        ps_mask     = ps.branch.bf.branchs.bfs.mask;        st_mask     = st.branch.bf.branchs.bfs.mask
        ps_prod     = ps.branch.bf.prods;                   st_prod     = st.branch.bf.prods
        ps_det_     = ps.branch.bf.det_;                    st_det_     = st.branch.bf.det_
        
        x_embed,  st_embed  = l_embed(R, ps_embed, st_embed)
        (x_jsen, st_jsen), pb_jsen = Zygote.pullback(y -> l_jsen(y,  ps_jsen,  st_jsen), x_embed)
        
        x_branch, st_branch = l_branch(x_embed, ps_branch, st_branch)

        (x_pooling, st_pooling), pb_pooling = Zygote.pullback(y -> l_pooling(y,  ps_pooling,  st_pooling),  x_branch)
        (x_corr,    st_corr),    pb_corr    = Zygote.pullback(y -> l_corr(y,     ps_corr,     st_corr),     x_pooling)
        (x_rsh,     st_reshape_),pb_rsh     = Zygote.pullback(y -> l_reshape_(y, ps_reshape_, st_reshape_),  x_corr)
        (x_dense,   st_dense),   pb_dense   = Zygote.pullback(y -> l_dense(y,    ps_dense,    st_dense),     x_rsh)
        (x_mask,    st_mask),    pb_mask    = Zygote.pullback(y -> l_mask(y,     ps_mask,     st_mask),      x_dense)
        (x_prod,    st_prod),    pb_prod    = Zygote.pullback(y -> l_prod(y,     ps_prod,     st_prod),      (x_mask, x_jsen))
        (x_det,     st_det_),    pb_det     = Zygote.pullback(y -> l_det_(y,     ps_det_,     st_det_),      x_prod)
    
        dp = similar(x_det); fill!(dp, one(eltype(x_det)))

        dx_det   = pb_det(   (dp,      NoTangent()) )[1]
        dx_prod  = pb_prod(  (dx_det,  NoTangent()) )[1]

        dp_jsen  = ACEpsi.gradPL_per_batch(x_embed, ps_jsen.P, ps_jsen.L, st_jsen.ΣA, dx_prod[2])

        dx_mask  = pb_mask(  (dx_prod[1],  NoTangent()) )[1]
        dx_dense = pb_dense( (dx_mask, NoTangent()) )[1]
        dx_rsh   = pb_rsh(   (dx_dense,NoTangent()) )[1]
        dx_corr  = pb_corr(  (dx_rsh,  NoTangent()) )[1]
        dx_pool  = pb_pooling((dx_corr,NoTangent()))[1]

        dp_dense = ACEpsi.dense_blocks(dx_mask, x_rsh, Nel)

        dp_branch = nothing
        if length(ps_branch) > 0 && hasproperty(ps_branch[1], :ζ)
            lens     = [length(st_branch[j].spec) for j in 1:length(ps_branch)]
            offsets  = cumsum(lens)
            lo       = [1; offsets[1:end-1] .+ 1]
            hi       = offsets
            ranges   = [lo[i]:hi[i] for i in eachindex(lo)]

            dp_branch = mapreduce(i ->
            ACEpsi.branch_blocks(l_branch.layers[i],
                             dx_pool[:, :, ranges[i]],
                             x_embed[i],
                             ps_branch[i],
                             st_branch[i]),
            vcat, 1:length(lens))
        end

        blocks = (dp_jsen, dp_branch, dp_dense)
        return vcat((blk for blk in blocks if blk !== nothing)...)
    else
        l_embed    = getlayer(model, :branch, :bf, :l_embed)
        l_branch   = getlayer(model, :branch, :bf, :branch)
        l_pooling  = getlayer(model, :branch, :bf, :pooling)
        l_corr     = getlayer(model, :branch, :bf, :corr)
        l_reshape_ = getlayer(model, :branch, :bf, :reshape_)
        l_dense    = getlayer(model, :branch, :bf, :dense)
        l_mask     = getlayer(model, :branch, :bf, :mask)
        l_det_     = getlayer(model, :branch, :bf, :det_)

        ps_embed    = ps.branch.bf.l_embed;     st_embed    = st.branch.bf.l_embed
        ps_branch   = ps.branch.bf.branch;      st_branch   = st.branch.bf.branch      
        ps_pooling  = ps.branch.bf.pooling;     st_pooling  = st.branch.bf.pooling
        ps_corr     = ps.branch.bf.corr;        st_corr     = st.branch.bf.corr
        ps_reshape_ = ps.branch.bf.reshape_;    st_reshape_ = st.branch.bf.reshape_
        ps_dense    = ps.branch.bf.dense;       st_dense    = st.branch.bf.dense
        ps_mask     = ps.branch.bf.mask;        st_mask     = st.branch.bf.mask
        ps_det_     = ps.branch.bf.det_;        st_det_     = st.branch.bf.det_

        x_embed,  st_embed  = l_embed(R, ps_embed, st_embed)
        x_branch, st_branch = l_branch(x_embed, ps_branch, st_branch)

        (x_pooling, st_pooling), pb_pooling = Zygote.pullback(y -> l_pooling(y,  ps_pooling,  st_pooling),  x_branch)
        (x_corr,    st_corr),    pb_corr    = Zygote.pullback(y -> l_corr(y,     ps_corr,     st_corr),     x_pooling)
        (x_rsh,     st_reshape_),pb_rsh     = Zygote.pullback(y -> l_reshape_(y, ps_reshape_, st_reshape_),  x_corr)
        (x_dense,   st_dense),   pb_dense   = Zygote.pullback(y -> l_dense(y,    ps_dense,    st_dense),     x_rsh)
        (x_mask,    st_mask),    pb_mask    = Zygote.pullback(y -> l_mask(y,     ps_mask,     st_mask),      x_dense)
        (x_det,     st_det_),    pb_det     = Zygote.pullback(y -> l_det_(y,     ps_det_,     st_det_),      x_mask)

        dp = similar(x_det); fill!(dp, one(eltype(x_det)))

        dx_det   = pb_det(   (dp,      NoTangent()) )[1]
        dx_mask  = pb_mask(  (dx_det,  NoTangent()) )[1]
        dx_dense = pb_dense( (dx_mask, NoTangent()) )[1]
        dx_rsh   = pb_rsh(   (dx_dense,NoTangent()) )[1]
        dx_corr  = pb_corr(  (dx_rsh,  NoTangent()) )[1]
        dx_pool  = pb_pooling((dx_corr,NoTangent()))[1]

        dp_dense = ACEpsi.dense_blocks(dx_mask, x_rsh, Nel)
    
        dp_branch = nothing
        if length(ps_branch) > 0 && hasproperty(ps_branch[1], :ζ)
            lens     = [length(st_branch[j].spec) for j in 1:length(ps_branch)]
            offsets  = cumsum(lens)
            lo       = [1; offsets[1:end-1] .+ 1]
            hi       = offsets
            ranges   = [lo[i]:hi[i] for i in eachindex(lo)]

            dp_branch = mapreduce(i ->
            ACEpsi.branch_blocks(l_branch.layers[i],
                             dx_pool[:, :, ranges[i]],
                             x_embed[i],
                             ps_branch[i],
                             st_branch[i]),
            vcat, 1:length(lens))
        end

        blocks = (dp_branch, dp_dense)
        return vcat((blk for blk in blocks if blk !== nothing)...)
    end
end


    using ACEpsi, Polynomials4ML, StaticArrays, Test
    using Polynomials4ML: natural_indices, degree, SparseProduct
    using ACEpsi.AtomicOrbitals: Nuc, make_nlms_spec, evaluate
    using ACEpsi: BackflowPooling, BFwf_lux, setupBFState, Jastrow, displayspec
    using ACEpsi.vmc: gradx, laplacian, grad_params, EmbeddingW!, _invmap, VMC_multilevel, wf_multilevel, VMC, gd_GradientByVMC, gd_GradientByVMC_multilevel, AdamW, SR, SumH, MHSampler
    using ACEbase.Testing: print_tf, fdtest
    using LuxCore
    using Lux
    using Zygote
    using Optimisers
    using Random
    using Printf
    using LinearAlgebra
    using BenchmarkTools
    using HyperDualNumbers: Hyper
    using Polynomials4ML.Utils: gensparse
    using ObjectPools


    Nel = 2
    X = randn(SVector{3, Float64}, Nel)
    Σ = [↑,↓]
    nuclei = SVector{1}([ Nuc(zeros(SVector{3, Float64}), Nel * 1.0)])

    spec_Be = [(n1 = 1, n2 = 1, l = 0),
            (n1 = 1, n2 = 2, l = 0),
            (n1 = 1, n2 = 3, l = 0),
            (n1 = 1, n2 = 1, l = 1),
            (n1 = 1, n2 = 2, l = 1),
            (n1 = 2, n2 = 1, l = 0),
            (n1 = 2, n2 = 2, l = 0),
            (n1 = 2, n2 = 3, l = 0),
            (n1 = 2, n2 = 1, l = 1),
            (n1 = 2, n2 = 2, l = 1),
            (n1 = 3, n2 = 1, l = 0),
            (n1 = 3, n2 = 2, l = 0),
            (n1 = 3, n2 = 3, l = 0),
            (n1 = 3, n2 = 1, l = 1),
            (n1 = 3, n2 = 2, l = 1)
            ]

    spec = [ spec_Be ]

    n1 = 5
    Pn = Polynomials4ML.legendre_basis(n1+1)
    Ylmdegree = 2
    totdegree = 20
    ζ = 10.0 * rand(length(spec))
    Dn = SlaterBasis(ζ)
    bYlm = RRlmBasis(Ylmdegree)

    totdegree = totdeg = 30

    ν = 2

    MaxIters = 10
    _spec = [ spec[1][1:8]]

    _TD = ACEpsi.TD.No_Decomposition()

    Nbf = 1
    speclist  = [1]
    sd_admissible = bb -> sum(b.s == '∅' for b in bb) == 1
    disspec = []
    bRnl = [AtomicOrbitalsRadials(Pn, SlaterBasis(10 * rand(length(_spec[j]))), _spec[speclist[j]]) for j = 1:length(_spec)]




    js = ACEpsi.JPauliNet(nuclei)
    # X -> (X-R[1], X-R[2], X-R[3])
    embed_layer = ACEpsi.embed_diff_layer(nuclei)
    # X -> (ϕ1(X-R[1]), ϕ2(X-R[2]), ϕ3(X-R[3])
    prodbasis_layer = ACEpsi.AtomicOrbitals.ProductBasisLayer(speclist, bRnl, bYlm, totdeg)
    # BackFlowPooling: (length(nuclei), nX, length(spec1 from totaldegree)) -> (nX, 3, length(nuclei) * length(spec1))
    aobasis_layer = ACEpsi.AtomicOrbitals.AtomicOrbitalsBasisLayer(prodbasis_layer, nuclei)
    pooling = BackflowPooling(aobasis_layer)
    pooling_layer = ACEpsi.lux(pooling)
    spec1p = ACEpsi.get_spec(nuclei, speclist, bRnl, bYlm, totdeg)
    tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]]
    default_admissible = bb -> (length(bb) == 0) || (sum(b.n1 - 1 for b in bb ) <= totdeg)

    specAA = gensparse(; NU = ν, tup2b = tup2b, admissible = default_admissible,
                        minvv = fill(0, ν),
                        maxvv = fill(length(spec1p), ν),
                        ordered = true)
    spec = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]

    # further restrict
    spec = [t for t in spec if sd_admissible([spec1p[t[j]] for j = 1:length(t)])]

    # define n-correlation
    corr1 = Polynomials4ML.SparseSymmProd(spec)




spec
spec1p
displayspec(spec, spec1p)




    # (nX, 3, length(nuclei), length(spec1 from totaldegree)) -> (nX, length(spec))
    corr_layer = Polynomials4ML.lux(corr1)

    l_hidden = Tuple(collect(Chain(; hidden1 = LinearLayer(length(corr1), Nel; use_cache = false), Mask = ACEpsi.MaskLayer(Nel), det = WrappedFunction(x::Matrix -> det(x))) for i = 1:Nbf))

    jastrow_layer = ACEpsi.lux(js)

    BFwf_chain = Chain(; diff = embed_layer, Pds = prodbasis_layer,
                        bA = pooling_layer, reshape = ACEpsi.myReshapeLayer((Nel, 3 * sum(length.(prodbasis_layer.sparsebasis)))),
                        bAA = corr_layer,
                        hidden = l_hidden[1], # BranchLayer(l_hidden...),
                        sum = WrappedFunction(sum))
    l = Chain(; branch = BranchLayer(; js = jastrow_layer, bf = BFwf_chain, ), prod = WrappedFunction(x -> x[1] * x[2]), logabs = WrappedFunction(x -> 2 * log(abs(x))) )



# ===
ps, st = setupBFState(MersenneTwister(1234), l, Σ)
Zygote.gradient(_X -> l(_X, ps, st)[1], X)

ham = SumH(nuclei)

physical_config = ACEpsi.vmc.Physical_config(nuclei, [1,1,1,1], [[2,2]])
sam = MHSampler(l, Nel, physical_config,
                Δt = 0.08,
                burnin  = 1000,
                nchains = 2000)

x = sam.x0
x = x[1:20]
grad_params(l, X, ps, st)

dps = pmap(x) do d
   grad_params(l, d, ps, st)
end

dps2 = [grad_params(l, xx, ps, st) for xx in x]

dps2 ≈ dps
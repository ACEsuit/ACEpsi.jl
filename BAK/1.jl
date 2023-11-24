wf = wf_list[1]
ps = ps_list[1]
st = st_list[1]
spec = spec_list[1]
spec1p = spec1p_list[1]
specAO = specAO_list[1]

wf2 = wf_list[end]
ps2 = ps_list[end]
st2 = st_list[end]
spec2 = spec_list[end]
spec1p2 = spec1p_list[end]
specAO2 = specAO_list[end]

X = randn(SVector{3, Float64}, Nel)

wf(X, ps,st)
wf2(X,ps2,st2)

specIk1 = reduce(vcat,[[(n1 = specAO[i].n1, n2 = specAO[i].n2, l = specAO[i].l, m = j) for j = -specAO[i].l:1:specAO[i].l] for i = 1:length(specAO)])
specIk2 = reduce(vcat,[[(n1 = specAO2[i].n1, n2 = specAO2[i].n2, l = specAO2[i].l, m = j) for j = -specAO2[i].l:1:specAO2[i].l] for i = 1:length(specAO2)])


size(W1)
size(W2)


wf = wf_list[end]
ps = ps_list[end]
spec = spec1p_list[end][reduce(vcat,spec_list[end])]

ps.ϕnlm.ζ .= reduce(vcat,st2.ϕnlm.ζ[1])

reduce(vcat,st2.ϕnlm.ζ[2])

W = zero(ps.hidden1.W)
for i = 1:size(W,1)
    for j = 1:4
        for z = 1:10
            W[i, (j-1) * 10 + z] = reduce(vcat,st2.ϕnlm.ζ[2])[j] * ps2.hidden1.W[i, z]
        end
    end
    W[i, 41:end] = ps2.hidden1.W[i, 11:end]
end

ps.hidden1.W .= W

st = st_list[end]

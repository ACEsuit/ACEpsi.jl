wf2 = wf_list[end]
ps2 = ps_list[end]
st2 = st_list[end]


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

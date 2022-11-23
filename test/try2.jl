using ACEcore.Utils: gensparse
function checkOrd(bb)
    if length(bb) == 1 || length(bb) == 0
       return true
    end
    
    for i = 1:length(bb) - 1
       if bb[i][2] > bb[i+1][2]
          return false
       end
    end
    return true
 end

K = 10
totdeg = 10
ν = 2
spec1p = [ (k, σ) for σ in [1, 2, 3] for k in 1:K ]  # (1, 2, 3) = (∅, ↑, ↓);

# generate the many-particle spec 
tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
spec1p = sort(spec1p)
@show spec1p
admissible = bb -> (length(bb) == 0) || (sum(b[1] - 1 for b in bb ) <= totdeg) && checkOrd(bb)

specAA = gensparse(; NU = ν, tup2b = tup2b, admissible = admissible,
                    minvv = fill(0, ν), 
                    maxvv = fill(length(spec1p), ν), 
                    ordered = true)

#@show specAA
spec = [ vv[vv .> 0] for vv in specAA][2:end]
for i = 1:length(spec)
    @show spec[i]
end
# @show length(spec)
# for i = 10:27
#     A = spec1p[spec[i][1]]
#     B = spec1p[spec[i][2]]
#     @assert mod(A[1] - 1, K) + mod(B[1] - 1, K) <= totdeg
#     #@show length(spec[i])
#     @assert checkOrd([A, B], K)
# end
@show length(spec)
for i = 31:300
    @show spec1p[spec[i][1]], spec1p[spec[i][2]]
end


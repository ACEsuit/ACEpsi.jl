

using Polynomials4ML, ForwardDiff, Test, ACEpsi 
using Polynomials4ML.Testing: println_slim, print_tf
using Polynomials4ML: evaluate, evaluate_ed, evaluate_ed2

# -------------- RnlExample -----------------
@info("Testing RnlExample basis")
bRnl = ACEpsi.AtomicOrbitals.RnlExample(5)

rr = 2 * rand(10) .- 1
Rnl = evaluate(bRnl, rr)
Rnl1, dRnl1 = evaluate_ed(bRnl, rr)
Rnl2, dRnl2, ddRnl2 = evaluate_ed2(bRnl, rr)

fdRnl = vcat([ ForwardDiff.derivative(r -> evaluate(bRnl, [r,]), r) 
               for r in rr ]...) 
fddRnl = vcat([ ForwardDiff.derivative(r -> evaluate_ed(bRnl, [r,])[2], r)
               for r in rr ]...) 

println_slim(@test Rnl ≈ Rnl1 ≈ Rnl2)
println_slim(@test dRnl1 ≈ dRnl2 ≈ fdRnl)
println_slim(@test ddRnl2 ≈ fddRnl)


# -------------- **** -----------------
using ACEbase.Testing: fdtest
using Zygote

@info("Test rrule")
using LinearAlgebra: dot 

for ntest = 1:30
    local rr
    local uu
    local Rnl
    local u
    
    rr = 2 .* randn(10) .- 1
    uu = 2 .* randn(10) .- 1
    _rr(t) = rr + t * uu
    Rnl = evaluate(bRnl, rr)
    u = randn(size(Rnl))
    F(t) = dot(u, evaluate(bRnl, _rr(t)))
    dF(t) = begin
        val, pb = Zygote.pullback(evaluate, bRnl, _rr(t))
        ∂BB = pb(u)[2] # pb(u)[1] returns NoTangent() for basis argument
        return sum( dot(∂BB[i], uu[i]) for i = 1:length(uu) )
    end
    print_tf(@test fdtest(F, dF, 0.0; verbose = false))
end
println()
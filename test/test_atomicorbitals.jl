

using Polynomials4ML, ForwardDiff, Test, ACEpsi 
using Polynomials4ML.Testing: println_slim
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


using StaticArrays: SA 

export ↑, ↓, spins


# Define the spin types and variables
const Spin = Char 
const ↑ = '↑'
const ↓ = '↓'
const ∅ = '∅'   # this is only for internal use
_spins = SA[↑, ↓]
_extspins = SA[↑, ↓, ∅]

spins() = _spins
extspins() = _extspins


"""
This function convert spin to corresponding integer value used in spec
"""
function spin2idx(σ)
   if σ == ↑
      return 1
   elseif σ == ↓
      return 2
   elseif σ == ∅
      return 3
   end
   error("illegal spin char for spin2idx")
end

"""
This function convert idx to corresponding spin string.
"""
function idx2spin(i)
   if i == 1
      return ↑
   elseif i == 2
      return ↓
   elseif i == 3
      return ∅
   end
   error("illegal integer value for idx2spin")
end


# TODO : deprecate these 
const spin2num = spin2idx
const num2spin = idx2spin

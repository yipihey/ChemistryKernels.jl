using ChemistryKernels, KernelAbstractions, Test
try; @eval using Metal; catch; end
include(joinpath(@__DIR__, "oracle.jl")); using .ChemOracle
include(joinpath(@__DIR__, "harness.jl"))
module UnitDeut
  using ChemistryKernels, KernelAbstractions
  using ChemistryKernels: MH, TEV_PER_K
  include(joinpath(@__DIR__, "..", "src", "rates_deuterium.jl"))
end
ChemOracle.set_flags!(); Ts = ChemOracle.tgrid()

@testset "rates_deuterium vs grackle" begin
  for rn in ("k50","k51","k52","k53","k54","k55","k56")
    Tall = collect(Ts)
    refall = [ChemOracle.rate(rn, t) for t in Tall]
    # Grackle's k55=1.08e-22 branch is a numerical sentinel, not the
    # Shavitt/Galli-Palla fit. Compare the shared fit above its 200 K boundary.
    keep = if rn == "k50"
      findall(>(0.0), refall)
    elseif rn == "k55"
      findall(>(200.0), Tall)
    else
      eachindex(Tall)
    end
    check_scalar_kernel(rn, getfield(UnitDeut, Symbol(rn, "_grid")),
                        refall[keep], Tall[keep])
  end
end

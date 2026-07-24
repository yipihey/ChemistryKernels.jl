using ChemistryKernels, KernelAbstractions, Test
try; @eval using Metal; catch; end
include(joinpath(@__DIR__, "oracle.jl")); using .ChemOracle
include(joinpath(@__DIR__, "harness.jl"))

module UnitAtomic
  using ChemistryKernels, KernelAbstractions
  using ChemistryKernels: MH, TEV_PER_K
  include(joinpath(@__DIR__, "..", "src", "rates_atomic.jl"))
end

ChemOracle.set_flags!()             # CaseB on
Ts = ChemOracle.tgrid()

@testset "rates_atomic vs grackle" begin
  for rn in ("k1","k2","k3","k4","k5","k6","k57","k58")
    Tall = collect(Ts)
    refall = [ChemOracle.rate(rn, t) for t in Tall]
    # Grackle encodes inactive branches as 1e-20. The modern model uses physical
    # zeros/rate tails, so retain the oracle gate only where its fit is active.
    keep = if rn == "k1"
      findall(!=(1.0e-20), refall)
    elseif rn in ("k3", "k5")
      findall(>(0.8 * 11605.0), Tall)
    elseif rn in ("k57", "k58")
      findall(>(3000.0), Tall)
    else
      eachindex(Tall)
    end
    check_scalar_kernel(rn, getfield(UnitAtomic, Symbol(rn, "_grid")), refall[keep], Tall[keep];
                        f32rtol = f32rtol_for(rn))
  end

  # The Abel ionisation fits have physical Boltzmann tails. They must pass
  # continuously below the old 1e-20 sentinel and underflow to zero on their
  # own; k6 is recombination and correctly rises rather than vanishing at low T.
  @test UnitAtomic.k1(0.0) == 0.0
  @test UnitAtomic.k3(0.0) == 0.0
  @test UnitAtomic.k5(0.0) == 0.0
  @test UnitAtomic.k1(3000.0) < 1.0e-20
  @test UnitAtomic.k3(5000.0) < 1.0e-30
  @test UnitAtomic.k5(10000.0) < 1.0e-30
  for f in (UnitAtomic.k1, UnitAtomic.k3, UnitAtomic.k5)
    @test issorted(f.(10.0 .^ range(0, 5; length=301)))
    Tedge = 0.8 * 11605.0
    @test isapprox(f(prevfloat(Tedge)), f(nextfloat(Tedge)); rtol=1e-12)
  end
  @test UnitAtomic.k6(100.0) > UnitAtomic.k6(1000.0) > 0.0
  @test UnitAtomic.k57(100.0) == 0.0
  @test UnitAtomic.k58(100.0) == 0.0
  @test UnitAtomic.k57(3000.0) < 1.0e-30
  @test UnitAtomic.k58(3000.0) < 1.0e-30
end

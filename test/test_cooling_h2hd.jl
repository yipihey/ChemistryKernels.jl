using ChemistryKernels, EmissionKernels, KernelAbstractions, Test
try; @eval using Metal; catch; end
include(joinpath(@__DIR__, "oracle.jl")); using .ChemOracle
include(joinpath(@__DIR__, "harness.jl"))

# H2/HD cooling coefficients live in EmissionKernels; its grid launchers are the targets.
ChemOracle.set_flags!()
Ts = ChemOracle.tgrid()
@testset "cooling_h2hd vs grackle" begin
  for nm in ("GAHI","GAH2","GAHe","GAHp","GAel","H2LTE","HDlte","HDlow")
    ref = [ChemOracle.cool(nm, t) for t in Ts]
    check_scalar_kernel(nm, getfield(EmissionKernels, Symbol(nm, "_grid")), ref, Ts; f32rtol = f32rtol_for(nm))
  end
end

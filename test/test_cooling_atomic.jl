using ChemistryKernels, EmissionKernels, KernelAbstractions, Test
try; @eval using Metal; catch; end
include(joinpath(@__DIR__, "oracle.jl")); using .ChemOracle
include(joinpath(@__DIR__, "harness.jl"))

# Atomic cooling coefficients live in EmissionKernels; its @scalarkernel grid launchers
# (ceHI_grid, …) are the parity targets against the C-grackle oracle.
ChemOracle.set_flags!()            # CaseB on
Ts = ChemOracle.tgrid()
@testset "cooling_atomic vs grackle" begin
  for nm in ("ceHI","ceHeI","ceHeII","ciHI","ciHeI","ciHeII","ciHeIS","reHII","reHeII1","reHeII2","reHeIII","brem")
    ref = [ChemOracle.cool(nm, t) for t in Ts]
    check_scalar_kernel(nm, getfield(EmissionKernels, Symbol(nm, "_grid")), ref, Ts; f32rtol = f32rtol_for(nm))
  end
end

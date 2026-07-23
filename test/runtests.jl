# ChemistryKernels test suite. Run directly against the test project (NOT via
# Pkg.test, whose sandbox does not inherit [sources]):
#   <julia> --project=test test/runtests.jl
#
# The grackle parity suites need the C-grackle oracle (oracle/libchem_oracle.dylib),
# which is macOS-only; they are GATED on `ChemOracle.available()`. The table-free
# suites (recombination, field, metal, CMB literals, equilibrium, network, driver,
# one-zone) always run and are what CI exercises on Linux.
using ChemistryKernels
using EmissionKernels
using Test

try; @eval using Metal; catch; end     # :metal backend where available (Apple Silicon)
try; @eval using CUDA;  catch; end     # :cuda backend where available (NVIDIA)

include("oracle.jl");  using .ChemOracle
include("harness.jl")

const ORACLE = ChemOracle.available()
ORACLE || @info "C-grackle oracle (libchem_oracle.dylib) absent — skipping grackle-parity " *
                "suites (macOS-only); running the table-free suites."

@testset "ChemistryKernels" begin
    # ── grackle-parity suites (macOS oracle) ─────────────────────────────────
    if ORACLE
        include("test_smoke.jl")
        include("test_rates_atomic.jl")
        include("test_rates_h2.jl")
        include("test_rates_deuterium.jl")
        include("test_cooling_atomic.jl")
        include("test_cooling_h2hd.jl")
        include("test_temperature.jl")
        include("test_edot.jl")
    end
    # ── table-free suites (always; the Linux/CI gate) ────────────────────────
    include("test_rates_cmb.jl")
    include("test_equilibrium.jl")
    include("test_network_step.jl")
    include("test_dense_threebody.jl")
    include("test_driver.jl")
    include("test_onezone.jl")
    include("test_recombination_mixing.jl")
    include("test_recombination_field.jl")
    include("test_cooling_metal.jl")
    include("test_rate_tables.jl")
    # ── dust physics (rates, T_dust, shielding, network_step, evolve_cell) ─────
    include("test_dust.jl")
    # ── performance knobs (dtfrac, workgroup_size, cell sort) ─────────────────
    include("test_performance.jl")
    # ── UInt16 log₂-encoded species storage ───────────────────────────────────
    include("test_u16_species.jl")
    # ── per-call sub-cycle cap (itcap) opt-in + resumable contract ────────────
    include("test_itcap.jl")
    # ── phase-space-binned solve (one stiff solve per occupied bin) ────────────
    include("test_binned.jl")
    # ── reduced fast analytic H+H2 mode (evolve_cell_fast) ─────────────────────
    include("test_fast.jl")
end

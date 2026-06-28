# test_itcap.jl — the per-call sub-cycle cap `itcap` is a caller knob, NOT a fixed
# constant: the default stays high (every regime correct out of the box), and a
# latency-bound caller (the galaxy-formation sim hot path) can pass a LOW cap to
# bound per-cell work, with the rare stiff straggler re-entered on the next host
# step (the resumable contract).  The cap governs the COUPLED cooling+chemistry
# sub-cycle, so both the species (BE) and the energy (cooling) integration are
# bounded identically.

module TestItcap
using Test
using ChemistryKernels
const CK = ChemistryKernels
const MH = CK.MH; const KB = CK.KBOLTZ; const fh = CK.FH_DEFAULT
e_of(T, xe) = (KB*T)/((5/3 - 1) * (1.0/(fh*(1 + xe) + 0.25*(1 - fh))) * MH)
cell(; nH=1.0, T=1e4, xHII=1e-3) = (nH*MH/fh, e_of(T, xHII), xHII*fh*(nH*MH/fh), 0.0, 0.0)

const DT = 3.15e12; const Z = 10.0

@testset "itcap caller knob + resumable contract" begin
    # a genuinely stiff cell (hot, far from equilibrium) that needs ≫10 sub-steps
    c = cell(nH=10.0, T=5e4, xHII=1e-3)

    @testset "default is the high watchdog (one call consumes dt)" begin
        e, HII, H2I, HDI, ttot = CK.evolve_cell(c..., DT, Z; fh=fh)
        @test ttot ≈ DT rtol=1e-6           # default cap (5000) finishes in one call
    end

    @testset "a low cap bounds work and reports partial progress" begin
        e, HII, H2I, HDI, ttot = CK.evolve_cell(c..., DT, Z; fh=fh, itcap=10)
        @test ttot < DT                     # capped: did NOT consume the whole step
        @test isfinite(e) && e > 0          # but the partial state is physical
        @test 0 ≤ HII ≤ fh*c[1]*(1 + 1e-6)
    end

    @testset "capped + re-entered recovers the uncapped result (cooling AND chem)" begin
        # NOTE: recovery is APPROXIMATE, not bit-identical — re-entry resets the fast
        # H⁻/H₂⁺ intermediaries and re-sizes the dtfrac sub-steps, and the uncapped
        # reference is itself only 10%-rule accurate.  The point: a low-cap straggler,
        # re-entered, consumes the whole step and lands near the uncapped answer (the
        # "tolerated error" that is recovered over the run).
        ref_e, ref_HII = CK.evolve_cell(c..., DT, Z; fh=fh)[1:2]   # uncapped reference
        rho, e, HII, H2I, HDI = c
        done = 0.0
        for _ in 1:10_000
            e, HII, H2I, HDI, tt = CK.evolve_cell(rho, e, HII, H2I, HDI,
                                                  DT - done, Z; fh=fh, itcap=10)
            done += tt
            done ≥ DT * (1 - 1e-9) && break
        end
        @test done ≈ DT rtol=1e-6                  # the straggler DID consume the step
        @test isapprox(HII, ref_HII; rtol=5e-2)    # chemistry recovered (≲5%)
        @test isapprox(e,   ref_e;   rtol=2e-1)    # cooling/energy recovered (≲20%, coarse)
    end

    @testset "solve_chem! exposes itcap (default == explicit high cap)" begin
        mk() = (fill(c[1], 4), fill(c[2], 4), fill(c[3], 4), fill(1e-30, 4))
        kw = (; a_value = 1/(1+Z), dt = DT, density_units = 1.0,
                length_units = 1.0, time_units = 1.0, fh = fh)
        r1, e1, h1, m1 = mk(); CK.solve_chem!(r1, e1, h1, m1; kw...)
        r2, e2, h2, m2 = mk(); CK.solve_chem!(r2, e2, h2, m2; kw..., itcap = CK._SUB_ITMAX)
        @test e1 ≈ e2 && h1 ≈ h2                    # default IS the high watchdog
        # a low cap still runs and stays physical
        r3, e3, h3, m3 = mk(); CK.solve_chem!(r3, e3, h3, m3; kw..., itcap = 100)
        @test all(isfinite, e3) && all(>(0), e3)
    end
end

end # module TestItcap

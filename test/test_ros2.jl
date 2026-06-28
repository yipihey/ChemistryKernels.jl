# test_ros2.jl — the adaptive ROS2 integrator (rosenbrock.jl): self-convergence,
# agreement with the fine-subcycler oracle across primordial regimes, robustness,
# and the deuterium path.  PRIMORDIAL-only (no dust/metals).
#
# Wrapped in a module so its helper names don't collide with the other suites when
# included from runtests.jl (shared Main scope).

module TestROS2
using Test
using ChemistryKernels
const CK = ChemistryKernels

const MH = CK.MH; const KB = CK.KBOLTZ
const fh = CK.FH_DEFAULT
e_of(T, xe) = (KB*T)/((5/3 - 1) * (1.0/(fh*(1 + xe) + 0.25*(1 - fh))) * MH)
cell(; nH=1.0, T=1e4, xHII=1e-3, xH2=0.0) =
    (nH*MH/fh, e_of(T, xHII), xHII*fh*(nH*MH/fh), xH2*fh*(nH*MH/fh), 0.0)
xof(HII, rho) = HII / (fh * rho)

const DT = 3.15e12          # 0.1 Myr macro-step
const Z  = 10.0

# representative primordial cells across the regimes that stress the integrator
const CASES = [
    ("cold neutral", cell(nH=1.0,  T=1e2,   xHII=1e-4)),
    ("warm partial", cell(nH=1.0,  T=8e3,   xHII=1e-2)),
    ("hot ionized",  cell(nH=1.0,  T=5e4,   xHII=0.5)),
    ("active stiff", cell(nH=10.0, T=1.5e4, xHII=0.1)),
    ("dense cool",   cell(nH=1e2,  T=3e3,   xHII=1e-3)),
    ("far from eq",  cell(nH=1.0,  T=3e4,   xHII=1e-4)),
]

@testset "ROS2 integrator" begin
    @testset "self-convergence (tighter rtol ⇒ stable final state)" begin
        for (name, c) in CASES
            r2 = CK.evolve_cell_ros2(c..., DT, Z; fh=fh, rtol=1e-2, itcap=500_000)
            r3 = CK.evolve_cell_ros2(c..., DT, Z; fh=fh, rtol=1e-3, itcap=500_000)
            r4 = CK.evolve_cell_ros2(c..., DT, Z; fh=fh, rtol=1e-4, itcap=500_000)
            x2 = xof(r2[2], c[1]); x3 = xof(r3[2], c[1]); x4 = xof(r4[2], c[1])
            # default (1e-2) within ~1.5% of converged; 1e-3 and 1e-4 essentially equal
            @test isapprox(x2, x4; rtol=2e-2) || abs(x2 - x4) < 1e-5
            @test isapprox(x3, x4; rtol=3e-3) || abs(x3 - x4) < 1e-5
            @test r2[5] ≈ DT rtol=1e-3        # consumed the whole macro-step
        end
    end

    @testset "agreement with fine subcycler oracle" begin
        for (name, c) in CASES
            ref = CK.evolve_cell(c..., DT, Z; fh=fh, dtfrac=2e-3, itcap=2_000_000)
            ros = CK.evolve_cell_ros2(c..., DT, Z; fh=fh, rtol=1e-2, itcap=500_000)
            xr = xof(ref[2], c[1]); xs = xof(ros[2], c[1])
            @test isapprox(xs, xr; rtol=3e-2) || abs(xs - xr) < 1e-4   # ≤3% on x_HII
            @test isapprox(ros[1], ref[1]; rtol=5e-2)                  # ≤5% on energy
        end
    end

    @testset "robustness / positivity" begin
        for (name, c) in CASES
            e, HII, H2I, HDI, ttot = CK.evolve_cell_ros2(c..., DT, Z; fh=fh)
            rho = c[1]
            @test e > 0 && isfinite(e)
            @test 0 ≤ HII ≤ fh*rho*(1 + 1e-6)            # x_HII ∈ [0,1]
            @test H2I ≥ 0 && isfinite(H2I)
            @test ttot ≈ DT rtol=1e-3
        end
    end

    @testset "deuterium path" begin
        rho, e, HII, H2I, _ = cell(nH=1.0, T=2e4, xHII=1e-2)
        HDI = 2.0e-5 * fh * rho * 3      # seed HD (×3 y-convention scale)
        out = CK.evolve_cell_ros2(rho, e, HII, H2I, HDI, DT, Z; fh=fh, deuterium=true)
        @test length(out) == 5
        @test out[1] > 0 && isfinite(out[1])             # energy
        @test out[4] ≥ 0 && isfinite(out[4])             # HDI stays physical
        @test out[5] ≈ DT rtol=1e-3
    end

    @testset "Float32 (GPU precision) runs and agrees" begin
        for (name, c) in CASES[1:4]
            c32 = Float32.(c)
            r64 = CK.evolve_cell_ros2(c...,   DT, Z; fh=fh, rtol=1e-2)
            r32 = CK.evolve_cell_ros2(c32..., Float32(DT), Float32(Z); fh=Float32(fh), rtol=1f-2)
            @test isfinite(r32[1]) && r32[1] > 0
            # f32 + coarse-FD Jacobian widens agreement on stiff cells; still ≲6%
            @test isapprox(xof(r32[2], c32[1]), xof(r64[2], c[1]); rtol=6e-2, atol=2e-4)
        end
    end
end

end # module TestROS2

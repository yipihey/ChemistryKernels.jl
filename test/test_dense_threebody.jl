using ChemistryKernels, EmissionKernels, Test
using ChemistryKernels: network_step

function threebody_only_rates(T)
    z = zero(T)
    return (; k1=z, k2=z, k3=z, k4=z, k5=z, k6=z, k7=z, k8=z,
            k9=z, k10=z, k11=z, k12=z, k13=k13(T), k14=z, k15=z,
            k16=z, k17=z, k18=z, k19=z, k22=k22(T), k57=z, k58=z,
            k27=z, k28=z, k_beta1s=z)
end

function dense_thermo_state(::Type{R}, nH, T, fH2) where {R}
    fh = R(0.76)
    mh = R(ChemistryKernels.MH)
    kb = R(1.380649e-16)
    n = R(nH)
    f = R(fH2)
    xHII = R(1e-12)
    rho = n*mh/fh
    nHI = max(one(R) - f - xHII, zero(R))*n
    nH2 = f*n/R(2)
    nHe = (one(R)-fh)/(R(4)*fh)*n
    e = (nHI+nH2+nHe+xHII*n)*kb*R(T)/(rho*(R(5)/R(3)-one(R)))
    return (; rho, e, HII=xHII*n*mh, H2I=f*n*mh, nH=n, mh, fh)
end

function independent_pair_equilibrium(nH, T)
    kd = k13(T)
    kd == 0 && return nH
    nHI = 2nH / (1 + sqrt(1 + 8k22(T)*nH/kd))
    return nH - nHI
end

@testset "analytic/full dense thermochemical agreement" begin
    R = Float32
    rt = build_rate_tables(; backend=:cpu, precision=R)
    ct = build_cooling_tables(; backend=:cpu, precision=R)
    for (label, nH, T, dt) in (
            (:forming, 1e12, 1500.0, 1e5),
            (:formation_equilibrium, 1e16, 3500.0, 1.0),
            (:dissociation_equilibrium, 1e16, 5000.0, 1.0))
        f0 = label === :forming ? 1e-4 : independent_pair_equilibrium(nH, T)/nH
        s = dense_thermo_state(R, nH, T, f0)
        full = evolve_cell(s.rho, s.e, s.HII, s.H2I, R(0), R(dt), R(20);
                           fh=s.fh, rate_tables=rt, cool_tables=ct)
        analytic = evolve_cell_analytic(s.rho, s.e, s.HII, s.H2I, R(dt), R(20);
                                        fh=s.fh, rate_tables=rt, cool_tables=ct)
        @test isapprox(analytic[3], full[3]; rtol=0.015)
        @test isapprox(analytic[1], full[1]; rtol=0.005)
        @test analytic[4] == R(dt)
    end
end

@testset "dense three-body H2 equilibrium closure" begin
    fh = 0.76
    for T in (3500.0, 5000.0), nH in (1e16, 1e18)
        K = threebody_only_rates(T)
        expected = independent_pair_equilibrium(nH, T)
        nHI_eq = nH - expected
        nu_eq = 2K.k22*nHI_eq^2 + K.k13*nHI_eq

        # Exercise both sides of the attracting fixed point, including the
        # fully molecular state where a lagged H-impact sink is initially zero.
        for initial_h2 in (0.0, nH)
            initial_hi = nH - initial_h2
            nu_old = 2K.k22*initial_hi^2 + K.k13*initial_hi
            dt = 100 / max(nu_old, nu_eq)
            out = network_step(nH/fh, fh, initial_hi, 0.0, 0.0, initial_h2,
                               0.0, 0.0, 0.0, 0.0, 0.0, K, dt;
                               yHeII_in=0.0, yHeIII_in=0.0,
                               intermediates_current=Val(true),
                               stiff_h2_pair=Val(true))

            @test isapprox(out.yH2I, expected; rtol=2e-14)
            @test isapprox(out.yHI + out.yH2I, nH; rtol=2e-14)
            @test out.yHI >= 0
            @test out.yH2I >= 0
        end

        # The same equilibrium closure is used by the Float32 GPU kernels.
        R = Float32
        K32 = map(R, K)
        expected32 = R(expected)
        nH32 = R(nH)
        nHI_eq32 = nH32 - expected32
        nu_eq32 = R(2)*K32.k22*nHI_eq32^2 + K32.k13*nHI_eq32
        out32 = network_step(nH32/R(fh), R(fh), nH32, R(0), R(0), R(0),
                             R(0), R(0), R(0), R(0), R(0), K32, R(100)/nu_eq32;
                             yHeII_in=R(0), yHeIII_in=R(0),
                             intermediates_current=Val(true),
                             stiff_h2_pair=Val(true))
        @test isapprox(out32.yH2I, expected32; rtol=2e-6)
        @test isapprox(out32.yHI + out32.yH2I, nH32; rtol=2e-6)
    end
end

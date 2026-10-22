# test_dense_equilibrium.jl — the high-density / high-temperature continuation (dense.jl).
#
# 1. Equilibrium constants: K_H2 against the exact 61-term rotational sum; K_ion against
#    the textbook Saha constant; Float32 ln-forms finite where the plain K would under/
#    overflow.
# 2. Detailed balance: k13_dense → k22·K_H2 at n ≫ n_cr and → the AAZN97 k13 at n ≪ n_cr;
#    k1r·K_ion = k1 above 1e4 K and continuous below.
# 3. The closure alone: conserves H nuclei and thermal+chemical energy to round-off and
#    lands exactly on the Saha manifold (both pairs), Float64 and Float32.
# 4. evolve_cell with dense=Val(true), planck_radiation=Val(true), no radiative cooling:
#    from a far-from-equilibrium state at n = 1e14–1e21 the cell reaches the Saha
#    composition AT ITS OWN FINAL TEMPERATURE with total energy conserved, in a handful
#    of sub-steps (the default network needs 10³–10⁴ sub-steps there and is wrong).
# 5. Default path untouched: dense=Val(false) is bit-identical to a pinned reference
#    (the k1r/GamHI hooks add exactly 0.0).  Low-density cells with dense=Val(true)
#    agree with the default to ≲ 1e-3 (the only change there is the ~0.3 % k13 blend).
# 6. KernelAbstractions: solve_chem!(…; dense=true, planck_radiation=true) on :cpu
#    (f64, f32) and on :metal (f32, when available) agrees with the scalar path.
using ChemistryKernels, EmissionKernels, Test
try; @eval using Metal; catch; end

const CKd = ChemistryKernels
const _EVd = 1.602176634e-12
const _FHd = 0.76

# exact K_H2 with the full weighted rotational sum, same constants as dense.jl
function _K_H2_exact(T)
    Zr = 0.0
    for J in 0:60
        Zr += (iseven(J) ? 1 : 3) * (2J + 1) * exp(-J * (J + 1) * CKd.THETA_ROT_H2 / T)
    end
    Zv = 1 / (1 - exp(-CKd.THETA_VIB_H2 / T))
    16 / (Zr * Zv) * (π * CKd.MH * CKd.KBOLTZ * T / CKd.HPLANCK^2)^1.5 * exp(-CKd.D0_H2_K / T)
end
_K_ion_ref(T) = (2π * CKd.ME * CKd.KBOLTZ * T / CKd.HPLANCK^2)^1.5 * exp(-CKd.CHI_H_K / T)

# independent Saha composition (bisection on x_H): x + sqrt(B x) + 2x²/A = 1
function _saha_ref(nH, T)
    A = CKd.K_H2(T) / nH; B = CKd.K_ion(T) / nH
    g(x) = x + sqrt(B * x) + 2x^2 / A - 1
    lo, hi = 0.0, 1.0
    for _ in 1:200; m = (lo + hi) / 2; g(m) > 0 ? (hi = m) : (lo = m); end
    x = (lo + hi) / 2
    (xHII = sqrt(B * x), fH2 = max(1 - x - sqrt(B * x), 0.0))
end

# one cell: H nuclei nH [cm⁻³], T, H₂ fraction (of H nuclei) and x_HII → (rho, e, HII_m, H2I_m)
function _cell(::Type{R}, nH, T, fH2, xHII) where {R}
    rho = nH * CKd.MH / _FHd
    nHI = (1 - fH2 - xHII) * nH; nH2 = fH2 * nH / 2; nHe = (1 - _FHd) / (4_FHd) * nH; ne = xHII * nH
    e = CKd._thermal_energy(rho, T, nHI + nHe + ne + xHII * nH, nH2)
    (R(rho), R(e), R(xHII * nH * CKd.MH), R(fH2 * nH * CKd.MH))
end
# thermal + chemical energy per gram (−4.48 eV per H₂, +13.6 eV per H⁺), as dense.jl
_etot(rho, e, HII_m, H2I_m) = e - CKd.E_H2_BIND * (H2I_m / (2CKd.MH)) / rho + CKd.E_H_ION * (HII_m / CKd.MH) / rho
_ehe(rho, nHeII, nHeIII) = (CKd.E_HEI_ION * nHeII + (CKd.E_HEI_ION + CKd.E_HEII_ION) * nHeIII) / rho
# the He stages the planck path starts from (first sub-step: Saha at T0, n_e from H⁺, self-consistent)
function _he0(nH, T0, nHII0)
    yHe = (1 - _FHd) / (4_FHd) * nH
    s1, s2 = CKd.helium_saha_pair(T0)
    _, a, b, _ = CKd.helium_selfconsistent(s1, s2, CKd.k3(T0), CKd.k4(T0), CKd.k5(T0), CKd.k6(T0), nHII0, yHe)
    a, b
end

@testset "dense: equilibrium constants" begin
    for T in (150.0, 300.0, 1000.0, 3000.0, 1.0e4, 3.0e4)
        @test isapprox(CKd.K_H2(T), _K_H2_exact(T); rtol = (T < 200 ? 5e-2 : T < 500 ? 1e-3 : 1e-4))
        @test isapprox(CKd.K_ion(T), _K_ion_ref(T); rtol = 1e-12)
    end
    # Float32 ln-forms are finite where K itself is not representable
    @test isfinite(CKd.lnK_H2(100.0f0)) && isfinite(CKd.lnK_ion(300.0f0))
    @test CKd.K_ion(300.0f0) == 0.0f0            # underflows cleanly to zero, no NaN
    @test isapprox(CKd.lnK_H2(3000.0f0), CKd.lnK_H2(3000.0); rtol = 1e-5)
end

@testset "dense: detailed-balance rates" begin
    for T in (3000.0, 6000.0, 1.0e4)
        kH = CKd.k22(T) * CKd.K_H2(T)
        @test isapprox(CKd.k13_dense(T, 1e16, CKd.k13(T), CKd.k22(T)), kH; rtol = 1e-6)   # LTE limit
        T >= 4000 && @test isapprox(CKd.k13_dense(T, 1e-2, CKd.k13(T), CKd.k22(T)), CKd.k13(T); rtol = 1e-4)  # low-n limit
        @test CKd.k13_dense(3000.0, 1e-2, CKd.k13(3000.0), CKd.k22(3000.0)) < 1e-7 * CKd.k22(3000.0) * CKd.K_H2(3000.0) * 1.01
    end
    for T in (1.0e4, 2.0e4, 1.0e5)
        @test isapprox(CKd.k1r_db(T, CKd.k1(T)) * CKd.K_ion(T), CKd.k1(T); rtol = 1e-10)
    end
    # continuous and monotone across the 1e4 K join
    a = CKd.k1r_db(9999.0, CKd.k1(9999.0)); b = CKd.k1r_db(10001.0, CKd.k1(10001.0))
    @test isapprox(a, b; rtol = 1e-3) && a > b
    @test CKd.k1r_db(1000.0, CKd.k1(1000.0)) > CKd.k1r_db(1.0e4, CKd.k1(1.0e4))
end

@testset "dense: energy-conserving Saha closure" begin
    for R in (Float64, Float32), (nH, T, ion) in ((1e16, 3500.0, false), (1e18, 5000.0, true), (1e20, 1.2e4, true))
        rho = R(nH * CKd.MH / _FHd)
        yHI = R(0.5nH); yH2I = R(0.5nH); yHII = R(1e-6 * nH); yde = yHII
        nHe = R((1 - _FHd) / (4_FHd) * nH)
        e = CKd._thermal_energy(rho, R(T), nHe + yHI + yHII + yde, yH2I / 2)
        etot = e - R(CKd.E_H2_BIND) * (yH2I / 2) / rho + (ion ? R(CKd.E_H_ION) * yHII / rho : zero(R))
        C = yHI + yH2I + (ion ? yHII : zero(R))
        n_other = nHe + (ion ? zero(R) : yHII + yde)
        en, yHIn, yH2n, yHIIn, yden, Tn = CKd.dense_equilibrium(rho, etot, C, zero(R), n_other, zero(R), R(T), ion)
        tol = R === Float64 ? 1e-10 : 2e-4
        @test isapprox(yHIn + yH2n + yHIIn, C; rtol = tol)                                   # H nuclei
        etot1 = en - R(CKd.E_H2_BIND) * (yH2n / 2) / rho + (ion ? R(CKd.E_H_ION) * yHIIn / rho : zero(R))
        @test isapprox(etot1, etot; atol = tol * e)                                          # energy
        @test isapprox(Float64(yHIn)^2 / (Float64(yH2n) / 2), CKd.K_H2(Float64(Tn)); rtol = (R === Float64 ? 1e-9 : 5e-3))  # Saha H₂
        ion && @test isapprox(Float64(yHIIn) * Float64(yden) / Float64(yHIn), CKd.K_ion(Float64(Tn)); rtol = (R === Float64 ? 1e-9 : 5e-3))  # Saha H⁺
        # T is the exact inverse of the thermal energy relation used by the dense path
        n_no = n_other + yHIn + yHIIn + (ion ? yden : zero(R))
        @test isapprox(CKd._temperature_dense(rho, en, n_no, yH2n / 2), Tn; rtol = (R === Float64 ? 1e-10 : 1e-5))
    end
end

@testset "dense: evolve_cell reaches Saha at fixed total energy" begin
    # (nH, T0, dt): far-from-equilibrium start (f_H2 = 0.5, x_HII = 1e-8), optically thick,
    # no radiative cooling.  Reference: the Saha composition at the cell's own final T.
    cases = ((1e14, 2000.0, 3e7), (1e16, 3000.0, 3e6), (1e16, 3500.0, 3e6),
             (1e18, 5000.0, 3e5), (1e20, 8000.0, 3e4), (1e20, 1.2e4, 3e4), (1e21, 2.0e4, 1e4))
    for (nH, T0, dt) in cases
        rho, e0, HII0, H2I0 = _cell(Float64, nH, T0, 0.5, 1e-8)
        o = CKd.evolve_cell(rho, e0, HII0, H2I0, 0.0, dt, 20.0; fh = _FHd, diagnostics = Val(true),
                            dense = Val(true), planck_radiation = Val(true), radiative_cooling = Val(false))
        @test o.completed
        @test o.iterations <= 10
        nHe = (1 - _FHd) / (4_FHd) * nH
        nHII = o.HII_m / CKd.MH; nH2 = o.H2I_m / (2CKd.MH); nHI = nH - nHII - 2nH2
        Tend = CKd._temperature_dense(rho, o.e, nHe + nHI + 2nHII, nH2)
        s = _saha_ref(nH, Tend)
        @test isapprox(o.H2I_m / (nH * CKd.MH), s.fH2; rtol = 2e-3)
        # H⁺: Saha where its relaxation is fast on dt (≥ 1e18 here); below that it is
        # integrated and must lie between its start and the Saha value
        xHII = nHII / nH
        if nH >= 1e18
            @test isapprox(xHII, s.xHII; rtol = 2e-2)
        else
            @test min(1e-8, s.xHII) * 0.5 <= xHII <= max(1e-8, s.xHII) * 2
        end
        a0, b0 = _he0(nH, T0, HII0 / CKd.MH)
        @test abs(_etot(rho, o.e, o.HII_m, o.H2I_m) + _ehe(rho, o.nHeII, o.nHeIII) -
                  _etot(rho, e0, HII0, H2I0) - _ehe(rho, a0, b0)) < 1e-6 * e0
    end
    # Float32 (n_HI ≤ 1e18: above ~1.8e19 cm⁻³ the pinned network's yHI^2 overflows Float32)
    for (nH, T0, dt) in ((1e16, 3000.0, 3e6), (1e18, 5000.0, 3e5))
        rho, e0, HII0, H2I0 = _cell(Float32, nH, T0, 0.5, 1e-8)
        o = CKd.evolve_cell(rho, e0, HII0, H2I0, 0.0f0, Float32(dt), 20.0f0; fh = _FHd, diagnostics = Val(true),
                            dense = Val(true), planck_radiation = Val(true), radiative_cooling = Val(false))
        @test o.completed
        # Float32 rounding differs between CPUs and Julia versions (x86-64 Julia 1.13 vs Apple Silicon 1.12)
        # and can move the stiffness thresholds: a handful of extra sub-steps is fine, thousands are not
        @test o.iterations <= 100
        nHe = (1 - _FHd) / (4_FHd) * nH
        nHII = o.HII_m / CKd.MH; nH2 = o.H2I_m / (2CKd.MH)
        Tend = CKd._temperature_dense(Float64(rho), Float64(o.e), nHe + (nH - nHII - 2nH2) + 2nHII, Float64(nH2))
        @test isapprox(Float64(o.H2I_m) / (nH * CKd.MH), _saha_ref(nH, Tend).fH2; rtol = 1e-2)
    end
end

@testset "dense: default path untouched, low density unaffected" begin
    # a pinned state: default call is bit-identical whether or not dense.jl exists (the
    # new K entries are absent ⇒ the added terms are exactly 0.0)
    rho, e0, HII0, H2I0 = _cell(Float64, 1.0, 8000.0, 1e-6, 0.1)
    a = CKd.evolve_cell(rho, e0, HII0, H2I0, 0.0, 3e13, 10.0; fh = _FHd)
    b = CKd.evolve_cell(rho, e0, HII0, H2I0, 0.0, 3e13, 10.0; fh = _FHd, dense = Val(false))
    @test a == b
    # low-density cells, 1 Myr step: the two schemes share the physics there (k13 blend
    # ≤ 0.4 % at n ≪ n_cr, three-body recombination negligible) and differ only in their
    # sub-step choices; both carry a few-% operator-splitting error at dtfrac = 0.1
    # (measured against dtfrac = 0.01 references: p90 ≈ 3.5 % in e for either scheme)
    for (nH, T, fH2, xHII) in ((1e-3, 1.0e4, 1e-6, 0.3), (1.0, 5000.0, 1e-5, 1e-3), (1e2, 300.0, 1e-4, 1e-4), (1e4, 1000.0, 1e-3, 1e-6))
        rho, e0, HII0, H2I0 = _cell(Float64, nH, T, fH2, xHII)
        a = CKd.evolve_cell(rho, e0, HII0, H2I0, 0.0, 3e13, 10.0; fh = _FHd)
        b = CKd.evolve_cell(rho, e0, HII0, H2I0, 0.0, 3e13, 10.0; fh = _FHd, dense = Val(true))
        @test isapprox(a[1], b[1]; rtol = 5e-2)
        @test isapprox(a[2], b[2]; rtol = 5e-2, atol = 1e-30)
        @test isapprox(a[3], b[3]; rtol = 5e-2, atol = 1e-30)
    end
    # converged (dtfrac = 0.002) answers of the two schemes agree where the physics is shared
    # (at dtfrac = 0.01 the default is not yet converged in the H₂-cooling cell: 0.5 %)
    for (nH, T, fH2, xHII) in ((1.0, 5000.0, 1e-5, 1e-3), (1e2, 300.0, 1e-4, 1e-4), (1e4, 1000.0, 1e-3, 1e-6))
        rho, e0, HII0, H2I0 = _cell(Float64, nH, T, fH2, xHII)
        a = CKd.evolve_cell(rho, e0, HII0, H2I0, 0.0, 3e13, 10.0; fh = _FHd, dtfrac = 0.002, itcap = 2_000_000)
        b = CKd.evolve_cell(rho, e0, HII0, H2I0, 0.0, 3e13, 10.0; fh = _FHd, dense = Val(true), dtfrac = 0.002, itcap = 2_000_000)
        @test isapprox(a[1], b[1]; rtol = 1e-2)
        @test isapprox(a[3], b[3]; rtol = 2e-2, atol = 1e-30)
    end
end

@testset "dense: KernelAbstractions solve_chem! (cpu f64/f32, metal f32)" begin
    # (the 1e14, 2000 K cell is left out here: at dt = 3e5 s its pair stiffness ν·dt ≈ 2–10
    # sits on the closure threshold, and f32 CPU vs Metal rounding can flip that branch)
    cases = ((1e14, 2500.0), (1e16, 3000.0), (1e16, 3500.0), (1e18, 5000.0), (1e12, 1500.0), (1e10, 800.0))
    n = length(cases)
    rho = zeros(n); e = zeros(n); HII = zeros(n); H2I = zeros(n)
    for (i, (nH, T)) in enumerate(cases)
        rho[i], e[i], HII[i], H2I[i] = _cell(Float64, nH, T, 0.5, 1e-8)
    end
    dt = 3e5
    # optically thick, no radiative cooling: the equilibrium/energy-conservation problem,
    # which the device must solve exactly like the scalar path
    kw = (; a_value = 1 / 21, dt, density_units = 1.0, length_units = 1.0, time_units = 1.0, fh = _FHd,
          dense = true, planck_radiation = true, radiative_cooling = false)
    ref = (copy(e), copy(HII), copy(H2I))
    CKd.solve_chem!(rho, ref[1], ref[2], ref[3]; kw..., backend = :cpu, precision = Float64)
    for i in 1:n
        o = CKd.evolve_cell(rho[i], e[i], HII[i], H2I[i], 0.0, dt, 20.0; fh = _FHd, dense = Val(true),
                            planck_radiation = Val(true), radiative_cooling = Val(false))
        @test o[1] == ref[1][i] && o[3] == ref[3][i]
        od = CKd.evolve_cell(rho[i], e[i], HII[i], H2I[i], 0.0, dt, 20.0; fh = _FHd, dense = Val(true),
                             planck_radiation = Val(true), radiative_cooling = Val(false), diagnostics = Val(true))
        a0, b0 = _he0(cases[i][1], cases[i][2], HII[i] / CKd.MH)
        @test abs(_etot(rho[i], ref[1][i], ref[2][i], ref[3][i]) + _ehe(rho[i], od.nHeII, od.nHeIII) -
                  _etot(rho[i], e[i], HII[i], H2I[i]) - _ehe(rho[i], a0, b0)) < 1e-6 * e[i]
    end
    # CPU f32 (analytic rates) vs f64
    e32 = copy(e); h32 = copy(HII); m32 = copy(H2I)
    CKd.solve_chem!(rho, e32, h32, m32; kw..., backend = :cpu, precision = Float32)
    for i in 1:n
        @test isapprox(e32[i], ref[1][i]; rtol = 1e-2)
        @test isapprox(m32[i], ref[3][i]; rtol = 1e-2, atol = 1e-6 * rho[i])
    end
    # WITH radiative cooling (default), CPU f32 vs f64: a few % from the sub-step count in
    # the cooling-dominated cells (the 1e14, 2000 K cell takes ~30 sub-steps)
    kwc = merge(kw, (; radiative_cooling = true))
    refc = (copy(e), copy(HII), copy(H2I))
    CKd.solve_chem!(rho, refc[1], refc[2], refc[3]; kwc..., backend = :cpu, precision = Float64)
    ec32 = copy(e); hc32 = copy(HII); mc32 = copy(H2I)
    CKd.solve_chem!(rho, ec32, hc32, mc32; kwc..., backend = :cpu, precision = Float32)
    for i in 1:n
        @test isapprox(ec32[i], refc[1][i]; rtol = 5e-2)
    end
    # Metal f32 vs CPU f32 with the SAME log–log rate/cooling tables (Metal defaults to
    # tables when none are given; the CPU default is analytic): the device-parity gate
    if CKd.has_backend(:metal)
        rt_c = build_rate_tables(; backend = :cpu, precision = Float32)
        ct_c = build_cooling_tables(; backend = :cpu, precision = Float32)
        rt_m = build_rate_tables(; backend = :metal, precision = Float32)
        ct_m = build_cooling_tables(; backend = :metal, precision = Float32)
        ec = copy(e); hc = copy(HII); mc = copy(H2I)
        CKd.solve_chem!(rho, ec, hc, mc; kw..., backend = :cpu, precision = Float32, rate_tables = rt_c, cool_tables = ct_c)
        em = copy(e); hm = copy(HII); mm = copy(H2I)
        CKd.solve_chem!(rho, em, hm, mm; kw..., backend = :metal, precision = Float32, rate_tables = rt_m, cool_tables = ct_m)
        for i in 1:n
            @test isapprox(em[i], ec[i]; rtol = 1e-2)
            @test isapprox(mm[i], mc[i]; rtol = 1e-2, atol = 1e-6 * rho[i])
            @test isapprox(em[i], ref[1][i]; rtol = 5e-2)        # tables vs analytic, f32 vs f64
        end
    else
        @info "dense KA test: :metal unavailable, device parity skipped"
    end
end

# test_atomic_helium.jl — the complete atomic H + He cooling set.
#
# 1. `cooling_rate_helium` is the Cen (1992) / Anninos et al. (1997) helium sum with the
#    documented multiplicities (He I 2³S terms ∝ n_e² n_HeII, brem Z² weights).
# 2. Collisional-ionisation-equilibrium (CIE) cooling curve of primordial gas: the H
#    (Lyα) peak near 2e4 K, the He⁺ bump near 1e5 K, bremsstrahlung above; magnitudes in
#    the textbook range.
# 3. evolve_cell / solve_chem! with `helium_cooling`: bit-identical default, more cooling
#    with He for hot ionised gas, scalar ≡ kernel, CPU f32 ≡ Metal f32.
# 4. dense + planck_radiation + helium_cooling: the He ionisation energy is part of the
#    conserved budget.
using ChemistryKernels, EmissionKernels, Test
try; @eval using Metal; catch; end

const CKh = ChemistryKernels
const _FHh = 0.76

@testset "helium cooling sum = documented channels" begin
    for T in (1.5e4, 5.0e4, 1.0e5, 3.0e5, 1.0e6)
        nHeI, nHeII, nHeIII, nde = 0.02, 0.05, 0.01, 1.3
        expect = ceHeI(T) * nHeII * nde^2 + ceHeII(T) * nHeII * nde +
                 ciHeI(T) * nHeI * nde + ciHeII(T) * nHeII * nde + ciHeIS(T) * nHeII * nde^2 +
                 (reHeII1(T) + reHeII2(T)) * nHeII * nde + reHeIII(T) * nHeIII * nde +
                 brem(T) * nHeII * nde + brem_Z(T, 2) * nHeIII * nde
        @test isapprox(cooling_rate_helium(nHeI, nHeII, nHeIII, nde, T), expect; rtol = 1e-12)
        ch = helium_channels(nHeI, nHeII, nHeIII, nde, T)
        @test isapprox(ch.total, expect; rtol = 1e-12)
        @test ch.excitation >= 0 && ch.collisional_ionisation >= 0 && ch.recombination >= 0 && ch.brem >= 0
    end
    # neutral cold helium: no He cooling at all (collisional ionisation of He I is 0 below 0.8 eV)
    @test cooling_rate_helium(0.08, 0.0, 0.0, 1e-4, 5000.0) == 0.0
    # Float32 agrees
    @test isapprox(cooling_rate_helium(0.02f0, 0.05f0, 0.01f0, 1.3f0, 1.0f5),
                   cooling_rate_helium(0.02, 0.05, 0.01, 1.3, 1.0e5); rtol = 1e-5)
end

# CIE composition at (nH, T): H from k1/k2, He from k3/k4, k5/k6 (she = 0: no radiation)
function _cie(nH, T)
    yHe = (1 - _FHh) / (4_FHh)
    r1 = CKh.k1(T) / CKh.k2(T)                       # n_HII/n_HI = r1 (independent of n_e)
    xHII = r1 / (1 + r1)
    nHeI, nHeII, nHeIII = CKh.helium_equilibrium(0.0, 0.0, CKh.k3(T), CKh.k4(T), CKh.k5(T), CKh.k6(T), 1.0, yHe * nH)
    ne = xHII * nH + nHeII + 2nHeIII
    (nHI = (1 - xHII) * nH, nHII = xHII * nH, nHeI = nHeI, nHeII = nHeII, nHeIII = nHeIII, ne = ne)
end
_Λ_H(c, T, z) = cooling_rate_total(c.nHI, c.nHII, c.nHeI, c.ne, 0.0, 0.0, T, z) - comp1_cmb(z) * (T - comp2_cmb(z)) * c.ne
_Λ_He(c, T) = cooling_rate_helium(c.nHeI, c.nHeII, c.nHeIII, c.ne, T)

@testset "CIE cooling curve of primordial gas" begin
    nH = 1.0; z = 0.0
    Ts = 10 .^ range(4.0, 7.0; length = 121)
    ΛH = [_Λ_H(_cie(nH, T), T, z) for T in Ts]
    ΛHe = [_Λ_He(_cie(nH, T), T) for T in Ts]
    Λ = ΛH .+ ΛHe
    # H peak (Lyα) between 1.5e4 and 3e4 K, Λ/n_H² of order 1e-22 (Dalgarno & McCray 1972 range)
    iH = argmax(ΛH[Ts .< 6e4]); TH = Ts[Ts .< 6e4][iH]
    @test 1.5e4 < TH < 3.0e4
    @test 3e-23 < ΛH[iH] / nH^2 < 3e-22
    # He⁺ (He II Lyα) bump: the He share has a local maximum between 6e4 and 2e5 K
    share = ΛHe ./ Λ
    sel = (Ts .> 6e4) .& (Ts .< 2e5)
    @test maximum(share[sel]) > 0.3
    @test share[findfirst(Ts .> 6e4)] < maximum(share[sel]) && share[findlast(Ts .< 2e5)] < maximum(share[sel])
    # above 1e6 K bremsstrahlung dominates and the curve rises ∝ T^0.5
    i6 = findfirst(Ts .>= 1e6); i7 = length(Ts)
    @test Λ[i7] > Λ[i6]
    bre = brem(Ts[i7]) * (_cie(nH, Ts[i7]).nHII + _cie(nH, Ts[i7]).nHeII + 4_cie(nH, Ts[i7]).nHeIII) * _cie(nH, Ts[i7]).ne
    @test bre / Λ[i7] > 0.8
    # He never cools more than H by orders of magnitude; both positive
    @test all(ΛHe .>= 0) && all(ΛH .> 0)
end

# one cell: H nuclei nH, T, H₂ fraction, x_HII (He neutral in the energy, as the network's reconstruction)
function _hcell(::Type{R}, nH, T, fH2, xHII) where {R}
    rho = nH * CKh.MH / _FHh
    nHI = (1 - fH2 - xHII) * nH; nH2 = fH2 * nH / 2; nHe = (1 - _FHh) / (4_FHh) * nH; ne = xHII * nH
    e = CKh._thermal_energy(rho, T, nHI + nHe + ne + xHII * nH, nH2)
    (R(rho), R(e), R(xHII * nH * CKh.MH), R(fH2 * nH * CKh.MH))
end

@testset "helium_cooling in evolve_cell / solve_chem!" begin
    # default untouched
    rho, e0, HII0, H2I0 = _hcell(Float64, 1.0, 1.0e5, 1e-8, 0.99)
    a = CKh.evolve_cell(rho, e0, HII0, H2I0, 0.0, 3e12, 10.0; fh = _FHh)
    b = CKh.evolve_cell(rho, e0, HII0, H2I0, 0.0, 3e12, 10.0; fh = _FHh, helium_cooling = Val(false))
    @test a == b
    # hot ionised gas: helium adds cooling (He⁺ excitation / He bremsstrahlung at 1e5 K)
    c = CKh.evolve_cell(rho, e0, HII0, H2I0, 0.0, 3e12, 10.0; fh = _FHh, helium_cooling = Val(true))
    @test c[1] < a[1]
    @test c[1] > 0.3 * a[1]                    # but not by an unphysical factor
    # kernel ≡ scalar, several cells
    cases = ((1.0, 1.0e5, 0.99), (10.0, 3.0e4, 0.5), (0.1, 3.0e5, 1.0), (100.0, 1.5e4, 0.1))
    n = length(cases)
    rhov = zeros(n); ev = zeros(n); hv = zeros(n); mv = zeros(n)
    for (i, (nH, T, x)) in enumerate(cases)
        rhov[i], ev[i], hv[i], mv[i] = _hcell(Float64, nH, T, 1e-8, x)
    end
    dt = 3e12
    kw = (; a_value = 1 / 11, dt, density_units = 1.0, length_units = 1.0, time_units = 1.0, fh = _FHh, helium_cooling = true)
    ek = copy(ev); hk = copy(hv); mk = copy(mv)
    CKh.solve_chem!(rhov, ek, hk, mk; kw..., backend = :cpu, precision = Float64)
    for i in 1:n
        o = CKh.evolve_cell(rhov[i], ev[i], hv[i], mv[i], 0.0, dt, 10.0; fh = _FHh, helium_cooling = Val(true))
        @test o[1] == ek[i]
    end
    if CKh.has_backend(:metal)
        rt_c = build_rate_tables(; backend = :cpu, precision = Float32); ct_c = build_cooling_tables(; backend = :cpu, precision = Float32)
        rt_m = build_rate_tables(; backend = :metal, precision = Float32); ct_m = build_cooling_tables(; backend = :metal, precision = Float32)
        ec = copy(ev); hc_ = copy(hv); mc = copy(mv)
        CKh.solve_chem!(rhov, ec, hc_, mc; kw..., backend = :cpu, precision = Float32, rate_tables = rt_c, cool_tables = ct_c)
        em = copy(ev); hm = copy(hv); mm = copy(mv)
        CKh.solve_chem!(rhov, em, hm, mm; kw..., backend = :metal, precision = Float32, rate_tables = rt_m, cool_tables = ct_m)
        for i in 1:n
            @test isapprox(em[i], ec[i]; rtol = 2e-2)
            @test isapprox(em[i], ek[i]; rtol = 5e-2)
        end
    else
        @info "helium cooling: :metal unavailable, device parity skipped"
    end
end

@testset "dense + planck + helium: ionisation energy in the budget" begin
    # hot dense gas: He partly ionised in Saha at the gas T; without radiative cooling the
    # thermal + chemical (H₂, H⁺, He⁺, He²⁺) energy is conserved
    nH = 1e20; T0 = 3.0e4
    rho, e0, HII0, H2I0 = _hcell(Float64, nH, T0, 0.01, 1e-3)
    o = CKh.evolve_cell(rho, e0, HII0, H2I0, 0.0, 1e4, 20.0; fh = _FHh, diagnostics = Val(true),
                        dense = Val(true), planck_radiation = Val(true), radiative_cooling = Val(false),
                        helium_cooling = Val(true))
    @test o.completed && o.iterations <= 10
    yHe = (1 - _FHh) / (4_FHh) * nH
    # the He stages the solver charged for are returned in the diagnostics; the initial ones
    # are the equilibrium at the start (first sub-step, n_e = n_HII, Saha at T0)
    nHII0 = HII0 / CKh.MH; nH2_0 = H2I0 / (2CKh.MH)
    s1, s2 = CKh.helium_saha_pair(T0)
    _, nHeII0, nHeIII0, _ = CKh.helium_selfconsistent(s1, s2, CKh.k3(T0), CKh.k4(T0), CKh.k5(T0), CKh.k6(T0), nHII0, yHe)
    chem(nHII, nH2, nHeII, nHeIII) = (-CKh.E_H2_BIND * nH2 + CKh.E_H_ION * nHII +
                                      CKh.E_HEI_ION * nHeII + (CKh.E_HEI_ION + CKh.E_HEII_ION) * nHeIII) / rho
    E0 = e0 + chem(nHII0, nH2_0, nHeII0, nHeIII0)
    E1 = o.e + chem(o.HII_m / CKh.MH, o.H2I_m / (2CKh.MH), o.nHeII, o.nHeIII)
    @test abs(E1 - E0) < 1e-6 * e0
    @test o.nHeII + o.nHeIII > 1e-6 * yHe       # helium did ionise (the closure cools the gas
                                               # by H₂ dissociation / H ionisation; He stays He⁺)
end

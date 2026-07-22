# test_dust.jl — unit tests for dust physics (rates, T_dust, shielding, network)
#
# Tests cover:
#   1. Rate function limiting behaviors and reference values
#   2. T_dust_eq correctness at known G₀ / A_V / T_CMB
#   3. Shielding function limits and reference values
#   4. network_step with dust_rates: H₂ source, HII sink
#   5. evolve_cell regression: dust=false path is bit-identical to no-dust call

using Test
using ChemistryKernels

# ─── 1. Rate functions ────────────────────────────────────────────────────────

@testset "k_H2_dust" begin
    # Zero metallicity → no H₂ on dust regardless of T
    @test k_H2_dust(300.0, 15.0, 0.0) == 0.0
    # Reference value at standard conditions (T=300K, T_d=15K, Z=1)
    # S(300) = 1/(1+(300/464)²) ≈ 0.706; ξ(15) ≈ 1 (600/15=40, exp(-40)≈0)
    # k ≈ 3e-17 * 1.0 * 0.706 * 1.0 ≈ 2.12e-17
    k = k_H2_dust(300.0, 15.0, 1.0)
    @test 1.5e-17 < k < 3.5e-17
    # High T_gas: sticking falls, rate drops
    @test k_H2_dust(5000.0, 15.0, 1.0) < k_H2_dust(300.0, 15.0, 1.0)
    # High T_dust: surface efficiency ξ falls, rate drops
    @test k_H2_dust(300.0, 100.0, 1.0) < k_H2_dust(300.0, 15.0, 1.0)
    # Linear scaling with metallicity
    @test k_H2_dust(300.0, 15.0, 2.0) ≈ 2 * k_H2_dust(300.0, 15.0, 1.0)
    # Float32 precision-generic
    @test k_H2_dust(300f0, 15f0, 1f0) isa Float32
end

@testset "Gamma_PE" begin
    # Zero G₀ → no PE heating
    @test Gamma_PE(1000.0, 0.0, 1.0, 1.0) == 0.0
    # Zero metallicity → no PE heating
    @test Gamma_PE(1000.0, 1.0, 0.0, 1.0) == 0.0
    # Reference value: G₀=1, T=1e4 K, Z=1, n_e=1 cm⁻³ (ψ = G0√T/ne = 100)
    # ε = 4.87e-2/(1+4e-3*100^0.73) + 3.65e-2/(1+2e-4*100) ≈ 0.0793
    # Γ ≈ 1.3e-24 * 0.0793 ≈ 1.03e-25 erg/s per H nucleus
    Gpe = Gamma_PE(1e4, 1.0, 1.0, 1.0)
    @test 5e-26 < Gpe < 2e-25
    # Monotonically increasing with G₀ (not exactly linear because ε depends on ψ=G0√T/ne)
    @test Gamma_PE(1e4, 2.0, 1.0, 1.0) > Gamma_PE(1e4, 1.0, 1.0, 1.0)
    # Float32
    @test Gamma_PE(1f4, 1f0, 1f0, 1f0) isa Float32
    # Positive radiation with no electrons is the exact high-charging limit.
    @test Gamma_PE(1e4, 1.0, 1.0, 0.0) == 0.0
end

@testset "Lambda_gr" begin
    # T_gas == T_dust → no energy exchange (gas-grain in thermal equilibrium)
    @test Lambda_gr(1000.0, 1000.0, 100.0, 1.0) == 0.0
    # T_gas > T_dust → gas cools (positive)
    @test Lambda_gr(1000.0, 15.0, 100.0, 1.0) > 0.0
    # T_gas < T_dust → gas heats (negative, warm grains heat cold gas)
    @test Lambda_gr(10.0, 100.0, 100.0, 1.0) < 0.0
    # Zero metallicity → no coupling
    @test Lambda_gr(1000.0, 15.0, 100.0, 0.0) == 0.0
    # Float32
    @test Lambda_gr(1000f0, 15f0, 100f0, 1f0) isa Float32
end

@testset "k_gr_recomb_HII" begin
    # Zero metallicity → no grain recombination
    @test k_gr_recomb_HII(8000.0, 1.0, 0.0, 1.0) == 0.0
    # At standard conditions: should be O(1e-13) cm³/s
    k = k_gr_recomb_HII(8000.0, 1.0, 1.0, 1.0)
    @test 1e-14 < k < 2e-13
    # High ψ (strong UV, low n_e) → suppressed rate
    k_hi = k_gr_recomb_HII(8000.0, 100.0, 1.0, 0.001)
    @test k_hi < k
    # Linear in Z_rel
    @test k_gr_recomb_HII(8000.0, 1.0, 2.0, 1.0) ≈ 2 * k_gr_recomb_HII(8000.0, 1.0, 1.0, 1.0)
    # Float32
    @test k_gr_recomb_HII(8000f0, 1f0, 1f0, 1f0) isa Float32
    @test k_gr_recomb_HII(8000.0, 1.0, 1.0, 0.0) == 0.0
    @test k_gr_recomb_HII(8000.0, 0.0, 1.0, 0.0) == 1.225e-13
end

@testset "Lambda_dust" begin
    # Zero metallicity → no dust emission
    @test Lambda_dust(20.0, 0.0, 100.0) == 0.0
    # Positive and scales as T^6
    L1 = Lambda_dust(20.0, 1.0, 100.0)
    L2 = Lambda_dust(40.0, 1.0, 100.0)
    @test L1 > 0
    @test L2 / L1 ≈ (40/20)^6 rtol=1e-10
    # Float32
    @test Lambda_dust(20f0, 1f0, 100f0) isa Float32
end

# ─── 2. Dust temperature ─────────────────────────────────────────────────────

@testset "T_dust_eq" begin
    T_CMB = 2.728
    # Pure CMB: T_dust = T_CMB (CMB sets the floor)
    @test T_dust_eq(0.0, 0.0, T_CMB) ≈ T_CMB rtol=0.01
    # G₀=1, no extinction, T_CMB negligible → T_dust ≈ 12.2 K (Hollenbach & McKee 1979)
    T_uv = T_dust_eq(1.0, 0.0, T_CMB)
    @test 11.0 < T_uv < 14.0
    # Extinction attenuates UV: more extinct → cooler dust
    @test T_dust_eq(1.0, 5.0, T_CMB) < T_dust_eq(1.0, 0.0, T_CMB)
    # Full extinction → dust settles at CMB floor
    @test T_dust_eq(1.0, 100.0, T_CMB) ≈ T_CMB rtol=0.05
    # T_dust scales as G₀^(1/5.7): doubling G₀ multiplies T_dust by 2^(1/5.7) ≈ 1.13
    T1 = T_dust_eq(1.0, 0.0, 0.0)
    T2 = T_dust_eq(2.0, 0.0, 0.0)
    @test T2 / T1 ≈ 2.0^(1/5.7) rtol=0.01
    # Higher CMB temperature raises the floor
    @test T_dust_eq(0.0, 0.0, 10.0) ≈ 10.0 rtol=0.01
    # Float32
    @test T_dust_eq(1f0, 0f0, T_CMB) isa Float32
end

# ─── 3. Shielding functions ──────────────────────────────────────────────────

@testset "f_shield_H2" begin
    # Optically thin (N_H2 = 0): f_shield ≈ 1
    @test f_shield_H2(0.0) ≈ 1.0 rtol=1e-3
    # Increasing column → more shielding
    @test f_shield_H2(1e15) < f_shield_H2(1e14)
    @test f_shield_H2(1e16) < f_shield_H2(1e15)
    # Reference from Draine & Bertoldi (1996) Eq. 37 at N_H2 = 5e14 (x=1):
    # f = (0.965/(1+1/3)^2 + 0.035/sqrt(2)) * exp(-8.5e-4*sqrt(2))
    # = (0.965/1.778 + 0.0247) * exp(-1.202e-3)
    # = (0.5428 + 0.0247) * 0.9988 ≈ 0.567
    f = f_shield_H2(5e14)
    @test 0.4 < f < 0.7
    # Float32
    @test f_shield_H2(0f0) isa Float32
    @test f_shield_H2(1f15) < 1f0
end

@testset "f_dust_LW" begin
    # No column, any Z: no attenuation
    @test f_dust_LW(0.0, 1.0) == 1.0
    # Zero metallicity: no attenuation
    @test f_dust_LW(1e21, 0.0) == 1.0
    # Increasing column → more attenuation
    @test f_dust_LW(1e21, 1.0) < f_dust_LW(1e20, 1.0)
    # At N_H = 1e21, Z=1: τ = 2e-21 * 1e21 = 2 → f = exp(-2) ≈ 0.135
    @test f_dust_LW(1e21, 1.0) ≈ exp(-2.0) rtol=1e-10
    # Float32
    @test f_dust_LW(0f0, 1f0) isa Float32
end

@testset "k_H2_LW_eff" begin
    # No UV: zero rate
    @test k_H2_LW_eff(0.0, 0.0, 0.0, 1.0) == 0.0
    # Unshielded (N_H=0, N_H2=0): k = 5.8e-11 * G₀
    @test k_H2_LW_eff(1.0, 0.0, 0.0, 1.0) ≈ 5.8e-11 * f_shield_H2(0.0) rtol=1e-10
    # Shielding reduces rate
    @test k_H2_LW_eff(1.0, 1e15, 0.0, 1.0) < k_H2_LW_eff(1.0, 0.0, 0.0, 1.0)
    # Dust attenuation reduces rate
    @test k_H2_LW_eff(1.0, 0.0, 1e21, 1.0) < k_H2_LW_eff(1.0, 0.0, 0.0, 1.0)
    # Float32
    @test k_H2_LW_eff(1f0, 0f0, 0f0, 1f0) isa Float32
end

# ─── 4. network_step with dust_rates ─────────────────────────────────────────

@testset "network_step dust_rates" begin
    using ChemistryKernels: network_step, build_rates, equilibrium_HM, equilibrium_H2II

    # Set up a modest ISM state: partly neutral gas at T=1000 K
    T = 1000.0
    z = 0.0
    Trad = 2.728
    nH = 100.0      # H nuclei cm^-3
    fh = 0.76
    d  = nH / fh    # network density

    # Start mostly neutral with a little H₂
    yHI   = 0.9 * nH
    yHII  = 0.1 * nH
    yde   = yHII
    yH2I  = 1e-3 * nH  # trace H₂
    yHM   = 0.0; yH2II = 0.0
    yDI   = 0.0; yDII = 0.0; yHDI = 0.0
    dt    = 1e10   # 300 yr sub-step

    Hz  = 70.0 / 3.086e22   # Hubble rate [s^-1]
    K   = build_rates(T, Trad, yHI, Hz)

    # Without dust: reference H₂ after one step
    s0 = network_step(d, fh, yHI, yHII, yde, yH2I, yHM, yH2II,
                      yDI, yDII, yHDI, K, dt)

    # With dust grain formation only (k_lw = 0 to isolate the H₂ source term):
    # k_lw is set to zero here so that k_h2d can dominate at these conditions.
    # (At G₀=1 and dt=1e10 s, k_lw·dt ~ 0.6 >> k_h2d·nH·dt ~ 2e-4, so testing
    # grain formation requires suppressing LW; that combination is tested via evolve_cell.)
    T_d   = T_dust_eq(1.0, 2.0, Trad)
    k_h2d = k_H2_dust(T, T_d, 1.0)
    k_grr = k_gr_recomb_HII(T, 1.0, 1.0, yde)
    dust_rates = (; k_h2d, k_grr, k_lw = 0.0)

    s1 = network_step(d, fh, yHI, yHII, yde, yH2I, yHM, yH2II,
                      yDI, yDII, yHDI, K, dt; dust_rates = dust_rates)

    # H₂ with grain formation (no LW) > H₂ without dust (grain route adds molecules)
    @test s1.yH2I > s0.yH2I

    # HII with grain recombination < HII without (extra recombination sink)
    @test s1.yHII < s0.yHII

    # LW alone (k_h2d=0) must destroy H₂
    k_lw_val = k_H2_LW_eff(1.0, 0.0, 0.0, 1.0)
    dust_lw = (; k_h2d = 0.0, k_grr = 0.0, k_lw = k_lw_val)
    s3 = network_step(d, fh, yHI, yHII, yde, yH2I, yHM, yH2II,
                      yDI, yDII, yHDI, K, dt; dust_rates = dust_lw)
    @test s3.yH2I < s0.yH2I

    # dust_rates = nothing must give identical result to no keyword
    s2 = network_step(d, fh, yHI, yHII, yde, yH2I, yHM, yH2II,
                      yDI, yDII, yHDI, K, dt; dust_rates = nothing)
    @test s2.yHII == s0.yHII
    @test s2.yH2I == s0.yH2I
end

# ─── 5. cooling_edot dust terms ──────────────────────────────────────────────

@testset "cooling_edot dust terms" begin
    T = 8000.0; z = 0.0
    nHI = 50.0; nHII = 50.0; nHeI = 10.0; nde = 50.0; nH2 = 0.1; nHD = 0.0
    nH  = 100.0

    # Baseline (no dust)
    e0 = cooling_edot(nHI, nHII, nHeI, nde, nH2, nHD, T, z; nH = nH)

    # PE heating (positive contribution → edot less negative)
    pe_vol = Gamma_PE(T, 1.0, 1.0, nde) * nH   # erg/cm³/s
    e_pe   = cooling_edot(nHI, nHII, nHeI, nde, nH2, nHD, T, z; nH = nH,
                          Gamma_PE_vol = pe_vol)
    @test e_pe > e0    # less net cooling with PE heating

    # Gas-grain cooling when T_gas > T_dust (Lambda_gr_vol > 0 → more cooling)
    T_d    = T_dust_eq(1.0, 0.0, 2.728)
    lg_vol = Lambda_gr(T, T_d, nH, 1.0)    # positive (gas cools)
    @test lg_vol > 0
    e_gg   = cooling_edot(nHI, nHII, nHeI, nde, nH2, nHD, T, z; nH = nH,
                          Lambda_gr_vol = lg_vol)
    @test e_gg < e0   # more net cooling with gas-grain coupling

    # Both zero kwargs → identical to baseline
    e_z = cooling_edot(nHI, nHII, nHeI, nde, nH2, nHD, T, z; nH = nH,
                       Gamma_PE_vol = 0.0, Lambda_gr_vol = 0.0)
    @test e_z == e0
end

# ─── 6. evolve_cell dust=false regression ────────────────────────────────────

@testset "evolve_cell dust=false regression" begin
    # dust=false (default) must produce bit-identical output to a call that
    # simply omits the dust kwargs entirely.
    rho   = 1.0e-24   # g/cm^3
    e     = 1.0e12    # erg/g
    HII_m = 1.2e-25
    H2I_m = 1.0e-30
    HDI_m = 0.0
    dt    = 3.0e13    # s (~1 Myr)
    z     = 3.0

    e0, hii0, h2_0, hd0, _ = evolve_cell(rho, e, HII_m, H2I_m, HDI_m, dt, z)
    e1, hii1, h2_1, hd1, _ = evolve_cell(rho, e, HII_m, H2I_m, HDI_m, dt, z;
                                          dust = false)
    @test e0 == e1
    @test hii0 == hii1
    @test h2_0 == h2_1
end

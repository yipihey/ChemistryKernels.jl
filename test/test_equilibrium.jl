using ChemistryKernels, Test
using ChemistryKernels: equilibrium_HM, equilibrium_H2II, equilibrium_HeH,
                        equilibrium_DII, helium_equilibrium

# equilibrium.jl carries no closed-form grackle oracle (HM/H2II/DII are scratch in
# the reduced run).  The transcription risk is purely STRUCTURAL — which rates,
# which species, the /2 and /3 molecular factors, the 2× stoichiometry.  We pin
# rates + species to distinct values and check each function against an
# INDEPENDENT literal re-evaluation of solve_rate_cool_g.F:2450-2520.  A wrong
# k-index, a dropped factor, or a swapped species then fails immediately.
@testset "equilibrium (pinned-rate transcription)" begin
    # Published HeH⁺ coefficients: Schleicher et al. (2008) formation/recombination
    # and Bovino et al. (2011) ab-initio proton transfer.
    THeH = 300.0
    @test ChemistryKernels.kHeH_ra_spont(THeH) ≈
          8e-20 * (THeH/300)^(-0.24) * exp(-THeH/4000)
    @test ChemistryKernels.kHeH_ra_stim_base(THeH) ≈
          3.2e-20*THeH^1.8/(1 + 0.1*THeH^2.04)*exp(-THeH/4000)
    @test ChemistryKernels.kHeH_H(THeH) ≈
          4.3489e-10*THeH^0.110373*exp(-31.5396/THeH)
    @test ChemistryKernels.kHeH_e(THeH) ≈ 3e-8*(THeH/300)^(-0.47)

    # distinct, non-degenerate values so any miswiring shifts the result
    yHI, yHII, yde, yHM, yH2I, yH2II = 0.7, 0.03, 0.03, 1.0e-9, 0.02, 1.0e-11
    yDI, yHDI = 6.8e-5*0.7, 1.0e-7
    # rates (arbitrary but distinct)
    k1,k2 = 1.1e-10, 2.2e-12
    k7,k8,k9,k10,k11 = 3.0e-16, 1.3e-9, 2.0e-10, 6.0e-10, 5.0e-11
    k14,k15,k16,k17,k18,k19 = 1.0e-9, 2.0e-9, 4.0e-9, 8.0e-10, 1.3e-6, 5.0e-7
    k50,k51,k52,k53,k54 = 9.0e-11, 9.5e-11, 1.0e-9, 1.5e-9, 2.0e-9
    k27,k28 = 7.0e-12, 4.0e-13

    # H⁻ : (k7·HI·de) / [(k8+k15)HI + (k16+k17)HII + k14·de + k19·H2II/2 + k27]
    HMref = (k7*yHI*yde) /
            ((k8+k15)*yHI + (k16+k17)*yHII + k14*yde + k19*yH2II/2 + k27)
    HMjl = equilibrium_HM(yHI, yHII, yde, yH2II, k7,k8,k14,k15,k16,k17,k19,k27)
    @test isapprox(HMjl, HMref; rtol = 1e-13)

    # H2⁺ : 2(k9·HI·HII + k11·H2I/2·HII + k17·HM·HII) /
    #       (k10·HI + k18·de + k19·HM + k28)             [k29=k30=0]
    H2IIref = 2*(k9*yHI*yHII + k11*yH2I/2*yHII + k17*yHM*yHII) /
              (k10*yHI + k18*yde + k19*yHM + k28)
    H2IIjl = equilibrium_H2II(yHI, yHII, yH2I, yde, yHM, k9,k10,k11,k17,k18,k19,k28)
    @test isapprox(H2IIjl, H2IIref; rtol = 1e-13)

    # D⁺ : (k1·DI·de + k50·HII·DI + 2·k53·HII·HDI/3) / (k2·de + k51·HI + k52·H2I/2)
    DIIref = (k1*yDI*yde + k50*yHII*yDI + 2*k53*yHII*yHDI/3) /
             (k2*yde + k51*yHI + k52*yH2I/2)
    DIIjl = equilibrium_DII(yDI, yde, yHI, yHII, yH2I, yHDI, k1,k2,k50,k51,k52,k53)
    @test isapprox(DIIjl, DIIref; rtol = 1e-13)

    # precision-genericity: f32 path runs and tracks f64 to the f32 floor
    HMf  = equilibrium_HM(Float32.((yHI,yHII,yde,yH2II))...,
                          Float32.((k7,k8,k14,k15,k16,k17,k19,k27))...)
    @test isapprox(Float64(HMf), HMref; rtol = 1e-5)
    H2IIf = equilibrium_H2II(Float32.((yHI,yHII,yH2I,yde,yHM))...,
                             Float32.((k9,k10,k11,k17,k18,k19,k28))...)
    @test isapprox(Float64(H2IIf), H2IIref; rtol = 1e-5)
    DIIf = equilibrium_DII(Float32.((yDI,yde,yHI,yHII,yH2I,yHDI))...,
                           Float32.((k1,k2,k50,k51,k52,k53))...)
    @test isapprox(Float64(DIIf), DIIref; rtol = 1e-5)

    # photo-destruction (k27) must increase H⁻ destruction ⇒ lower HM
    HM_nocmb = equilibrium_HM(yHI, yHII, yde, yH2II, k7,k8,k14,k15,k16,k17,k19, 0.0)
    @test HM_nocmb > HMjl

    # Disconnected exact-zero networks are valid equilibria, not 0/0 failures.
    @test equilibrium_HM(0.7, 0.0, 0.0, 0.0, ntuple(_ -> 0.0, 8)...) == 0.0
    @test equilibrium_H2II(0.7, 0.0, 0.0, 0.0, 0.0, ntuple(_ -> 0.0, 7)...) == 0.0

    # HeH⁺: radiative association divided by proton-transfer,
    # dissociative-recombination, and photodissociation losses.
    nHeI, nHII, ne, nHI = 0.06, 2e-4, 2e-4, 0.75
    kra, kH, ke, gph = 8e-20, 7e-10, 3e-8, 2e-11
    refHeH = kra*nHeI*nHII / (kH*nHI + ke*ne + gph)
    @test equilibrium_HeH(nHeI, nHII, ne, nHI, kra, kH, ke, gph) ≈ refHeH
    @test equilibrium_HeH(nHeI, 0.0, ne, nHI, kra, kH, ke, gph) == 0.0
    @test equilibrium_DII(0.0, 0.0, 0.7, 0.0, 0.0, 0.0, ntuple(_ -> 0.0, 6)...) == 0.0
    he0 = helium_equilibrium(0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.08)
    @test he0 == (0.08, 0.0, 0.0)
    @test all(isfinite, he0)
end

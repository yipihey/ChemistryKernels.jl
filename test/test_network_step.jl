using ChemistryKernels, Test
using ChemistryKernels: network_step, equilibrium_HM, equilibrium_H2II, equilibrium_DII

# network_step has no closed grackle oracle (step_rate_g is internal); the
# transcription risk is structural (which rate in which scoef/acoef, the /2,/3,2×
# molecular factors, the Gauss-Seidel ordering).  We pin all rates + a state and
# check each provisional against an INDEPENDENT literal re-evaluation of
# solve_rate_cool_g.F:2336-2581.  End-to-end accuracy vs grackle is the Wave-5
# one-zone gate.
@testset "network_step (pinned-rate transcription)" begin
    d, fh = 1.0, 0.76
    # state (grackle mass-equiv convention: yH2I=2·n(H2), yH2II=2·n(H2⁺), yHDI=3·n(HD))
    yHI, yHII, yde = 0.70, 0.03, 0.03
    yH2I, yHM, yH2II = 0.02, 1.0e-9, 1.0e-11
    yDI, yDII, yHDI = 6.8e-5*0.70, 6.8e-5*0.03, 1.0e-7
    dt = 1.0e10
    yHeI = (1-fh)*d

    K = (; k1=1.1e-10, k2=2.2e-12, k3=1.7e-13, k4=1.0e-12, k5=1.0e-13, k6=1.0e-12,
         k7=3.0e-16, k8=1.3e-9, k9=2.0e-10, k10=6.0e-10, k11=5.0e-11, k12=4.0e-9,
         k13=1.0e-9, k14=1.0e-9, k15=2.0e-9, k16=4.0e-9, k17=8.0e-10, k18=1.3e-6,
         k19=5.0e-7, k22=1.3e-32, k57=1.0e-15, k58=1.0e-16, k27=7.0e-12, k28=4.0e-13,
         k50=9.0e-11, k51=9.5e-11, k52=1.0e-9, k53=1.5e-9, k54=2.0e-9, k55=2.5e-9,
         k56=3.0e-9, k_beta1s=0.0)
    k = K  # shorthand

    out = network_step(d, fh, yHI,yHII,yde,yH2I,yHM,yH2II,yDI,yDII,yHDI, K, dt;
                       deuterium=true, yHeII_in=0.0, yHeIII_in=0.0)

    # --- independent re-evaluation (literal Fortran) ---
    HMp = equilibrium_HM(yHI,yHII,yde,yH2II, k.k7,k.k8,k.k14,k.k15,k.k16,k.k17,k.k19,k.k27)
    H2IIeq = equilibrium_H2II(yHI,yHII,yH2I,yde,HMp,
                              k.k9,k.k10,k.k11,k.k17,k.k18,k.k19,k.k28)
    nH2II = H2IIeq/2
    Hatomic = fh*d - yH2I - HMp - H2IIeq - yHDI/3
    qion = k.k1*yde + k.k57*yHI + k.k58*yHeI/4 + k.k_beta1s
    sc_other = k.k10*H2IIeq*yHI/2 + k.k28*nH2II
    acHII = k.k2*yde + k.k9*yHI + k.k11*yH2I/2 + k.k16*yHM + k.k17*yHM
    HIIp = (yHII + (qion*Hatomic + sc_other)*dt) / (1 + (qion + acHII)*dt)
    HIIp = clamp(HIIp, 0.0, Hatomic)

    H2Ip = 2*(k.k8*yHM*yHI + k.k10*H2IIeq*yHI/2 + k.k19*H2IIeq*yHM/2 + k.k22*yHI*yHI^2)*dt + yH2I
    H2Ip /= 1 + (k.k13*yHI + k.k11*yHII + k.k12*yde)*dt

    # deuterium
    DIp = (k.k2*yDII*yde + k.k51*yDII*yHI + 2*k.k55*yHDI*yHI/3)*dt + yDI
    DIp /= 1 + (k.k1*yde + k.k50*yHII + k.k54*yH2I/2 + k.k56*yHM)*dt
    @test isapprox(out.yDI, max(DIp,0.0); rtol=1e-12)

    DIIp = equilibrium_DII(yDI,yde,yHI,yHII,yH2I,yHDI, k.k1,k.k2,k.k50,k.k51,k.k52,k.k53)
    @test isapprox(out.yDII, max(DIIp,0.0); rtol=1e-12)

    HDIp = 3*(k.k52*yDII*yH2I/2/2 + k.k54*yDI*yH2I/2/2 + 2*k.k56*yDI*yHM/2)*dt + yHDI
    HDIp /= 1 + (k.k53*yHII + k.k55*yHI)*dt
    @test isapprox(out.yHDI, max(HDIp,0.0); rtol=1e-12)

    # Conservative final H assignment and charge balance.
    Hbudget = max(fh*d - max(HDIp,0.0)/3, 0.0)
    H2I_n = clamp(H2Ip, 0.0, Hbudget)
    Hrem = Hbudget - H2I_n
    HM_n = clamp(HMp, 0.0, Hrem); Hrem -= HM_n
    H2II_n = clamp(H2IIeq, 0.0, Hrem); Hrem -= H2II_n
    HII_n = clamp(HIIp, 0.0, Hrem)
    HI_n = Hrem - HII_n
    @test isapprox(out.yHI, HI_n; rtol=1e-12)
    @test isapprox(out.yHII, HII_n; rtol=1e-12)
    @test isapprox(out.yH2I, H2I_n; rtol=1e-12)
    @test isapprox(out.yHM, HM_n; rtol=1e-12)
    @test isapprox(out.yH2II, H2II_n; rtol=1e-12)
    de_n = max(HII_n - HM_n + H2II_n/2, 0.0)
    @test isapprox(out.yde, de_n; rtol=1e-12)

    # f32 path runs and tracks
    o32 = network_step(Float32(d), Float32(fh),
                       Float32.((yHI,yHII,yde,yH2I,yHM,yH2II,yDI,yDII,yHDI))...,
                       map(Float32, K), Float32(dt); deuterium=true,
                       yHeII_in=0.0f0, yHeIII_in=0.0f0)
    @test isapprox(Float64(o32.yHI),  out.yHI;  rtol=1e-4)
    @test isapprox(Float64(o32.yHII), out.yHII; rtol=1e-4)
end

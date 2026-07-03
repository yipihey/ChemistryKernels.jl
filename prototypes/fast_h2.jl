#!/usr/bin/env julia
# fast_h2.jl — prototype of a FAST ANALYTIC reduced (x_HII, x_H2, e) chemistry mode.
#
# The existing fast high-z path is Compton cooling + hydrogen recombination/ionization
# (analytic Peebles k2, C-weighted CMB photo-ionization, implicit Compton) — it tracks
# the thermal + ionization history z=1000→20 well.  This explores ADDING H2 formation +
# cooling to that path so the SAME cheap solver also follows the first cooling halos.
#
# Reduced state (per H nucleus): x_HII = n_HII/n_H, x_H2 = n_H2/n_H, and e [erg/g].
#   * HI from conservation: n_HI = n_H(1 − x_HII − 2 x_H2)
#   * n_e = n_HII (He neutral by z≲1000)
#   * H⁻ and H2⁺ collapsed to their algebraic quasi-equilibria (closed form) → the H2
#     formation rate is analytic in (n_HI, n_e, n_HII, T): the H⁻ channel k8·n(H⁻)·n_HI
#     (dominant in the IGM + first halos), the H2⁺ channel, and 3-body k22 (dense halos).
#   * H2 cooling = the existing Galli–Palla low-density↔LTE interpolation (cooling_edot).
#   * Energy: implicit Compton (stiff at high z) + all cooling.
# This is the reduced network MINUS the deuterium network and MINUS advecting H⁻/H2⁺ —
# i.e. 2 chemical ODEs + energy, vs the full coupled sweep.  We validate it reproduces
# the full `evolve_cell` network on the IGM history AND a minihalo collapse.
#
# Run:  julia --project=ChemistryKernels.jl/test ChemistryKernels.jl/prototypes/fast_h2.jl

using ChemistryKernels
const CK = ChemistryKernels
using Printf

const MH=1.6726e-24; const KB=1.380649e-16; const FH=0.76; const GAMMA=5/3
const H0=71.0e5/3.0857e24; const OMB=0.0456; const OMM=0.27; const OML=0.73; const HH=0.71
nHbar(z)=FH*OMB*1.8788e-29*HH^2*(1+z)^3/MH                 # mean H number density [cm^-3]
Hofz(z)=H0*sqrt(OMM*(1+z)^3+OML)
tcos(z)=(2/(3*H0*sqrt(OML)))*asinh(sqrt(OML/OMM)*(1+z)^(-1.5))   # matter+Λ cosmic time [s]
nHe(nH)=(1-FH)/(4*FH)*nH

# temperature from the reduced state (reuse the network's H2 γ-corrected relation)
Tof(nH,nHII,nH2,e) = begin
    nHI=max(nH-nHII-2nH2,1e-30)
    CK.gas_temperature(nH*MH/FH, e, nHI, nHII, nHe(nH), 0.0, 0.0, nHII, 0.0, nH2, 0.0)
end

# ── one reduced analytic step (physical CGS), z & nH fixed across the step ──
function fast_step(nHII, nH2, e, nH, z, dt; nH2II_prev=0.0)
    Trad = CK.comp2_cmb(z); Hz = Hofz(z)
    nHI = max(nH - nHII - 2nH2, 1e-30); ne = max(nHII, 1e-30)
    T = Tof(nH,nHII,nH2,e)
    # rate coefficients (matter T; CMB rates at Trad)
    k1=CK.k1(T); k2=CK.peebles_k2(T,nHI,Hz)
    kb1s=CK.beta1s_freq(Trad)*k2/(CK.recfast_alpha(T)*1e6)      # C-weighted CMB photo-ion of H
    k7=CK.k7(T); k8=CK.k8(T); k9=CK.k9(T); k10=CK.k10(T); k11=CK.k11(T); k12=CK.k12(T)
    k13=CK.k13(T); k14=CK.k14(T); k15=CK.k15(T); k16=CK.k16(T); k17=CK.k17(T)
    k18=CK.k18(T); k19=CK.k19(T); k22=CK.k22(T); k27=CK.k27_cmb(Trad); k28=CK.k28_cmb(Trad)
    k57=CK.k57(T); k58=CK.k58(T)
    # H⁻ and H2⁺ algebraic quasi-equilibria (network convention: yH2I=2·n(H2), yH2II=2·n(H2⁺))
    yH2II = 2nH2II_prev
    nHM  = CK.equilibrium_HM(nHI, nHII, ne, yH2II, k7,k8,k14,k15,k16,k17,k19,k27)
    yH2II= CK.equilibrium_H2II(nHI, nHII, 2nH2, ne, nHM, k9,k10,k11,k17,k18,k19,k28)
    nH2II= yH2II/2
    nHM  = CK.equilibrium_HM(nHI, nHII, ne, yH2II, k7,k8,k14,k15,k16,k17,k19,k27)   # 1 refine
    # H2 formation (H⁻ channel + H2⁺ channel + 3-body) & destruction — implicit
    form = k8*nHM*nHI + k10*nH2II*nHI + k19*nH2II*nHM + k22*nHI^3
    dest = k13*nHI + k11*nHII + k12*ne
    nH2n = (nH2 + form*dt)/(1 + dest*dt)
    # HII: recombination (Peebles k2) − collisional/CMB ionization — implicit in nHII
    # HII: recomb (Peebles k2) balanced by CMB/collisional ionization INCLUDING the
    # k57/k58 H-H,H-He collisional-ionization FLOOR (~1e-20) — with n_HI^2 this sets the
    # relic electron freeze-out that catalyzes halo H2 (matches Grackle + the full network).
    scH  = (k1*ne + kb1s)*nHI + k10*nH2II*nHI + k28*nH2II + k57*nHI^2 + k58*nHI*nHe(nH)
    acH  = k2*ne + k9*nHI + k11*nH2 + k16*nHM + k17*nHM
    nHIIn= (nHII + scH*dt)/(1 + acH*dt)
    # energy: cooling_edot (all channels incl H2/atomic/Compton) with implicit Compton split
    edot = CK.cooling_edot(nHI, nHII, nHe(nH), ne, nH2, 0.0, T, z; nH=nH)
    rho  = nH*MH/FH; Tc=Trad; c1=CK.comp1_cmb(z)
    edot_c = -c1*(T-Tc)*ne; edot_rest = edot - edot_c
    Kc = c1*ne*(T/e)/rho
    en = (Kc*dt > 1) ? (e + ((c1*ne*Tc+edot_rest)/rho)*dt)/(1+Kc*dt) : e + (edot/rho)*dt
    return nHIIn, max(nH2n,1e-30), max(en,1e-30), nH2II
end

# reference: the full reduced network (evolve_cell), same z & nH fixed across the step
function ref_step(nHII, nH2, e, nH, z, dt)
    rho = nH*MH/FH
    en, hii, h2, _, _ = CK.evolve_cell(rho, e, nHII*MH, 2nH2*MH, 0.0, dt, z;
                                       fh=FH, deuterium=false)
    return hii/MH, h2/(2MH), en
end

# ── IGM history z=1000→20 ──
function run_igm(; N=80)
    zs = exp.(range(log(1000.0), log(20.0), length=N))
    # start Compton-locked, mostly ionized residual after recombination
    T0=CK.comp2_cmb(zs[1]); e0=KB*T0/((GAMMA-1)*1.22*MH)
    nH=nHbar(zs[1]); xHII=0.05; xH2=1e-8
    sF=(nHII=xHII*nH, nH2=xH2*nH, e=e0); sR=sF; nh2ii=0.0
    println("\nIGM history (reduced 'fast+H2'  vs  full network):")
    @printf("%6s | %10s %10s | %9s %9s | %10s %10s\n","z","x_e fast","x_e full","T fast","T full","fH2 fast","fH2 full")
    for i in 2:N
        z=zs[i]; nHn=nHbar(z); dt=tcos(z)-tcos(zs[i-1]); ad=(nHn/nHbar(zs[i-1]))^(GAMMA-1)
        # adiabatic expansion between steps (both models), then the chemistry step
        eF=sF.e*ad; eR=sR.e*ad
        nHIIf,nH2f,eFn,nh2ii = fast_step(sF.nHII*nHn/nHbar(zs[i-1]), sF.nH2*nHn/nHbar(zs[i-1]), eF, nHn, z, dt; nH2II_prev=nh2ii)
        nHIIr,nH2r,eRn = ref_step(sR.nHII*nHn/nHbar(zs[i-1]), sR.nH2*nHn/nHbar(zs[i-1]), eR, nHn, z, dt)
        sF=(nHII=nHIIf,nH2=nH2f,e=eFn); sR=(nHII=nHIIr,nH2=nH2r,e=eRn)
        if i in (2, N÷4, N÷2, 3N÷4, N-3, N)
            @printf("%6.0f | %10.3e %10.3e | %9.1f %9.1f | %10.3e %10.3e\n",
                z, sF.nHII/nHn, sR.nHII/nHn, Tof(nHn,sF.nHII,sF.nH2,sF.e), Tof(nHn,sR.nHII,sR.nH2,sR.e),
                sF.nH2/nHn, sR.nH2/nHn)
        end
    end
end

# ── minihalo collapse at fixed z: n_H rises n̄→1e4, adiabatic heating + chem + cooling ──
function run_halo(; zc=25.0, nmax=1e4, N=600)
    nH0=nHbar(zc); ns=exp.(range(log(nH0), log(nmax), length=N))
    # seed with the IGM freeze-out state at zc (roughly): x_e~2e-4, fH2~2e-6, T~adiabatic
    T0=max(CK.comp2_cmb(zc)*0.5, 60.0); e0=KB*T0/((GAMMA-1)*1.22*MH)
    xHII=2e-4; xH2=2e-6
    sF=(nHII=xHII*ns[1], nH2=xH2*ns[1], e=e0); sR=sF; nh2ii=0.0
    tff(n)=1/sqrt(6.67e-8*n*MH/FH)                 # free-fall time at density n
    println("\nMinihalo collapse at z=$zc (reduced vs full network):")
    @printf("%10s | %9s %9s | %10s %10s | %9s %9s\n","n_H[cm^-3]","T fast","T full","fH2 fast","fH2 full","xe fast","xe full")
    for i in 2:N
        nH=ns[i]; nprev=ns[i-1]; dt=0.3*tff(nH); ad=(nH/nprev)^(GAMMA-1)
        # compress: number densities scale ∝ nH, energy heats adiabatically
        sc=nH/nprev
        eF=sF.e*ad; eR=sR.e*ad
        nHIIf,nH2f,eFn,nh2ii=fast_step(sF.nHII*sc, sF.nH2*sc, eF, nH, zc, dt; nH2II_prev=nh2ii)
        nHIIr,nH2r,eRn=ref_step(sR.nHII*sc, sR.nH2*sc, eR, nH, zc, dt)
        sF=(nHII=nHIIf,nH2=nH2f,e=eFn); sR=(nHII=nHIIr,nH2=nH2r,e=eRn)
        if i in (2, N÷4, N÷2, 3N÷4, N)
            @printf("%10.2e | %9.1f %9.1f | %10.3e %10.3e | %9.2e %9.2e\n",
                nH, Tof(nH,sF.nHII,sF.nH2,sF.e), Tof(nH,sR.nHII,sR.nH2,sR.e),
                sF.nH2/nH, sR.nH2/nH, sF.nHII/nH, sR.nHII/nH)
        end
    end
end

run_igm()
run_halo()

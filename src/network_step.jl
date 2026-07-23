# network_step.jl — ONE linearly-implicit backward-Euler sweep of the v2026
# reduced primordial+D network: one backward-Euler sweep of the network, with
#   H⁻, H₂⁺, D⁺ as algebraic-equilibrium intermediaries; helium in collisional-
#   radiative ionisation equilibrium (or, optionally, advected He⁺ — see the He
#   block below); nₑ from charge conservation; optional dust physics (H₂ on
#   grains, LW photodissociation with shielding, grain-assisted HII recombination)
#   enabled by passing a `dust_rates` NamedTuple.
#
# Each provisional Xⁿ⁺¹ = (s·dt + Xⁿ)/(1 + a·dt) with s = formation, a =
# destruction frequency.  Species use the network's mass-equivalent convention
# (same as equilibrium.jl): yHI=n_HI, yHII=n_HII, yde=n_e, yHM=n_HM, yHeI=n_HeI, and
# yH2I=2·n(H2), yH2II=2·n(H2⁺), yHDI=3·n(HD) — so every literal /2,/3,2× is
# consistent with that convention.  Pure & allocation-free (AD-friendly); the
# per-cell KA launcher lives in solve.jl.
#
# Gauss-Seidel ordering: HIp/HIIp/dep/H2Ip from the OLD state; HMp from OLD;
# H2IIp from the NEW provisionals (uses dep for its e⁻ destruction); the
# deuterium block from OLD; and the charge-conservation nₑ at the end uses the
# NEW HII but the OLD HM/H2II.

export network_step

@inline function _molecular_intermediates(yHI, yHII, yde, yH2I, K)
    R = typeof(yHI)
    # H- and H2+ are algebraic, so their solution must not depend on a retained
    # value from the previous subcycle or caller. Two fixed Gauss-Seidel passes
    # resolve their weak mutual k19 coupling while remaining allocation-free.
    HM0 = equilibrium_HM(yHI, yHII, yde, zero(R), K.k7, K.k8, K.k14, K.k15,
                         K.k16, K.k17, K.k19, K.k27)
    H2II0 = equilibrium_H2II(yHI, yHII, yH2I, yde, HM0, K.k9, K.k10, K.k11,
                             K.k17, K.k18, K.k19, K.k28)
    HM = equilibrium_HM(yHI, yHII, yde, H2II0, K.k7, K.k8, K.k14, K.k15,
                        K.k16, K.k17, K.k19, K.k27)
    H2II = equilibrium_H2II(yHI, yHII, yH2I, yde, HM, K.k9, K.k10, K.k11,
                            K.k17, K.k18, K.k19, K.k28)
    return HM, H2II
end

"""
    _threebody_h2_equilibrium(hydrogen_budget, k22, k13)

H₂ nuclei abundance at equilibrium between three-body formation
`3H → H₂ + H` and H-impact dissociation `H₂ + H → 3H`.  Both the input and
return value use the network convention in which `yH2I = 2n(H₂)`.
"""
@inline function _threebody_h2_equilibrium(hydrogen_budget, k22, k13)
    R = typeof(hydrogen_budget)
    z = zero(R)
    C = max(hydrogen_budget, z)
    k22 <= z && return z
    k13 <= z && return C

    # At equilibrium 2k22*nHI^3 = k13*nHI*yH2 and nHI+yH2=C.  This form of
    # the positive quadratic root avoids subtractive cancellation when the
    # equilibrium molecular fraction is small.
    nHI = R(2)*C / (one(R) + sqrt(one(R) + R(8)*k22*C/k13))
    return clamp(C - nHI, z, C)
end

@inline function _use_threebody_h2_equilibrium(yHI, yH2I, k22, k13,
                                                h2_source, h2_loss, dt)
    R = typeof(yHI)
    z = zero(R)
    pair_budget = max(yHI + yH2I, z)
    H2I_eq = _threebody_h2_equilibrium(pair_budget, k22, k13)
    HI_eq = pair_budget - H2I_eq
    pair_source = R(2)*k22*yHI*yHI^2
    pair_sink = k13*yHI*yH2I
    pair_source_eq = R(2)*k22*HI_eq*HI_eq^2
    pair_sink_eq = k13*HI_eq*H2I_eq
    pair_activity = max(pair_source + pair_sink,
                        pair_source_eq + pair_sink_eq)
    other_activity = max(h2_source - pair_source, z) +
                     max(h2_loss - k13*yHI, z)*yH2I
    pair_frequency = max(R(2)*k22*yHI^2 + k13*yHI,
                         R(2)*k22*HI_eq^2 + k13*HI_eq)
    return pair_activity > z &&
           other_activity <= R(0.01)*pair_activity &&
           pair_frequency*dt >= R(10)
end

"""
    network_step(d, fh, yHI, yHII, yde, yH2I, yHM, yH2II, yDI, yDII, yHDI, K, dt;
                 deuterium = false, dust_rates = nothing,
                 stiff_h2_pair = Val(false))

One backward-Euler sweep. `d` = total density (same units as the species), `fh` =
hydrogen mass fraction, `K` = NamedTuple of rate coefficients `k1..k58` plus the
CMB photo-rates `k27`,`k28` (all in the network's per-density-unit convention),
`dt` = the subcycle step. Returns a NamedTuple of the updated species. Pure.

`dust_rates`, when not `nothing`, is a NamedTuple `(; k_h2d, k_grr, k_lw)`:
  - `k_h2d` [cm³/s] — H₂ formation on dust (Cazaux & Tielens 2004)
  - `k_grr` [cm³/s] — grain-assisted HII recombination (Weingartner & Draine 2001)
  - `k_lw`  [s⁻¹]  — effective LW H₂ photodissociation (Draine & Bertoldi 1996)

With `stiff_h2_pair=Val(true)`, a substep that resolves many relaxation times
and is dominated by three-body formation plus H-impact dissociation is placed
directly on that pair's conservative equilibrium manifold.  Higher-level
subcyclers enable this guard; the default remains the pinned literal one-sweep
transcription for standalone callers.
"""
@inline function network_step(d, fh, yHI, yHII, yde, yH2I, yHM, yH2II,
                              yDI, yDII, yHDI, K, dt; deuterium::Bool = false,
                              yHeII_in = nothing, yHeIII_in = nothing,
                              GamHI = 0.0, GamHeI = 0.0, GamHeII = 0.0,
                              dust_rates = nothing,
                              intermediates_current::Val{IC} = Val(false),
                              stiff_h2_pair::Val{SP} = Val(false)) where {IC,SP}
    R    = typeof(yHI)
    z    = zero(R)
    two  = R(2); half = R(0.5); three = R(3); four = R(4)

    # Helium ionisation.  Two modes (both in the mass-equivalent ×4 convention,
    # yHeX = 4·n(HeX)):
    #   • Equilibrium (default, yHeII_in===nothing): collisional-radiative ionisation
    #     equilibrium — every up/down channel balanced.  For each stage the ratio of
    #     successive ions is  (collisional ion·nₑ + photo) / (recombination·nₑ):
    #         n_HeII /n_HeI  = K.she1/nₑ + k3/k4 + Γ_HeI /(k4·nₑ)
    #         n_HeIII/n_HeII = K.she2/nₑ + k5/k6 + Γ_HeII/(k6·nₑ)
    #     The radiation/CMB term is the Saha factor (K.she*, detailed balance at
    #     T_rad — exact at z≳3000 & z≲1700, ~3% low only in the He⁺→He⁰ freeze-out);
    #     k3/k5 = collisional ionisation, k4/k6 = recombination (Abel/Hui-Gnedin, at
    #     T_matter); Γ = optional external photoionisation rate [s⁻¹] (e.g. a UV
    #     background, 0 by default).  Cold CMB + cold matter ⇒ He neutral ⇒ no cost
    #     at late times; hot gas ⇒ collisional-ionisation equilibrium.  Solved
    #     semi-implicitly off the OLD nₑ.
    #   • Evolved (yHeII_in given): the caller (evolve_cell_mixing) has integrated
    #     He⁺ with the full rate equation (incl. the He I radiative-transfer
    #     freeze-out); we just consume its yHeII/yHeIII for the electron balance.
    nHe4 = (one(R) - R(fh)) * d                  # total He in ×4 convention
    if yHeII_in === nothing
        _, nHeII, nHeIII = helium_equilibrium(K.she1, K.she2, K.k3, K.k4, K.k5, K.k6,
                                              max(yde, z), nHe4 / four;
                                              GamHeI = GamHeI, GamHeII = GamHeII)
        yHeII  = four * nHeII                     # back to ×4 convention
        yHeIII = four * nHeIII
    else
        yHeII   = R(yHeII_in)
        yHeIII  = R(yHeIII_in)
    end
    yHeI = max(nHe4 - yHeII - yHeIII, z)         # neutral He (×4)

    k1=K.k1; k2=K.k2; k3=K.k3; k4=K.k4; k5=K.k5; k6=K.k6; k7=K.k7; k8=K.k8
    k9=K.k9; k10=K.k10; k11=K.k11; k12=K.k12; k13=K.k13; k14=K.k14; k15=K.k15
    k16=K.k16; k17=K.k17; k18=K.k18; k19=K.k19; k22=K.k22; k57=K.k57; k58=K.k58
    k27=K.k27; k28=K.k28; k_beta1s=K.k_beta1s
    # `network_step` is also a low-level public primitive used with hand-built
    # pinned rate bundles. Missing HeH⁺ keys mean an explicitly disabled
    # channel; every production assembler supplies the non-zero defaults.
    kHeH_ra=get(K, :kHeH_ra, z)
    kHeH_H=get(K, :kHeH_H, z)
    kHeH_e=get(K, :kHeH_e, z)
    gamma_HeH=get(K, :gamma_HeH, z)

    # ── (C) HI / HII / e⁻ / H2 with molecular terms ──────────────────────────
    # H⁻ and H₂⁺ are fast algebraic-equilibrium intermediaries. We evaluate them
    # from the OLD state FIRST and substitute the equilibrium H₂⁺ (H2IIeq) into the
    # HI/HII/e⁻/H2 source terms — INCLUDING the CMB photodissociation return k28
    # (H₂⁺+γ → HI + HII) in the HI/HII equations.
    #
    # DELIBERATE EXTENSION beyond the original network (Abel, Anninos, Zhang &
    # Norman 1997): it drops the H₂⁺ photodissociation (k28) return to HI/HII and
    # uses the lagged H₂⁺ density because H₂⁺ is trace at z<100, its galaxy-
    # formation regime.  During recombination (z≈1000-1200), however, the radiative
    # association k9 (HI+HII→H₂⁺) reaches ~1.5% of the net recombination rate; the
    # CMB then photodissociates that H₂⁺ straight back (k28≈330 s⁻¹), so the cycle
    # is very nearly null for HII.  Without crediting the k28/k10 return, the k9
    # term leaks HII and biases x_e ~1-1.5% low at z≈1000-1100.  We add it for the
    # recombination epoch.  Closing the cycle (the only net HII sink via H₂⁺ is the
    # dissociative k18·de branch) recovers the full network to <0.25% of HyRec
    # across z=700-1100.  H₂⁺ being trace at low z, the change is negligible in the
    # original galaxy-formation regime.
    if IC
        HMp, H2IIeq = yHM, yH2II
    else
        # Preserve the pinned standalone one-sweep transcription. Higher-level
        # subcyclers pass their state-independent algebraic solution with IC=true.
        HMp = equilibrium_HM(yHI, yHII, yde, yH2II, k7, k8, k14, k15,
                             k16, k17, k19, k27)
        H2IIeq = equilibrium_H2II(yHI, yHII, yH2I, yde, HMp, k9, k10, k11,
                                  k17, k18, k19, k28)
    end
    nH2II  = H2IIeq / two          # n(H₂⁺); H2IIeq carries the 2× mass-equiv convention
    # Hirata & Padmanabhan (2006) HeH⁺ route. The newly produced H₂⁺ occupies
    # low vibrational levels and overwhelmingly charge-transfers to H₂; treat
    # the two-step sequence as a direct H₂ source. HeH⁺ itself remains a trace
    # algebraic intermediate and therefore adds no advected/storage field.
    nHeH = equilibrium_HeH(yHeI/four, yHII, yde, yHI,
                           kHeH_ra, kHeH_H, kHeH_e, gamma_HeH)

    # 1,2) Coupled HI/HII update. At recombination redshifts the β₁s exchange is
    # extremely stiff. Updating HI and HII as two lagged fixed points makes the neutral
    # fraction alternate between successive substeps. Instead use H conservation to
    # substitute HI = H_atomic - HII in every one-body ionisation source and solve the
    # resulting backward-Euler equation once. The H-H ionisation term freezes one HI
    # factor in qion, consistent with the other semi-implicit nonlinear terms.
    h_in_hd = deuterium ? yHDI/three : zero(R)
    Hatomic = max(R(fh)*d - yH2I - HMp - H2IIeq - h_in_hd, z)
    qion = k1*yde + k57*yHI + k58*yHeI/four + k_beta1s + R(GamHI)
    sc_other = k10*H2IIeq*yHI/two + k28*nH2II
    ac_grr = dust_rates !== nothing ? dust_rates.k_grr * yde : zero(R)
    ac = k2*yde + k9*yHI + k11*yH2I/two + k16*yHM + k17*yHM + ac_grr
    HIIp = (yHII + (qion*Hatomic + sc_other)*dt) /
           (one(R) + (qion + ac)*dt)
    HIIp = clamp(HIIp, z, Hatomic)
    HIp = Hatomic - HIIp

    # 3) e⁻ provisional — used ONLY for downstream consistency
    sc = k8*yHM*yHI + k15*yHM*yHI + k17*yHM*yHII + k57*yHI*yHI + k58*yHI*yHeI/four +
         k_beta1s*yHI
    ac = -(k1*yHI - k2*yHII + k3*yHeI/four - k6*yHeIII/four +
           k5*yHeII/four - k4*yHeII/four + k14*yHM - k7*yHI - k18*H2IIeq/two)
    dep = (sc*dt + yde) / (one(R) + ac*dt)

    # 7) H2  (formation via H₂⁺, H⁻ channels and dust surface reactions;
    #     destruction via LW photodissociation when dust_rates supplied)
    sc_dust = dust_rates !== nothing ? two * dust_rates.k_h2d * yHI^2 : zero(R)
    sc = two*(k8*yHM*yHI + k10*H2IIeq*yHI/two + k19*H2IIeq*yHM/two +
              kHeH_H*nHeH*yHI + k22*yHI*yHI^2) +
         sc_dust
    ac_lw = dust_rates !== nothing ? dust_rates.k_lw : zero(R)
    ac = k13*yHI + k11*yHII + k12*yde + ac_lw
    H2Ip = (sc*dt + yH2I) / (one(R) + ac*dt)
    if SP && _use_threebody_h2_equilibrium(yHI, yH2I, k22, k13, sc, ac, dt)
        # A frozen-yHI backward-Euler sweep can overshoot the physical fixed
        # point when the pair is extremely stiff, then become trapped at
        # yHI=0 where H-impact dissociation also vanishes. In the asymptotic
        # pair-dominated regime solve the conservative balance directly.
        H2Ip = _threebody_h2_equilibrium(yHI + yH2I, k22, k13)
    end

    # 8,9) store the consistent old-state equilibrium H₂⁺ (H⁻ already in HMp above)
    H2IIp = H2IIeq

    # ── (D) deuterium (OLD state) ────────────────────────────────────────────
    if deuterium
        k50=K.k50; k51=K.k51; k52=K.k52; k53=K.k53; k54=K.k54; k55=K.k55; k56=K.k56
        # 1) DI
        sc = k2*yDII*yde + k51*yDII*yHI + two*k55*yHDI*yHI/three
        ac = k1*yde + k50*yHII + k54*yH2I/two + k56*yHM
        DIp = (sc*dt + yDI) / (one(R) + ac*dt)
        # 2) DII equilibrium (OLD)
        DIIp = equilibrium_DII(yDI, yde, yHI, yHII, yH2I, yHDI,
                               k1, k2, k50, k51, k52, k53)
        # 3) HDI (OLD DI,DII)
        sc = three*(k52*yDII*yH2I/two/two + k54*yDI*yH2I/two/two +
                    two*k56*yDI*yHM/two)
        ac = k53*yHII + k55*yHI
        HDIp = (sc*dt + yHDI) / (one(R) + ac*dt)
    else
        DIp = yDI; DIIp = yDII; HDIp = yHDI
    end

    # ── (E) field assignment + charge-conservation nₑ ────────────────────────
    DI_n   = max(DIp,  z)
    DII_n  = max(DIIp, z)
    HDI_n  = max(HDIp, z)
    # Exact H-nuclei conservation, including the one H nucleus in each HD molecule.
    # H₂ is the only non-trace molecular reservoir; clip trace intermediaries into the
    # remaining budget and derive the two atomic stages from what remains.
    Hbudget = max(R(fh)*d - (deuterium ? HDI_n/three : z), z)
    H2I_n  = clamp(H2Ip, z, Hbudget)
    Hrem   = Hbudget - H2I_n
    HM_n   = clamp(HMp, z, Hrem)
    Hrem  -= HM_n
    H2II_n = clamp(H2IIp, z, Hrem)
    Hatomic_n = Hrem - H2II_n
    HII_n = clamp(HIIp, z, Hatomic_n)
    HI_n  = Hatomic_n - HII_n
    # nₑ from charge conservation, using the new conservative H state and the
    # algebraic intermediaries returned by this same step.
    de_n = max(HII_n + yHeII/four + yHeIII/two - HM_n + H2II_n/two, z)

    return (; yHI = HI_n, yHII = HII_n, yde = de_n, yH2I = H2I_n,
            yHM = HM_n, yH2II = H2II_n, yDI = DI_n, yDII = DII_n, yHDI = HDI_n)
end

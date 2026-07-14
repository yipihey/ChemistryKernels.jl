# fast.jl — the FAST analytic (H + H₂) chemistry mode.
#
# A reduced sibling of `evolve_cell`: it advects only x_HII and x_H2 (+ energy), with
# everything else closed-form — HI by conservation, n_e = n_HII (helium neutral), and
# H⁻/H₂⁺ as algebraic quasi-equilibria.  It drops the deuterium network AND the helium
# ionization sub-network of the full solver, and folds the species update inline (no
# per-substep `network_step` call), so it is meaningfully cheaper per substep while
# reproducing the full network across z=1000→20 AND the first cooling halos.
#
# The relic electron freeze-out that catalyzes halo H₂ is captured by the k57/k58
# H-H / H-He collisional-ionization FLOOR (~1e-20): with n_HI² at halo densities it is
# the persistent electron source that balances recombination at x_e ≈ √(k57 n_HI²/k2)
# — validated against the full network and the Grackle reduced oracle.
#
# Same contract as `evolve_cell`: pure, allocation-free, precision-generic (R=typeof(e)),
# GPU-safe.  Returns (e, HII_m, H2I_m, ttot); no HDI (primordial H+H₂ only).

export evolve_cell_fast, evolve_cell_analytic

# Riccati closed form for  dy/dt = scH − k2·y²  (recombination vs an ionization source):
# relaxes y toward y_eq = √(scH/k2) analytically over the whole step — no subcycle.
@inline function _riccati(y0::R, k2::R, scH::R, dt::R) where {R}
    # scH is the true ionization source (k57·n_HI² + …), legitimately ~1e-26 at IGM
    # densities — do NOT floor it (a floor spuriously raises y_eq and re-ionizes the gas).
    scH <= zero(R) && return y0 / (one(R) + k2*y0*dt)      # pure recombination (no source)
    yeq = sqrt(scH / k2)
    th  = tanh(min(k2 * yeq * dt, R(30)))
    return yeq * (y0 + yeq*th) / (yeq + y0*th)
end

# Linear-source Riccati:  dy/dt = q + p·(C − y) − k2·y²  (exact over the whole step).
# The photoionisation / collisional-e ionisation source is LINEAR in the neutral
# fraction (kb1s·n_HI + k1·n_e·n_HI), so with H-nuclei conservation n_HI = C − y it is
# a −p·y depletion, NOT a frozen constant.  Freezing it (the plain `_riccati` above,
# which lumps kb1s·n_HI into scH at the start-of-substep n_HI and then drives y all the
# way to √(scH/k2)) OVERSHOOTS at high z where the source is strong and the step is
# large (cooling-limited): y bounces, i.e. the spurious x_HII step-to-step oscillation
# at z≳1600 where H is Saha-pinned to the CMB.  Treating the linear term implicitly
# removes the overshoot — the equilibrium is the true radiative-balance (Saha/Peebles)
# ionisation, reached monotonically.  `q` keeps the genuinely-nonlinear collisional
# floor (k57·n_HI² + k58·…) frozen (∝n_HI², negligible at high z, ~constant at collapse
# where n_HI≈C).  Reduces to `_riccati(scH=q)` when p→0 (the cold collapse regime).
@inline function _riccati2(y0::R, k2::R, p::R, q::R, C::R, dt::R) where {R}
    d = q + p*C                                            # dy/dt = d − p·y − k2·y²
    (p <= zero(R) && q <= zero(R)) && return y0 / (one(R) + k2*y0*dt)   # pure recomb
    Δ   = sqrt(p*p + R(4)*k2*d)
    yeq = (Δ - p) / (R(2)*k2)                              # positive equilibrium root
    ym  = -(Δ + p) / (R(2)*k2)                             # negative root (< 0)
    E   = exp(-min(Δ*dt, R(60)))                           # e^{−Δ·dt}, Δ = k2·(yeq−ym)
    return (yeq*(y0 - ym) - ym*(y0 - yeq)*E) / ((y0 - ym) - (y0 - yeq)*E)
end

# H₂ formation/dissociation chemical heating (erg cm⁻³ s⁻¹; >0 heats).  Binding
# energy is released when H₂ forms (→ gas) and absorbed when it dissociates.
# Per-channel energies (Palla, Salpeter & Stahler 1983 / Omukai 2000): 3-body
# 4.48 eV, H⁻ 3.53 eV, H₂⁺ 1.83 eV; collisional dissociation absorbs 4.48 eV.
# Formation heating is SUPPRESSED below the H₂ critical density (the vibrationally-
# excited nascent H₂ radiates the energy before collisions thermalise it — the same
# n/(n+n_cr) that saturates the H₂ cooling to LTE); dissociation cooling is always
# thermalised (kinetic).  This 3-body heating sets the ~1000–2000 K collapse plateau.
const _ERG_PER_EV = 1.602176634e-12
@inline function _h2_chem_heat(nHI::R, nHII, nde, nH2, nHM, nH2II, nH,
                               k8, k10, k11, k12, k13, k22) where {R}
    fform = k8  * nHM   * nHI                       # H⁻ channel   (3.53 eV)
    gform = k10 * nH2II * nHI                       # H₂⁺ channel  (1.83 eV)
    tform = k22 * nHI * nHI * nHI                   # 3-body       (4.48 eV)
    diss  = (k13*nHI + k11*nHII + k12*nde) * nH2    # dissociation (4.48 eV absorbed)
    fcr   = nH / (nH + R(1.0e8))                    # critical-density heating efficiency
    return R(_ERG_PER_EV) *
           (fcr*(R(3.53)*fform + R(1.83)*gform + R(4.48)*tform) - R(4.48)*diss)
end

"""
    evolve_cell_fast(rho, e, HII_m, H2I_m, dt, z; hubble, Om, OL, fh,
                     hubble_expansion, adot_over_a, cool_tables, itcap, dtfrac)
        -> (e, HII_m, H2I_m, ttot)

Sub-cycle one cell over `dt` [s] with the reduced H+H₂ network (see file header).
`rho` [g/cm³], `e` [erg/g], `HII_m`/`H2I_m` = species mass densities ρ·x (network
convention, `H2I_m` = 2·n(H₂)·m_H).  Pure; ~2× cheaper per substep than `evolve_cell`.
"""
@inline function evolve_cell_fast(rho, e, HII_m, H2I_m, dt, z;
                                  hubble = 71.0, Om = 0.27, OL = 0.73,
                                  fh = FH_DEFAULT, hubble_expansion::Bool = false,
                                  adot_over_a = NaN, rate_tables = nothing,
                                  cool_tables = nothing,
                                  itcap::Int = _SUB_ITMAX, dtfrac::Real = 0.1)
    R    = typeof(e)
    mh   = R(MH); tiny = R(_SUB_TINY)
    # domain guard: f16 hydro can hand rho <= 0 / e <= 0 / X*rho < 0 at extreme
    # shock and void frontiers; unguarded they reach device sqrt/log/pow
    # (DomainError aborts the kernel).  Clamp to physical-positive; the T floor
    # in gas_temperature then makes the cell a benign ~1 K near-vacuum.
    rho  = ifelse(rho > R(1.0e-35), rho, R(1.0e-35))   # ifelse: NaN>x is false
    e    = ifelse(e > tiny, e, tiny)                    # (max() passes NaN through)
    HII_m = ifelse(HII_m > zero(R), HII_m, zero(R))
    H2I_m = ifelse(H2I_m > zero(R), H2I_m, zero(R))
    d    = rho / mh
    z0   = R(z)
    Hz0  = hubble_z_of(z0; hubble = hubble, Om = Om, OL = OL)
    Hz_ad = isnan(adot_over_a) ? Hz0 : R(adot_over_a)
    evolve_z = !isnan(adot_over_a)
    fhd  = R(fh) * d                        # total H-nuclei number density n_H

    yHeI = (one(R) - R(fh)) * d              # neutral helium (×4 convention: yHeI = 4·n_He)
    yHII = HII_m / mh
    yH2I = H2I_m / mh                        # = 2·n(H₂)
    yHI  = max(fhd - yHII - yH2I, tiny)
    yde  = yHII                              # n_e = n_HII (helium neutral)

    Tc0  = comp2_cmb(z0); c10 = comp1_cmb(z0)
    f    = R(dtfrac)
    ttot = zero(R); iter = 0
    @inbounds while ttot < dt && iter < itcap
        iter += 1
        rem = dt - ttot
        if evolve_z
            zt = (one(R) + z0) * exp(-Hz_ad * ttot) - one(R)
            Tc = comp2_cmb(zt); c1 = comp1_cmb(zt); Hz = hubble_z_of(zt; hubble=hubble, Om=Om, OL=OL)
        else
            zt = z0; Tc = Tc0; c1 = c10; Hz = Hz0
        end

        T = gas_temperature(rho, e, yHI, yHII, yHeI/R(4), tiny, tiny, yde,
                            tiny, yH2I/R(2), tiny; gamma = GAMMA_DEFAULT)

        # ── reduced rate set (~17): H recomb/ionization + BOTH H₂-formation channels in
        # closed form.  The sub-dominant H⁻ destruction terms (k14,k16-k19) are dropped
        # (they matter only when x_HII~1 at recombination, where f_H2≪1e-6 anyway).
        if rate_tables === nothing
            k2   = peebles_k2(T, yHI, Hz)
            kb1s = beta1s_freq(Tc) * k2 / (recfast_alpha(T) * R(1.0e6))
            k1v=k1(T); k7v=k7(T); k8v=k8(T); k9v=k9(T); k10v=k10(T); k11v=k11(T); k12v=k12(T)
            k13v=k13(T); k15v=k15(T); k22v=k22(T); k57v=k57(T); k58v=k58(T)
            k27v=k27_cmb(Tc); k28v=k28_cmb(Tc)
        else
            K = table_rates(rate_tables, T, yHI, Hz, cmb_rates(Tc))
            k2=K.k2; kb1s=K.k_beta1s; k1v=K.k1; k7v=K.k7; k8v=K.k8; k9v=K.k9; k10v=K.k10
            k11v=K.k11; k12v=K.k12; k13v=K.k13; k15v=K.k15; k22v=K.k22; k57v=K.k57; k58v=K.k58
            k27v=K.k27; k28v=K.k28
        end

        # cooling (atomic + H₂ + Compton; no HD/metals in this mode)
        edot = cooling_edot(yHI, yHII, yHeI/R(4), yde, yH2I/R(2), zero(R), T, zt;
                            nH = fhd, cool_tables = cool_tables)
        if T <= R(1.01)*R(MIN_TEMPERATURE) && edot < zero(R); edot = zero(R); end
        hubble_expansion && (edot -= R(2) * Hz_ad * e * rho)

        # ── HII: recomb (k2) vs the k57/k58 collisional-ionization floor + CMB photo-ion ──
        scH = k1v*yHI*yde + k57v*yHI*yHI + k58v*yHI*yHeI/R(4) + kb1s*yHI
        acH = k2*yde
        # ── H₂: H⁻ channel + H₂⁺ channel (both closed-form) + 3-body, vs dissociation ──
        #   n(H⁻)  = k7·n_HI·n_e / [(k8+k15)·n_HI + k27]         (→ H₂ via k8)
        #   n(H₂⁺) = k9·n_HI·n_HII / [k10·n_HI + k28]            (→ H₂ via k10; k28 kills it at high z)
        nHM   = k7v*yHI*yde  / ((k8v + k15v)*yHI + k27v + tiny)
        nH2II = k9v*yHI*yHII / (k10v*yHI + k28v + tiny)
        scH2  = R(2)*(k8v*nHM*yHI + k10v*nH2II*yHI + k22v*yHI*yHI*yHI)
        acH2  = k13v*yHI + k11v*yHII + k12v*yde

        # substep size: 10%-change on n_e and e ONLY (as the full network does) — H₂ is
        # updated implicitly (backward-Euler, unconditionally stable), so it needs NO step
        # limit; limiting on H₂ would force tiny substeps while it builds and kill the speed.
        dHIIdt = scH - acH*yHII
        dtit = min(_step_f(yde, dHIIdt, f), rem, R(0.5)*dt)
        edot_c = -c1 * (T - Tc) * yde; edot_rest = edot - edot_c
        Kc = c1 * yde * (T / e) / rho; stiff = Kc * rem > one(R)
        de_spec = (stiff ? edot_rest : edot) / rho
        dtit = min(dtit, _step_f(e, de_spec, f))

        # energy update (implicit Compton when stiff)
        if stiff
            B = (c1*yde*Tc + edot_rest) / rho
            e = (e + B*dtit) / (one(R) + Kc*dtit)
        else
            e = e + (edot/rho)*dtit
        end
        e = max(e, tiny)

        # species update (backward-Euler, inline) + conservation
        yHII = (yHII + scH*dtit) / (one(R) + acH*dtit)
        yH2I = (yH2I + scH2*dtit) / (one(R) + acH2*dtit)
        # keep physical: 0 ≤ yH2I, yHII with yHII + yH2I ≤ n_H
        yH2I = yH2I < tiny ? tiny : (yH2I > fhd ? fhd : yH2I)
        yHII = yHII < tiny ? tiny : (yHII > fhd - yH2I ? fhd - yH2I : yHII)
        yHI  = max(fhd - yHII - yH2I, tiny)
        yde  = yHII

        ttot += dtit
    end
    return e, yHII*mh, yH2I*mh, ttot
end

"""
    evolve_cell_analytic(rho, e, HII_m, H2I_m, dt, z; hubble, Om, OL, fh,
                         hubble_expansion, adot_over_a, cool_tables, itcap, dtfrac,
                         xHeII_in = NaN)
        -> (e, HII_m, H2I_m, ttot, xHeII)

`xHeII_in` (optional, n_HeII/n_H): when supplied the caller CARRIES He⁺ across steps
and it is EVOLVED — 3-level Saha while He recombination is fast (z>3000), the HyRec-2
He I rate (`helium_HeI_rate_AB`, semi-forbidden 2³P + 2¹P-escape) for the slow
z≈2000-2500 freeze-out — and returned as the 5th value.  Default `NaN` ⇒ He neutral
(collapse) or, for hot/shock gas (Tc>5000K), the stateless Saha QSSA.

Fully closed-form variant of [`evolve_cell_fast`](@ref) — **no stiff/BDF sub-cycling
of the chemistry**.  Each step advances x_HII by the exact Riccati solution
(recombination vs the k57/k58 ionization floor), x_H2 by the exact backward-Euler
balance of formation (H⁻ + H₂⁺ + 3-body) against collisional dissociation
(k13·n_HI + k11·n_HII + k12·n_e — linear in n_H2, so still one closed-form division),
and the energy by the analytic Compton exponential + explicit non-Compton cooling.
The step is limited ONLY by the non-Compton cooling time (the Compton stiffness is
handled analytically), so Compton-locked gas takes a single step.  Same accuracy as
`evolve_cell_fast` at a fraction of the substeps; **this is the default fast path**
(`evolve_cell_fast` is the accurate subcycled fallback).  On an RTX A6000 (128³, f32)
it runs at ~4 Gcell/s — ~4× the full `evolve_cell` network.

`rate_tables` (an optional `RateTables` from `build_rate_tables`) swaps the per-iteration
analytic rate fits for a log–log table lookup; it helps the full network but is a wash-to-
slower here (this mode is dominated by the cooling/temperature transcendentals and its low
iteration count, not the rate fits), so the fits remain the default.
"""
@inline function evolve_cell_analytic(rho, e, HII_m, H2I_m, dt, z;
                                      hubble = 71.0, Om = 0.27, OL = 0.73,
                                      fh = FH_DEFAULT, hubble_expansion::Bool = false,
                                      adot_over_a = NaN, rate_tables = nothing,
                                      cool_tables = nothing,
                                      itcap::Int = _SUB_ITMAX, dtfrac::Real = 0.1,
                                      xHeII_in = NaN)
    R    = typeof(e)
    mh   = R(MH); tiny = R(_SUB_TINY)
    # domain guard: f16 hydro can hand rho <= 0 / e <= 0 / X*rho < 0 at extreme
    # shock and void frontiers; unguarded they reach device sqrt/log/pow
    # (DomainError aborts the kernel).  Clamp to physical-positive; the T floor
    # in gas_temperature then makes the cell a benign ~1 K near-vacuum.
    rho  = ifelse(rho > R(1.0e-35), rho, R(1.0e-35))   # ifelse: NaN>x is false
    e    = ifelse(e > tiny, e, tiny)                    # (max() passes NaN through)
    HII_m = ifelse(HII_m > zero(R), HII_m, zero(R))
    H2I_m = ifelse(H2I_m > zero(R), H2I_m, zero(R))
    d    = rho / mh
    z0   = R(z)
    Hz0  = hubble_z_of(z0; hubble = hubble, Om = Om, OL = OL)
    Hz_ad = isnan(adot_over_a) ? Hz0 : R(adot_over_a)
    evolve_z = !isnan(adot_over_a)
    fhd  = R(fh) * d

    yHeI = (one(R) - R(fh)) * d
    yHII = HII_m / mh
    yH2I = H2I_m / mh
    yHI  = max(fhd - yHII - yH2I, tiny)
    yde  = yHII
    # He⁺ freeze-out state (cosmological recombination): when xHeII_in is supplied the
    # caller CARRIES He⁺ across steps and we EVOLVE it (HyRec-2 He I rate below), which
    # captures the z≈2000-2500 slow freeze-out that Saha alone misses.  fHe = n_He/n_H.
    track_He = !isnan(xHeII_in)
    fHe   = (one(R) - R(fh)) / (R(4) * R(fh))
    nH_h  = fhd
    xHeII = track_He ? R(xHeII_in) : zero(R)

    Tc0  = comp2_cmb(z0); c10 = comp1_cmb(z0)
    f    = R(dtfrac)
    ttot = zero(R); iter = 0
    @inbounds while ttot < dt && iter < itcap
        iter += 1
        rem = dt - ttot
        if evolve_z
            zt = (one(R) + z0) * exp(-Hz_ad * ttot) - one(R)
            Tc = comp2_cmb(zt); c1 = comp1_cmb(zt); Hz = hubble_z_of(zt; hubble=hubble, Om=Om, OL=OL)
        else
            zt = z0; Tc = Tc0; c1 = c10; Hz = Hz0
        end

        T = gas_temperature(rho, e, yHI, yHII, yHeI/R(4), tiny, tiny, yde,
                            tiny, yH2I/R(2), tiny; gamma = GAMMA_DEFAULT)

        # rate coefficients: lean analytic fits by default; the optional log–log rate
        # table (uniform across the warp ⇒ the branch compiles away) swaps the ~13
        # per-iteration transcendental fits for a branchless interpolation — the GPU hot
        # path.  Only the pure-T rates this mode needs are pulled from the NamedTuple.
        if rate_tables === nothing
            # H recombination: RECFAST-v2 fudge (α_B×1.125 + the Hswitch Gaussian on the
            # Lyα K-factor) — the multilevel-cascade speed-up that brings the effective-
            # 3-level Peebles rate to HyRec/CosmoRec (closes the freeze-out tail).  Applied
            # only in the H-recombination epoch (evolve_z & z<1600) where the RADIATIVE
            # recombination dominates and the fudge is calibrated; at z>1600 (He era, H
            # Saha) the collisional terms dominate and the fudge is neither calibrated nor
            # cleanly cancelling, so leave it off.  Un-fudged in the collapse (dense gas).
            k2   = (evolve_z && zt < R(1600)) ?
                   peebles_k2_mixing(T, yHI, yHI, Hz; fudge = R(1.125),
                                     gauss = R(recfast_gauss_factor(zt))) :
                   peebles_k2(T, yHI, Hz)
            kb1s = beta1s_freq(Tc) * k2 / (recfast_alpha(T) * R(1.0e6))
            k1v=k1(T); k7v=k7(T); k8v=k8(T); k9v=k9(T); k10v=k10(T); k15v=k15(T)
            k11v=k11(T); k12v=k12(T); k13v=k13(T); k14v=k14(T); k16v=k16(T); k17v=k17(T); k19v=k19(T)
            k22v=k22(T); k57v=k57(T); k58v=k58(T); k27v=k27_cmb(Tc); k28v=k28_cmb(Tc)
        else
            K = table_rates(rate_tables, T, yHI, Hz, cmb_rates(Tc))
            k2=K.k2; kb1s=K.k_beta1s; k1v=K.k1; k7v=K.k7; k8v=K.k8; k9v=K.k9
            k10v=K.k10; k15v=K.k15; k11v=K.k11; k12v=K.k12; k13v=K.k13
            k14v=K.k14; k16v=K.k16; k17v=K.k17; k19v=K.k19
            k22v=K.k22; k57v=K.k57; k58v=K.k58; k27v=K.k27; k28v=K.k28
        end
        # He electron contribution to n_e — drives BOTH the H recombination (Riccati
        # source) and the Compton cooling.  track_He: fold the CARRIED He⁺ (+ He²⁺ Saha
        # on top); else the stateless Saha QSSA for hot/shock-ionized gas (Tc>5000).
        # Both inert for the collapse (He neutral) — the analytic-fit path only for now.
        if track_He
            _shh1, _shh2 = helium_saha_pair(Tc)
            if zt > R(4500)
                # He fully in Saha equilibrium and He²⁺ is the MAJORITY — reconstructing
                # it from the tiny He⁺ (n_HeIII = n_HeII·s2/n_e) amplifies any error and
                # (with n_e ≈ n_HII, missing the 2·He²⁺ electrons) overshot n_HeIII past
                # n_He.  Solve the 3-level He + n_e balance self-consistently instead (a
                # few fixed points; He is a small perturbation on n_e ⇒ fast) — the exact
                # Saha x_e, and it re-anchors the carried He⁺ to equilibrium each step.
                _neh = max(yHII + xHeII*nH_h, tiny)
                for _ in 1:4
                    _r1 = _shh1/_neh; _r2 = _shh2/_neh; _den = one(R) + _r1 + _r1*_r2
                    _neh = yHII + fHe*(_r1 + R(2)*_r1*_r2)/_den * nH_h
                end
                _r1 = _shh1/_neh; _r2 = _shh2/_neh; _den = one(R) + _r1 + _r1*_r2
                xHeII = fHe*_r1/_den
                yde   = _neh
            else
                # He²⁺ negligible (z≲4500): carried He⁺ + its (tiny) He²⁺ Saha tail.
                _neh    = max(yHII + xHeII*nH_h, tiny)
                _xHeIII = xHeII * _shh2 / _neh
                yde     = yHII + (xHeII + R(2)*_xHeIII) * nH_h
            end
        elseif Tc > R(5000) && rate_tables === nothing
            _shh1, _shh2 = helium_saha_pair(Tc)
            _, nHeII, nHeIII = helium_equilibrium(_shh1, _shh2, k3(T), k4(T), k5(T), k6(T),
                                                  yde, yHeI/R(4))
            yde = yHII + nHeII + R(2)*nHeIII
        end

        # cooling & the Compton/non-Compton split
        edot = cooling_edot(yHI, yHII, yHeI/R(4), yde, yH2I/R(2), zero(R), T, zt;
                            nH = fhd, cool_tables = cool_tables)
        # H₂ formation heating + dissociation cooling (start-of-substep QSSA rates)
        nHMh   = k7v*yHI*yde  / ((k8v+k15v)*yHI + (k16v+k17v)*yHII + k14v*yde + k27v + tiny)
        nH2IIh = k9v*yHI*yHII / (k10v*yHI + k28v + tiny)
        edot += _h2_chem_heat(yHI, yHII, yde, yH2I/R(2), nHMh, nH2IIh, fhd,
                              k8v, k10v, k11v, k12v, k13v, k22v)
        if T <= R(1.01)*R(MIN_TEMPERATURE) && edot < zero(R); edot = zero(R); end
        edot_c    = -c1 * (T - Tc) * yde
        edot_rest = edot - edot_c
        hubble_expansion && (edot_rest -= R(2) * Hz_ad * e * rho)
        Kc      = c1 * yde * (T / e) / rho          # Compton specific frequency
        de_rest = edot_rest / rho

        # Step limited by (a) non-Compton cooling and (b) the Compton T-change — the Compton
        # is integrated analytically, but its ΔT must stay ≲f·T so the T-dependent rates (and
        # the x_e–T feedback) stay accurate.  This is LOOSE where T≈T_cmb (Compton-locked) or
        # Kc is small (decoupled) — big steps survive there; it only bites in the transition.
        dT     = abs(Tc - T)
        exmax  = dT > tiny ? f*T/dT : one(R)
        dtc_c  = (exmax < one(R) && Kc > tiny) ? -log(one(R) - exmax)/Kc : typemax(R)
        dtc = min(_step_f(e, de_rest, f), dtc_c, rem)

        # ── energy: analytic Compton relaxation + explicit non-Compton cooling over dtc ──
        ex  = -expm1(-Kc*dtc)                       # 1 − e^{−Kc·dtc}  (stable)
        g   = Kc > tiny ? ex/Kc : dtc               # ∫₀^{dtc} e^{−Kc t}dt → dtc as Kc→0
        e   = e + (e*(Tc/T) - e)*ex + de_rest*g     # exact for de/dt = −Kc(e−e_cmb) + de_rest
        e   = max(e, tiny)

        # ── x_HII: linear-source Riccati (recomb k2 vs the CMB photo-ion + collisional
        #    source).  The photoionisation kb1s·n_HI and collisional-e k1·n_e·n_HI are
        #    LINEAR in n_HI = C − y ⇒ carried as the implicit −p·y depletion (p = kb1s +
        #    k1·n_e); this removes the frozen-source overshoot that made x_HII oscillate
        #    step-to-step at z≳1600 (H Saha-pinned).  The genuinely-nonlinear collisional
        #    floor k57·n_HI² + k58·n_HI·n_HeI stays frozen in q.  C = n_HI + n_HII. ──
        pion = kb1s + k1v*yde
        qion = k57v*yHI*yHI + k58v*yHI*yHeI/R(4)
        yHII = _riccati2(yHII, k2, pion, qion, yHI + yHII, dtc)
        yde  = yHII        # ADVANCE n_e = n_HII from the Riccati BEFORE forming H₂:
                           # H⁻/H₂ formation ∝ n_e, so using the recombined end-of-step
                           # electron density (not the stale start value) removes the
                           # coarse-Δt H₂ over-formation at essentially zero cost.
        # ── x_H2: formation (H⁻ + H₂⁺ closed-form channels + 3-body) vs collisional
        #    dissociation, backward-Euler — unconditionally stable, still fully closed-form
        #    (one division, NO subcycle).  acH2 = k13·n_HI + k11·n_HII + k12·n_e : the
        #    destruction the old pure-formation quadrature FROZE; it matters once the
        #    compressing core warms toward ~few×10³ K (H₂ + H → 3H), pulling f_H2 back
        #    toward its formation/dissociation balance instead of accumulating unbounded.
        # Advance the H₂-formation rates to the POST-cooling temperature.  Rates are
        # evaluated at the start-of-substep T, but the gas COOLED over the substep and
        # spends its time at the lower T; k7 (H+e→H⁻) ∝ T^0.95 and k8 (H⁻+H→H₂) both
        # drop with T, so the start-T value over-forms H₂.  Refresh T from the updated
        # energy and re-evaluate ONLY k7,k8 (the H⁻ channel = 99.9% of formation) — this
        # is enough to make the reduced solver's CONVERGED H₂ match the full network's
        # bit-for-bit, at 2 rate re-evals (not the full ~13).  (Table path keeps start-T.)
        if rate_tables === nothing
            Ta = gas_temperature(rho, e, yHI, yHII, yHeI/R(4), tiny, tiny, yde, tiny, yH2I/R(2), tiny; gamma = GAMMA_DEFAULT)
            k7v = k7(Ta); k8v = k8(Ta)
        end
        # H⁻ / H₂⁺ quasi-steady-state.  The H⁻ denominator now carries the FULL
        # destruction set (matching the network's equilibrium_HM): associative
        # detachment (k8+k15)·n_HI + mutual neutralization by the ionized species
        # (k16+k17)·n_HII + k14·n_e + k19·n_H2II + CMB photodetachment k27.  These
        # ionized-species sinks are ~0.1% of k8·n_HI at collapse ionization (x_e≪1)
        # so they don't shift the cold-collapse H₂, but they make the H⁻ equilibrium
        # exact in the recombination era (x_e→1) too.  n_H2II first for the k19 term.
        nH2II = k9v*yHI*yHII / (k10v*yHI + k28v + tiny)
        nHM   = k7v*yHI*yde  / ((k8v + k15v)*yHI + (k16v + k17v)*yHII +
                                 k14v*yde + k19v*nH2II + k27v + tiny)
        # H₂ formation with EXACT n_HI depletion (Bernoulli), then dissociation as an
        # operator-split linear sink.  Formation is nonlinear in n_HI: linear (H⁻ via
        # k8·n_HM, H₂⁺ via k10·n_H2II) with coeff `af` + 3-body ∝ n_HI³ (k22).  With
        # H-nuclei conservation n_HI + yH2I = C, dn_HI/dt = −2(af·n_HI + k22·n_HI³) is
        # a Bernoulli eq; w = n_HI⁻² linearises it (w′ = 4·af·w + 4·k22), integrated
        # EXACTLY over the substep → the fully-molecular transition (n_HI→0) is captured
        # instead of the frozen-n_HI over-formation.  Reduces to the old formation
        # backward-Euler at low density (af·dt≪1, k22→0).
        af    = k8v*nHM + k10v*nH2II                     # linear formation coeff (rate = af·n_HI)
        CH    = yHI + yH2I                               # conserved H nuclei (not in H⁺), y-conv
        w0    = one(R) / (yHI*yHI)
        e4a   = exp(min(R(4)*af*dtc, R(60)))             # cap → fully molecular (no overflow)
        gfac  = af > tiny ? (e4a - one(R))/af : R(4)*dtc  # (e^{4af·dt}−1)/af  → 4·dt as af→0 (no blow-up)
        wN    = w0*e4a + k22v*gfac                       # w(dt) = n_HI(dt)⁻²  (exact Bernoulli)
        yH2I  = CH - one(R)/sqrt(wN)                     # H₂ formed = n_HI consumed
        acH2  = k13v*yHI + k11v*yHII + k12v*yde          # dissociation (≈0 below ~8000 K)
        yH2I  = yH2I / (one(R) + acH2*dtc)               # operator-split backward-Euler sink

        yH2I = yH2I < tiny ? tiny : (yH2I > fhd ? fhd : yH2I)
        yHII = yHII < tiny ? tiny : (yHII > fhd - yH2I ? fhd - yH2I : yHII)
        yHI  = max(fhd - yHII - yH2I, tiny)
        yde  = yHII
        # advance He⁺ over the substep.  Above z≈4500 He²⁺ is non-negligible and He is in
        # Saha equilibrium — that is solved self-consistently in the START-of-substep fold
        # (xHeII already re-anchored to equilibrium there), so nothing to do at the end.
        # Below z≈4500 (He²⁺≈0) He⁺→He⁰ freezes out ⇒ evolve xHeII with the HyRec-2 He I
        # rate (backward-Euler), which captures the slow z≈2000-2500 freeze-out (semi-
        # forbidden 2³P / 2¹P-escape) with no hard Saha→rate seam in the transition region.
        if track_He && zt <= R(4500)
            # He⁺ freeze-out: HyRec-2 He I rate, backward-Euler.  The rate's H-continuum
            # enhancement — the term that COMPLETES He recombination — is ∝ 1/xH1 (neutral
            # H).  At z≳1600 H is Saha-pinned to the CMB and barely recombining (xH1 ~ 1e-8),
            # but the reduced H Riccati, though now monotone (linear-source form), still
            # carries round-off there; the He completion is exquisitely sensitive to it.
            # Feed the He rate the SMOOTH H Saha neutral fraction instead:
            #   n_HI/n_H = x_HII·x_e·n_H / S_H,  S_H = n_Q·e^{−χ_H/T_cmb}.
            # This is exact where H is in radiative equilibrium (the whole He era) and
            # costs one exp — the H-He-electron coupling the full network gets from its
            # n_e-consistent charge balance, without reinventing the network.  Below
            # z≈1600 (H freeze-out, the fudge epoch) H leaves Saha ⇒ real neutral frac.
            if zt > R(1600)
                _SH  = (R(_REC_CR)*Tc)^R(1.5) * R(1.0e-6) * exp(-R(_CHI_H_K)/Tc)
                _xh1 = min(max(yHII*yde/(nH_h*_SH), R(1.0e-12)), one(R))
            else
                _xh1 = yHI/nH_h
            end
            _A, _B = helium_HeI_rate_AB(Tc, nH_h, Hz, _xh1, xHeII, fHe)
            xHeII  = (xHeII + _A*dtc) / (one(R) + _B*dtc)
        end
        ttot += dtc
    end
    return e, yHII*mh, yH2I*mh, ttot, xHeII
end

# ── GPU / batched solve path for the analytic mode ───────────────────────────
# One thread per cell; no metals/dust/deuterium.  The analytic solver's low, uniform
# iteration count (the Compton stiffness is integrated in closed form) minimizes warp
# divergence — the reason this is the fast GPU chemistry path.
@kernel function _evolve_analytic_k!(e, HII, H2I, @Const(rho), du, vu2, tu, dt, z,
                                     hubble, Om, OL, fh, hexp, aoa, rtab, ctab, dtfrac, itcap)
    i = @index(Global)
    @inbounds begin
        en, hii, h2, _, _ = evolve_cell_analytic(rho[i]*du, e[i]*vu2, HII[i]*du, H2I[i]*du,
                                              dt*tu, z; hubble=hubble, Om=Om, OL=OL, fh=fh,
                                              hubble_expansion=hexp, adot_over_a=aoa,
                                              rate_tables=rtab, cool_tables=ctab,
                                              itcap=itcap, dtfrac=dtfrac)
        e[i]   = en  / vu2
        HII[i] = hii / du
        H2I[i] = h2  / du
    end
end

"""
    solve_chem_analytic!(rho, e_int, HII, H2I; a_value, dt, density_units, length_units,
                         time_units, hubble=71, Om=0.27, OL=0.73, fh=0.76,
                         hubble_expansion=false, adot_over_a=NaN, cool_tables=nothing,
                         dtfrac=0.1, itcap=_SUB_ITMAX, workgroup_size=0,
                         backend=:cpu, precision=Float64)

Batched [`evolve_cell_analytic`](@ref) over every cell (KA kernel; `:cpu`/`:cuda`/`:metal`).
Primordial H+H₂ only (no deuterium/dust/metals); `e_int`, `HII`, `H2I` updated in place.
"""
function solve_chem_analytic!(rho::AbstractVector, e_int::AbstractVector,
                              HII::AbstractVector, H2I::AbstractVector;
                              a_value::Real, dt::Real, density_units::Real,
                              length_units::Real, time_units::Real,
                              hubble::Real = 71.0, Om::Real = 0.27, OL::Real = 0.73,
                              fh::Real = 0.76, hubble_expansion::Bool = false,
                              adot_over_a::Real = NaN, rate_tables = nothing,
                              cool_tables = nothing,
                              dtfrac::Real = 0.1, itcap::Int = _SUB_ITMAX, workgroup_size::Int = 0,
                              backend::Symbol = :cpu, precision::Type = Float64)
    n  = length(rho)
    @assert length(e_int) == n && length(HII) == n && length(H2I) == n
    P  = precision
    be = ChemistryKernels.backend(backend)
    du = P(density_units); vu2 = P((length_units/time_units)^2); tu = P(time_units)
    z  = P(1.0/a_value - 1.0)
    d_rho = to_device(be, collect(rho),   P); d_e   = to_device(be, collect(e_int), P)
    d_HII = to_device(be, collect(HII),   P); d_H2I = to_device(be, collect(H2I),   P)
    k! = workgroup_size > 0 ? _evolve_analytic_k!(be, workgroup_size) : _evolve_analytic_k!(be)
    k!(d_e, d_HII, d_H2I, d_rho, du, vu2, tu, P(dt), z, P(hubble), P(Om), P(OL), P(fh),
       hubble_expansion, P(adot_over_a), rate_tables, cool_tables, P(dtfrac), itcap; ndrange = n)
    e_int .= to_host(d_e); HII .= to_host(d_HII); H2I .= to_host(d_H2I)
    return nothing
end
export solve_chem_analytic!

"""
    solve_chem_analytic_device!(rho, e_int, HII, H2I; …, backend=:cuda, precision=Float32)

Zero-copy variant of [`solve_chem_analytic!`](@ref): all arrays are ALREADY device
arrays of element type `precision`; the kernel runs in place, no host round-trip.
"""
function solve_chem_analytic_device!(rho, e_int, HII, H2I;
                                     a_value::Real, dt::Real, density_units::Real,
                                     length_units::Real, time_units::Real,
                                     hubble::Real = 71.0, Om::Real = 0.27, OL::Real = 0.73,
                                     fh::Real = 0.76, hubble_expansion::Bool = false,
                                     adot_over_a::Real = NaN, rate_tables = nothing,
                                     cool_tables = nothing,
                                     dtfrac::Real = 0.1, itcap::Int = _SUB_ITMAX,
                                     workgroup_size::Int = 0, backend::Symbol = :cuda,
                                     precision::Type = Float32)
    n  = length(rho); P = precision; be = ChemistryKernels.backend(backend)
    du = P(density_units); vu2 = P((length_units/time_units)^2); tu = P(time_units)
    z  = P(1.0/a_value - 1.0)
    k! = workgroup_size > 0 ? _evolve_analytic_k!(be, workgroup_size) : _evolve_analytic_k!(be)
    k!(e_int, HII, H2I, rho, du, vu2, tu, P(dt), z, P(hubble), P(Om), P(OL), P(fh),
       hubble_expansion, P(adot_over_a), rate_tables, cool_tables, P(dtfrac), itcap; ndrange = n)
    KA.synchronize(be)
    return nothing
end
export solve_chem_analytic_device!

# ── UInt16 log₂-encoded species variant of the analytic path ──────────────────
# HII/H2I are carried as UInt16 log₂-encoded mass FRACTIONS (X = ρ_species/ρ_total)
# via the `log2_species.jl` codec (2 B/cell vs 4 B f32; ~0.12 %/ULP over 33 dex).  The
# kernel decodes to physical CGS at load, runs `evolve_cell_analytic`, and re-encodes at
# store; energy and rho stay floating-point.  Primordial H+H₂ only (no deuterium/dust/metals).
@kernel function _evolve_analytic_k_u16!(e, HII_u16, H2I_u16, @Const(rho), du, vu2, tu, dt, z,
                                         hubble, Om, OL, fh, hexp, aoa, rtab, ctab, dtfrac, itcap)
    i = @index(Global)
    @inbounds begin
        T = eltype(e)
        # clamp: f16 hydro can hand rho ≤ 0; the re-encode below divides by r
        # and log2's the ratio (device DomainError without the guard)
        r0 = rho[i] * du
        r = ifelse(r0 > T(1.0e-35), r0, T(1.0e-35))        # NaN-safe (physical CGS)
        hii_m = decode_log2sp(T, HII_u16[i]) * r           # UInt16 fraction → CGS mass density
        h2i_m = decode_log2sp(T, H2I_u16[i]) * r
        en, hii, h2, _, _ = evolve_cell_analytic(r, e[i]*vu2, hii_m, h2i_m, dt*tu, z;
                                              hubble=hubble, Om=Om, OL=OL, fh=fh,
                                              hubble_expansion=hexp, adot_over_a=aoa,
                                              rate_tables=rtab, cool_tables=ctab,
                                              itcap=itcap, dtfrac=dtfrac)
        e[i]       = en / vu2
        HII_u16[i] = encode_log2sp(hii / r)                # CGS mass density → fraction → UInt16
        H2I_u16[i] = encode_log2sp(h2  / r)
    end
end

"""
    solve_chem_analytic_u16!(rho, e_int, HII_u16, H2I_u16; a_value, dt, density_units,
                             length_units, time_units, …, backend=:cpu, precision=Float32)

UInt16 log₂-encoded-species variant of [`solve_chem_analytic!`](@ref).  `HII_u16`,
`H2I_u16` are `AbstractVector{UInt16}` storing log₂-encoded **mass fractions**
X = ρ_species/ρ_total (codec: u=0 → X≈7.73e-34, u=65535 → X=1.0; ≈0.12 %/ULP).  Host
round-trip; prepare/recover with `encode_log2sp_vec(HII ./ rho)` / `decode_log2sp_vec`.
2 B/species vs 4 B f32; primordial H+H₂ only.
"""
function solve_chem_analytic_u16!(rho::AbstractVector, e_int::AbstractVector,
                                  HII_u16::AbstractVector{UInt16},
                                  H2I_u16::AbstractVector{UInt16};
                                  a_value::Real, dt::Real, density_units::Real,
                                  length_units::Real, time_units::Real,
                                  hubble::Real = 71.0, Om::Real = 0.27, OL::Real = 0.73,
                                  fh::Real = 0.76, hubble_expansion::Bool = false,
                                  adot_over_a::Real = NaN, rate_tables = nothing,
                                  cool_tables = nothing,
                                  dtfrac::Real = 0.1, itcap::Int = _SUB_ITMAX, workgroup_size::Int = 0,
                                  backend::Symbol = :cpu, precision::Type = Float32)
    n  = length(rho)
    @assert length(e_int) == n && length(HII_u16) == n && length(H2I_u16) == n
    P  = precision
    be = ChemistryKernels.backend(backend)
    du = P(density_units); vu2 = P((length_units/time_units)^2); tu = P(time_units)
    z  = P(1.0/a_value - 1.0)
    d_rho = to_device(be, collect(rho),   P); d_e = to_device(be, collect(e_int), P)
    d_HII = to_device(be, collect(HII_u16), UInt16); d_H2I = to_device(be, collect(H2I_u16), UInt16)
    k! = workgroup_size > 0 ? _evolve_analytic_k_u16!(be, workgroup_size) : _evolve_analytic_k_u16!(be)
    k!(d_e, d_HII, d_H2I, d_rho, du, vu2, tu, P(dt), z, P(hubble), P(Om), P(OL), P(fh),
       hubble_expansion, P(adot_over_a), rate_tables, cool_tables, P(dtfrac), itcap; ndrange = n)
    e_int .= to_host(d_e); HII_u16 .= to_host(d_HII); H2I_u16 .= to_host(d_H2I)
    return nothing
end
export solve_chem_analytic_u16!

"""
    solve_chem_analytic_device_u16!(rho, e_int, HII_u16, H2I_u16; …, backend=:cuda, precision=Float32)

Zero-copy variant of [`solve_chem_analytic_u16!`](@ref): `rho`/`e_int` are device arrays of
element type `precision`, `HII_u16`/`H2I_u16` are device `UInt16` arrays.  No host round-trip.
"""
function solve_chem_analytic_device_u16!(rho, e_int, HII_u16, H2I_u16;
                                         a_value::Real, dt::Real, density_units::Real,
                                         length_units::Real, time_units::Real,
                                         hubble::Real = 71.0, Om::Real = 0.27, OL::Real = 0.73,
                                         fh::Real = 0.76, hubble_expansion::Bool = false,
                                         adot_over_a::Real = NaN, rate_tables = nothing,
                                         cool_tables = nothing,
                                         dtfrac::Real = 0.1, itcap::Int = _SUB_ITMAX,
                                         workgroup_size::Int = 0, backend::Symbol = :cuda,
                                         precision::Type = Float32)
    n  = length(rho); P = precision; be = ChemistryKernels.backend(backend)
    du = P(density_units); vu2 = P((length_units/time_units)^2); tu = P(time_units)
    z  = P(1.0/a_value - 1.0)
    k! = workgroup_size > 0 ? _evolve_analytic_k_u16!(be, workgroup_size) : _evolve_analytic_k_u16!(be)
    k!(e_int, HII_u16, H2I_u16, rho, du, vu2, tu, P(dt), z, P(hubble), P(Om), P(OL), P(fh),
       hubble_expansion, P(adot_over_a), rate_tables, cool_tables, P(dtfrac), itcap; ndrange = n)
    KA.synchronize(be)
    return nothing
end
export solve_chem_analytic_device_u16!

# ── hybrid analytic ⇄ full-network per-cell dispatch ─────────────────────────
# The analytic path (evolve_cell_analytic) is a reduced primordial H+H₂ network:
# fast, great for IC generation, but MISSING 3-body H₂ formation and collisional
# H₂ dissociation — the physics that governs the dense (n ≳ 1e8 cm⁻³) collapsing
# core.  This kernel switches to the FULL network (evolve_cell) per cell where the
# physical density r exceeds `rho_switch` (CGS g/cm³), keeping the cheap analytic
# path everywhere else.  Same 2-species (HII, H2I) u16 contract as the analytic
# path — the full network reconstructs the intermediates (HM, H2II, He, e)
# internally.  Warp divergence is minimal in a collapse (density is spatially
# coherent → warps are all-dense or all-analytic).
@kernel function _evolve_hybrid_k_u16!(e, HII_u16, H2I_u16, @Const(rho), du, vu2, tu, dt, z,
                                       hubble, Om, OL, fh, hexp, aoa, rtab, ctab, dtfrac, itcap,
                                       rho_switch)
    i = @index(Global)
    @inbounds begin
        T = eltype(e)
        r0 = rho[i] * du
        r = ifelse(r0 > T(1.0e-35), r0, T(1.0e-35))        # NaN-safe (physical CGS)
        hii_m = decode_log2sp(T, HII_u16[i]) * r
        h2i_m = decode_log2sp(T, H2I_u16[i]) * r
        local en, hii, h2
        if r > T(rho_switch)                               # dense core → FULL network
            en, hii, h2, _, _ = evolve_cell(r, e[i]*vu2, hii_m, h2i_m, zero(T), dt*tu, z;
                                            hubble=hubble, Om=Om, OL=OL, fh=fh, deuterium=false,
                                            hubble_expansion=hexp, adot_over_a=aoa,
                                            rate_tables=rtab, cool_tables=ctab,
                                            itcap=itcap, dtfrac=dtfrac)
        else                                               # envelope → analytic IC path
            en, hii, h2, _, _ = evolve_cell_analytic(r, e[i]*vu2, hii_m, h2i_m, dt*tu, z;
                                            hubble=hubble, Om=Om, OL=OL, fh=fh,
                                            hubble_expansion=hexp, adot_over_a=aoa,
                                            rate_tables=rtab, cool_tables=ctab,
                                            itcap=itcap, dtfrac=dtfrac)
        end
        e[i]       = en / vu2
        HII_u16[i] = encode_log2sp(hii / r)
        H2I_u16[i] = encode_log2sp(h2  / r)
    end
end

"""
    solve_chem_hybrid_device_u16!(rho, e_int, HII_u16, H2I_u16; rho_switch, …)

Per-cell dispatch between the analytic reduced network and the full `evolve_cell`
network: cells with physical density `> rho_switch` (CGS g/cm³) run the FULL
network (3-body H₂ formation + collisional dissociation — the dense-core physics
the analytic path omits), all others run the cheap analytic path.  Same device
buffers and 2-species (HII, H2I) u16 contract as
[`solve_chem_analytic_device_u16!`](@ref).  `rho_switch → 0` ⇒ full network
everywhere; `rho_switch → ∞` ⇒ pure analytic (identical to the analytic solver).
"""
function solve_chem_hybrid_device_u16!(rho, e_int, HII_u16, H2I_u16;
                                       rho_switch::Real,
                                       a_value::Real, dt::Real, density_units::Real,
                                       length_units::Real, time_units::Real,
                                       hubble::Real = 71.0, Om::Real = 0.27, OL::Real = 0.73,
                                       fh::Real = 0.76, hubble_expansion::Bool = false,
                                       adot_over_a::Real = NaN, rate_tables = nothing,
                                       cool_tables = nothing,
                                       dtfrac::Real = 0.1, itcap::Int = _SUB_ITMAX,
                                       workgroup_size::Int = 0, backend::Symbol = :cuda,
                                       precision::Type = Float32)
    n  = length(rho); P = precision; be = ChemistryKernels.backend(backend)
    du = P(density_units); vu2 = P((length_units/time_units)^2); tu = P(time_units)
    z  = P(1.0/a_value - 1.0)
    k! = workgroup_size > 0 ? _evolve_hybrid_k_u16!(be, workgroup_size) : _evolve_hybrid_k_u16!(be)
    k!(e_int, HII_u16, H2I_u16, rho, du, vu2, tu, P(dt), z, P(hubble), P(Om), P(OL), P(fh),
       hubble_expansion, P(adot_over_a), rate_tables, cool_tables, P(dtfrac), itcap,
       P(rho_switch); ndrange = n)
    KA.synchronize(be)
    return nothing
end
export solve_chem_hybrid_device_u16!

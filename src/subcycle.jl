# subcycle.jl — the per-cell sub-cycling driver, specialized to the v2026 reduced
# model.  Repeats, until the requested macro step `dt` is consumed, a self-limited
# sub-step `dtit` that:
#   1. evaluates T, all rates (with the Peebles k2 override), edot, and the
#      net e⁻/HI rates;
#   2. sizes dtit to a ≤10% change in n_e and the thermal energy (alternate
#      H2/no-species limiters are available for scalar convergence diagnosis);
#   3. advances the energy — IMPLICITLY for the stiff CMB-Compton term when it is
#      stiff (K·Δt>1), else explicitly;
#   4. advances the species by one backward-Euler sweep (`network_step`).
#
# Everything in physical CGS (number densities [cm⁻³], e [erg/g], t [s]).  State
# is carried in the mass-equivalent y-convention (yH2I=2·n(H2) etc., as in
# network_step.jl), with the network "density" d = ρ/m_H so yHeI=(1−fh)·d.  Pure &
# allocation-free (isbits NamedTuples) ⇒ runs in a KA kernel and is AD-ready.

export build_rates, evolve_cell

const _SUB_ITMAX = 5_000         # subcycle cap (bounds GPU kernel time; well-behaved
                                 # cells converge in ≪100 steps — this is a watchdog).
                                 # The default is kept HIGH so every regime is correct
                                 # out of the box (the recombination epoch z≈1100 needs
                                 # ≫100 substeps per macro-step).  A latency-bound caller
                                 # (the galaxy-formation sim hot path, z≈10) passes a low
                                 # `itcap` (e.g. 100) to bound per-cell work: ~1.8× faster
                                 # there with the median AND p90 accuracy unchanged, the
                                 # rare stiff straggler re-entered (partial state +
                                 # remaining dt) on the next host step.
const _SUB_TINY  = 1.0e-20

# Fraction-change sub-step from a rate: step ≤ f·X/rate.  No constraint (Inf)
# when the rate is ~0.  NOTE: unlike a code-unit scheme (whose edot/dedot are
# O(1), so an absolute floor is harmless), our rates are PHYSICAL CGS (volumetric
# ė ~ 1e-30), so an absolute floor would EXCEED real rates and corrupt them — we
# guard the division instead, never the value.
@inline _step_f(X, rate, f) = abs(rate) > zero(rate) ? abs(f * X / rate) : typemax(rate)
@inline _step10(X, rate) = _step_f(X, rate, oftype(rate, 0.1))   # backward-compat alias

# A vanishing species must not freeze the subcycler while it is being produced.
# Formation is bounded through the complementary HII/electron equation; this limit is
# only needed when the explicit rate would deplete the current neutral reservoir.
@inline _depletion_step_f(X, rate, f) =
    rate < zero(rate) ? _step_f(X, rate, f) : typemax(rate)

# `dt` is commonly a large physical time in Float32. Adding its final remainder to
# `ttot` can round back below `dt`, leaving the loop to spin until `itcap`. Preserve
# the exact completion state when the selected substep consumes that remainder.
@inline function _advance_subcycle_time(ttot, dtit, dt)
    rem = dt - ttot
    return dtit >= rem ? dt : ttot + dtit
end

"""
    build_rates(T, Trad, nHI, Hz; deuterium=false)

NamedTuple of every reaction-rate coefficient the network needs, at gas
temperature `T` and CMB temperature `Trad`. k2 is the Peebles override (needs the
neutral-H density `nHI` and Hubble rate `Hz`); k27/k28 are the CMB photo-rates;
all others are the Wave-1 analytic fits. Pure.
"""
# The reaction-rate coefficients that depend ONLY on the CMB radiation temperature
# Trad (β₁s, k27, k28, HeH⁺ radiation terms, the two He Saha factors). Trad is
# frozen across a sub-cycle
# unless the host supplies its expansion rate (evolve_z), so `evolve_cell` builds this
# ONCE per cell and reuses it every iteration — hoisting ~5 transcendentals/iter out
# of the per-iteration `build_rates`.  isbits NamedTuple ⇒ free in a GPU kernel.
@inline function cmb_rates(Trad)
    return (; b1s = beta1s_freq(Trad), k27 = k27_cmb(Trad), k28 = k28_cmb(Trad),
            gamma_HeH = gamma_HeH_cmb(Trad), HeH_stim = HeH_stim_factor(Trad),
            she = helium_saha_pair(Trad))
end

# Convenience: build the full rate set from (T, Trad, …) by computing the Trad terms
# inline (callers that don't sub-cycle).  The hot path passes a precomputed `cmb_rates`
# to `_build_rates_cr` directly (this 4-arg form and that one differ only in arg meaning,
# so the cr core gets its own name to avoid a same-arity method clash).
@inline build_rates(T, Trad, nHI, Hz; deuterium::Bool = false) =
    _build_rates_cr(T, nHI, Hz, cmb_rates(Trad); deuterium = deuterium)

@inline function _build_rates_cr(T, nHI, Hz, cr; deuterium::Bool = false)
    R = typeof(T)
    k2_val = peebles_k2(T, nHI, Hz)
    # C-weighted β₁s: matches k2=α_B×C so equilibrium gives true Saha (C cancels).
    # At high z (xe≈1, nHI≈0): C→1, k_beta1s→β₁s (drives Saha).
    # At z≈1200 (C≈0.006): k_beta1s negligible vs recombination → freeze-out preserved.
    # β₁s (=`cr.b1s`) is CMB photoionisation of H(1s) evaluated at the RADIATION
    # temperature Trad, not the matter T.  At recombination T≈Trad so this is unchanged;
    # under a low-z UV background the gas heats to T≫Trad, and beta1s_freq(T) would
    # otherwise spuriously drive H to Saha equilibrium at the hot matter temperature (no
    # CMB photons exist to do that — Trad is cold).  The Peebles C-factor (k2/α_B) stays
    # at the matter T.
    k_b1s = cr.b1s * k2_val / (recfast_alpha(T) * R(1.0e6))
    she1, she2 = cr.she     # He Saha factors at Trad (→ fully neutral He at low z)
    kHeH_ra = kHeH_ra_spont(T) + kHeH_ra_stim_base(T) * cr.HeH_stim
    base = (; k1=k1(T), k2=k2_val, k3=k3(T), k4=k4(T), k5=k5(T),
            k6=k6(T), k7=k7(T), k8=k8(T), k9=k9(T), k10=k10(T), k11=k11(T),
            k12=k12(T), k13=k13(T), k14=k14(T), k15=k15(T), k16=k16(T), k17=k17(T),
            k18=k18(T), k19=k19(T), k22=k22(T), k57=k57(T), k58=k58(T),
            k27=cr.k27, k28=cr.k28, k_beta1s=k_b1s,
            kHeH_ra=kHeH_ra, kHeH_H=kHeH_H(T), kHeH_e=kHeH_e(T),
            gamma_HeH=cr.gamma_HeH,
            she1=she1, she2=she2)
    deuterium || return base
    return merge(base, (; k50=k50(T), k51=k51(T), k52=k52(T), k53=k53(T),
                        k54=k54(T), k55=k55(T), k56=k56(T)))
end

# Net rate-of-change of n_e and n_HI (molecular branch, reduced: no
# radiation/dust/shielding).  Used only to size the chemistry sub-step.
@inline function _de_hi_dot(yHI, yHII, yde, yH2I, yHM, yH2II, yHeI, yHeII, yHeIII, K;
                            GamHI = zero(typeof(yHI)))
    R = typeof(yHI)
    # Γ_HI (external UV-background photoionisation, 0 by default) destroys HI and
    # frees an electron — include it so the sub-step limiter resolves photoionisation.
    HIdot = -K.k1*yHI*yde - K.k7*yHI*yde - K.k8*yHM*yHI - K.k9*yHII*yHI -
            K.k10*yH2II*yHI/R(2) - R(2)*K.k22*yHI^3 + K.k2*yHII*yde +
            R(2)*K.k13*yHI*yH2I/R(2) + K.k11*yHII*yH2I/R(2) +
            R(2)*K.k12*yde*yH2I/R(2) + K.k14*yHM*yde + K.k15*yHM*yHI +
            R(2)*K.k16*yHM*yHII + R(2)*K.k18*yH2II*yde/R(2) + K.k19*yH2II*yHM/R(2) -
            K.k57*yHI*yHI - K.k58*yHI*yHeI/R(4) - K.k_beta1s*yHI - R(GamHI)*yHI
    dedot = K.k1*yHI*yde + K.k3*yHeI*yde/R(4) + K.k5*yHeII*yde/R(4) +
            K.k8*yHM*yHI + K.k15*yHM*yHI + K.k17*yHM*yHII + K.k14*yHM*yde -
            K.k2*yHII*yde - K.k4*yHeII*yde/R(4) - K.k6*yHeIII*yde/R(4) -
            K.k7*yHI*yde - K.k18*yH2II*yde/R(2) + K.k57*yHI*yHI + K.k58*yHI*yHeI/R(4) +
            K.k_beta1s*yHI + R(GamHI)*yHI
    return dedot, HIdot
end

# Net H2 rate evaluated from the same algebraic H-/H2+ intermediaries and source /
# destruction coefficients used by `network_step`. `yH2I` counts H nuclei in H2,
# so this rate has the same mass-equivalent convention as the evolved state.
@inline function _h2_dot(yHI, yHII, yde, yH2I, yHM, yH2II, nHeI, K;
                         k_h2d = zero(typeof(yHI)), k_lw = zero(typeof(yHI)))
    R = typeof(yHI)
    HMp, H2IIeq = _molecular_intermediates(yHI, yHII, yde, yH2I, K)
    nHeH = equilibrium_HeH(nHeI, yHII, yde, yHI, K.kHeH_ra, K.kHeH_H,
                           K.kHeH_e, K.gamma_HeH)
    source = R(2) * (K.k8*HMp*yHI + K.k10*H2IIeq*yHI/R(2) +
                     K.k19*H2IIeq*HMp/R(2) + K.kHeH_H*nHeH*yHI +
                     K.k22*yHI^3) +
             R(2) * R(k_h2d) * yHI^2
    destruction = K.k13*yHI + K.k11*yHII + K.k12*yde + R(k_lw)
    return source - destruction*yH2I
end

"""
    evolve_cell(rho, e, HII_m, H2I_m, HDI_m, dt, z; hubble, Om, OL, fh, deuterium,
                dust, Z_rel, G0, A_V, N_H, N_H2)

Sub-cycle one cell over macro-step `dt` [s].  `rho` = gas mass density [g/cm³],
`e` = specific internal energy [erg/g]; `HII_m`/`H2I_m`/`HDI_m` = species MASS
densities [g/cm³] (ρ·x).  Returns the updated `(e, HII_m, H2I_m, HDI_m)`. Pure.

Dust physics is enabled by `dust = true`, which requires:
- `Z_rel`  : metallicity relative to solar (sets dust-to-gas ratio)
- `G0`     : FUV field [Habing units] (for PE heating, grain charging, T_dust)
- `A_V`    : visual extinction [mag] (attenuates UV heating of dust and LW field)
- `N_H`    : H column density [cm⁻²] (dust LW attenuation; 0 → skip)
- `N_H2`   : H₂ column density [cm⁻²] (H₂ self-shielding; 0 → skip)
"""
@inline function evolve_cell(rho, e, HII_m, H2I_m, HDI_m, dt, z;
                             hubble = 71.0, Om = 0.27, OL = 0.73,
                             fh = FH_DEFAULT, deuterium::Bool = false,
                             hubble_expansion::Bool = false,
                             adot_over_a = NaN, metals = nothing,
                             rate_tables = nothing, cool_tables = nothing,
                             itcap::Int = _SUB_ITMAX,
                             dtfrac::Real = 0.1,
                             max_substep_fraction::Real = 0.5,
                             species_limiter::Val{SL} = Val(:electron),
                             h2_fraction_floor::Real = 1.0e-12,
                             stiff_h2_pair::Val{SP} = Val(true),
                             diagnostics::Val{D} = Val(false),
                             dust::Bool = false,
                             Z_rel::Real = 0.0, G0::Real = 0.0,
                             A_V::Real = 0.0, N_H::Real = 0.0,
                             N_H2::Real = 0.0) where {SL,SP,D}
    R    = typeof(e)
    mh   = R(MH); tiny = R(_SUB_TINY)
    # domain guard (match evolve_cell_analytic/fast): f16/f32 hydro + dual-energy
    # can hand rho ≤ 0, e ≤ 0, or X·rho < 0 at shock/void frontiers and thermal-
    # underflow cells.  Unguarded, e = 0 makes T/e = Inf in the Compton frequency
    # Kc → NaN, and ONE such cell poisons the CFL/dt reduction and kills the run.
    # Clamp to physical-positive; the T floor then makes such a cell a benign ~1 K.
    rho   = ifelse(rho > R(1.0e-35), rho, R(1.0e-35))   # ifelse: NaN>x is false
    e     = ifelse(e > tiny, e, tiny)                    # (max() would pass NaN through)
    HII_m = ifelse(HII_m > zero(R), HII_m, zero(R))
    H2I_m = ifelse(H2I_m > zero(R), H2I_m, zero(R))
    deuterium && (HDI_m = ifelse(HDI_m > zero(R), HDI_m, zero(R)))
    d    = rho / mh                       # network density (∝ n)
    z0   = R(z)                           # redshift at step BEGIN
    Hz0  = hubble_z_of(z0; hubble = hubble, Om = Om, OL = OL)
    # ȧ/a [1/s] for the ADIABATIC term: analytic by default, OR a caller-supplied value
    # (Enzo's own CosmologyComputeExpansionFactor at the step endpoints, ln(a1/a0)/Δt)
    # so the adiabatic integral matches the host's expansion EXACTLY.  (a1≈a0 on sub-
    # resolution steps → 0, i.e. no expansion: fine.)
    Hz_ad = isnan(adot_over_a) ? Hz0 : R(adot_over_a)
    # When the host supplies its expansion rate (cosmological one-zone use), evolve the
    # redshift ACROSS the macro-step inside the sub-cycle: z(t)=(1+z0)exp(-ȧ/a·t)−1.
    # The CMB Compton target T_cmb(z), the Compton coefficient, and the recombination
    # H(z) then track z continuously instead of being frozen at z0 — essential for the
    # host's large (CIC_MAXEXP) steps, accurate in both the Compton-locked (high-z) and
    # decoupled (low-z) limits.  When no rate is supplied (default), z is held at z0.
    evolve_z = !isnan(adot_over_a)
    yHeI = (one(R) - R(fh)) * d

    # advected species → y-convention number densities; nₑ = n_HII initially
    yHII  = HII_m / mh
    yH2I  = H2I_m / mh                     # = 2·n(H2)
    yHDI  = deuterium ? HDI_m / mh : zero(R)
    yHI   = max((R(fh)*rho - HII_m - H2I_m) / mh, tiny)
    yde   = yHII
    yHM   = tiny; yH2II = tiny
    yDI   = deuterium ? R(DTOH_SEED)*yHI  : zero(R)
    yDII  = deuterium ? R(DTOH_SEED)*yHII : zero(R)

    # CMB/Trad quantities are frozen across the sub-cycle unless the host hands us its
    # expansion rate (evolve_z).  Precompute them ONCE — the CMB temperature Tc, the
    # Compton coefficient c1, the Hubble rate, and the Trad-only rate coefficients
    # (β₁s, k27, k28, He Saha) — so the per-iteration `build_rates` skips ~5
    # transcendentals.  When evolve_z, they're recomputed from zt inside the loop.
    Tc0  = comp2_cmb(z0)
    c10  = comp1_cmb(z0)
    cr0  = cmb_rates(Tc0)

    # Dust: hoist the loop-invariant pieces out of the sub-cycle (same idea as the
    # cmb_rates hoist above).  k_lw (LW photodissociation × self-shielding × dust
    # attenuation) depends ONLY on the per-cell constants (G0, N_H, N_H2, Z_rel) →
    # fully invariant.  T_dust = T_dust_eq(G0, A_V, Tc) is invariant too whenever the
    # CMB is frozen (the common local-equilibrium case); it is refreshed inside the
    # loop only when evolve_z moves Tc.  This keeps ~1 exp + 2 pow (T_dust) and the
    # shielding sqrt/exp out of every sub-step, so the dust path does not reintroduce
    # the transcendentals the rate tables were built to remove.
    if dust
        n_H_phys = R(fh) * d
        k_lw0    = k_H2_LW_eff(R(G0), R(N_H2), R(N_H), R(Z_rel))   # invariant
        T_d0     = T_dust_eq(R(G0), R(A_V), Tc0)                   # invariant unless evolve_z
    else
        n_H_phys = zero(R); k_lw0 = zero(R); T_d0 = zero(R)
    end

    f    = R(dtfrac)                       # fraction-change tolerance (default 0.1 = 10%)
    ttot = zero(R)
    iter = 0
    electron_shorter_steps = 0
    h2_shorter_steps = 0
    electron_selected_steps = 0
    h2_selected_steps = 0
    minimum_electron_dt_fraction = one(R)
    minimum_h2_dt_fraction = one(R)
    minimum_h2_over_electron = typemax(R)
    while ttot < dt && iter < itcap        # itcap bounds THIS call (resumable: caller re-enters
        iter += 1                          # with the partial state + remaining dt for the stragglers)
        rem = dt - ttot

        # redshift at the current point in the sub-cycle (frozen at z0 unless the host
        # handed us its expansion rate, in which case z evolves across the macro-step).
        if evolve_z
            zt = (one(R) + z0) * exp(-Hz_ad * ttot) - one(R)
            Tc = comp2_cmb(zt); c1 = comp1_cmb(zt)
            Hz = hubble_z_of(zt; hubble = hubble, Om = Om, OL = OL)
            cr = cmb_rates(Tc)
        else
            zt = z0; Tc = Tc0; c1 = c10; Hz = Hz0; cr = cr0
        end

        # temperature from the current state (number densities; nH2=yH2I/2 etc.)
        T = gas_temperature(rho, e, yHI, yHII, yHeI/R(4), tiny, tiny, yde,
                            yHM, yH2I/R(2), yH2II/R(2); gamma = GAMMA_DEFAULT)
        Trad = Tc
        # rate coefficients: analytic fits (default/reference) or the opt-in log–log table.
        K  = rate_tables === nothing ? _build_rates_cr(T, yHI, Hz, cr; deuterium = deuterium) :
                                       table_rates(rate_tables, T, yHI, Hz, cr; deuterium = deuterium)
        # H- and H2+ are algebraic intermediaries, not persistent state. Refresh them
        # from the same state/rates used by cooling, chemistry heating, and timestep
        # estimates so re-entering `evolve_cell` is equivalent to internal refinement.
        yHM_rate, yH2II_rate =
            _molecular_intermediates(yHI, yHII, yde, yH2I, K)

        # Dust physics: per-sub-step rate coefficients (the T- and nₑ-dependent ones).
        # T_dust and k_lw are hoisted above; T_d is only refreshed here when evolve_z
        # moves the CMB temperature.  All zero / nothing when dust = false (no overhead).
        if dust
            T_d          = evolve_z ? T_dust_eq(R(G0), R(A_V), Tc) : T_d0
            k_h2d        = k_H2_dust(T, T_d, R(Z_rel))
            k_grr        = k_gr_recomb_HII(T, R(G0), R(Z_rel), yde)
            Gamma_pe_vol = Gamma_PE(T, R(G0), R(Z_rel), yde) * n_H_phys
            Lambda_gg_vol = Lambda_gr(T, T_d, n_H_phys, R(Z_rel))
            dust_rates   = (; k_h2d, k_grr, k_lw = k_lw0)
        else
            Gamma_pe_vol  = zero(R)
            Lambda_gg_vol = zero(R)
            # Use a zeros NamedTuple (NOT `nothing`) so `dust_rates` has ONE concrete type in
            # both branches.  `dust` is a runtime kwarg, so `nothing | NamedTuple` would be a
            # type-unstable Union that poisons network_step's return and boxes on the GPU
            # (gpu_gc_pool_alloc → InvalidIRError).  network_step multiplies these coefficients
            # in, so zeros are bit-identical to the `nothing` (no-dust) path, at zero cost.
            dust_rates    = (; k_h2d = zero(R), k_grr = zero(R), k_lw = zero(R))
        end

        # cooling rate (volumetric, signed) + temstart shutoff (no cooling at
        # the temperature floor — set to exactly 0, not a spurious-sign tiny).
        nHD  = yHDI / R(3)
        edot = cooling_edot(yHI, yHII, yHeI/R(4), yde, yH2I/R(2), nHD, T, zt;
                            nH = R(fh)*d, metals = metals, cool_tables = cool_tables,
                            Gamma_PE_vol = Gamma_pe_vol, Lambda_gr_vol = Lambda_gg_vol)
        # H₂ formation heating + dissociation cooling (uses the explicitly-evolved
        # H⁻/H₂⁺; yH2II carries the 2× convention → n(H₂⁺)=yH2II/2, n(H⁻)=yHM).
        edot += _h2_chem_heat(yHI, yHII, yde, yH2I/R(2),
                              yHM_rate, yH2II_rate/R(2), R(fh)*d,
                              K.k8, K.k10, K.k11, K.k12, K.k13, K.k22)
        if T <= R(1.01)*R(MIN_TEMPERATURE) && edot < zero(R)
            edot = zero(R)
        end
        # adiabatic Hubble cooling: de/dt = -2H·e (γ=5/3); volumetric = ×ρ.
        # Only active when the host hands adiabatic cooling to the kernel.
        if hubble_expansion
            edot -= R(2) * Hz_ad * e * rho
        end

        dedot, _ = _de_hi_dot(yHI, yHII, yde, yH2I, yHM_rate, yH2II_rate,
                              yHeI, tiny, tiny, K)
        dt_electron = _step_f(yde, dedot, f)
        dt_h2 = if SL === :h2 || D
            h2dot = _h2_dot(yHI, yHII, yde, yH2I, yHM_rate, yH2II_rate,
                            yHeI/R(4), K;
                            k_h2d=dust_rates.k_h2d, k_lw=dust_rates.k_lw)
            h2_reference = max(yH2I, R(h2_fraction_floor) * R(fh) * d)
            _step_f(h2_reference, h2dot, f)
        else
            typemax(R)
        end
        # `network_step` solves the stiff atomic HI/HII pair conservatively with a
        # positivity-preserving backward-Euler update. Do not apply an additional
        # explicit fractional limiter to HI. The alternate compile-time selectors are
        # for convergence diagnosis; `:electron` remains the validated default.
        dt_species = if SL === :electron
            dt_electron
        elseif SL === :h2
            dt_h2
        elseif SL === :none
            typemax(R)
        else
            error("unknown chemistry species limiter")
        end
        dt_macro = min(rem, R(max_substep_fraction)*dt)
        # Resolve a moderately stiff H₂ formation/dissociation pair even when
        # its net rate is near zero. A net fractional limiter cannot see this
        # equilibrium cancellation, yet freezing nHI and T across the ordered
        # sweep gives an O(νΔt) trajectory error. Extremely stiff spans retain
        # the direct conservative equilibrium closure in network_step instead.
        h2_pair_frequency = R(2)*K.k22*yHI^2 + K.k13*yHI +
                            K.k11*yHII + K.k12*yde
        h2_source = R(2)*(K.k8*yHM_rate*yHI +
                          K.k10*yH2II_rate*yHI/R(2) +
                          K.k19*yH2II_rate*yHM_rate/R(2) +
                          K.kHeH_H*equilibrium_HeH(yHeI/R(4), yHII, yde, yHI,
                                                  K.kHeH_ra, K.kHeH_H,
                                                  K.kHeH_e, K.gamma_HeH)*yHI +
                          K.k22*yHI*yHI^2) +
                    R(2)*dust_rates.k_h2d*yHI^2
        h2_loss = K.k13*yHI + K.k11*yHII + K.k12*yde +
                  dust_rates.k_lw
        direct_h2_pair = SP &&
            _use_threebody_h2_equilibrium(yHI, yH2I, K.k22, K.k13,
                                          h2_source, h2_loss, dt_macro)
        dt_h2_pair = SP && !direct_h2_pair ?
                     R(0.2) / max(h2_pair_frequency, tiny) : typemax(R)
        dtit = min(dt_species, dt_h2_pair, dt_macro)

        # energy sub-step + CMB-Compton stiffness split
        edot_c    = -c1 * (T - Tc) * yde            # Compton part (volumetric)
        edot_rest = edot - edot_c
        Kc        = c1 * yde * (T / e) / rho        # specific Compton frequency
        stiff     = Kc * rem > one(R)
        de_spec   = (stiff ? edot_rest : edot) / rho
        dt_energy = _step_f(e, de_spec, f)
        dtit = min(dtit, dt_energy)

        if D
            electron_shorter_steps += dt_electron < dt_h2
            h2_shorter_steps += dt_h2 < dt_electron
            electron_selected_steps += dtit == dt_electron
            h2_selected_steps += dtit == dt_h2
            minimum_electron_dt_fraction =
                min(minimum_electron_dt_fraction, dt_electron / max(rem, eps(R)))
            minimum_h2_dt_fraction =
                min(minimum_h2_dt_fraction, dt_h2 / max(rem, eps(R)))
            minimum_h2_over_electron =
                min(minimum_h2_over_electron, dt_h2 / max(dt_electron, eps(R)))
        end

        # energy update
        if stiff
            B = (c1*yde*Tc + edot_rest) / rho       # specific source
            e = (e + B*dtit) / (one(R) + Kc*dtit)
        else
            e = e + (edot/rho)*dtit
        end
        e = max(e, tiny)

        # species update (one backward-Euler sweep)
        s = network_step(d, fh, yHI, yHII, yde, yH2I, yHM_rate, yH2II_rate,
                         yDI, yDII, yHDI, K, dtit; deuterium = deuterium,
                         dust_rates = dust_rates, intermediates_current=Val(true),
                         stiff_h2_pair=stiff_h2_pair)
        yHI=s.yHI; yHII=s.yHII; yde=s.yde; yH2I=s.yH2I; yHM=s.yHM
        yH2II=s.yH2II; yDI=s.yDI; yDII=s.yDII; yHDI=s.yHDI

        ttot = _advance_subcycle_time(ttot, dtit, dt)
    end

    if D
        return (; e, HII_m=yHII*mh, H2I_m=yH2I*mh,
                HDI_m=(deuterium ? yHDI*mh : HDI_m),
                consumed=ttot, iterations=iter, completed=ttot >= dt,
                electron_shorter_steps, h2_shorter_steps,
                electron_selected_steps, h2_selected_steps,
                minimum_electron_dt_fraction, minimum_h2_dt_fraction,
                minimum_h2_over_electron)
    end
    return e, yHII*mh, yH2I*mh, (deuterium ? yHDI*mh : HDI_m), ttot   # ttot = consumed (≤ dt)
end

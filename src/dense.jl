# dense.jl — continuation of the reduced network to high densities and temperatures
# (protostellar collapse: n ≳ 1e8 cm⁻³, T ≳ 2000 K).  Opt-in through
# `dense = Val(true)` in `evolve_cell` (and `dense = true` in `solve_chem!`); the
# default path is untouched and bit-identical.
#
# What the default network gets wrong there (measured, see test_dense_equilibrium.jl):
# its fixed point for H₂ ⇌ 2H is 2·k22·n_HI³ = k13·n_HI·n_H2 with the AAZN97 k13, which
# is the LOW-density dissociation rate (zero below 3480 K) and not the detailed-balance
# partner of k22 — so at n ≥ 1e14 the gas ends up fully molecular where Saha gives
# f_H2 = 0.02–0.8, and the sub-cycler hits its 5000-step cap.  H⁺ has no three-body
# recombination, so its fixed point is coronal (k1/k2) instead of Saha.
#
# The dense path adds (all pure, allocation-free, precision-generic, GPU-safe):
#  * K_H2(T) = n_H²/n_H2 and K_ion(T) = n_H⁺·n_e/n_H from partition functions
#    (ln forms, so nothing under/overflows in Float32).
#  * k13(T, n): log-interpolation between the low-density fit (rates_h2.jl) and the
#    LTE limit k22·K_H2 (detailed balance), MSM96 form with n_cr(T) — see `k13_dense`.
#  * k1r = k1/K_ion: three-body recombination H⁺ + 2e → H + e, the detailed-balance
#    inverse of collisional ionisation (`k1r_db`), with its 13.6 eV heating.
#  * `dense_equilibrium`: when the H₂ pair (and, if also stiff and three-body dominated
#    or under `planck_radiation`, the ionisation pair) relaxes much faster than the
#    step, place the state on the Saha manifold at the temperature that CONSERVES
#    thermal + chemical energy (a scalar root find), instead of sub-cycling thousands
#    of times.  This is the chemo-thermal coupling titan.jl's PrimordialChem.step_coupled
#    solves implicitly.
#  * Helium (planck_radiation): Saha at the gas temperature with n_e solved self-consistently
#    (`helium_selfconsistent` — the lagged scheme oscillates when He supplies the electrons),
#    inside the closure with its ionisation energy; `helium_cooling = Val(true)` adds the
#    Cen (1992) / Anninos et al. (1997) He cooling set (EmissionKernels.cooling_rate_helium).
#  * In evolve_cell (dense = Val(true)): the H₂ binding energy is charged from the actual
#    Δn_H₂ of each sub-step (the radiated, non-thermalised part of the formation energy
#    stays explicit); the energy limiter is divided by |d ln ė/d ln e| of the radiative
#    rate (Lyα: ≈ 13 at 9000 K); trace H₂ (≤ 1e-3 of H) is limited by its share of the
#    cooling instead of the 0.2/ν pair rule; with `planck_radiation` the electron limiter
#    is referenced to the Saha value.  Measured against dtfrac = 0.01 references on a
#    low-density set (n = 1e-4–1e4, T = 50–3e4 K, 1 Myr): p90 error in e 2.1 % (default
#    3.5 %), at 1.5–2× the default's cost per cell; at n ≥ 1e14 the closure replaces
#    10³–10⁴ sub-steps by 2.
#
# Constants: the equilibrium constants use the package constants (MH, KBOLTZ) and
# h = 6.62606896e-27 (CODATA 2006, matching them).  titan.jl uses the atomic-H mass
# 1.6735575e-24 g; K_H2 differs by (m ratio)^1.5 = 0.09 %.
#
# TO VERIFY against the sources before production use (flagged, not hidden):
#  * n_cr(T) of `k13_dense`: log10 n_cr = 3.0 − 0.416 x − 0.327 x², x = log10(T/1e4)
#    (Martin, Schwarz & Mandy 1996 as quoted by Glover & Abel 2008) — from memory.
#  * k1 (Janev et al. 1987 via AAZN97) is a GROUND-state ionisation fit, so k1r is the
#    recombination into the ground state only; the total three-body coefficient
#    (capture into excited levels, Hinnov & Hirschberg) is orders of magnitude larger.
#    The pair's equilibrium is Saha either way; only the relaxation time is slow.

export lnK_H2, K_H2, lnK_ion, K_ion, k13_dense, k1r_db, dense_equilibrium, helium_selfconsistent

const HPLANCK   = 6.62606896e-27                    # [erg s]
const _EV_ERG   = 1.602176634e-12                   # as fast.jl's _ERG_PER_EV
const THETA_ROT_H2 = 85.3                           # [K]  rigid rotor
const THETA_VIB_H2 = 6332.0                         # [K]  harmonic vibration
const D0_H2_K   = 4.478 * _EV_ERG / KBOLTZ          # [K]  dissociation energy (spectroscopic)
const CHI_H_K   = 13.598 * _EV_ERG / KBOLTZ         # [K]  ionisation energy
const E_H2_BIND = 4.48 * _EV_ERG                    # [erg] as in _h2_chem_heat
const E_H_ION   = 2.18e-11                          # [erg] 13.6 eV, as EmissionKernels.ciHI
const E_HEI_ION  = 3.94e-11                         # [erg] 24.59 eV, as EmissionKernels.ciHeI
const E_HEII_ION = 8.72e-11                         # [erg] 54.42 eV, as EmissionKernels.ciHeII
# ln of the translational factors (π m_H k/h²) and (2π m_e k/h²), times T^{3/2}
const _LN_TR_H2  = log(PI_G * MH * KBOLTZ / HPLANCK^2)
const _LN_TR_E   = log(2 * PI_G * ME * KBOLTZ / HPLANCK^2)

"""
    lnK_H2(T)

ln of the equilibrium constant of H₂ ⇌ 2H, K = n_H²/n_H₂ [cm⁻³]:
K = 16/(Z_rot Z_vib) (π m_H k T/h²)^{3/2} e^{−D0/kT} (electron spin 2 per atom, nuclear
spin 2 per atom; ortho:para weights 3:1 inside Z_rot).  Z_rot uses the Mulholland
expansion 2·(T/θ + 1/3 + θ/15T + 4θ²/315T²), equal to the 61-term weighted sum to
7e-4 at 300 K and < 1e-4 above 1000 K (4.5 % at 150 K, where the ortho/para alternation
is no longer negligible); K_H2 matters only where H₂ dissociates, T ≳ 1000 K.  Pure.
"""
@inline function lnK_H2(T)
    R = typeof(T)
    t = T / R(THETA_ROT_H2)
    Zr = R(2) * (t + R(1/3) + R(1/15) / t + R(4/315) / (t * t))
    lnZv = -log(-expm1(-R(THETA_VIB_H2) / T))                 # ln 1/(1 − e^{−θv/T})
    return R(log(16.0)) - log(Zr) - lnZv + R(1.5) * (R(_LN_TR_H2) + log(T)) - R(D0_H2_K) / T
end
@inline K_H2(T) = exp(lnK_H2(T))

"""
    lnK_ion(T)

ln of the Saha constant of H ⇌ H⁺ + e⁻, K = n_H⁺ n_e/n_H = (2π m_e k T/h²)^{3/2}
e^{−χ/kT} [cm⁻³] (statistical weights g_p g_e/g_H = 1·2/2; ground state only). Pure.
"""
@inline function lnK_ion(T)
    R = typeof(T)
    return R(1.5) * (R(_LN_TR_E) + log(T)) - R(CHI_H_K) / T
end
@inline K_ion(T) = exp(lnK_ion(T))

# Critical density of the H₂ + H dissociation rate (MSM96 form; TO VERIFY, see header).
@inline function _ln_ncr_h2h(T)
    R = typeof(T)
    x = log10(T / R(1e4))
    return R(log(10.0)) * (R(3.0) - R(0.416) * x - R(0.327) * x * x)
end

"""
    k13_dense(T, nH, k13_low, k22)

H₂ + H → 3H rate at H-nucleus density `nH` [cm⁻³]: log-interpolation between the
low-density coefficient `k13_low` (the network's k13) and the LTE limit k_H = k22·K_H₂
(the detailed-balance partner of the three-body formation rate, so the pair's fixed
point is the partition-function equilibrium):
    ln k13 = (ln k_L + (n/n_cr) ln k_H)/(1 + n/n_cr).
`k13_low` is floored at 1e-8·k_H (AAZN97's fit is exactly 0 below 3480 K; a log
interpolation needs a finite low end — 1e-8·k_H is dynamically irrelevant at n ≪ n_cr).
Pure.
"""
@inline function k13_dense(T, nH, k13_low, k22)
    R = typeof(T)
    lnkH = log(k22) + lnK_H2(T)
    lnkL = max(log(max(k13_low, zero(R))), lnkH - R(8 * log(10.0)))
    r = nH / exp(_ln_ncr_h2h(T))                                # n / n_cr
    return exp((lnkL + r * lnkH) / (one(R) + r))
end

# k1r below the Janev fit's validity: power law from [Tmin, 2 Tmin] (ChemNet.reverse_k rule).
const _K1R_TMIN     = 1.0e4
const _LN_K1R_TMIN  = log(k1(_K1R_TMIN)) - lnK_ion(_K1R_TMIN)
const _K1R_SLOPE    = (log(k1(2 * _K1R_TMIN)) - lnK_ion(2 * _K1R_TMIN) - _LN_K1R_TMIN) / log(2.0)

"""
    k1r_db(T, k1)

Three-body recombination H⁺ + e + e → H + e [cm⁶ s⁻¹] as the detailed-balance inverse
of the collisional ionisation coefficient `k1`: k1r = k1/K_ion, formed in logarithms.
Below 1e4 K (the fit's validity; k1/K would amplify the fit's error by e^{χ/kT}) the
exothermic reverse is extrapolated as the power law fitted on [1e4, 2e4] K
(slope $(round(_K1R_SLOPE; digits = 3))). Pure.
"""
@inline function k1r_db(T, k1v)
    R = typeof(T)
    if T >= R(_K1R_TMIN)
        k1v > zero(R) || return zero(R)
        return exp(log(k1v) - lnK_ion(T))
    else
        return exp(R(_LN_K1R_TMIN) + R(_K1R_SLOPE) * log(T / R(_K1R_TMIN)))
    end
end

"""
    helium_selfconsistent(she1, she2, k3, k4, k5, k6, ne_other, nHe) -> (nHeI, nHeII, nHeIII, ne)

Helium ionisation equilibrium (`helium_equilibrium`) with the electron density solved
self-consistently: n_e = ne_other + n_HeII + 2 n_HeIII, `ne_other` the electrons from the
hydrogen species (H⁺ + H₂⁺ − H⁻).  The network's semi-implicit He update off the OLD n_e is
fine while hydrogen supplies the electrons (recombination epoch), but with the Saha factors
at the gas temperature and weakly ionised hydrogen (dense collapse, x_H⁺ ~ 1e-8) helium
supplies its own electrons and the lagged iteration oscillates between ~40 % and ~1e-6
ionised.  g(n_e) = n_e − ne_other − n_HeII(n_e) − 2 n_HeIII(n_e) is monotone: 40 bisections
on [ne_other, ne_other + 2 nHe].  Pure, fixed work (GPU-safe).
"""
@inline function helium_selfconsistent(she1, she2, k3, k4, k5, k6, ne_other, nHe)
    R = typeof(nHe)
    neo = max(ne_other, zero(R))
    lo = neo; hi = neo + R(2) * nHe
    nHeI = nHe; nHeII = zero(R); nHeIII = zero(R); ne = lo
    for _ in 1:40
        ne = R(0.5) * (lo + hi)
        nHeI, nHeII, nHeIII = helium_equilibrium(she1, she2, k3, k4, k5, k6, ne, nHe)
        g = ne - neo - nHeII - R(2) * nHeIII
        g > zero(R) ? (hi = ne) : (lo = ne)
    end
    return nHeI, nHeII, nHeIII, neo + nHeII + R(2) * nHeIII
end

"""
    _thermal_energy(rho, T, n_no_h2, nH2tot)

Thermal energy per gram at `T`: e = k T (1.5 n_no_h2 + nH2tot·Γ_H₂⁻¹(T))/ρ with the H₂
rotation/vibration term of `gas_temperature` (same gating), evaluated at T itself.
`gas_temperature` evaluates that term at the UNCORRECTED temperature (AAZN97's one-pass
approximation, ~1 % at 3000–6000 K in molecular gas); `_temperature_dense` is the exact
inverse of this function and is what the dense path uses, so energy conservation in the
closure is exact.  Pure.
"""
@inline function _thermal_energy(rho, T, n_no_h2, nH2tot)
    R = typeof(T)
    GammaH2Inv = R(2.5)
    if nH2tot > R(1.0e-3) * n_no_h2
        x = R(6100.0) / T
        if x < R(10.0)
            ex = exp(x)
            GammaH2Inv = R(0.5) * (R(5) + R(2) * x * x * ex / (ex - one(R))^2)
        end
    end
    return R(KBOLTZ) * T * (n_no_h2 * R(1.5) + nH2tot * GammaH2Inv) / rho
end

# Saha composition of the hydrogen budget C (number density of H nuclei in H, H₂, and,
# when `ion`, H⁺) at temperature T.  `e_o` = electrons not from H⁺ (He, H₂⁺, −H⁻), fixed.
# Returns (yHI, yH2I, yHII) in the network convention (yH2I = 2 n_H₂).  Pure.
@inline function _saha_composition(C, e_o, T, lnKh2, lnKion, ion::Bool)
    R = typeof(T)
    z = zero(R)
    C > z || return z, z, z
    if !ion
        # closed form: nHI² = K yH2/2, nHI + yH2 = C  (as _threebody_h2_equilibrium with k22 = 1, k13 = K)
        q = exp(clamp(log(R(8) * C) - lnKh2, R(-80), R(80)))       # 8C/K
        nHI = R(2) * C / (one(R) + sqrt(one(R) + q))
        return nHI, max(C - nHI, z), z
    end
    # with H⁺: g(nHI) = nHI + yH2(nHI) + yHII(nHI) − C, monotone; bisection in ln nHI
    lo = log(C) - R(60); hi = log(C)
    nHI = C; yH2 = z; yHII = z
    for _ in 1:50
        m = R(0.5) * (lo + hi)
        nHI = exp(m)
        yH2 = exp(clamp(R(log(2.0)) + R(2) * m - lnKh2, R(-80), R(80)))
        s = exp(clamp(R(0.5) * (R(log(4.0)) + lnKion + m), R(-80), R(80)))   # sqrt(4 K n_HI)
        yHII = R(0.5) * s * (s / (e_o + hypot(e_o, s)))                        # 2Kn/(e_o + sqrt(e_o² + 4Kn))
        g = nHI + yH2 + yHII - C
        g > z ? (hi = m) : (lo = m)
    end
    yH2 = min(yH2, C); yHII = min(yHII, max(C - yH2, z))
    return max(C - yH2 - yHII, z), yH2, yHII
end

"""
    _temperature_dense(rho, e, n_no_h2, nH2tot)

Exact inverse of `_thermal_energy` (fixed-point iteration on the H₂ γ term, contraction
≤ 0.2, 12 iterations), started from the `gas_temperature`-style one-pass estimate.
Identical to `gas_temperature` whenever the H₂ term is gated off.  Pure.
"""
@inline function _temperature_dense(rho, e, n_no_h2, nH2tot)
    R = typeof(e)
    kB = R(KBOLTZ)
    n_tot = n_no_h2 + nH2tot
    (rho > zero(R) && n_tot > zero(R)) || return R(MIN_TEMPERATURE)
    T = clamp(rho * e / (kB * (R(1.5) * n_no_h2 + R(2.5) * nH2tot)), one(R), R(1.0e9))
    nH2tot > R(1.0e-3) * n_no_h2 || return max(T, R(MIN_TEMPERATURE))
    for _ in 1:12
        x = R(6100.0) / T
        GammaH2Inv = R(2.5)
        if x < R(10.0)
            ex = exp(x)
            GammaH2Inv = R(0.5) * (R(5) + R(2) * x * x * ex / (ex - one(R))^2)
        end
        T = clamp(rho * e / (kB * (R(1.5) * n_no_h2 + GammaH2Inv * nH2tot)), one(R), R(1.0e9))
    end
    return max(T, R(MIN_TEMPERATURE))
end

"""
    dense_equilibrium(rho, e_tot, C, e_o, n_other, nH2II, T0, ion::Bool)
        -> (e, yHI, yH2I, yHII, yde, T)

Energy-conserving equilibrium closure.  `C` = H nuclei [cm⁻³] in the pair(s) to be
equilibrated (H + H₂, plus H⁺ when `ion`); `e_o` = electrons not from H⁺; `n_other` =
other particles (He stages, H⁻, H⁺ when !ion, electrons when !ion … everything that is
neither the pair's H, H₂ nor, when `ion`, its H⁺/e⁻); `nH2II` = n(H₂⁺) (for the H₂ γ).
`e_tot` = thermal + chemical energy per gram, with the chemical energy −4.48 eV per H₂
and (when `ion`) +13.6 eV per H⁺ (as the network's heating terms).  Finds T with
e_th(T, y(T)) + e_chem(y(T)) = e_tot by bracketing + 40 bisections in ln T from `T0`
(monotone: both terms increase with T), y(T) the Saha composition.  Pure; fixed work.
"""
# Residual of the energy equation at trial temperature T (a plain function, not a closure:
# a closure capturing `R = typeof(e_tot)` as a value makes every `R(…)` inside it a dynamic
# call, which de-infers evolve_cell's state (15× slower on the CPU, no GPU compile)).
# With `nHe > 0` (dense + planck_radiation) helium is part of the equilibrium: its stages at the
# trial T (Saha at T plus the frozen collisional k3–k6) with n_e solved self-consistently,
# its electrons in the H⁺ Saha and in the particle count, and its ionisation energy
# (24.59 eV per He⁺, 79.0 eV per He²⁺) in the budget.  `ne_h` = electrons from H⁺ (+e_o)
# of the outer state for the He solve (one lag inside the closure; the energy charged is
# exact for the stages actually returned).
@inline function _dense_residual(T, rho, e_tot, C, e_o, n_other, nH2II, ion::Bool,
                                 nHe, k3, k4, k5, k6, ne_h)
    R = typeof(T)
    nHeII = zero(R); nHeIII = zero(R)
    if nHe > zero(R)
        she1, she2 = helium_saha_pair(T)
        _, nHeII, nHeIII, _ = helium_selfconsistent(she1, she2, k3, k4, k5, k6, ne_h, nHe)
    end
    nHeE = nHeII + R(2) * nHeIII
    e_o_eff = e_o + nHeE
    yHI, yH2, yHII = _saha_composition(C, e_o_eff, T, lnK_H2(T), lnK_ion(T), ion)
    n_no_h2 = n_other + yHI + yHII + (ion ? yHII + e_o_eff : nHeE)
    eth = _thermal_energy(rho, T, n_no_h2, yH2 / R(2) + nH2II)
    f = eth - R(E_H2_BIND) * (yH2 / R(2)) / rho + (ion ? R(E_H_ION) * yHII / rho : zero(R)) +
        (R(E_HEI_ION) * nHeII + (R(E_HEI_ION) + R(E_HEII_ION)) * nHeIII) / rho - e_tot
    return f, yHI, yH2, yHII, nHeII, nHeIII
end

@inline function dense_equilibrium(rho, e_tot, C, e_o, n_other, nH2II, T0, ion::Bool,
                                   nHe = zero(typeof(e_tot)), k3 = zero(typeof(e_tot)), k4 = zero(typeof(e_tot)),
                                   k5 = zero(typeof(e_tot)), k6 = zero(typeof(e_tot)), ne_h = zero(typeof(e_tot)))
    R = typeof(e_tot)
    res(T) = _dense_residual(T, rho, e_tot, C, e_o, n_other, nH2II, ion, nHe, k3, k4, k5, k6, ne_h)
    lo = T0 / R(3); hi = T0 * R(3)
    for _ in 1:12
        (res(lo)[1] > zero(R) && lo > R(1.0)) || break
        lo /= R(3)
    end
    for _ in 1:12
        (res(hi)[1] < zero(R) && hi < R(1.0e9)) || break
        hi *= R(3)
    end
    llo = log(lo); lhi = log(hi)
    for _ in 1:40
        m = R(0.5) * (llo + lhi)
        res(exp(m))[1] > zero(R) ? (lhi = m) : (llo = m)
    end
    T = max(exp(R(0.5) * (llo + lhi)), R(MIN_TEMPERATURE))
    _, yHI, yH2, yHII, nHeII, nHeIII = res(T)
    nHeE = nHeII + R(2) * nHeIII
    e = e_tot + R(E_H2_BIND) * (yH2 / R(2)) / rho - (ion ? R(E_H_ION) * yHII / rho : zero(R)) -
        (R(E_HEI_ION) * nHeII + (R(E_HEI_ION) + R(E_HEII_ION)) * nHeIII) / rho
    return max(e, R(_SUB_TINY)), yHI, yH2, yHII, (ion ? yHII + e_o + nHeE : zero(R)), T, nHeII, nHeIII
end

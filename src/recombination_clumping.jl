# recombination_clumping.jl — density-dependent Lyα-mixing recombination.
#
# Extends the Peebles C-factor in `peebles_k2` to account for small-scale baryon
# clumping: the Lyα escape rate R_α (Sobolev escape integral) depends on the
# neutral density *averaged over the Lyα mean-free-path volume* rather than the
# local cell density.  The host MHD code supplies that mean density as a per-cell
# field; we interpolate with mixing fraction f_α(z) from a user table.
#
# Core change (Eq. ★ from the brief):
#   n1s_eff = f_α · n_smoothed + (1-f_α) · n_local
# Replace n_local → n1s_eff in the Λ_2γ term (KL) of the Peebles C-factor; keep
# n_local in the β_e photoionisation term (KB) — "only the escape is non-local".
#
# When f_α = 0 (FA_ZERO default): n1s_eff = n_local → peebles_k2_mixing is
# *bit-identical* to peebles_k2; solve_chem_mixing! is bit-identical to solve_chem!
#
# Rate backend: RecFast analytic α_B (first cut). The `recfast_alpha` function is
# the seam; swap in a LogTable of HyRec values without touching the kernel.
#
# Reference: Jedamzik, Abel & Ali-Haïmoud (2025); see also Peebles (1968),
# Ma & Bertschinger (1995), and recombination.jl for the base RECFAST constants.

export recfast_gauss_factor, recfast_v2_kl_factor, peebles_k2_mixing, n1s_effective
export build_rates_mixing, table_rates_mixing, evolve_cell_mixing
export solve_chem_mixing!, solve_chem_mixing_device!
export evolve_cell_analytic_mixing
export solve_chem_analytic_mixing!, solve_chem_analytic_mixing_device!

# recfast_alpha is defined in recombination.jl (loaded before this file).

# ── RECFAST fudge + v2 Gaussian correction ───────────────────────────────────
#
# The RECFAST recombination "fudge" is NOT a correction to the Λ₂γ escape rate.
# Verified against the canonical codes (HyRec-2 hydrogen.c::rec_TLA_dxHIIdlna and
# CAMB recfast.f90::ION):
#
#   * The fudge `fu` multiplies the case-B recombination coefficient α_B.
#     In the Peebles C-factor it appears as
#         C_eff = fu·(1 + KL) / (1 + KL + fu·KB)
#     i.e. it scales the whole rate and the photoionization term KB in the
#     denominator — NOT the Λ₂γ term KL.  (HyRec puts `Fudge·α_B` on both the
#     recombination prefactor and β; CAMB folds the same `fu` into the C-factor.
#     The two forms are algebraically identical: k2 = fu·α_B·(1+KL)/(1+KL+fu·KB).)
#   * RECFAST v1 uses fu = 1.14 (flat).  RECFAST v2 (CAMB Hswitch=True) uses
#     fu = 1.125 PLUS a multiplicative double-Gaussian correction `gauss(z)` on
#     the Lyα escape factor K = (λ³/8πH)·gauss — which scales BOTH KL and KB.
#
# HyRec's own PEEBLES mode (fu=1) reproduces our previous "v1" error profile
# (+8% at z=700, falling to <1% at z=1100): that growing low-z tail is the
# intrinsic error of the three-level atom, not a bug.  Applying fu=1.14 to α_B
# collapses it to <1.5% everywhere — that is the physically correct fix.

const _RECFAST_V1_FUDGE = 1.14    # RECFAST v1 (flat fudge on α_B)
const _RECFAST_V2_FUDGE = 1.125   # RECFAST v2 (fudge on α_B; Gaussian on K)

"""
    recfast_gauss_factor(z) -> floating-point type of `z`

RECFAST v2 multiplicative correction to the Lyα escape factor K (CAMB
`Hswitch=True`; recfast.f90 line `K = CK/Hz*(1 + AGauss1·… + AGauss2·…)`):

  gauss = 1 + G₁(z) + G₂(z)
  G₁ = -0.14  × exp(-((ln(1+z) - 7.28) / 0.18)²)   [peak z ≈ 1449]
  G₂ =  0.079 × exp(-((ln(1+z) - 6.73) / 0.33)²)   [peak z ≈  836]

This scales K — and therefore BOTH KL and KB in the Peebles C-factor — bringing
x_e(z) within ~0.1-0.3% of HyRec. Returns 1.0 for RECFAST v1 (no Hswitch).
"""
@inline function recfast_gauss_factor(z::Real)
    zr = float(z)
    R = typeof(zr)
    lnzp1 = log(one(R) + zr)
    d1 = (lnzp1 - R(7.28)) / R(0.18)
    d2 = (lnzp1 - R(6.73)) / R(0.33)
    g1 = -R(0.14) * exp(-(d1*d1))
    g2 =  R(0.079) * exp(-(d2*d2))
    return one(R) + g1 + g2
end

# Backward-compatible alias (deprecated): the old name implied the correction
# applied to the Λ₂γ "KL" term, which was incorrect.  Kept so external callers
# don't break; returns the K-factor Gaussian correction.
@inline recfast_v2_kl_factor(z::Real) = recfast_gauss_factor(z)

# ── Eq. ★ : effective neutral density ────────────────────────────────────────

"""
    n1s_effective(nHI_local, n_smoothed, Xe_mean, f_alpha, ::Val{SN}) -> same units

Effective neutral-H number density for the Sobolev escape integral (Eq. ★):
    n1s_eff = f_α · n1s_smoothed + (1-f_α) · n1s_local.

`n_smoothed` is the smoothed H field from the host, interpreted as:
  - SN=true  : already the smoothed *neutral* density  (n1s_smoothed = n_smoothed)
  - SN=false : total smoothed H density; approximate  n1s_smoothed ≈ n_smoothed·(1−Xe_mean)
    using the global mean ionisation fraction Xe_mean (accurate because x_e varies
    slowly across the mixing length near the recombination epoch).

Implemented as a single fused multiply-add: branch-free in the hot path. Pure.
"""
@inline function n1s_effective(nHI_local::T, n_smoothed::T, Xe_mean::T, f_alpha::T,
                               ::Val{SN}) where {T,SN}
    n1s_sm = SN ? n_smoothed : n_smoothed * (one(T) - Xe_mean)
    return muladd(f_alpha, n1s_sm - nHI_local, nHI_local)
end

# ── Generalised Peebles k2 ────────────────────────────────────────────────────

"""
    peebles_k2_mixing(T, nHI_local, nHI_eff, Hz; fudge=1, gauss=1) -> k2 [cm³/s]

CaseB H recombination rate with the RECFAST Peebles C-factor, using an effective
neutral density `nHI_eff` [cm⁻³] for the Λ_2γ escape term (KL) but keeping the
local density `nHI_local` for the β_e photoionisation term (KB):

    K  = gauss · λ³/(8π·Hz)                  (Lyα Sobolev escape; gauss = v2 correction)
    KL = K · Λ_2γ · n1s_eff                  (mixing density; non-local escape)
    KB = K · β_e  · n1s_local                (local density for β_e)
    C  = fudge · (1 + KL) / (1 + KL + fudge·KB)
    k2 = α_B · C                             [cm³/s]

This is the exact RECFAST recombination coefficient (verified against HyRec-2
`rec_TLA_dxHIIdlna` and CAMB `recfast.f90`):
  * `fudge` (fu) multiplies α_B — it scales the whole rate and the KB term in
    the C-factor denominator, NOT the Λ₂γ term KL.  fudge=1 → pure Peebles
    (HyRec PEEBLES); 1.14 → RECFAST v1; 1.125 → RECFAST v2 base.
  * `gauss` is the multiplicative Gaussian correction on the Lyα K-factor
    (CAMB Hswitch; `recfast_gauss_factor(z)`), scaling BOTH KL and KB. =1 for v1.

When nHI_eff = nHI_local and fudge = gauss = 1 this is bit-identical to
`peebles_k2`. Pure.
"""
@inline function peebles_k2_mixing(T::Real, nHI_local::Real, nHI_eff::Real, Hz::Real;
                                   fudge::Real = one(typeof(T)),
                                   gauss::Real = one(typeof(T)))
    R = typeof(T)
    aB           = recfast_alpha(T)
    fu           = R(fudge)
    n1s_local_m3 = R(nHI_local) * R(1.0e6)        # cm⁻³ → m⁻³
    n1s_eff_m3   = R(nHI_eff)   * R(1.0e6)
    bet = aB * (R(_REC_CR) * T)^R(1.5) * exp(-R(_REC_CDB) / T)
    K   = R(gauss) * R(_REC_LAM)^3 / (R(8.0) * R(π) * R(Hz))  # v2 Gaussian scales K
    KL  = K * R(_REC_A8) * n1s_eff_m3
    KB  = K * bet        * n1s_local_m3
    C   = fu * (one(R) + KL) / (one(R) + KL + fu * KB)        # fudge on α_B (RECFAST)
    return aB * R(1.0e6) * C                                  # m³/s → cm³/s
end

# ── Rate assembler (identical to build_rates but k2 uses mixing) ─────────────

"""
    build_rates_mixing(T, Trad, nHI, nHI_eff, Hz; fudge=1, gauss=1, deuterium=false) -> NamedTuple

Like `build_rates` but substitutes `peebles_k2_mixing(T, nHI, nHI_eff, Hz; fudge, gauss)`
for k2. All other rates are identical. Pure.
"""
@inline function build_rates_mixing(T, Trad, nHI, nHI_eff, Hz;
                                    fudge::Real = one(typeof(T)),
                                    gauss::Real = one(typeof(T)),
                                    deuterium::Bool = false)
    R      = typeof(T)
    k2_val = peebles_k2_mixing(T, nHI, nHI_eff, Hz; fudge=fudge, gauss=gauss)
    # β₁s = CMB photoionisation of H(1s): evaluate at Trad (see build_rates) so it does
    # NOT spuriously Saha-ionise UV-heated low-z gas where T≫Trad.
    k_b1s  = beta1s_freq(Trad) * k2_val / (recfast_alpha(T) * R(1.0e6))
    she1, she2 = helium_saha_pair(Trad)
    kHeH_ra = kHeH_ra_spont(T) +
              kHeH_ra_stim_base(T) * HeH_stim_factor(Trad)
    base = (; k1=k1(T), k2=k2_val,
            k3=k3(T), k4=k4(T), k5=k5(T),
            k6=k6(T), k7=k7(T), k8=k8(T), k9=k9(T), k10=k10(T), k11=k11(T),
            k12=k12(T), k13=k13(T), k14=k14(T), k15=k15(T), k16=k16(T), k17=k17(T),
            k18=k18(T), k19=k19(T), k22=k22(T), k57=k57(T), k58=k58(T),
            k27=k27_cmb(Trad), k28=k28_cmb(Trad), k_beta1s=k_b1s,
            kHeH_ra=kHeH_ra, kHeH_H=kHeH_H(T), kHeH_e=kHeH_e(T),
            gamma_HeH=gamma_HeH_cmb(Trad),
            she1=she1, she2=she2)
    deuterium || return base
    return merge(base, (; k50=k50(T), k51=k51(T), k52=k52(T), k53=k53(T),
                        k54=k54(T), k55=k55(T), k56=k56(T)))
end

"""
    table_rates_mixing(rt, T, nHI, nHI_eff, Hz, cr; fudge=1, gauss=1)

Table-backed counterpart of [`build_rates_mixing`](@ref). Pure-temperature rates
come from `rt`; the density-dependent Peebles factor is rebuilt per cell using
the local and smoothed neutral densities. This is the production GPU hot path.
"""
@inline function table_rates_mixing(rt::RateTables, T, nHI, nHI_eff, Hz, cr;
                                    fudge::Real = one(typeof(T)),
                                    gauss::Real = one(typeof(T)))
    R = typeof(T)
    s = (log10(T) - R(rt.x0)) * R(rt.invdx)
    smax = R(rt.N - 1)
    s = clamp(s, zero(R), smax - eps(R)*smax)
    b = unsafe_trunc(Int, s)
    f = s - R(b)
    i = b + 1
    L = rt.logk
    N = rt.N
    @inline rd(c) = _rt_lookup(L, N, i, f, c)

    aB = rd(1)
    bet = rd(2)
    fu = R(fudge)
    Kf = R(gauss) * R(_REC_LAM)^3 / (R(8.0) * R(pi) * Hz)
    KL = Kf * R(_REC_A8) * nHI_eff * R(1.0e6)
    KB = Kf * bet * nHI * R(1.0e6)
    C = fu * (one(R) + KL) / (one(R) + KL + fu * KB)
    k2_val = aB * R(1.0e6) * C
    k_b1s = cr.b1s * k2_val / (aB * R(1.0e6))
    she1, she2 = cr.she

    Tev = T / R(11605.0)
    k3v  = Tev <= R(0.8)  ? zero(R) : rd(4)
    k5v  = Tev <= R(0.8)  ? zero(R) : rd(6)
    k11v = Tev <= R(0.3)  ? zero(R) : rd(12)
    k12v = Tev <= R(0.3)  ? zero(R) : rd(13)
    k13v = Tev <= R(0.3)  ? zero(R) : rd(14)
    k14v = Tev <= R(0.04) ? zero(R) : rd(15)
    kHeH_ra = rd(31) + rd(32) * cr.HeH_stim
    return (; k1=rd(3), k2=k2_val, k3=k3v, k4=rd(5), k5=k5v, k6=rd(7),
            k7=rd(8), k8=rd(9), k9=rd(10), k10=rd(11), k11=k11v, k12=k12v,
            k13=k13v, k14=k14v, k15=rd(16), k16=rd(17), k17=rd(18),
            k18=rd(19), k19=rd(20), k22=rd(21), k57=rd(22), k58=rd(23),
            k27=cr.k27, k28=cr.k28, k_beta1s=k_b1s,
            kHeH_ra=kHeH_ra, kHeH_H=rd(33), kHeH_e=rd(34),
            gamma_HeH=cr.gamma_HeH, she1=she1, she2=she2)
end

# ── Fast analytic Ly-alpha mixing path ───────────────────────────────────────

"""
    evolve_cell_analytic_mixing(rho, e, HII_m, H2I_m, n_sm_cgs, dt, z; ...)

Closed-form primordial H/H2 chemistry with density-dependent Ly-alpha escape.
This is the recombination-era counterpart of [`evolve_cell_analytic`](@ref): it
keeps the analytic Compton, HII Riccati, and H2 updates, while replacing the
local Peebles coefficient with [`peebles_k2_mixing`](@ref). `n_sm_cgs` is the
host-supplied smoothed H number density, or smoothed neutral-H number density
when `smoothed_is_neutral=Val(true)`.

The routine is pure and allocation-free. With `f_alpha=0`, `fudge=gauss=1`, it
reduces to `evolve_cell_analytic`.
"""
@inline function evolve_cell_analytic_mixing(rho, e, HII_m, H2I_m,
                                             n_sm_cgs, dt, z;
                                             f_alpha = zero(typeof(e)),
                                             Xe_mean = zero(typeof(e)),
                                             smoothed_is_neutral::Val{SN} = Val(false),
                                             fudge = one(typeof(e)),
                                             gauss = one(typeof(e)),
                                             hubble = 71.0, Om = 0.27, OL = 0.73,
                                             fh = FH_DEFAULT,
                                             hubble_expansion::Bool = false,
                                             adot_over_a = NaN,
                                             rate_tables = nothing,
                                             cool_tables = nothing,
                                             itcap::Int = _SUB_ITMAX,
                                             dtfrac::Real = 0.1) where {SN}
    R = typeof(e)
    # Preserve the exact no-mixing contract by using the canonical analytic
    # kernel when the escape correction is disabled. This also ensures that
    # improvements to its H2 and high-z helium closures automatically carry
    # into this entry point instead of being duplicated here.
    if R(f_alpha) == zero(R) && R(fudge) == one(R) && R(gauss) == one(R)
        en, hii, h2, ttot, _ = evolve_cell_analytic(
            rho, e, HII_m, H2I_m, dt, z;
            hubble=hubble, Om=Om, OL=OL, fh=fh,
            hubble_expansion=hubble_expansion, adot_over_a=adot_over_a,
            rate_tables=rate_tables, cool_tables=cool_tables,
            itcap=itcap, dtfrac=dtfrac)
        return en, hii, h2, ttot
    end
    mh = R(MH)
    zeroR = zero(R)
    d = rho / mh
    z0 = R(z)
    Hz0 = hubble_z_of(z0; hubble=hubble, Om=Om, OL=OL)
    Hz_ad = isnan(adot_over_a) ? Hz0 : R(adot_over_a)
    evolve_z = !isnan(adot_over_a)
    fhd = R(fh) * d

    yHeI = (one(R) - R(fh)) * d
    yHII = max(HII_m / mh, zeroR)
    yH2I = max(H2I_m / mh, zeroR)
    yHI = max(fhd - yHII - yH2I, zeroR)
    yde = yHII

    fa = R(f_alpha)
    Xem = R(Xe_mean)
    nsm = R(n_sm_cgs)
    fud = R(fudge)
    gss = R(gauss)
    Tc0 = comp2_cmb(z0)
    c10 = comp1_cmb(z0)
    f = R(dtfrac)
    ttot = zero(R)
    iter = 0

    @inbounds while ttot < dt && iter < itcap
        iter += 1
        rem = dt - ttot
        if evolve_z
            zt = (one(R) + z0) * exp(-Hz_ad * ttot) - one(R)
            Tc = comp2_cmb(zt)
            c1 = comp1_cmb(zt)
            Hz = hubble_z_of(zt; hubble=hubble, Om=Om, OL=OL)
        else
            zt = z0
            Tc = Tc0
            c1 = c10
            Hz = Hz0
        end

        T = gas_temperature(rho, e, yHI, yHII, yHeI/R(4), zeroR, zeroR, yde,
                            zeroR, yH2I/R(2), zeroR; gamma=GAMMA_DEFAULT)
        nHI_eff = n1s_effective(yHI, nsm, Xem, fa, smoothed_is_neutral)
        if rate_tables === nothing
            k2v = peebles_k2_mixing(T, yHI, nHI_eff, Hz; fudge=fud, gauss=gss)
            kb1s = beta1s_freq(Tc) * k2v / (recfast_alpha(T) * R(1.0e6))
            k1v = k1(T)
            k7v = k7(T)
            k8v = k8(T)
            k9v = k9(T)
            k10v = k10(T)
            k15v = k15(T)
            k22v = k22(T)
            k57v = k57(T)
            k58v = k58(T)
            k27v = k27_cmb(Tc)
            k28v = k28_cmb(Tc)
        else
            K = table_rates_mixing(rate_tables, T, yHI, nHI_eff, Hz, cmb_rates(Tc);
                                   fudge=fud, gauss=gss)
            k2v = K.k2
            kb1s = K.k_beta1s
            k1v = K.k1
            k7v = K.k7
            k8v = K.k8
            k9v = K.k9
            k10v = K.k10
            k15v = K.k15
            k22v = K.k22
            k57v = K.k57
            k58v = K.k58
            k27v = K.k27
            k28v = K.k28
        end

        edot = cooling_edot(yHI, yHII, yHeI/R(4), yde, yH2I/R(2), zero(R), T, zt;
                            nH=fhd, cool_tables=cool_tables)
        if T <= R(1.01)*R(MIN_TEMPERATURE) && edot < zero(R)
            edot = zero(R)
        end
        edot_c = -c1 * (T - Tc) * yde
        edot_rest = edot - edot_c
        hubble_expansion && (edot_rest -= R(2) * Hz_ad * e * rho)
        Kc = (e > zeroR && rho > zeroR) ? c1 * yde * (T / e) / rho : zeroR
        de_rest = edot_rest / rho

        dT = abs(Tc - T)
        exmax = dT > zeroR ? f*T/dT : one(R)
        dtc_c = (exmax < one(R) && Kc > zeroR) ? -log(one(R) - exmax)/Kc : typemax(R)
        dtc = min(_step_f(e, de_rest, f), dtc_c, rem)

        ex = -expm1(-Kc*dtc)
        g = Kc > zeroR ? ex/Kc : dtc
        e = e + (e*(Tc/T) - e)*ex + de_rest*g
        e = max(e, zeroR)

        ion_source = k57v*yHI*yHI + k58v*yHI*yHeI/R(4) + k1v*yHI*yde
        htot = max(fhd - yH2I, zeroR)
        yHII = _riccati_linear(yHII, k2v, kb1s, kb1s*htot + ion_source, dtc)

        nHM = _equilibrium_ratio(k7v*yHI*yde, (k8v + k15v)*yHI + k27v)
        nH2II = _equilibrium_ratio(k9v*yHI*yHII, k10v*yHI + k28v)
        yH2I += R(2)*(k8v*nHM*yHI + k10v*nH2II*yHI + k22v*yHI*yHI*yHI) * dtc

        yH2I = clamp(yH2I, zeroR, fhd)
        yHII = clamp(yHII, zeroR, max(fhd - yH2I, zeroR))
        yHI = max(fhd - yHII - yH2I, zeroR)
        yde = yHII
        ttot = _advance_subcycle_time(ttot, dtc, dt)
    end

    return e, yHII*mh, yH2I*mh, ttot
end

# ── Per-cell mixing subcycler ─────────────────────────────────────────────────

"""
    evolve_cell_mixing(rho, e, HII_m, H2I_m, HDI_m, n_sm_cgs, dt, z;
                       f_alpha, Xe_mean, smoothed_is_neutral, hubble, Om, OL,
                       fh, deuterium, diagnostics=Val(false))
        -> (e, HII_m, H2I_m, HDI_m, HeII_m)

Sub-cycle one cell over macro-step `dt` [s] with Lyα-mixing recombination.
Identical to `evolve_cell` except `build_rates_mixing` is called each substep with
`nHI_eff` computed from the per-cell smoothed density `n_sm_cgs` [cm⁻³].

`n_sm_cgs` is the smoothed H number density from the host (physical CGS). Interpreted
as total n_H when `smoothed_is_neutral=Val(false)` (default; approximates n1s via
global Xe_mean), or as n1s directly when `Val(true)`. `diagnostics=Val(true)`
returns a named tuple with the final state, consumed time, iteration count, and
timestep-limiter counters for scalar convergence analysis. Pure; allocation-free.
"""
@inline function evolve_cell_mixing(rho, e, HII_m, H2I_m, HDI_m,
                                    n_sm_cgs, dt, z;
                                    f_alpha    = zero(typeof(e)),
                                    Xe_mean    = zero(typeof(e)),
                                    smoothed_is_neutral::Val{SN} = Val(false),
                                    fudge      = one(typeof(e)),
                                    gauss      = one(typeof(e)),
                                    hubble = 71.0, Om = 0.27, OL = 0.73,
                                    fh = FH_DEFAULT,
                                    deuterium::Bool = false,
                                    helium::Bool = false,
                                    HeII_m = zero(typeof(e)),
                                    hubble_expansion::Bool = false,
                                    uvb::Bool = false,
                                    GamHI = zero(typeof(e)), GamHeI = zero(typeof(e)),
                                    GamHeII = zero(typeof(e)),
                                    piHI = zero(typeof(e)), piHeI = zero(typeof(e)),
                                    piHeII = zero(typeof(e)),
                                    metals = nothing,
                                    rate_tables = nothing,
                                    dtfrac::Real = 0.1,
                                    itcap::Int = _SUB_ITMAX,
                                    species_limiter::Val{SL} = Val(:gated_electron),
                                    h2_fraction_floor::Real = 1.0e-12,
                                    diagnostics::Val{D} = Val(false)) where {SN,SL,D}
    R    = typeof(e)
    mh   = R(MH); zeroR = zero(R)
    d    = rho / mh
    Hz   = hubble_z_of(R(z); hubble = hubble, Om = Om, OL = OL)
    Tc   = comp2_cmb(R(z))
    c1   = comp1_cmb(R(z))
    nHe4 = (one(R) - R(fh)) * d                # total He in ×4 (mass-equiv) convention
    nHe  = nHe4 / R(4)                          # total He number density [cm⁻³]
    nH_h = R(fh) * d                            # hydrogen number density [cm⁻³]
    fHe  = nH_h > zeroR ? nHe / nH_h : zeroR    # n_He/n_H
    # He⁺ number density carried as state (HeII_m is the He⁺ MASS density; He mass =
    # 4·m_H, so n(He⁺) = HeII_m/(4·m_H) and yHeII(×4) = HeII_m/m_H).
    nHeII = helium ? max((HeII_m / mh) / R(4), zeroR) : zeroR
    yHeI  = nHe4                                # neutral-He reservoir for cooling/T

    yHII  = max(HII_m / mh, zeroR)
    yH2I  = max(H2I_m / mh, zeroR)
    yHDI  = deuterium ? max(HDI_m / mh, zeroR) : zeroR
    yHI   = max((R(fh)*rho - HII_m - H2I_m) / mh, zeroR)
    yde   = yHII
    yHM   = zeroR; yH2II = zeroR
    yDI   = deuterium ? R(DTOH_SEED)*yHI  : zeroR
    yDII  = deuterium ? R(DTOH_SEED)*yHII : zeroR

    fa  = R(f_alpha)
    Xem = R(Xe_mean)
    nsm = R(n_sm_cgs)
    fud = R(fudge)
    gss = R(gauss)
    # UV-background photoionisation [s⁻¹] and photoheating [erg s⁻¹ per ion] for this
    # step (all 0 unless a UVB was supplied to solve_chem_mixing!).
    gHI = R(GamHI); gHeI = R(GamHeI); gHeII = R(GamHeII)
    pHI = R(piHI); pHeI = R(piHeI); pHeII = R(piHeII)
    cr = cmb_rates(Tc)
    frac = R(dtfrac)

    ttot = zero(R)
    iter = 0
    hi_formation_steps = 0
    hi_depletion_steps = 0
    hi_rate_sign_flips = 0
    previous_hi_sign = 0
    electron_limited_steps = 0
    h2_limited_steps = 0
    electron_shorter_steps = 0
    h2_shorter_steps = 0
    neutral_restrictive_steps = 0
    energy_limited_steps = 0
    remainder_limited_steps = 0
    halfstep_limited_steps = 0
    minimum_dt_fraction = one(R)
    minimum_electron_dt_fraction = one(R)
    minimum_h2_dt_fraction = one(R)
    minimum_h2_over_electron = typemax(R)
    minimum_neutral_fraction = yHI / max(nH_h, eps(R))
    maximum_neutral_fraction = minimum_neutral_fraction
    while ttot < dt && iter < itcap
        iter += 1
        rem = dt - ttot

        T = gas_temperature(rho, e, yHI, yHII, yHeI/R(4), zeroR, zeroR, yde,
                            yHM, yH2I/R(2), yH2II/R(2); gamma = GAMMA_DEFAULT)
        Trad = Tc

        # effective neutral density for the Sobolev escape rate
        nHI_eff = n1s_effective(yHI, nsm, Xem, fa, smoothed_is_neutral)
        K = rate_tables === nothing ?
            build_rates_mixing(T, Trad, yHI, nHI_eff, Hz;
                               fudge = fud, gauss = gss, deuterium = deuterium) :
            table_rates_mixing(rate_tables, T, yHI, nHI_eff, Hz, cr;
                               fudge = fud, gauss = gss)
        yHM_rate, yH2II_rate =
            _molecular_intermediates(yHI, yHII, yde, yH2I, K)

        # UV-background He photoionisation equilibrium (default He path).  Solve the
        # collisional-radiative + photo He equilibrium ONCE, up front, so this substep's
        # cooling, photoheating and electron balance all see the SAME He state (and we
        # hand it to network_step instead of letting it re-solve).  helium=true carries
        # its own advected He⁺ (Task B will add Γ there); with no UVB this branch is
        # skipped → the original default path is bit-identical.
        uvb_eq    = uvb && !helium
        nHeII_now = zero(R)
        yHeII_eq  = zero(R); yHeIII_eq = zero(R)
        if uvb_eq
            ne0 = max(yde, zeroR)
            _, nHeII_e, nHeIII_e =
                helium_equilibrium(K.she1, K.she2, K.k3, K.k4, K.k5, K.k6, ne0, nHe;
                                   GamHeI = gHeI, GamHeII = gHeII)
            yHeII_eq  = R(4) * nHeII_e                 # ×4 mass-equiv convention
            yHeIII_eq = R(4) * nHeIII_e
            yHeI      = max(nHe4 - yHeII_eq - yHeIII_eq, zeroR)
            nHeII_now = nHeII_e
        elseif helium
            nHeII_now = nHeII                          # carried (start-of-substep) He⁺
        end

        nHD  = yHDI / R(3)
        edot = cooling_edot(yHI, yHII, yHeI/R(4), yde, yH2I/R(2), nHD, T, R(z);
                            nH = R(fh)*d, metals = metals)
        if T <= R(1.01)*R(MIN_TEMPERATURE) && edot < zero(R)
            edot = zero(R)
        end
        # UV-background photoheating [erg cm⁻³ s⁻¹], a +edot source (energy per
        # photoionisation × number density of that species; n_HeI = yHeI/4).  Applied
        # after the floor shutoff so heating can lift gas off the temperature floor.
        edot += pHI*yHI + pHeI*(yHeI/R(4)) + pHeII*nHeII_now
        if hubble_expansion
            edot -= R(2) * Hz * e * rho
        end

        dedot, HIdot = _de_hi_dot(yHI, yHII, yde, yH2I, yHM_rate, yH2II_rate,
                                  yHeI, zeroR, zeroR, K; GamHI = gHI)
        dt_electron = _step_f(yde, dedot, frac)
        dt_h2 = if SL === :h2 || D
            h2dot = _h2_dot(yHI, yHII, yde, yH2I, yHM_rate, yH2II_rate,
                            yHeI/R(4), K)
            h2_reference = max(yH2I, R(h2_fraction_floor) * nH_h)
            _step_f(h2_reference, h2dot, frac)
        else
            typemax(R)
        end
        dt_neutral = _depletion_step_f(yHI, HIdot, frac)
        dt_halfstep = R(0.5) * dt

        edot_c    = -c1 * (T - Tc) * yde
        edot_rest = edot - edot_c
        Kc        = (e > zeroR && rho > zeroR) ? c1 * yde * (T / e) / rho : zeroR
        stiff     = Kc * rem > one(R)
        de_spec   = (stiff ? edot_rest : edot) / rho
        dt_energy = _step_f(e, de_spec, frac)
        # The atomic HI/HII update in `network_step` is already a conservative,
        # positivity-preserving backward-Euler solve. The explicit HI fractional
        # limiter is both redundant and pathological in Float32 when the equilibrium
        # neutral fraction lies below the `fh*rho - HII_m` subtraction quantum.
        # In nearly fully ionized atomic gas, enormous one-way ionization and
        # recombination rates cancel only in the coupled backward-Euler update.
        # Bypass the explicit electron limiter only when it would select a
        # pathological step in that regime. Retain it once neutral/molecular
        # chemistry becomes nonlinear.
        neutral_fraction = yHI / max(nH_h, eps(R))
        atomic_equilibrium = neutral_fraction <= R(1e-4) &&
                             dt_electron < R(0.01) * min(rem, dt_halfstep)
        dt_species = if SL === :gated_electron
            atomic_equilibrium ? typemax(R) : dt_electron
        elseif SL === :electron
            dt_electron
        elseif SL === :h2
            dt_h2
        elseif SL === :none
            typemax(R)
        else
            error("unknown mixing chemistry species limiter")
        end
        dtit = min(dt_species, dt_energy, rem, dt_halfstep)

        if D
            hi_sign = HIdot > zeroR ? 1 : (HIdot < zeroR ? -1 : 0)
            hi_formation_steps += hi_sign > 0
            hi_depletion_steps += hi_sign < 0
            hi_rate_sign_flips += previous_hi_sign != 0 && hi_sign != 0 &&
                                  hi_sign != previous_hi_sign
            hi_sign != 0 && (previous_hi_sign = hi_sign)
            electron_limited_steps += dtit == dt_electron
            h2_limited_steps += dtit == dt_h2
            electron_shorter_steps += dt_electron < dt_h2
            h2_shorter_steps += dt_h2 < dt_electron
            neutral_restrictive_steps += dt_neutral < dtit
            energy_limited_steps += dtit == dt_energy
            remainder_limited_steps += dtit == rem
            halfstep_limited_steps += dtit == dt_halfstep
            minimum_dt_fraction = min(minimum_dt_fraction, dtit / max(rem, eps(R)))
            minimum_electron_dt_fraction =
                min(minimum_electron_dt_fraction, dt_electron / max(rem, eps(R)))
            minimum_h2_dt_fraction =
                min(minimum_h2_dt_fraction, dt_h2 / max(rem, eps(R)))
            minimum_h2_over_electron =
                min(minimum_h2_over_electron, dt_h2 / max(dt_electron, eps(R)))
        end

        if stiff
            B = (c1*yde*Tc + edot_rest) / rho
            e = (e + B*dtit) / (one(R) + Kc*dtit)
        else
            e = e + (edot/rho)*dtit
        end
        e = max(e, zeroR)

        # ── Helium ionisation (evolved He⁺ with the full He I freeze-out) ─────
        # He³⁺/He²⁺ and the z≳3000 He⁺ plateau are fast ⇒ 3-level Saha (exact).
        # At z≲3000 (He²⁺≈0) He⁺→He⁰ freezes out ⇒ integrate He⁺ with the
        # HyRec He I rate (helium_HeI_rate_AB), backward-Euler.  Carried in nHeII.
        if helium
            if R(z) > R(3000.0)
                # Saha populations expressed as multiplication-only weights.  This
                # remains finite for an exactly neutral state (ne=0) and for rates
                # that have physically underflowed to zero.
                _, nHeII, nHeIII = helium_equilibrium(
                    K.she1, K.she2, zeroR, one(R), zeroR, one(R), yde, nHe)
            else
                if nH_h > zeroR
                    xHeII = nHeII / nH_h
                    A, B  = helium_HeI_rate_AB(Trad, nH_h, Hz, yHI/nH_h, xHeII, fHe)
                    nHeII = ((xHeII + A*dtit) / (one(R) + B*dtit)) * nH_h
                else
                    nHeII = zeroR
                end
                nHeIII = _equilibrium_ratio(nHeII * K.she2, yde) # He²⁺ Saha (≈0 here)
            end
            nHeIII = clamp(nHeIII, zeroR, max(nHe - nHeII, zeroR))
            nHeII = clamp(nHeII, zeroR, max(nHe - nHeIII, zeroR))
            yHeII_x  = R(4) * nHeII                            # ×4 mass-equiv convention
            yHeIII_x = R(4) * nHeIII
            yHeI     = max(nHe4 - yHeII_x - yHeIII_x, zeroR)  # for next substep's cooling/T
            s = network_step(d, fh, yHI, yHII, yde, yH2I, yHM_rate, yH2II_rate,
                             yDI, yDII, yHDI, K, dtit; deuterium = deuterium,
                             yHeII_in = yHeII_x, yHeIII_in = yHeIII_x, GamHI = gHI,
                             intermediates_current=Val(true))
        elseif uvb_eq
            # consume the up-front He equilibrium (already includes Γ_HeI/Γ_HeII)
            s = network_step(d, fh, yHI, yHII, yde, yH2I, yHM_rate, yH2II_rate,
                             yDI, yDII, yHDI, K, dtit; deuterium = deuterium,
                             yHeII_in = yHeII_eq, yHeIII_in = yHeIII_eq, GamHI = gHI,
                             intermediates_current=Val(true))
        else
            s = network_step(d, fh, yHI, yHII, yde, yH2I, yHM_rate, yH2II_rate,
                             yDI, yDII, yHDI, K, dtit; deuterium = deuterium,
                             GamHI = gHI, GamHeI = gHeI, GamHeII = gHeII,
                             intermediates_current=Val(true))
        end
        yHI=s.yHI; yHII=s.yHII; yde=s.yde; yH2I=s.yH2I; yHM=s.yHM
        yH2II=s.yH2II; yDI=s.yDI; yDII=s.yDII; yHDI=s.yHDI
        if D
            neutral_fraction = yHI / max(nH_h, eps(R))
            minimum_neutral_fraction = min(minimum_neutral_fraction, neutral_fraction)
            maximum_neutral_fraction = max(maximum_neutral_fraction, neutral_fraction)
        end

        # Enforce H-nuclei conservation (only with a UVB, so the validated no-UVB path
        # is bit-identical).  The operator-split, Gauss-Seidel backward-Euler updates HI
        # and HII as separate fixed points; under strong photoionisation their sum drifts
        # a few % from the true H budget.  Renormalise the H species (each y counts H
        # nuclei: yH2I=2·nH₂, yH2II=2·nH₂⁺) to the exact total — the standard
        # `make_consistent` step.  nₑ is re-derived from charge balance next substep.
        if uvb
            SH = yHI + yHII + yH2I + yHM + yH2II
            if SH > zeroR
                fH = (R(fh) * d) / SH
                yHI *= fH; yHII *= fH; yH2I *= fH; yHM *= fH; yH2II *= fH
            else
                yHI = R(fh) * d
                yHII = zeroR; yH2I = zeroR; yHM = zeroR; yH2II = zeroR
            end
        end

        ttot = _advance_subcycle_time(ttot, dtit, dt)
    end

    if D
        return (; e, HII_m=yHII*mh, H2I_m=yH2I*mh,
                HDI_m=(deuterium ? yHDI*mh : HDI_m),
                HeII_m=(helium ? R(4)*nHeII*mh : HeII_m),
                consumed=ttot, iterations=iter, completed=ttot >= dt,
                hi_formation_steps, hi_depletion_steps, hi_rate_sign_flips,
                electron_limited_steps, h2_limited_steps,
                electron_shorter_steps, h2_shorter_steps,
                neutral_restrictive_steps,
                energy_limited_steps, remainder_limited_steps,
                halfstep_limited_steps, minimum_dt_fraction,
                minimum_electron_dt_fraction, minimum_h2_dt_fraction,
                minimum_h2_over_electron,
                minimum_neutral_fraction, maximum_neutral_fraction)
    end
    return e, yHII*mh, yH2I*mh, (deuterium ? yHDI*mh : HDI_m),
           (helium ? R(4)*nHeII*mh : HeII_m)
end

# ── KA kernel ────────────────────────────────────────────────────────────────

@kernel function _evolve_analytic_mixing_k!(e, HII, H2I, @Const(rho), @Const(n_sm),
                                            nsm_scalar, du, vu2, tu, dt, z,
                                            f_alpha, Xe_mean, fudge, gauss,
                                            hubble, Om, OL, fh, hub_exp, aoa,
                                            rate_tables, cool_tables, dtfrac, itcap,
                                            ::Val{NS}, ::Val{SN}) where {NS,SN}
    i = @index(Global)
    @inbounds begin
        T = eltype(e)
        nsm_code = NS ? T(nsm_scalar) : n_sm[i]
        # SN=false carries smoothed baryon mass; SN=true carries neutral-H mass.
        nsm_h = nsm_code * du * (SN ? one(T) : T(fh)) / T(MH)
        en, hii, h2, _ = evolve_cell_analytic_mixing(
            rho[i] * du, e[i] * vu2, HII[i] * du, H2I[i] * du,
            nsm_h, dt * tu, z;
            f_alpha=T(f_alpha), Xe_mean=T(Xe_mean),
            smoothed_is_neutral=Val(SN), fudge=T(fudge), gauss=T(gauss),
            hubble=T(hubble), Om=T(Om), OL=T(OL), fh=T(fh),
            hubble_expansion=hub_exp, adot_over_a=T(aoa),
            rate_tables=rate_tables, cool_tables=cool_tables,
            dtfrac=T(dtfrac), itcap=itcap)
        e[i] = en / vu2
        HII[i] = hii / du
        H2I[i] = h2 / du
    end
end

@kernel function _evolve_mixing_k!(e, HII, H2I, HDI, HeII, @Const(rho), @Const(n_sm),
                                   du, vu2, tu, dt, z,
                                   f_alpha, Xe_mean, fudge, gauss,
                                   hubble, Om, OL, fh, deut, hel, hub_exp,
                                   uvb_on, GamHI, GamHeI, GamHeII, piHI, piHeI, piHeII,
                                   @Const(aC), @Const(aO), @Const(aSi), @Const(aFe), hasmetals,
                                   rate_tables, dtfrac, itcap, ::Val{SN}) where {SN}
    i = @index(Global)
    @inbounds begin
        T    = eltype(e)
        hd_in    = deut ? HDI[i]*du : zero(T)
        he_in    = hel  ? HeII[i]*du : zero(T)
        # SN=false carries smoothed baryon mass; SN=true carries neutral-H mass.
        n_sm_cgs = n_sm[i] * du * (SN ? one(T) : T(fh)) / T(MH)
        mab = hasmetals ? MetalAbundances{T}(aC[i], aO[i], aSi[i], aFe[i]) :
                          MetalAbundances{T}()
        en, hii, h2, hd, he = evolve_cell_mixing(
            rho[i]*du, e[i]*vu2, HII[i]*du, H2I[i]*du, hd_in,
            n_sm_cgs, dt*tu, z;
            f_alpha  = T(f_alpha),
            Xe_mean  = T(Xe_mean),
            smoothed_is_neutral = Val(SN),
            fudge = T(fudge), gauss = T(gauss),
            hubble   = T(hubble), Om = T(Om), OL = T(OL),
            fh       = T(fh), deuterium = deut,
            helium = hel, HeII_m = he_in, hubble_expansion = hub_exp,
            uvb = uvb_on,
            GamHI = T(GamHI), GamHeI = T(GamHeI), GamHeII = T(GamHeII),
            piHI = T(piHI), piHeI = T(piHeI), piHeII = T(piHeII), metals = mab,
            rate_tables = rate_tables, dtfrac = T(dtfrac), itcap = itcap)
        e[i]   = en  / vu2
        HII[i] = hii / du
        H2I[i] = h2  / du
        deut && (HDI[i] = hd / du)
        hel  && (HeII[i] = he / du)
    end
end

@kernel function _evolve_mixing_primordial_k!(e, HII, H2I, @Const(rho), @Const(n_sm),
                                              nsm_scalar, du, vu2, tu, dt, z,
                                              f_alpha, Xe_mean, fudge, gauss,
                                              hubble, Om, OL, fh, hub_exp,
                                              rate_tables, dtfrac, itcap,
                                              ::Val{NS}, ::Val{SN}) where {NS,SN}
    i = @index(Global)
    @inbounds begin
        T = eltype(e)
        nsm_code = NS ? T(nsm_scalar) : n_sm[i]
        nsm_h = nsm_code * du * (SN ? one(T) : T(fh)) / T(MH)
        en, hii, h2, _, _ = evolve_cell_mixing(
            rho[i] * du, e[i] * vu2, HII[i] * du, H2I[i] * du, zero(T),
            nsm_h, dt * tu, z;
            f_alpha = T(f_alpha), Xe_mean = T(Xe_mean),
            smoothed_is_neutral = Val(SN), fudge = T(fudge), gauss = T(gauss),
            hubble = T(hubble), Om = T(Om), OL = T(OL), fh = T(fh),
            deuterium = false, helium = false, hubble_expansion = hub_exp,
            rate_tables = rate_tables, dtfrac = T(dtfrac), itcap = itcap)
        e[i] = en / vu2
        HII[i] = hii / du
        H2I[i] = h2 / du
    end
end

# ── Public API ────────────────────────────────────────────────────────────────

"""
    solve_chem_mixing!(rho, e_int, HII, H2I, n_smoothed; [HDI,]
                       a_value, dt, density_units, length_units, time_units,
                       fa_table, Xe_mean, smoothed_is_neutral,
                       hubble, Om, OL, fh, deuterium, backend, precision)

Evolve the v2026 reduced chemistry/cooling with Lyα-mixing recombination over `dt`
(code time units) for every cell.  Mirrors `solve_chem!` but takes one extra
positional argument:

  `n_smoothed` — smoothed baryon mass density [same code units as `rho`], or smoothed
  neutral-H mass density when `smoothed_is_neutral=true`, pre-computed by the host
  MHD code over the Lyα mean-free-path volume.

Keyword arguments:
  `uvb`                 — optional metagalactic UV/X-ray background (`UVBackground`,
                           e.g. `fg20_uvb()`).  When given, its rates at this step's `z`
                           are threaded into the network: Γ_HI photoionises H (HI→HII+e
                           in `network_step`), Γ_HeI/Γ_HeII drive the He ionisation
                           equilibrium, and piHI/piHeI/piHeII photoheat the gas (a +edot
                           source).  `nothing` (default) ⇒ primordial-only, bit-identical
                           to the no-UVB path.
  `fa_table`            — `FAlphaTable` with f_α(z). Default: `FA_ZERO` (f_α ≡ 0,
                           no mixing, bit-identical to `solve_chem!`).
  `Xe_mean`             — global mean free-electron fraction n_e/n_H for this step.
                           Used to convert smoothed n_H → n1s when smoothed_is_neutral=false.
  `smoothed_is_neutral`  — if `true`, `n_smoothed` is already the smoothed neutral
                           density n1s; if `false` (default), approximate n1s_smoothed ≈
                           n_smoothed × (1 − Xe_mean).
  `recfast_fudge`       — RECFAST fudge `fu` on α_B (enters the C-factor as
                           fu·(1+KL)/(1+KL+fu·KB)). Default 1.0 (the pure-Peebles
                           default; = HyRec PEEBLES mode). Set to 1.14 for
                           RECFAST v1.  Overridden to 1.125 when `recfast_hswitch`.
  `recfast_hswitch`     — if `true`, use RECFAST v2: fudge fu = 1.125 on α_B PLUS
                           a multiplicative Gaussian correction gauss(z) =
                           1 + G₁(z) + G₂(z) on the Lyα escape factor K (two
                           Gaussians in ln(1+z), CAMB 1.6.6 defaults; scales both
                           KL and KB).  Brings x_e(z) within ~0.1-0.3% of HyRec.
                           Default: `false`.
  `HeII`, `helium`      — helium ionisation handling:
                           • DEFAULT (`helium=false`, no `HeII`): He I/II/III in
                             Saha equilibrium with the CMB each step.  Total x_e is
                             correct to <0.1% at z≳3000 and z≲1700, but ~3% LOW in
                             the He⁺→He⁰ freeze-out window z≈2000-2500 (Saha has no
                             radiative-transfer delay).  This is the shipped default:
                             He⁺⁺/He⁺ and the deep-recombination x_e are exact, and
                             the only error is the quantified ~3% transient at z≈2000.
                           • OPT-IN (`helium=true` + an advected `HeII` vector, the
                             He⁺ MASS density = 4·n(He⁺)·m_H): evolve He⁺ with the
                             full He I recombination (HyRec radiative transfer),
                             capturing the freeze-out → total x_e <0.1% vs HyRec
                             across z=1900-8000.  He⁺⁺ stays Saha (always fast);
                             report total x_e via `total_electron_fraction(xHII,
                             xHeII, nH, Trad)`.  Needed only when x_e in z≈2000-2500
                             must be better than ~3%.
"""
function solve_chem_mixing!(rho::AbstractVector, e_int::AbstractVector,
                            HII::AbstractVector, H2I::AbstractVector,
                            n_smoothed::AbstractVector;
                            HDI::Union{Nothing,AbstractVector} = nothing,
                            HeII::Union{Nothing,AbstractVector} = nothing,
                            a_value::Real, dt::Real,
                            density_units::Real, length_units::Real, time_units::Real,
                            fa_table::FAlphaTable = FA_ZERO,
                            Xe_mean::Real = 0.0,
                            smoothed_is_neutral::Bool = false,
                            recfast_fudge::Real = 1.0,
                            recfast_hswitch::Bool = false,
                            hubble_expansion::Bool = false,
                            uvb::Union{Nothing,UVBackground} = nothing,
                            metals = nothing,
                            hubble::Real = 71.0, Om::Real = 0.27, OL::Real = 0.73,
                            fh::Real = 0.76, deuterium::Bool = false,
                            helium::Bool = false,
                            rate_tables = nothing,
                            dtfrac::Real = 0.1,
                            itcap::Int = _SUB_ITMAX,
                            backend::Symbol = :cpu, precision::Type = Float64)
    n = length(rho)
    @assert length(e_int) == n && length(HII) == n && length(H2I) == n
    @assert length(n_smoothed) == n
    deut = deuterium && HDI !== nothing
    deut && @assert length(HDI) == n
    hel = helium && HeII !== nothing
    hel && @assert length(HeII) == n
    hasmetals = metals !== nothing
    if hasmetals
        @assert length(metals.C)==n && length(metals.O)==n &&
                length(metals.Si)==n && length(metals.Fe)==n
    end

    P   = precision
    be  = ChemistryKernels.backend(backend)
    rate_tables = _resolve_rate_tables(backend, P, rate_tables)
    du  = P(density_units)
    vu2 = P((length_units / time_units)^2)
    tu  = P(time_units)
    z   = P(1.0 / a_value - 1.0)

    # f_α for this step (scalar, from the redshift table)
    f_alpha = P(fa_at(fa_table, Float64(z)))

    # RECFAST fudge on α_B (C_eff = fu·(1+KL)/(1+KL+fu·KB)) and the v2 Gaussian
    # correction on the Lyα escape K (scales both KL and KB):
    #   recfast_hswitch=false: fudge = recfast_fudge (1.0 = pure Peebles), no Gaussian
    #   recfast_hswitch=true:  fudge = 1.125 (RECFAST v2) + Gaussian gauss(z)
    fudge = P(recfast_hswitch ? _RECFAST_V2_FUDGE : recfast_fudge)
    gauss = P(recfast_hswitch ? recfast_gauss_factor(Float64(z)) : 1.0)

    # UV-background rates for this step (scalars, evaluated once at z).  Mapping:
    # uvb_rates → (k24=Γ_HI, k25=Γ_HeII, k26=Γ_HeI, piHI, piHeI, piHeII).
    uvb_on = uvb !== nothing
    gHI = gHeI = gHeII = zero(P); pHI = pHeI = pHeII = zero(P)
    if uvb_on
        (k24, k25, k26, qHI, qHeI, qHeII) = uvb_rates(uvb, Float64(z))
        gHI = P(k24); gHeI = P(k26); gHeII = P(k25)
        pHI = P(qHI); pHeI = P(qHeI); pHeII = P(qHeII)
    end

    d_rho = to_device(be, collect(rho),        P)
    d_e   = to_device(be, collect(e_int),       P)
    d_HII = to_device(be, collect(HII),         P)
    d_H2I = to_device(be, collect(H2I),         P)
    d_HDI = deut ? to_device(be, collect(HDI),  P) : device_zeros(be, P, (n,))
    d_HeII = hel ? to_device(be, collect(HeII), P) : device_zeros(be, P, (n,))
    d_nsm = to_device(be, collect(n_smoothed),  P)

    d_aC = hasmetals ? to_device(be, collect(metals.C),  P) : device_zeros(be, P, (n,))
    d_aO = hasmetals ? to_device(be, collect(metals.O),  P) : device_zeros(be, P, (n,))
    d_aSi= hasmetals ? to_device(be, collect(metals.Si), P) : device_zeros(be, P, (n,))
    d_aFe= hasmetals ? to_device(be, collect(metals.Fe), P) : device_zeros(be, P, (n,))

    SN = smoothed_is_neutral
    _evolve_mixing_k!(be)(d_e, d_HII, d_H2I, d_HDI, d_HeII, d_rho, d_nsm,
                          du, vu2, tu, P(dt), z,
                          f_alpha, P(Xe_mean), fudge, gauss,
                          P(hubble), P(Om), P(OL), P(fh), deut, hel, hubble_expansion,
                          uvb_on, gHI, gHeI, gHeII, pHI, pHeI, pHeII,
                          d_aC, d_aO, d_aSi, d_aFe, hasmetals,
                          rate_tables, P(dtfrac), itcap, Val(SN);
                          ndrange = n)

    e_int      .= to_host(d_e)
    HII        .= to_host(d_HII)
    H2I        .= to_host(d_H2I)
    deut && (HDI .= to_host(d_HDI))
    hel  && (HeII .= to_host(d_HeII))
    return nothing
end

"""
    solve_chem_mixing_device!(rho, e_int, HII, H2I, n_smoothed; ...)

Zero-copy primordial H/H2 device path for Lyalpha-mixing recombination. All state
arrays already reside on `backend` and are updated in place. `n_smoothed` may be a
device vector or a scalar code-unit density. No full-grid pad or staging arrays are
allocated. Pass a device-resident `rate_tables` for the production table-backed path.
"""
function solve_chem_mixing_device!(rho, e_int, HII, H2I, n_smoothed;
                                   a_value::Real, dt::Real,
                                   density_units::Real, length_units::Real,
                                   time_units::Real,
                                   f_alpha::Real = 0.0,
                                   Xe_mean::Real = 0.0,
                                   smoothed_is_neutral::Bool = false,
                                   recfast_fudge::Real = 1.0,
                                   recfast_hswitch::Bool = false,
                                   hubble_expansion::Bool = false,
                                   hubble::Real = 71.0, Om::Real = 0.27,
                                   OL::Real = 0.73, fh::Real = 0.76,
                                   rate_tables = nothing,
                                   dtfrac::Real = 0.1,
                                   itcap::Int = _SUB_ITMAX,
                                   workgroup_size::Int = 0,
                                   backend::Symbol = :cuda,
                                   precision::Type = Float64)
    n = length(rho)
    @assert length(e_int) == n && length(HII) == n && length(H2I) == n
    scalar_nsm = n_smoothed isa Real
    scalar_nsm || @assert length(n_smoothed) == n

    P = precision
    be = ChemistryKernels.backend(backend)
    rate_tables = _resolve_rate_tables(backend, P, rate_tables)
    du = P(density_units)
    vu2 = P((length_units / time_units)^2)
    tu = P(time_units)
    z = P(1.0 / a_value - 1.0)
    fudge = P(recfast_hswitch ? _RECFAST_V2_FUDGE : recfast_fudge)
    gauss = P(recfast_hswitch ? recfast_gauss_factor(Float64(z)) : 1.0)
    nsm_arg = scalar_nsm ? rho : n_smoothed
    nsm_scalar = scalar_nsm ? P(n_smoothed) : zero(P)
    k! = workgroup_size > 0 ?
         _evolve_mixing_primordial_k!(be, workgroup_size) :
         _evolve_mixing_primordial_k!(be)
    k!(e_int, HII, H2I, rho, nsm_arg, nsm_scalar,
       du, vu2, tu, P(dt), z, P(f_alpha), P(Xe_mean), fudge, gauss,
       P(hubble), P(Om), P(OL), P(fh), hubble_expansion,
       rate_tables, P(dtfrac), itcap,
       Val(scalar_nsm), Val(smoothed_is_neutral); ndrange = n)
    KA.synchronize(be)
    return nothing
end

"""
    solve_chem_analytic_mixing!(rho, e_int, HII, H2I, n_smoothed; ...)

Host-array entry point for the fast analytic Ly-alpha mixing solver. State is
copied to the selected backend, advanced in one allocation-free device kernel,
and copied back. Use [`solve_chem_analytic_mixing_device!`](@ref) when the state
already resides on the device.
"""
function solve_chem_analytic_mixing!(rho, e_int, HII, H2I, n_smoothed;
                                     a_value::Real, dt::Real,
                                     density_units::Real, length_units::Real,
                                     time_units::Real,
                                     f_alpha::Real = 0.0,
                                     Xe_mean::Real = 0.0,
                                     smoothed_is_neutral::Bool = false,
                                     recfast_fudge::Real = 1.0,
                                     recfast_hswitch::Bool = false,
                                     hubble_expansion::Bool = false,
                                     adot_over_a::Real = NaN,
                                     hubble::Real = 71.0, Om::Real = 0.27,
                                     OL::Real = 0.73, fh::Real = 0.76,
                                     rate_tables = nothing,
                                     cool_tables = nothing,
                                     dtfrac::Real = 0.1,
                                     itcap::Int = _SUB_ITMAX,
                                     workgroup_size::Int = 0,
                                     backend::Symbol = :cpu,
                                     precision::Type = Float64)
    n = length(rho)
    @assert length(e_int) == n && length(HII) == n && length(H2I) == n
    scalar_nsm = n_smoothed isa Real
    scalar_nsm || @assert length(n_smoothed) == n

    P = precision
    be = ChemistryKernels.backend(backend)
    rate_tables, cool_tables = _resolve_tables(backend, P, rate_tables, cool_tables)
    d_rho = to_device(be, collect(rho), P)
    d_e = to_device(be, collect(e_int), P)
    d_HII = to_device(be, collect(HII), P)
    d_H2I = to_device(be, collect(H2I), P)
    d_nsm = scalar_nsm ? d_rho : to_device(be, collect(n_smoothed), P)
    rt = rate_tables === nothing ? nothing : rate_tables
    ct = cool_tables === nothing ? nothing : cool_tables

    solve_chem_analytic_mixing_device!(d_rho, d_e, d_HII, d_H2I,
                                       scalar_nsm ? P(n_smoothed) : d_nsm;
        a_value=a_value, dt=dt, density_units=density_units,
        length_units=length_units, time_units=time_units,
        f_alpha=f_alpha, Xe_mean=Xe_mean,
        smoothed_is_neutral=smoothed_is_neutral,
        recfast_fudge=recfast_fudge, recfast_hswitch=recfast_hswitch,
        hubble_expansion=hubble_expansion, adot_over_a=adot_over_a,
        hubble=hubble, Om=Om, OL=OL, fh=fh,
        rate_tables=rt, cool_tables=ct, dtfrac=dtfrac, itcap=itcap,
        workgroup_size=workgroup_size, backend=backend, precision=P)

    e_int .= to_host(d_e)
    HII .= to_host(d_HII)
    H2I .= to_host(d_H2I)
    return nothing
end

"""
    solve_chem_analytic_mixing_device!(rho, e_int, HII, H2I, n_smoothed; ...)

Zero-copy device entry point for analytic primordial H/H2 chemistry with a
smoothed neutral-density field in the Peebles C-factor. All state arrays are
updated in place. `n_smoothed` may be a device vector or a scalar code density.
"""
function solve_chem_analytic_mixing_device!(rho, e_int, HII, H2I, n_smoothed;
                                            a_value::Real, dt::Real,
                                            density_units::Real, length_units::Real,
                                            time_units::Real,
                                            f_alpha::Real = 0.0,
                                            Xe_mean::Real = 0.0,
                                            smoothed_is_neutral::Bool = false,
                                            recfast_fudge::Real = 1.0,
                                            recfast_hswitch::Bool = false,
                                            hubble_expansion::Bool = false,
                                            adot_over_a::Real = NaN,
                                            hubble::Real = 71.0, Om::Real = 0.27,
                                            OL::Real = 0.73, fh::Real = 0.76,
                                            rate_tables = nothing,
                                            cool_tables = nothing,
                                            dtfrac::Real = 0.1,
                                            itcap::Int = _SUB_ITMAX,
                                            workgroup_size::Int = 0,
                                            backend::Symbol = :cuda,
                                            precision::Type = Float32)
    n = length(rho)
    @assert length(e_int) == n && length(HII) == n && length(H2I) == n
    scalar_nsm = n_smoothed isa Real
    scalar_nsm || @assert length(n_smoothed) == n

    P = precision
    be = ChemistryKernels.backend(backend)
    rate_tables, cool_tables = _resolve_tables(backend, P, rate_tables, cool_tables)
    du = P(density_units)
    vu2 = P((length_units / time_units)^2)
    tu = P(time_units)
    z = P(1.0 / a_value - 1.0)
    fudge = P(recfast_hswitch ? _RECFAST_V2_FUDGE : recfast_fudge)
    gauss = P(recfast_hswitch ? recfast_gauss_factor(Float64(z)) : 1.0)
    nsm_arg = scalar_nsm ? rho : n_smoothed
    nsm_scalar = scalar_nsm ? P(n_smoothed) : zero(P)
    k! = workgroup_size > 0 ?
         _evolve_analytic_mixing_k!(be, workgroup_size) :
         _evolve_analytic_mixing_k!(be)
    k!(e_int, HII, H2I, rho, nsm_arg, nsm_scalar,
       du, vu2, tu, P(dt), z, P(f_alpha), P(Xe_mean), fudge, gauss,
       P(hubble), P(Om), P(OL), P(fh), hubble_expansion, P(adot_over_a),
       rate_tables, cool_tables, P(dtfrac), itcap,
       Val(scalar_nsm), Val(smoothed_is_neutral); ndrange=n)
    KA.synchronize(be)
    return nothing
end

# rates_cmb.jl — CMB photo-destruction rates for H-, H2+, and HeH+.
#
# These rates are not tabulated functions of gas temperature; they are
# evaluated per-call as literal analytic formulas of the CMB radiation
# temperature Trad = 2.73*(1+z) [K], following the original network of
# Abel/Anninos et al. 1997:
#
#   k27: H- + γ_CMB → H + e       (Galli & Palla 1998, H4; de Jong 1972)
#   k28: H2+ + γ_CMB → H + H+     (Galli & Palla 1998, H9, LTE; Argyros 1974 /
#                                   Stancil 1994 — LTE because the CMB keeps H2+
#                                   vibrational levels thermally excited)
#
# Units: CGS s^-1.  The host multiplies by time_units before handing the rate
# to the solver, exactly as the UV-background photo-rates are handled (k27 *=
# time_units).  These functions return the raw CGS value WITHOUT the time_units
# factor.

# ── k27 : H- + γ_CMB → H + e  (GP98 H4) ─────────────────────────────────────
# Rate (per CGS, before the time_units factor tu):
#   k27 = 1.1e-1 * Trad^2.13 * exp(-8823.0 / Trad)
@inline function k27_cmb(Trad::Real)
    R = typeof(Trad)
    return R(1.1e-1) * Trad^R(2.13) * exp(-R(8823.0) / Trad)
end
@scalarkernel k27_cmb

# ── k28 : H2+ + γ_CMB → H + H+  (GP98 H9, LTE) ─────────────────────────────
# Rate (per CGS, before the time_units factor tu):
#   k28 = 1.63e7 * exp(-32400.0 / Trad)
@inline function k28_cmb(Trad::Real)
    R = typeof(Trad)
    return R(1.63e7) * exp(-R(32400.0) / Trad)
end
@scalarkernel k28_cmb

# ── HeH⁺ + γ_CMB → He + H⁺ / He⁺ + H ───────────────────────────────────────
# Sum of reactions 58 and 59 in Schleicher et al. (2008, A&A 490, 521).
# The first channel is the inverse of radiative association and dominates in
# the dark-age regime.  Raw CGS rate, s⁻¹.
@inline function gamma_HeH_cmb(Trad::Real)
    R = typeof(Trad)
    return R(220.0) * Trad^R(0.9) * exp(-R(22740.0) / Trad) +
           R(7.8e3) * Trad^R(1.2) * exp(-R(240000.0) / Trad)
end
@scalarkernel gamma_HeH_cmb

# CMB stimulated-radiative-association enhancement used with
# `kHeH_ra_stim_base(T)`.
@inline function HeH_stim_factor(Trad::Real)
    R = typeof(Trad)
    return one(R) + R(2.0e-4) * Trad^R(1.1)
end
@scalarkernel HeH_stim_factor

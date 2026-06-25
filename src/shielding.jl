# shielding.jl — H₂ self-shielding and dust attenuation of the Lyman-Werner band.
#
# The Lyman-Werner (LW) UV flux (11.2–13.6 eV) photodissociates H₂.  Two
# processes reduce the effective rate below the optically-thin value G₀ · k_LW0:
#   1. H₂ self-shielding  — the H₂ absorption lines saturate in the column.
#   2. Dust attenuation   — continuum absorption by dust grains.
#
# Both are multiplicative factors on the unshielded dissociation rate
# k_LW0 = 5.8e-11 s⁻¹ per Habing unit (Draine 1978, Black & van Dishoeck 1987).
#
# References:
#   Draine & Bertoldi (1996) ApJ 468 269   — H₂ self-shielding function
#   Black & van Dishoeck (1987)            — unshielded LW rate coefficient
#   Draine (1978) ApJS 36 595              — grain cross section in LW band

export f_shield_H2, f_dust_LW, k_H2_LW_eff

# ── H₂ self-shielding factor ─────────────────────────────────────────────────
# Draine & Bertoldi (1996), ApJ 468 269, Eq. 37.  The factor f_shield ∈ [0, 1]
# approaches 1 when N(H₂) → 0 (optically thin) and drops as the H₂ Lyman-Werner
# absorption lines saturate.  b = 3 km/s (b₅ = 3) is the standard choice for
# warm ISM gas (thermal broadening at ~few hundred K plus turbulence).
#
# `N_H2` is the H₂ column density [cm⁻²] along the line of sight to the UV
# source.  Passing N_H2 = 0 gives f_shield ≈ 1 (optically thin; the -8.5e-4
# exponential correction is ~0.1% at x=0 and can be ignored).
@inline function f_shield_H2(N_H2::Real)
    R  = typeof(N_H2)
    x  = N_H2 / R(5e14)
    sx = sqrt(one(R) + x)
    # Draine & Bertoldi (1996) Eq. 37, b₅ = 3 (b = 3 km/s)
    f1 = R(0.965) / (one(R) + x / R(3))^2        # Doppler core self-shielding
    f2 = R(0.035) / sx                            # Lorentzian damping wings
    return (f1 + f2) * exp(-R(8.5e-4) * sx)
end

# ── Dust attenuation of the Lyman-Werner band ────────────────────────────────
# Continuum dust opacity in the LW band: σ_LW ≈ 2e-21 cm² per H atom (grain
# cross section at ~1000 Å, standard MRN distribution, Draine 1978).  Scales
# with Z_rel = Z/Z_☉ (dust-to-gas proportional to metallicity).
#
# `N_H` is the total H column density [cm⁻²].  Passing N_H = 0 returns 1.0.
@inline function f_dust_LW(N_H::Real, Z_rel::Real)
    R = typeof(N_H)
    return exp(-R(2e-21) * N_H * Z_rel)
end

# ── Effective LW H₂ photodissociation rate [s⁻¹] ────────────────────────────
# Combines the unshielded LW rate (proportional to G₀) with both shielding
# factors.  Returns 0 when G₀ = 0 (no UV field).
#
# Arguments
# ---------
# G0    : FUV field intensity [Habing units]
# N_H2  : H₂ column density [cm⁻²] (0 → no self-shielding)
# N_H   : Total H column density [cm⁻²] (0 → no dust attenuation)
# Z_rel : Metallicity relative to solar
@inline function k_H2_LW_eff(G0::Real, N_H2::Real, N_H::Real, Z_rel::Real)
    R  = typeof(G0)
    fs = f_shield_H2(N_H2)
    fd = f_dust_LW(N_H, Z_rel)
    # k_LW0 = 5.8e-11 s⁻¹ per Habing unit (Black & van Dishoeck 1987)
    return R(5.8e-11) * G0 * fs * fd
end

# rates_dust.jl — precision-generic dust-chemistry rate and heating/cooling kernels.
#
# All functions are pure scalars of their physical arguments (CGS) and return
# CGS values.  Pattern follows rates_atomic.jl: R = typeof(first_arg); every
# numeric literal cast to R; no allocation; GPU-safe.
#
# References:
#   Cazaux & Tielens (2004) ApJ 604 222         — H₂ on dust
#   Bakes & Tielens (1994) ApJ 427 822          — photoelectric heating
#   Hollenbach & McKee (1989) ApJ 342 306       — gas-grain cooling
#   Weingartner & Draine (2001) ApJS 134 263    — grain-assisted recombination

export k_H2_dust, Gamma_PE, Lambda_gr, k_gr_recomb_HII, Lambda_dust

# ── H₂ formation on dust grains ──────────────────────────────────────────────
# Cazaux & Tielens (2004). The rate coefficient [cm³/s] is factored into the
# thermal velocity × dust cross-section (∝ √T_gas / 300), a sticking coefficient
# S(T_gas) that falls off above ~464 K (Hollenbach & McKee 1979), and a surface
# recombination efficiency ξ(T_dust) that falls off when thermal hopping evaporates
# physisorbed H atoms before they meet (Langmuir-Hinshelwood mechanism).
# Scaled by Z_rel = Z/Z_☉ (dust-to-gas ratio proportional to metallicity).
@inline function k_H2_dust(T_gas::Real, T_dust::Real, Z_rel::Real)
    R = typeof(T_gas)
    # Sticking coefficient: atoms bounce off hot grains (Hollenbach & McKee 1979)
    S = inv(one(R) + (T_gas / R(464))^2)
    # Surface recombination efficiency: physisorption sites populated when T_dust ≲ 30 K
    T_d = max(T_dust, R(1))          # guard against T_dust ≤ 0
    ξ   = inv(one(R) + R(1e4) * exp(-R(600) / T_d))
    return R(3e-17) * sqrt(T_gas / R(300)) * S * ξ * Z_rel
end

# ── Photoelectric heating rate per H nucleus [erg/s] ─────────────────────────
# Bakes & Tielens (1994), Eq. 2. UV photons eject electrons from PAHs/small
# grains; the heating efficiency ε depends on the grain charging parameter
# ψ = G₀√T / nₑ.  Multiply by n_H to get the volumetric heating rate [erg/cm³/s].
@inline function Gamma_PE(T_gas::Real, G0::Real, Z_rel::Real, n_e::Real)
    R = typeof(T_gas)
    ψ   = G0 * sqrt(T_gas) / max(n_e, R(1e-20))
    # Two-branch efficiency: collisional de-excitation (first) and recombination (second)
    ε_1 = R(4.87e-2) / (one(R) + R(4e-3) * ψ^R(0.73))
    ε_2 = R(3.65e-2) * (T_gas / R(1e4))^R(0.7) / (one(R) + R(2e-4) * ψ)
    # 1.3e-24 erg/s per H atom per Habing field unit; Z_rel scales the grain abundance
    return R(1.3e-24) * (ε_1 + ε_2) * G0 * Z_rel
end

# ── Gas-grain collisional coupling [erg/cm³/s] ───────────────────────────────
# Hollenbach & McKee (1989), positive when gas is hotter than dust (gas cools).
# The sign flips and the gas heats when T_gas < T_dust (e.g., in warm PDR skins).
@inline function Lambda_gr(T_gas::Real, T_dust::Real, n_H::Real, Z_rel::Real)
    R = typeof(T_gas)
    return R(2e-33) * sqrt(T_gas) * (T_gas - T_dust) * n_H^2 * Z_rel
end

# ── Grain-assisted recombination of HII [cm³/s] ──────────────────────────────
# Weingartner & Draine (2001), ApJS 134 263, Table 2 (standard RV=3.1 MRN
# dust, H⁺ case).  The charging parameter ψ = G₀√T/nₑ controls how positively
# charged the grains are: at low ψ grains attract H⁺ (fast); at high ψ they
# repel it (slow).  Cross terms in T and ψ are dropped (second-order correction
# over 100–10^4 K); clamped at ψ_max to prevent overflow in strong-field regions.
@inline function k_gr_recomb_HII(T_gas::Real, G0::Real, Z_rel::Real, n_e::Real)
    R = typeof(T_gas)
    ψ     = G0 * sqrt(T_gas) / max(n_e, R(1e-20))
    psi_c = min(ψ, R(1e8))
    return R(1.225e-13) * Z_rel / (one(R) + R(8.074e-6) * psi_c^R(1.378))
end

# ── Dust thermal emission [erg/cm³/s] ────────────────────────────────────────
# Modified blackbody with κ ∝ ν² → Λ ∝ n_H · Z_rel · T_dust^6 (Hollenbach &
# McKee 1979, Krumholz et al. 2011).  This is the energy the dust radiates away;
# it does NOT directly enter the GAS energy equation (gas couples to dust only
# through Lambda_gr).  Exported as a diagnostic and for energy-balance checks.
@inline function Lambda_dust(T_dust::Real, Z_rel::Real, n_H::Real)
    R = typeof(T_dust)
    return R(2.0e-27) * n_H * Z_rel * T_dust^R(6)
end

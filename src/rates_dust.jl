# rates_dust.jl — dust-chemistry RATE COEFFICIENTS for the gas-phase network.
#
# Only pure rate coefficients live here (cm³/s or s⁻¹): H₂ formation on grains,
# grain-assisted HII recombination, and the LW photodissociation infrastructure.
# The THERMAL channels (PE heating, gas-grain coupling, dust emission) live in
# EmissionKernels (cooling_dust.jl) alongside metal_cooling_rate and are imported
# in ChemistryKernels.jl; subcycle.jl calls Gamma_PE / Lambda_gr from there.
#
# Pattern follows rates_atomic.jl: R = typeof(first_arg); every numeric literal
# cast to R; no allocation; GPU-safe.
#
# References:
#   Cazaux & Tielens (2004) ApJ 604 222         — H₂ on dust
#   Weingartner & Draine (2001) ApJS 134 263    — grain-assisted recombination

export k_H2_dust, k_gr_recomb_HII

# ── H₂ formation on dust grains ──────────────────────────────────────────────
# Cazaux & Tielens (2004). The rate coefficient [cm³/s] is factored into the
# thermal velocity × dust cross-section (∝ √T_gas / 300), a sticking coefficient
# S(T_gas) that falls off above ~464 K (Hollenbach & McKee 1979), and a surface
# recombination efficiency ξ(T_dust) that falls off when thermal hopping evaporates
# physisorbed H atoms before they meet (Langmuir-Hinshelwood mechanism).
# Scaled by Z_rel = Z/Z_☉ (dust-to-gas ratio proportional to metallicity).
@inline function k_H2_dust(T_gas::Real, T_dust::Real, Z_rel::Real)
    R = typeof(T_gas)
    (T_gas > zero(R) && Z_rel > zero(R)) || return zero(R)
    # Sticking coefficient: atoms bounce off hot grains (Hollenbach & McKee 1979)
    S = inv(one(R) + (T_gas / R(464))^2)
    # Surface recombination efficiency: physisorption sites populated when T_dust ≲ 30 K
    T_d = max(T_dust, R(1))          # guard against T_dust ≤ 0
    ξ   = inv(one(R) + R(1e4) * exp(-R(600) / T_d))
    return R(3e-17) * sqrt(T_gas / R(300)) * S * ξ * Z_rel
end

# ── Grain-assisted recombination of HII [cm³/s] ──────────────────────────────
# Weingartner & Draine (2001), ApJS 134 263, Table 2 (standard RV=3.1 MRN
# dust, H⁺ case).  The charging parameter ψ = G₀√T/nₑ controls how positively
# charged the grains are: at low ψ grains attract H⁺ (fast); at high ψ they
# repel it (slow).  Cross terms in T and ψ are dropped (second-order correction
# over 100–10^4 K); clamped at ψ_max to prevent overflow in strong-field regions.
@inline function k_gr_recomb_HII(T_gas::Real, G0::Real, Z_rel::Real, n_e::Real)
    R = typeof(T_gas)
    Z_rel > zero(R) || return zero(R)
    G0 <= zero(R) && return R(1.225e-13) * Z_rel
    (T_gas > zero(R) && n_e > zero(R)) || return zero(R)
    ψ     = G0 * sqrt(T_gas) / n_e
    psi_c = min(ψ, R(1e8))
    return R(1.225e-13) * Z_rel / (one(R) + R(8.074e-6) * psi_c^R(1.378))
end

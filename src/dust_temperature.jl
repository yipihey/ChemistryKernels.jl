# dust_temperature.jl — local equilibrium dust temperature.
#
# T_dust is determined algebraically by balancing attenuated UV absorption plus
# CMB photon absorption against modified-blackbody grain emission (κ ∝ ν², so
# emission ∝ T_dust^5.7 rather than T^4 — Hollenbach & McKee 1979).
#
# "Local equilibrium" means T_dust is recomputed per sub-step from the local
# radiation field (G₀, A_V) and CMB temperature, without evolving it as a
# separate advected field and without RHD.  Gas-grain collisional coupling
# (Lambda_gr) affects the GAS temperature, not T_dust at this level; the
# correction to T_dust from gas-grain coupling is deferred (important only at
# n_H ≳ 10⁵ cm⁻³, Hollenbach & McKee 1989).
#
# Reference: Hollenbach & McKee (1979) ApJS 41 555, their Eq. 3.

export T_dust_eq

"""
    T_dust_eq(G0, A_V, T_CMB) → T_dust [K]

Local equilibrium dust temperature from attenuated FUV field `G0` [Habing
units], visual extinction `A_V` [mag], and CMB temperature `T_CMB` [K].

The effective radiation field combines attenuated UV (`G0 · exp(-2.5·A_V)`)
with the CMB contribution, expressed as an equivalent Habing field via
`(T_CMB/2.728)^5.7`.  The grain emission law `Λ ∝ T_dust^5.7` (modified
blackbody, κ ∝ ν²) then gives `T_dust = 12.2 · G0_eff^(1/5.7)` (Hollenbach
& McKee 1979); the CMB floor ensures `T_dust ≥ T_CMB`.

Precision-generic: works with Float32, Float64, and dual numbers.
"""
@inline function T_dust_eq(G0::Real, A_V::Real, T_CMB::Real)
    R = typeof(G0)
    # Attenuated FUV field: dust extinction at LW/optical wavelengths
    G0_att = G0 * exp(-R(2.5) * A_V)
    # CMB contribution as equivalent Habing field (sets the floor at T_CMB)
    G0_cmb = (T_CMB / R(2.728))^R(5.7)
    G0_eff = G0_att + G0_cmb
    # Hollenbach & McKee (1979): T_dust = 12.2 · G_eff^(1/5.7)
    T_d = R(12.2) * G0_eff^(one(R) / R(5.7))
    return max(T_CMB, T_d)
end

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

Energy balance: T_d^5.7 = 12.2^5.7 · G0 · exp(-2.5·A_V) + T_CMB^5.7.
The 12.2^5.7 prefactor gives T_d = 12.2 K at G₀ = 1, A_V = 0, T_CMB = 0
(Hollenbach & McKee 1979).  The CMB^5.7 term provides the exact floor:
T_dust → T_CMB when G₀ → 0 (no UV heating).

Precision-generic: works with Float32, Float64, and dual numbers.
"""
@inline function T_dust_eq(G0::Real, A_V::Real, T_CMB::Real)
    R = typeof(G0)
    # T_d^5.7 = 12.2^5.7 * G0 * exp(-2.5*A_V)  +  T_CMB^5.7
    # → T_d = T_CMB when G0 = 0; T_d = 12.2 when G0 = 1, T_CMB = 0.
    uv_term  = R(12.2)^R(5.7) * G0 * exp(-R(2.5) * A_V)
    cmb_term = R(T_CMB)^R(5.7)
    return (uv_term + cmb_term)^(one(R) / R(5.7))
end

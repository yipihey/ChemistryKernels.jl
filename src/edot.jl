# edot.jl — net radiative cooling/heating rate of the v2026 reduced network.
#
# The per-channel physics (H/He collisional excitation & ionisation, recombination,
# bremsstrahlung, H2, HD, CMB-Compton, metal fine-structure) now lives in the
# foundation package EmissionKernels.  `cooling_edot` is the network's view of it: the
# negative of `EmissionKernels.cooling_rate_total` (the summed radiative cooling, He
# omitted as in the reduced model).  Bit-identical to the legacy assembler by
# construction — same expression, same term order. Pure & allocation-free.

export cooling_edot

"""
    cooling_edot(nHI, nHII, nHeI, nde, nH2, nHD, T, z; ih2optical=false, nH=nothing,
                 metals=nothing, Gamma_PE_vol=0, Lambda_gr_vol=0)

Net volumetric energy rate ė [erg cm⁻³ s⁻¹] (cooling ⇒ negative) for the reduced
network at gas temperature `T` [K] and redshift `z`. Number densities are physical
[cm⁻³]; `nH2`/`nHD` are H2 and HD *molecule* densities; `metals` an optional
`MetalAbundances` (n(X)/n_H per cell) and `nH` the H-nucleus density. Delegates to
`EmissionKernels.cooling_rate_total`. Pure.

Dust contributions to the GAS energy budget (both volumetric [erg/cm³/s]):
- `Gamma_PE_vol` — photoelectric heating (positive, heats gas)
- `Lambda_gr_vol` — gas-grain collisional coupling (positive = gas cools when T_gas > T_dust)
"""
@inline function cooling_edot(nHI, nHII, nHeI, nde, nH2, nHD, T, z;
                              ih2optical::Bool = false, nH = nothing, metals = nothing,
                              cool_tables = nothing,
                              Gamma_PE_vol = zero(typeof(T)),
                              Lambda_gr_vol = zero(typeof(T)))
    # analytic fits (default/reference) or the opt-in log–log cooling table.
    gas_cool = cool_tables === nothing ?
        cooling_rate_total(nHI, nHII, nHeI, nde, nH2, nHD, T, z;
                           ih2optical = ih2optical, nH = nH, metals = metals) :
        cooling_rate_total_tab(cool_tables, nHI, nHII, nHeI, nde, nH2, nHD, T, z;
                               ih2optical = ih2optical, nH = nH, metals = metals)
    # Net rate: gas cooling (negative sign) + PE heating − gas-grain coupling.
    # Lambda_gr_vol > 0 when T_gas > T_dust (gas loses energy), < 0 when gas gains.
    return -(gas_cool + Lambda_gr_vol) + Gamma_PE_vol
end

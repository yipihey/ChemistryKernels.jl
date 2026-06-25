# ChemistryKernels.jl

A **table-free, pure-Julia, GPU-ready, differentiable** primordial (+deuterium, +metal-line
cooling) chemistry & cooling network — a modern, drop-in-style alternative to
[grackle](https://grackle.readthedocs.io) for cosmological / astrophysical simulation
codes (RAMSES, Gadget-4, Arepo, Enzo, …) and for Julia hosts.

Every reaction rate and cooling coefficient is evaluated **directly from its analytic fit**
(no log-T tables, no interpolation). One precision-generic `@kernel` per cell runs on the
CPU (Float64 / Float32) and on GPU (Metal / CUDA via package extensions), and the math core
is allocation-free and pure, so it is **autodiff-friendly** (ForwardDiff today; Enzyme-ready).

Physics lineage: the reduced **Abel, Anninos, Zhang & Norman (1997)** network (the original
1990s Enzo chemistry that grackle descends from), re-grounded on the primary papers:
- Advects HII, H₂, HD; H⁻/H₂⁺/D⁺ and He stages in algebraic/ionization equilibrium;
  nₑ from charge conservation.
- Cooling/emission (H/He collisional, recombination, bremsstrahlung, H₂, HD, CMB-Compton,
  **metal fine-structure with independent Fe for non-solar [α/Fe]**) lives in the companion
  [`EmissionKernels.jl`](https://github.com/yipihey/EmissionKernels.jl).
- Optional FG20 metagalactic UV background; HyRec-validated recombination incl. an optional
  Lyα-mixing mode for early-universe / PMF work.

> **Scope vs grackle.** This is the *reduced* network + fine-structure metal-line cooling +
> dust physics.  It does **not** (yet) include full tabulated metal cooling above 10⁴ K or
> radiative-transfer heating. See [`docs/integration.md`](docs/integration.md) for the full
> capability/units reference and [`docs/grackle_interface.md`](docs/grackle_interface.md)
> for how to expose a grackle-compatible C/Fortran interface.

## Install (custom registry — not in Julia's General registry)

```julia
pkg> registry add https://github.com/yipihey/VespaRegistry
pkg> add ChemistryKernels
```

The registry resolves the `EmissionKernels` dependency automatically — no manual `[sources]`
needed. (To track a development branch instead: `pkg> dev https://github.com/yipihey/ChemistryKernels.jl`.)

## Quick start (one-zone)

```julia
using ChemistryKernels

# Per-cell arrays (any length N). MASS densities ρ·x in code units; here code units = CGS.
rho = [1.0e-24]              # g/cm³
e   = [1.0e12]              # specific internal energy [erg/g]
HII = [0.12e-24]           # HII mass density (= x_HII · ρ · fh)
H2I = [1.0e-30]            # H₂ mass density (mass-equiv: 2·n(H₂)·m_H)

solve_chem!(rho, e, HII, H2I;          # evolves e, HII, H2I in place
            a_value = 1.0, dt = 3.0e13,
            density_units = 1.0, length_units = 1.0, time_units = 1.0,
            fh = 0.76)
```

Add deuterium (`HDI` field + `deuterium=true`), a UV background (`uvb = fg20_uvb()`), metal
cooling (`metals = (C=…, O=…, Si=…, Fe=…)` per-cell number abundances n(X)/n_H), or run on a
GPU (`backend = :metal`/`:cuda`, `precision = Float32`, or zero-copy via
`solve_chem_device!`). Full reference: [`docs/integration.md`](docs/integration.md).

## Dust and Lyman-Werner shielding

Enable dust physics with `dust = true`.  Per-cell arrays are required for
`Z_rel` (metallicity relative to solar), `G0` (FUV field in Habing units), and
`A_V` (visual extinction in mag).  Column-density arrays `N_H` [cm⁻²] and `N_H2`
[cm⁻²] are optional: when supplied they activate dust LW attenuation and H₂
self-shielding respectively; when omitted those factors default to no attenuation.

```julia
using ChemistryKernels

n = 1000
rho = fill(1.67e-22, n)   # g/cm³  (~100 H/cm³ at solar metallicity)
e   = fill(1.0e11,   n)   # erg/g  (~1000 K)
HII = fill(1.67e-26, n)   # HII mass density (x_HII ≈ 1e-4)
H2I = fill(1.67e-28, n)   # H₂ mass density  (x_H2  ≈ 1e-6)

# Dust per-cell arrays (all required when dust=true)
Z_rel = fill(0.3,   n)    # 0.3 Z_⊙ (sets dust-to-gas ratio for all dust rates)
G0    = fill(1.0,   n)    # 1 Habing (ISRF strength)
A_V   = fill(2.0,   n)    # 2 mag visual extinction (shields UV + LW field)

# Optional: column densities for LW shielding (omit → no self-shielding)
N_H   = fill(1e21,  n)    # H  column [cm⁻²] — dust LW attenuation
N_H2  = fill(1e14,  n)    # H₂ column [cm⁻²] — H₂ self-shielding (Draine & Bertoldi 1996)

solve_chem!(rho, e, HII, H2I;
            a_value = 1.0, dt = 3.15e13,          # ~1 Myr
            density_units = 1.0, length_units = 1.0, time_units = 1.0,
            dust  = true,
            Z_rel = Z_rel, G0 = G0, A_V = A_V,
            N_H   = N_H,   N_H2 = N_H2)
```

The dust path adds five effects per sub-step (all zero-cost when `dust = false`):
| Effect | Source | Enters |
|---|---|---|
| H₂ formation on grains (`k_H2_dust`) | Cazaux & Tielens (2004) | H₂ source term |
| Grain-assisted HII recombination (`k_gr_recomb_HII`) | Weingartner & Draine (2001) | HII sink |
| LW photodissociation with shielding (`k_H2_LW_eff`) | Draine & Bertoldi (1996) | H₂ sink |
| Photoelectric heating (`Gamma_PE`) | Bakes & Tielens (1994) | gas energy source |
| Gas-grain coupling (`Lambda_gr`) | Hollenbach & McKee (1989) | gas energy sink/source |

Dust equilibrium temperature `T_dust_eq(G0, A_V, T_CMB)` (Hollenbach & McKee 1979)
is computed once per macro-step (hoisted out of the sub-cycle). Low-density cells
(n_H ≲ 10⁵ cm⁻³) use pure local equilibrium; the gas-grain coupling correction to
T_dust at higher density is a planned Phase-2 addition.

## Public API (summary)

| Function | Purpose |
|---|---|
| `solve_chem!(rho,e,HII,H2I,[HDI]; …, metals, dust, Z_rel, G0, A_V, N_H, N_H2, backend, precision)` | evolve chemistry+cooling over `dt`, in place |
| `solve_chem_device!(…)` | zero-copy variant for device-resident arrays (no host round-trip) |
| `solve_chem_mixing!(…)` | recombination with optional Lyα-mixing (early-universe) |
| `cooling_edot(nHI,nHII,nHeI,nde,nH2,nHD,T,z; metals,nH,Gamma_PE_vol,Lambda_gr_vol)` | net volumetric cooling rate `[erg/s/cm³]` |
| `temperature_from_reduced(rho,e,HII,H2I; fh)` | gas temperature from the reduced state |
| `fg20_uvb()` | the Faucher-Giguère 2020 UV background |
| `T_dust_eq(G0, A_V, T_CMB)` | local equilibrium dust temperature `[K]` |
| `k_H2_dust(T_gas, T_dust, Z_rel)` | H₂ formation rate on grains `[cm³/s]` |
| `k_gr_recomb_HII(T_gas, G0, Z_rel, n_e)` | grain-assisted HII recombination `[cm³/s]` |
| `k_H2_LW_eff(G0, N_H2, N_H, Z_rel)` | LW H₂ photodissociation with shielding `[s⁻¹]` |
| `Gamma_PE(T_gas, G0, Z_rel, n_e)` | photoelectric heating per H nucleus `[erg/s]` (× n_H → volumetric) |
| `Lambda_gr(T_gas, T_dust, n_H, Z_rel)` | gas-grain collisional coupling `[erg/cm³/s]` |
| `Lambda_dust(T_dust, Z_rel, n_H)` | dust thermal emission `[erg/cm³/s]` (diagnostic) |
| (EmissionKernels) `emiss_*`, `metal_line_emissivities`, `MetalAbundances`, `metal_abund` | per-channel / per-line emission |

## Validation

The table-free formulas are parity-checked against grackle's own analytic rates via a
C-grackle oracle (macOS suite); recombination is validated to <0.1% vs CAMB/HyRec; the
metal-line layer is parity-checked vs the Glover & Jappsen / `metal_cooling.pro` reference.
CI runs the table-free suites on Linux (the oracle parity suites are macOS-gated).

## License

University of Illinois/NCSA Open Source License (the Enzo Public License) — see `LICENSE`.
Extracted from the Vespa/EnzoNG project (`github.com/yipihey/enzo-dev`).

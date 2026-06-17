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

> **Scope vs grackle.** This is the *reduced* network + fine-structure metal-line cooling.
> It does **not** (yet) include dust, full tabulated metal cooling above 10⁴ K, or
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

## Public API (summary)

| Function | Purpose |
|---|---|
| `solve_chem!(rho,e,HII,H2I,[HDI]; …, metals, uvb, backend, precision)` | evolve chemistry+cooling over `dt`, in place |
| `solve_chem_device!(…)` | zero-copy variant for device-resident arrays (no host round-trip) |
| `solve_chem_mixing!(…)` | recombination with optional Lyα-mixing (early-universe) |
| `cooling_edot(nHI,nHII,nHeI,nde,nH2,nHD,T,z; metals,nH)` | net volumetric cooling rate `[erg/s/cm³]` |
| `temperature_from_reduced(rho,e,HII,H2I; fh)` | gas temperature from the reduced state |
| `fg20_uvb()` | the Faucher-Giguère 2020 UV background |
| (EmissionKernels) `emiss_*`, `metal_line_emissivities`, `MetalAbundances`, `metal_abund` | per-channel / per-line emission |

## Validation

The table-free formulas are parity-checked against grackle's own analytic rates via a
C-grackle oracle (macOS suite); recombination is validated to <0.1% vs CAMB/HyRec; the
metal-line layer is parity-checked vs the Glover & Jappsen / `metal_cooling.pro` reference.
CI runs the table-free suites on Linux (the oracle parity suites are macOS-gated).

## License

University of Illinois/NCSA Open Source License (the Enzo Public License) — see `LICENSE`.
Extracted from the Vespa/EnzoNG project (`github.com/yipihey/enzo-dev`).

# Integrating ChemistryKernels.jl into a simulation code

This guide covers the units/field conventions and the call patterns for driving the
network from a host code. For a C/Fortran (grackle-style) host, read this first, then
[`grackle_interface.md`](grackle_interface.md).

## The state a cell carries

The network advects three (optionally five) **mass-density** fields plus the specific
internal energy. All densities are `ρ·x` in the host's *code units* (set by
`density_units`, `length_units`, `time_units`); with those all `= 1.0` everything is CGS.

| field | meaning | convention |
|---|---|---|
| `rho`  | total gas mass density | read-only |
| `e`    | specific internal energy `[erg/g]` (code-unit `(L/t)²`) | evolved |
| `HII`  | ionized-H mass density `= x_HII · fh · ρ` | evolved |
| `H2I`  | H₂ mass density | **mass-equivalent: `2·n(H₂)·m_H`** |
| `HDI`  | HD mass density (if `deuterium=true`) | **`3·n(HD)·m_H`** |
| `HeII` | He⁺ mass density (if `helium=true`, mixing path) | `4·n(He⁺)·m_H` |

Neutral HI is reconstructed from conservation (`fh·ρ − HII − H2I`); nₑ from charge
conservation; H⁻/H₂⁺/D⁺ and He stages from algebraic/ionization equilibrium. `fh` is the
hydrogen mass fraction (default 0.76). The redshift is `z = 1/a_value − 1`, which sets the
CMB temperature for Compton and the cooling floor.

## Driving it

`solve_chem!` operates on flat per-cell arrays of any length `N` — pass your grid's cells
(interior only; skip ghosts) as contiguous vectors (or reshaped views):

```julia
solve_chem!(rho, e, HII, H2I [, HDI];
            a_value, dt,                       # cosmology time / step (code time units)
            density_units, length_units, time_units,
            hubble=71.0, Om=0.27, OL=0.73, fh=0.76,
            deuterium=false,
            hubble_expansion=false,            # adiabatic + Compton handled internally
            adot_over_a=NaN,                   # supply ȧ/a for exact cosmological steps
            metals=nothing,                    # see below
            uvb=nothing,                       # e.g. fg20_uvb()
            backend=:cpu, precision=Float64)
```

It sub-cycles each cell internally (10%-change limiter, stiff CMB-Compton split) and writes
`e`, `HII`, `H2I` (and `HDI`) back in place. It is **per-cell independent** — trivially
parallel; the kernel runs the whole array on the chosen backend.

## Analytic and hybrid paths

`solve_chem_analytic!` is a reduced primordial H + H₂ solver. It reconstructs H I and
electrons by conservation, closes H⁻ and H₂⁺ in quasi-steady state, integrates H II with a
linear-source Riccati solution, and integrates CMB-Compton exchange exponentially.
Non-Compton cooling and the temperature-dependent rate coefficients are frozen within each
substep; `dtfrac` controls their re-evaluation. This path deliberately excludes HD, dust,
metals, and the general UV background.

The current regression envelope is:

| problem | tested range | comparison gate |
|---|---|---|
| mean primordial IGM | z=1200→20 | H II and energy within 5% of `solve_chem!` at z=20; H₂ within ×3 |
| minihalo collapse | z=25, nH through 10⁴ cm⁻³ | H II within 5%, H₂ within 25%, terminal T=60–200 K |
| H recombination | z=900–1300 | <1% at z=1000–1100; <1.5% over the tested wider window |
| helium-aware recombination | z=1900–8000 | Saha at fully ionized epochs plus carried He II through HyRec-derived He I freeze-out |

These are validation windows, not runtime cutoffs. Compare with `solve_chem!` when applying
the analytic closure elsewhere. For collapse approaching three-body densities
(nH≈10⁸ cm⁻³), `solve_chem_hybrid_device_u16!` can switch dense cells to the full network;
validate `rho_switch` for the target problem. `solve_chem_analytic_mixing!` adds the
host-calibrated Lyα escape-density closure; with `f_alpha=0` it reduces to the local
analytic kernel.

Implementation and validation: [`src/fast.jl`](../src/fast.jl),
[`src/recombination_clumping.jl`](../src/recombination_clumping.jl), and
[`test/test_fast.jl`](../test/test_fast.jl).

### Metals (independent [α/Fe])

Pass per-cell number abundances `n(X)/n_H` as a NamedTuple of arrays:

```julia
metals = (C = aC, O = aO, Si = aSi, Fe = aFe)   # each a length-N vector
solve_chem!(...; metals=metals, fh=fh)
```

`nothing` (default) ⇒ primordial-only, bit-identical to no-metals. The metal cooling is
linear in each abundance, so you can build the per-cell vector from separately tracked
yield channels with `metal_abund(; solar, ccsn, ia, agb)` (see `EmissionKernels`).

### GPU

`backend=:metal` (Float32) or `:cuda` after `using Metal`/`using CUDA`. For
device-resident host fields, `solve_chem_device!(...)` runs the kernel in place on your
`CuArray`/`MtlArray`s — no host round-trip, no allocation.

### Differentiability

The math core is pure/allocation-free; `cooling_edot`, `metal_cooling_rate`, and the rate
fits are ForwardDiff-friendly (∂/∂T, ∂/∂abundance). Use for implicit solvers / SBI.

## Cooling rate directly

If you only want the net cooling (e.g. for your own integrator or a cooling-time):

```julia
Λ = cooling_edot(nHI, nHII, nHeI, nde, nH2, nHD, T, z; metals=ab, nH=nH)  # [erg/s/cm³], <0 ⇒ cooling
T = temperature_from_reduced(rho, e, HII, H2I; fh=fh)
```

## What's covered / not

Covered: H/He collisional excitation & ionization, recombination, bremsstrahlung, H₂
(Galli-Palla), HD, CMB-Compton, metal fine-structure (C/O/Si/Fe) with live-nₑ ion balance,
FG20 UVB photo-ionization/heating, HyRec-accurate recombination (+ optional Lyα mixing).
**Not** (yet): full tabulated metal cooling above 10⁴ K (the fine-structure channel tapers
out there), radiative-transfer heating, and general non-equilibrium He/H⁻ advection.

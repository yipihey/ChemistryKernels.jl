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
conservation; H⁻/H₂⁺/HeH⁺/D⁺ and He stages from algebraic/ionization equilibrium. `fh` is the
hydrogen mass fraction (default 0.76). The redshift is `z = 1/a_value − 1`, which sets the
CMB temperature for Compton and the cooling floor.

The default gas-phase H₂ formation network includes all three charged-particle
routes:

```text
H + e⁻ → H⁻ + γ;       H⁻ + H → H₂ + e⁻
H + H⁺ → H₂⁺ + γ;      H₂⁺ + H → H₂ + H⁺
He + H⁺ → HeH⁺ + γ;    HeH⁺ + H → He + H₂⁺ → H₂
```

The last route follows Hirata & Padmanabhan (2006). HeH⁺ is a trace,
short-lived intermediary solved in quasi-steady state, so it adds no advected
field and does not weaken the two-species storage advantage. Formation includes
spontaneous and CMB-stimulated radiative association and both CMB
photodissociation branches; destruction by electrons is included. The
`HeH⁺ + H` coefficient uses the low-temperature ab-initio calculation of
Bovino et al. (2011), frozen at its 1000 K validity boundary because this route
is negligible in hot gas.

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

## Full and reduced analytic paths

Use `solve_chem!` as the default, including primordial collapse into the
three-body regime. The full network now refreshes its algebraic intermediaries
every substep and stabilizes the asymptotic pair

```text
3 H ⇌ H₂ + H
```

without changing the advected state. When three-body formation (`k22`) and
H-impact dissociation (`k13`) supply more than 99% of the instantaneous H₂
activity and a substep spans at least ten pair-relaxation times, the network
places H₂ on the conservative fixed point

```text
2 k22 nHI³ = k13 nHI yH2,       nHI + yH2 = constant,
```

where `yH2 = 2 n(H₂)`. This removes the fully molecular absorbing state of a
lagged-HI backward-Euler sweep. Competing H⁻, H₂⁺, ion, LW, or dust channels
automatically keep the general network update active. The implementation and
Float64/Float32 regressions are in
[`src/network_step.jl`](../src/network_step.jl) and
[`test/test_dense_threebody.jl`](../test/test_dense_threebody.jl).

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
| three-body formation | nH=10¹² cm⁻³, T=1500 K | H₂ within 1.5%; energy within 0.5% |
| three-body/dissociation balance | nH=10¹⁶ cm⁻³, T=3500 and 5000 K | H₂ within 1.5%; energy within 0.5% |

These are validation windows, not runtime cutoffs, but they are the evidence boundary.
The dense comparison is now an equivalent solve: the analytic path uses the same
coupled H₂ source/sink backward-Euler update as the full network and resolves
the pair-relaxation time when frozen coefficients would otherwise move the
formation/dissociation fixed point.

For 262,144 Float32 cells on the current Apple GPU, the analytic/full results are
`fH2=6.199×10⁻⁴ / 6.189×10⁻⁴` at `nH=10¹² cm⁻³`, 1500 K;
`0.7127 / 0.7124` at `nH=10¹⁶ cm⁻³`, 3500 K; and
`0.03666 / 0.03699` at `nH=10¹⁶ cm⁻³`, 5000 K. Energy differs by at most
0.2% in these cases. In the recorded run, the analytic path is respectively
1.98×, 1.10×, and 5.85× faster; the full path sustains approximately 256,
274, and 15.2 Mcell/s. The first two kernels complete in about one millisecond
and their ratio is correspondingly sensitive to device scheduling. On the CPU
used for the same run, the analytic speed-ups are 2.46×, 2.51×, and 1.12×.

At `nH=10¹⁸ cm⁻³`, the full Metal path reaches about 237 Mcell/s at 3500 K;
the coupled 5000 K thermal trajectory falls below 1 Mcell/s because thermal
subcycling dominates. The analytic path can exhaust its subcycle budget at
that density and is not part of the validated comparison. Reproduce both the
throughput and final-state comparison with
[`benchmark/dense_threebody.jl`](../benchmark/dense_threebody.jl); hardware and the
thermal trajectory determine absolute throughput.

The chemistry fixed point remains valid above the density where the present thermal model
is complete. The built-in H₂ cooling has the low-density/LTE collisional bridge but no
column-dependent H₂ line trapping or collision-induced continuum treatment. Supply those
effects in the host, or validate the thermal trajectory separately, before interpreting a
one-zone result at `nH≈10¹⁸ cm⁻³`.

`solve_chem_analytic_mixing!` adds the host-calibrated Lyα escape-density closure; with
`f_alpha=0` it reduces to the local analytic kernel.

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

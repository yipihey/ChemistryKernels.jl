# Exposing a grackle-compatible C / Fortran interface (recipe — *not provided*)

RAMSES, Gadget-4, Arepo and Enzo already call **grackle's C API in-process**. The lowest-
friction way to adopt ChemistryKernels as a drop-in is therefore to build a **C-ABI shared
library from this Julia package that mirrors grackle's public functions**, so a host code
keeps its existing call sites and merely relinks.

> This document explains *how to build that shim*. It deliberately does **not** ship a
> working shim — the marshalling and the grackle struct layout are host/version specific,
> and the species mapping (below) is a project decision. The pieces below are a sketch.

## 1. Target the grackle **API**, not its binary ABI

Provide a header that is *source-compatible* with `grackle.h` (`initialize_chemistry_data`,
`solve_chemistry`, `calculate_cooling_time`, `calculate_temperature`,
`calculate_pressure`, `calculate_gamma`, and the `chemistry_data` / `code_units` /
`grackle_field_data` structs). A code that `#include`s your header and links your `.so`
relinks with ~zero call-site changes. Do **not** try to spoof grackle's exact SONAME/ABI —
API-compatible relink is robust; binary ABI spoofing is brittle and version-locked.

## 2. `Base.@ccallable` entry points (sketch — not provided)

Mark thin Julia wrappers `@ccallable`; each `unsafe_wrap`s the host's field pointers
(zero-copy), converts code units, and calls `solve_chem!`:

```julia
# SKETCH — illustrative only, not a working shim.
Base.@ccallable function solve_chemistry(u::Ptr{CodeUnits},
                                         fd::Ptr{GrackleFieldData},
                                         dt::Cdouble)::Cint
    f = unsafe_load(fd)
    n = prod(unsafe_wrap_dims(f))                       # grid_dimension product
    HII = unsafe_wrap(Array, f.HII_density, n)          # … and rho, e, H2I, …
    # map code_units (density/length/time/a_value) → solve_chem! kwargs, then:
    solve_chem!(rho, e, HII, H2I; a_value=u.a, dt=dt,
                density_units=u.density_units, length_units=u.length_units,
                time_units=u.time_units, metals=metals_from(f))
    return 1                                             # grackle SUCCESS
end
Base.@ccallable calculate_temperature(...)::Cint  # → temperature_from_reduced
Base.@ccallable calculate_cooling_time(...)::Cint  # → e / cooling_edot
```

## 3. Build the library

- **`PackageCompiler.create_library`** (mature): emits `libchemistrykernels.so` + a C
  header + a bundled Julia runtime. Link it like any `.so`.
- **`juliac`** (Julia ≥1.12, experimental): AOT-compiles the `@ccallable` entry points into
  a smaller, trimmed, fast-startup `.so` without the full runtime — the better artifact as
  it matures.

Ship per-platform artifacts (linux-x86_64, linux-aarch64, macos-arm64) via GitHub Releases
(and optionally a JLL).

## 4. C/C++ and Fortran call patterns

Keep one scientific ABI: a small C header with separate Float32/Float64 entry points,
fixed-width integers, a pointer plus element count for every field, and a plain
`ck_units` struct. The wrapper must not retain a host pointer after returning. Initialize
the bundled Julia runtime once per process (normally once per MPI rank), warm the selected
solver, and finalize only at shutdown.

For modern Fortran (2003+), provide an `ISO_C_BINDING` module with `bind(C)` interfaces to
those exact symbols. Use `integer(c_size_t), value` for lengths, `real(c_float)` or
`real(c_double)` for arrays, and `real(c_double), value` for the timestep. A contiguous
multidimensional Fortran field can be passed as a flat native-order array: cells are
independent, so C versus Fortran indexing order does not change the chemistry.

For a legacy Fortran 77/90 host, keep compiler spelling and by-reference conventions out
of the scientific library. Add either (a) one small C shim exporting the required trailing-
underscore symbols and dereferencing scalar arguments or, preferably, (b) one modern
Fortran `bind(C)` bridge module called from the legacy routines. Test this bridge with the
same compiler flags and integer-width options as the production executable.

For an existing RAMSES/grackle integration, the Fortran module can mirror
`grackle_fortran_interface` so `cooling_module` call sites remain stable while the C shim
performs the reduced-species mapping.

## 5. Runtime and threading contract

- The host owns all arrays; `unsafe_wrap(...; own=false)` borrows them only for the call.
- Do not call Julia finalization between timesteps. Cache rate/cooling tables and compiled
  method state for the life of the process.
- Start with serialized entry into the library. Enable concurrent host calls only after
  testing the exact Julia-thread, OpenMP, and MPI configuration used in production.
- Return a stable integer error code across the ABI. Catch Julia exceptions at the wrapper
  boundary and copy a diagnostic into a caller-provided buffer or a thread-local query.
- Expose units and species conventions in the C header; never infer them from a host name.

## 6. Species mapping — the one project decision

grackle's `primordial_chemistry` modes advect more species (full He stages, H⁻, H₂⁺) than
this reduced network does. Two options for the shim:
- **Reduced-mode (recommended first):** consume grackle's field arrays, map the ones we
  advect (HII, H2I, HDI), and *reconstruct* the equilibrium species (H⁻, H₂⁺, D⁺, He
  stages) on output so the host's arrays stay populated — documenting that those are
  equilibrium values.
- **Full parity:** extend the network to advect the full set (only if a target code needs
  non-equilibrium He/H⁻).

## 7. Trust anchor

Ship a **conformance suite** that runs a grid of (T, ρ, z, species, metallicity) states
through both real `libgrackle` and the shim and asserts agreement to a documented
tolerance, plus a compatibility matrix of which grackle features/modes match and the known
gaps (dust, full metal tables, RT heating). This is what convinces adopters to switch.

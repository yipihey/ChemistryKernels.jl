"""
    ChemistryKernels

A table-free, KernelAbstractions.jl implementation of the primordial +
deuterium chemistry/cooling network of **Abel, Anninos, Zhang & Norman (1997,
New Astronomy 2, 181)** and **Anninos, Zhang, Abel & Norman (1997, New Astronomy
2, 209)** — the original Enzo primordial chemistry (the same physics later
packaged as the `grackle` library).  Reduced model: advect HII, H2I, HDI;
H⁻/H₂⁺/HeH⁺/D⁺ in algebraic equilibrium; helium in ionisation equilibrium (or
advected He⁺); nₑ from charge conservation; primordial only by default. The
default H₂ network includes the Hirata & Padmanabhan (2006) HeH⁺ catalytic
route with the updated Bovino et al. (2011) low-temperature proton-transfer rate.

Dust physics (enabled by `dust=true` in `solve_chem!` / `evolve_cell`): H₂
formation on grain surfaces (Cazaux & Tielens 2004), grain-assisted HII
recombination (Weingartner & Draine 2001), and Lyman-Werner H₂ photodissociation
with self-shielding (Draine & Bertoldi 1996) and dust attenuation live in
`rates_dust.jl` / `shielding.jl` here.  The thermal channels — photoelectric
heating (Bakes & Tielens 1994) and gas-grain collisional coupling (Hollenbach &
McKee 1989) — live in `EmissionKernels.cooling_dust` (imported as `Gamma_PE` and
`Lambda_gr`) alongside the metal-line cooling, keeping the architecture consistent.
Local equilibrium `T_dust` (Hollenbach & McKee 1979, `dust_temperature.jl`);
no RHD required.

Also implements density-dependent Lyα-mixing recombination (`solve_chem_mixing!`)
for early-Universe / PMF science, where the Peebles C-factor escape rate uses a
host-supplied smoothed neutral density instead of the cell-local value.

Design contract (mirrors `PPMKernels`/`PoissonKernels`):

  * **One source, two devices.** Every compute kernel is a precision-generic
    `@kernel` parameterised on `T = eltype(output)`. The CPU backend runs f64 and
    f32; Metal runs f32-only. f32 CPU↔Metal agreement is the parity gate. Rate and
    cooling coefficients are checked against the published Abel/Anninos et al.
    (1997) analytic fits (and, for recombination, against HyRec-2).

  * **Analytic CPU reference; tabulated GPU production path.** CPU calls evaluate every
    rate/cooling coefficient directly from its published fit by default. GPU extensions
    may provide cached log–log tables when none are supplied; this is both the fastest
    path and avoids generating enormous device programs from the analytic fits. Explicit
    `rate_tables` / `cool_tables` always take precedence.

  * **f32-safe representation.** State is carried as physical abundances relative
    to the hydrogen number density (`x_i = n_i/n_H`, dimensionless and O(1) for
    the dominant species) with reaction frequencies `k·n_H` (s⁻¹), keeping
    products in f32 range without coupling to any host code-unit system.

  * **AD-friendly.** The math core (rates, cooling, network update) is written as
    pure, allocation-free functions of `(state, T)`; mutation is confined to the
    outer driver, so Phase-3 differentiability (Enzyme) is a later add-on.

  * **Backend by name.** `backend(:cpu)` always works; `backend(:metal)` resolves
    after `using Metal`. Allocation/host-transfer go through `device_zeros` /
    `to_device` / `to_host`, specialised by the Metal extension.
"""
module ChemistryKernels

using KernelAbstractions
const KA = KernelAbstractions
import Adapt

# Radiative-channel physics (cooling coefficients, metal lines, CMB Compton) now lives
# in the foundation package EmissionKernels; ChemistryKernels depends on it and its
# `cooling_edot` delegates to `EmissionKernels.cooling_rate_total`.  Import the names the
# network uses bare (comp*_cmb in the subcycle, MetalAbundances in the kernels) and the
# cooling sum; the backend machinery is NOT imported (each package owns its own).
using EmissionKernels: ceHI, ceHeI, ceHeII, ciHI, ciHeI, ciHeII, ciHeIS,
      reHII, reHeII1, reHeII2, reHeIII, brem,
      GAHI, GAH2, GAHe, GAHp, GAel, H2LTE, HDlte, HDlow,
      comp1_cmb, comp2_cmb,
      MetalAbundances, metal_abund, metal_cooling_rate, cooling_rate_total, cooling_rate_total_tab,
      Gamma_PE, Lambda_gr, Lambda_dust

export backend, has_backend, device_zeros, to_device, to_host
# re-export the cooling/metal surface so `using ChemistryKernels` still sees it
export ceHI, ceHeI, ceHeII, ciHI, ciHeI, ciHeII, ciHeIS,
       reHII, reHeII1, reHeII2, reHeIII, brem,
       GAHI, GAH2, GAHe, GAHp, GAel, H2LTE, HDlte, HDlow, comp1_cmb, comp2_cmb,
       MetalAbundances, metal_abund, metal_cooling_rate,
       Gamma_PE, Lambda_gr, Lambda_dust

# ── backend registry ─────────────────────────────────────────────────────────
const _BACKENDS = Dict{Symbol,Any}(:cpu => CPU())

"Register a KernelAbstractions backend under `name` (used by the Metal extension)."
register_backend!(name::Symbol, be) = (_BACKENDS[name] = be)

"True when backend `name` is available (`:metal` needs `using Metal` first)."
has_backend(name::Symbol) = haskey(_BACKENDS, name)

# Backend extensions may provide persistent default tables. CPU and backends that do
# not opt in retain the exact analytic reference (`nothing`). These hooks live here so
# every public host/device launcher can resolve the same policy without depending on a
# particular GPU package.
_default_rate_tables(::Val, ::Type) = nothing
_default_cool_tables(::Val, ::Type) = nothing

@inline _resolve_rate_tables(name::Symbol, P::Type, rt) =
    rt === nothing ? _default_rate_tables(Val(name), P) : rt
@inline function _resolve_tables(name::Symbol, P::Type, rt, ct)
    return (_resolve_rate_tables(name, P, rt),
            ct === nothing ? _default_cool_tables(Val(name), P) : ct)
end

"""
    backend(name::Symbol = :cpu)

The KernelAbstractions backend registered under `name`. `:cpu` is always
available; `:metal` requires `using Metal` (Apple Silicon) to have loaded the
`ChemistryKernelsMetalExt` extension.
"""
function backend(name::Symbol = :cpu)
    return get(_BACKENDS, name) do
        error("Chemistry backend :$name is not available. " *
              (name === :metal ? "Run `using Metal` first (Apple Silicon only)." :
               "Known backends: $(collect(keys(_BACKENDS)))."))
    end
end

# ── device array helpers (specialised by the Metal extension) ────────────────
"A zero-filled array of element type `T` and shape `dims` on backend `be`."
device_zeros(::CPU, ::Type{T}, dims::Dims) where {T} = zeros(T, dims)

"Copy host array `a` onto backend `be`, converting to element type `T`."
function to_device(be, a::AbstractArray, ::Type{T} = eltype(a)) where {T}
    d = device_zeros(be, T, size(a))
    copyto!(d, convert(Array{T}, a))
    return d
end

"`to_host(a)` — a plain host `Array` copy of a device array; synchronizes first."
function to_host(a::AbstractArray)
    KA.synchronize(KA.get_backend(a))
    return Array(a)
end

# ── component sources (added as each Wave lands) ─────────────────────────────
include("constants.jl")
include("kernelgen.jl")
include("uv_background.jl")
include("representation.jl")
# Wave 1 — table-free rate + cooling formula kernels.
include("rates_atomic.jl")
include("rates_h2.jl")
include("rates_deuterium.jl")
include("rates_cmb.jl")
# Wave 1 dust extension — grain surface rate coefficients (H₂-on-dust, grain-assisted
# HII recombination), local equilibrium T_dust, and LW shielding.  The thermal channels
# (Gamma_PE, Lambda_gr, Lambda_dust) live in EmissionKernels (imported above).
include("rates_dust.jl")
include("dust_temperature.jl")
include("shielding.jl")
# Wave 1 compact storage — UInt16 log₂-encoded species fractions (2 B/cell vs 4 B f32).
# Codec: u=0 → X≈7.73e-34, u=65535 → X=1.0; ≈0.12%/ULP; 33 dex range.
include("log2_species.jl")
# Wave 1 cooling coefficients (ceHI…brem, GA*, HD*, comp*_cmb) + the metal lines now
# live in EmissionKernels (imported above).
# Wave 2 — local composed: temperature (mmw/H2-γ) + algebraic equilibrium species.
include("temperature.jl")
include("equilibrium.jl")
# Wave 3 — assemblers: cooling rate + one backward-Euler network sweep.
include("edot.jl")
include("network_step.jl")
# Wave 4 — driver: Peebles recombination, the sub-cycle, and the host boundary.
include("recombination.jl")
include("rate_tables.jl")     # optional log–log rate lookup (uses recombination + rates_*)
include("subcycle.jl")
include("fast.jl")                # reduced fast analytic H+H2 mode (evolve_cell_fast)
include("solve.jl")
include("binned.jl")              # phase-space-binned solve (one stiff solve per occupied bin)
# Wave 5 — Lyα-mixing recombination for early-Universe / PMF science.
include("tables.jl")
include("recombination_clumping.jl")
# Host-side utilities (cell sorting for GPU warp coherence, etc.).
include("utils.jl")

end # module ChemistryKernels

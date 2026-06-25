# solve.jl — host boundary + device launcher.  Converts a host's code-unit fields
# to physical CGS, sub-cycles every cell on any backend, and writes the evolved
# fields back — the reduced-network step of the Abel/Anninos et al. 1997 network.
# Unit convention = comoving_coordinates=0: ρ_cgs = field·density_units (NO a³),
# e_cgs = e·(length/time)², t_s = dt·time_units, z = 1/a_value − 1 (sets the CMB).

export solve_chem!

# Per-cell: convert code units → CGS, evolve, convert back. Keeps fields in place.
@kernel function _evolve_k!(e, HII, H2I, HDI, @Const(rho),
                            du, vu2, tu, dt, z, hubble, Om, OL, fh, deut, hexp, aoa,
                            @Const(aC), @Const(aO), @Const(aSi), @Const(aFe), hasmetals, rtab, ctab,
                            @Const(aZrel), @Const(aG0), @Const(aAV),
                            @Const(aNH), @Const(aNH2), hasdust)
    i = @index(Global)
    @inbounds begin
        T   = eltype(e)
        hd_in = deut ? HDI[i]*du : zero(T)
        # metals: per-cell number abundances n(X)/n_H (dimensionless, no unit conv).
        # Always a concrete MetalAbundances{T}; zeros (hasmetals=false) ⇒ the metal
        # term short-circuits to exactly 0 (bit-identical, zero-cost).
        mab = hasmetals ? MetalAbundances{T}(aC[i], aO[i], aSi[i], aFe[i]) :
                          MetalAbundances{T}()
        # dust: per-cell physical parameters (dimensionless / CGS column densities).
        # zeros (hasdust=false) → dust=false branch inside evolve_cell, zero-cost.
        Z_rel = hasdust ? aZrel[i] : zero(T)
        G0_i  = hasdust ? aG0[i]  : zero(T)
        AV_i  = hasdust ? aAV[i]  : zero(T)
        NH_i  = hasdust ? aNH[i]  : zero(T)
        NH2_i = hasdust ? aNH2[i] : zero(T)
        en, hii, h2, hd, _ = evolve_cell(rho[i]*du, e[i]*vu2, HII[i]*du, H2I[i]*du,
                                      hd_in, dt*tu, z; hubble=hubble, Om=Om, OL=OL,
                                      fh=fh, deuterium=deut,
                                      hubble_expansion=hexp, adot_over_a=aoa, metals=mab,
                                      rate_tables=rtab, cool_tables=ctab,
                                      dust=hasdust, Z_rel=Z_rel, G0=G0_i,
                                      A_V=AV_i, N_H=NH_i, N_H2=NH2_i)
        e[i]   = en  / vu2
        HII[i] = hii / du
        H2I[i] = h2  / du
        deut && (HDI[i] = hd / du)
    end
end

"""
    solve_chem!(rho, e_int, HII, H2I, [HDI]; a_value, dt, density_units,
                length_units, time_units, hubble=71, Om=0.27, OL=0.73, fh=0.76,
                deuterium=false, dust=false, Z_rel=nothing, G0=nothing, A_V=nothing,
                N_H=nothing, N_H2=nothing, backend=:cpu, precision=Float64)

Evolve the v2026 reduced primordial+D chemistry/cooling over `dt` (code time
units) for every cell, updating `e_int`, `HII`, `H2I` (and `HDI` if `deuterium`)
in place.  `rho` is read-only; `HII`/`H2I`/`HDI` are MASS densities ρ·x (the
network's mass-equivalent convention) in the host code units defined by
`density_units`/`length_units`/`time_units`.  The engine is a KA kernel
(`backend=:cpu` or `:metal`) at `precision` (Float64/Float32).

When `dust = true`, per-cell dust parameters must be supplied as length-`n` vectors:
- `Z_rel` : metallicity relative to solar (dust-to-gas ratio ∝ Z_rel)
- `G0`    : FUV field [Habing units]
- `A_V`   : visual extinction [mag]
- `N_H`   : H column density [cm⁻²] (optional; `nothing` → zeros → no LW dust shielding)
- `N_H2`  : H₂ column density [cm⁻²] (optional; `nothing` → zeros → no self-shielding)
"""
function solve_chem!(rho::AbstractVector, e_int::AbstractVector,
                     HII::AbstractVector, H2I::AbstractVector,
                     HDI::Union{Nothing,AbstractVector} = nothing;
                     a_value::Real, dt::Real, density_units::Real,
                     length_units::Real, time_units::Real,
                     hubble::Real = 71.0, Om::Real = 0.27, OL::Real = 0.73,
                     fh::Real = 0.76, deuterium::Bool = false,
                     hubble_expansion::Bool = false, adot_over_a::Real = NaN,
                     metals = nothing, rate_tables = nothing, cool_tables = nothing,
                     dust::Bool = false,
                     Z_rel = nothing, G0 = nothing, A_V = nothing,
                     N_H = nothing, N_H2 = nothing,
                     backend::Symbol = :cpu, precision::Type = Float64)
    n  = length(rho)
    @assert length(e_int) == n && length(HII) == n && length(H2I) == n
    deut = deuterium && HDI !== nothing
    deut && @assert length(HDI) == n

    P   = precision
    be  = ChemistryKernels.backend(backend)
    du  = P(density_units)
    vu2 = P((length_units / time_units)^2)
    tu  = P(time_units)
    z   = P(1.0 / a_value - 1.0)

    # metals: optional NamedTuple (C,O,Si,Fe) of per-cell number abundances n(X)/n_H.
    hasmetals = metals !== nothing
    if hasmetals
        @assert length(metals.C)==n && length(metals.O)==n &&
                length(metals.Si)==n && length(metals.Fe)==n
    end
    d_aC = hasmetals ? to_device(be, collect(metals.C),  P) : device_zeros(be, P, (n,))
    d_aO = hasmetals ? to_device(be, collect(metals.O),  P) : device_zeros(be, P, (n,))
    d_aSi= hasmetals ? to_device(be, collect(metals.Si), P) : device_zeros(be, P, (n,))
    d_aFe= hasmetals ? to_device(be, collect(metals.Fe), P) : device_zeros(be, P, (n,))

    # dust: optional per-cell physical parameters. When dust=false, zero pads are
    # uploaded but the hasdust=false flag prevents any use inside the kernel.
    hasdust = dust
    if hasdust
        @assert Z_rel !== nothing && G0 !== nothing && A_V !== nothing
        @assert length(Z_rel)==n && length(G0)==n && length(A_V)==n
        @assert N_H  === nothing || length(N_H)  == n
        @assert N_H2 === nothing || length(N_H2) == n
    end
    d_Zrel = hasdust ? to_device(be, collect(Z_rel), P) : device_zeros(be, P, (n,))
    d_G0   = hasdust ? to_device(be, collect(G0),    P) : device_zeros(be, P, (n,))
    d_AV   = hasdust ? to_device(be, collect(A_V),   P) : device_zeros(be, P, (n,))
    d_NH   = (hasdust && N_H  !== nothing) ? to_device(be, collect(N_H),  P) : device_zeros(be, P, (n,))
    d_NH2  = (hasdust && N_H2 !== nothing) ? to_device(be, collect(N_H2), P) : device_zeros(be, P, (n,))

    d_rho = to_device(be, collect(rho),   P)
    d_e   = to_device(be, collect(e_int), P)
    d_HII = to_device(be, collect(HII),   P)
    d_H2I = to_device(be, collect(H2I),   P)
    d_HDI = deut ? to_device(be, collect(HDI), P) : device_zeros(be, P, (n,))

    _evolve_k!(be)(d_e, d_HII, d_H2I, d_HDI, d_rho, du, vu2, tu,
                   P(dt), z, P(hubble), P(Om), P(OL), P(fh), deut,
                   hubble_expansion, P(adot_over_a),
                   d_aC, d_aO, d_aSi, d_aFe, hasmetals, rate_tables, cool_tables,
                   d_Zrel, d_G0, d_AV, d_NH, d_NH2, hasdust; ndrange = n)

    e_int .= to_host(d_e)
    HII   .= to_host(d_HII)
    H2I   .= to_host(d_H2I)
    deut && (HDI .= to_host(d_HDI))
    return nothing
end

"""
    solve_chem_device!(rho, e_int, HII, H2I, [HDI]; a_value, dt, density_units,
                       length_units, time_units, …, backend=:cuda, precision=Float64)

Zero-copy variant of [`solve_chem!`](@ref): the arrays are ALREADY device arrays
(e.g. a CuArray view onto a host code's device-resident state) of element type
`precision`.  Launches the per-cell evolve kernel IN PLACE — no host round-trip, no
allocation beyond a zero `HDI` pad when `deuterium=false`.  Caller owns the arrays
and any unit/precision conversion.  `e_int`/`HII`/`H2I`/`HDI` are updated in place.
"""
function solve_chem_device!(rho, e_int, HII, H2I, HDI = nothing;
                            a_value::Real, dt::Real, density_units::Real,
                            length_units::Real, time_units::Real,
                            hubble::Real = 71.0, Om::Real = 0.27, OL::Real = 0.73,
                            fh::Real = 0.76, deuterium::Bool = false,
                            hubble_expansion::Bool = false, adot_over_a::Real = NaN,
                            rate_tables = nothing, cool_tables = nothing,
                            backend::Symbol = :cuda, precision::Type = Float64)
    n  = length(rho)
    P  = precision
    be = ChemistryKernels.backend(backend)
    du  = P(density_units); vu2 = P((length_units / time_units)^2)
    tu  = P(time_units);    z   = P(1.0 / a_value - 1.0)
    deut = deuterium && HDI !== nothing
    d_HDI = deut ? HDI : device_zeros(be, P, (n,))
    dz = device_zeros(be, P, (n,))     # metal and dust abundance pads (zero-copy path: no dust/metals)
    _evolve_k!(be)(e_int, HII, H2I, d_HDI, rho, du, vu2, tu,
                   P(dt), z, P(hubble), P(Om), P(OL), P(fh), deut,
                   hubble_expansion, P(adot_over_a),
                   dz, dz, dz, dz, false, rate_tables, cool_tables,
                   dz, dz, dz, dz, dz, false; ndrange = n)
    KA.synchronize(be)
    return nothing
end
export solve_chem_device!

# fast.jl — the FAST analytic (H + H₂) chemistry mode.
#
# A reduced sibling of `evolve_cell`: it advects only x_HII and x_H2 (+ energy), with
# everything else closed-form — HI by conservation, n_e = n_HII (helium neutral), and
# H⁻/H₂⁺ as algebraic quasi-equilibria.  It drops the deuterium network AND the helium
# ionization sub-network of the full solver, and folds the species update inline (no
# per-substep `network_step` call), so it is meaningfully cheaper per substep while
# reproducing the full network across z=1000→20 AND the first cooling halos.
#
# The relic electron freeze-out that catalyzes halo H₂ is captured by the k57/k58
# H-H / H-He collisional-ionization FLOOR (~1e-20): with n_HI² at halo densities it is
# the persistent electron source that balances recombination at x_e ≈ √(k57 n_HI²/k2)
# — validated against the full network and the Grackle reduced oracle.
#
# Same contract as `evolve_cell`: pure, allocation-free, precision-generic (R=typeof(e)),
# GPU-safe.  Returns (e, HII_m, H2I_m, ttot); no HDI (primordial H+H₂ only).

export evolve_cell_fast, evolve_cell_analytic

# Riccati closed form for  dy/dt = scH − k2·y²  (recombination vs an ionization source):
# relaxes y toward y_eq = √(scH/k2) analytically over the whole step — no subcycle.
@inline function _riccati(y0::R, k2::R, scH::R, dt::R) where {R}
    # scH is the true ionization source (k57·n_HI² + …), legitimately ~1e-26 at IGM
    # densities — do NOT floor it (a floor spuriously raises y_eq and re-ionizes the gas).
    scH <= zero(R) && return y0 / (one(R) + k2*y0*dt)      # pure recombination (no source)
    yeq = sqrt(scH / k2)
    th  = tanh(min(k2 * yeq * dt, R(30)))
    return yeq * (y0 + yeq*th) / (yeq + y0*th)
end

"""
    evolve_cell_fast(rho, e, HII_m, H2I_m, dt, z; hubble, Om, OL, fh,
                     hubble_expansion, adot_over_a, cool_tables, itcap, dtfrac)
        -> (e, HII_m, H2I_m, ttot)

Sub-cycle one cell over `dt` [s] with the reduced H+H₂ network (see file header).
`rho` [g/cm³], `e` [erg/g], `HII_m`/`H2I_m` = species mass densities ρ·x (network
convention, `H2I_m` = 2·n(H₂)·m_H).  Pure; ~2× cheaper per substep than `evolve_cell`.
"""
@inline function evolve_cell_fast(rho, e, HII_m, H2I_m, dt, z;
                                  hubble = 71.0, Om = 0.27, OL = 0.73,
                                  fh = FH_DEFAULT, hubble_expansion::Bool = false,
                                  adot_over_a = NaN, rate_tables = nothing,
                                  cool_tables = nothing,
                                  itcap::Int = _SUB_ITMAX, dtfrac::Real = 0.1)
    R    = typeof(e)
    mh   = R(MH); tiny = R(_SUB_TINY)
    # domain guard: f16 hydro can hand rho <= 0 / e <= 0 / X*rho < 0 at extreme
    # shock and void frontiers; unguarded they reach device sqrt/log/pow
    # (DomainError aborts the kernel).  Clamp to physical-positive; the T floor
    # in gas_temperature then makes the cell a benign ~1 K near-vacuum.
    rho  = ifelse(rho > R(1.0e-35), rho, R(1.0e-35))   # ifelse: NaN>x is false
    e    = ifelse(e > tiny, e, tiny)                    # (max() passes NaN through)
    HII_m = ifelse(HII_m > zero(R), HII_m, zero(R))
    H2I_m = ifelse(H2I_m > zero(R), H2I_m, zero(R))
    d    = rho / mh
    z0   = R(z)
    Hz0  = hubble_z_of(z0; hubble = hubble, Om = Om, OL = OL)
    Hz_ad = isnan(adot_over_a) ? Hz0 : R(adot_over_a)
    evolve_z = !isnan(adot_over_a)
    fhd  = R(fh) * d                        # total H-nuclei number density n_H

    yHeI = (one(R) - R(fh)) * d              # neutral helium (×4 convention: yHeI = 4·n_He)
    yHII = HII_m / mh
    yH2I = H2I_m / mh                        # = 2·n(H₂)
    yHI  = max(fhd - yHII - yH2I, tiny)
    yde  = yHII                              # n_e = n_HII (helium neutral)

    Tc0  = comp2_cmb(z0); c10 = comp1_cmb(z0)
    f    = R(dtfrac)
    ttot = zero(R); iter = 0
    @inbounds while ttot < dt && iter < itcap
        iter += 1
        rem = dt - ttot
        if evolve_z
            zt = (one(R) + z0) * exp(-Hz_ad * ttot) - one(R)
            Tc = comp2_cmb(zt); c1 = comp1_cmb(zt); Hz = hubble_z_of(zt; hubble=hubble, Om=Om, OL=OL)
        else
            zt = z0; Tc = Tc0; c1 = c10; Hz = Hz0
        end

        T = gas_temperature(rho, e, yHI, yHII, yHeI/R(4), tiny, tiny, yde,
                            tiny, yH2I/R(2), tiny; gamma = GAMMA_DEFAULT)

        # ── reduced rate set (~17): H recomb/ionization + BOTH H₂-formation channels in
        # closed form.  The sub-dominant H⁻ destruction terms (k14,k16-k19) are dropped
        # (they matter only when x_HII~1 at recombination, where f_H2≪1e-6 anyway).
        if rate_tables === nothing
            k2   = peebles_k2(T, yHI, Hz)
            kb1s = beta1s_freq(Tc) * k2 / (recfast_alpha(T) * R(1.0e6))
            k1v=k1(T); k7v=k7(T); k8v=k8(T); k9v=k9(T); k10v=k10(T); k11v=k11(T); k12v=k12(T)
            k13v=k13(T); k15v=k15(T); k22v=k22(T); k57v=k57(T); k58v=k58(T)
            k27v=k27_cmb(Tc); k28v=k28_cmb(Tc)
        else
            K = table_rates(rate_tables, T, yHI, Hz, cmb_rates(Tc))
            k2=K.k2; kb1s=K.k_beta1s; k1v=K.k1; k7v=K.k7; k8v=K.k8; k9v=K.k9; k10v=K.k10
            k11v=K.k11; k12v=K.k12; k13v=K.k13; k15v=K.k15; k22v=K.k22; k57v=K.k57; k58v=K.k58
            k27v=K.k27; k28v=K.k28
        end

        # cooling (atomic + H₂ + Compton; no HD/metals in this mode)
        edot = cooling_edot(yHI, yHII, yHeI/R(4), yde, yH2I/R(2), zero(R), T, zt;
                            nH = fhd, cool_tables = cool_tables)
        if T <= R(1.01)*R(MIN_TEMPERATURE) && edot < zero(R); edot = zero(R); end
        hubble_expansion && (edot -= R(2) * Hz_ad * e * rho)

        # ── HII: recomb (k2) vs the k57/k58 collisional-ionization floor + CMB photo-ion ──
        scH = k1v*yHI*yde + k57v*yHI*yHI + k58v*yHI*yHeI/R(4) + kb1s*yHI
        acH = k2*yde
        # ── H₂: H⁻ channel + H₂⁺ channel (both closed-form) + 3-body, vs dissociation ──
        #   n(H⁻)  = k7·n_HI·n_e / [(k8+k15)·n_HI + k27]         (→ H₂ via k8)
        #   n(H₂⁺) = k9·n_HI·n_HII / [k10·n_HI + k28]            (→ H₂ via k10; k28 kills it at high z)
        nHM   = k7v*yHI*yde  / ((k8v + k15v)*yHI + k27v + tiny)
        nH2II = k9v*yHI*yHII / (k10v*yHI + k28v + tiny)
        scH2  = R(2)*(k8v*nHM*yHI + k10v*nH2II*yHI + k22v*yHI*yHI*yHI)
        acH2  = k13v*yHI + k11v*yHII + k12v*yde

        # substep size: 10%-change on n_e and e ONLY (as the full network does) — H₂ is
        # updated implicitly (backward-Euler, unconditionally stable), so it needs NO step
        # limit; limiting on H₂ would force tiny substeps while it builds and kill the speed.
        dHIIdt = scH - acH*yHII
        dtit = min(_step_f(yde, dHIIdt, f), rem, R(0.5)*dt)
        edot_c = -c1 * (T - Tc) * yde; edot_rest = edot - edot_c
        Kc = c1 * yde * (T / e) / rho; stiff = Kc * rem > one(R)
        de_spec = (stiff ? edot_rest : edot) / rho
        dtit = min(dtit, _step_f(e, de_spec, f))

        # energy update (implicit Compton when stiff)
        if stiff
            B = (c1*yde*Tc + edot_rest) / rho
            e = (e + B*dtit) / (one(R) + Kc*dtit)
        else
            e = e + (edot/rho)*dtit
        end
        e = max(e, tiny)

        # species update (backward-Euler, inline) + conservation
        yHII = (yHII + scH*dtit) / (one(R) + acH*dtit)
        yH2I = (yH2I + scH2*dtit) / (one(R) + acH2*dtit)
        # keep physical: 0 ≤ yH2I, yHII with yHII + yH2I ≤ n_H
        yH2I = yH2I < tiny ? tiny : (yH2I > fhd ? fhd : yH2I)
        yHII = yHII < tiny ? tiny : (yHII > fhd - yH2I ? fhd - yH2I : yHII)
        yHI  = max(fhd - yHII - yH2I, tiny)
        yde  = yHII

        ttot += dtit
    end
    return e, yHII*mh, yH2I*mh, ttot
end

"""
    evolve_cell_analytic(rho, e, HII_m, H2I_m, dt, z; hubble, Om, OL, fh,
                         hubble_expansion, adot_over_a, cool_tables, itcap, dtfrac)
        -> (e, HII_m, H2I_m, ttot)

Fully closed-form variant of [`evolve_cell_fast`](@ref) — **no stiff/BDF sub-cycling
of the chemistry**.  Each step advances x_HII by the exact Riccati solution
(recombination vs the k57/k58 ionization floor), x_H2 by a pure formation quadrature
(H₂ is only formed below ~10⁴ K — the collisional-dissociation channels are frozen),
and the energy by the analytic Compton exponential + explicit non-Compton cooling.
The step is limited ONLY by the non-Compton cooling time (the Compton stiffness is
handled analytically), so Compton-locked gas takes a single step.  Same accuracy as
`evolve_cell_fast` at a fraction of the substeps; **this is the default fast path**
(`evolve_cell_fast` is the accurate subcycled fallback).  On an RTX A6000 (128³, f32)
it runs at ~4 Gcell/s — ~4× the full `evolve_cell` network.

`rate_tables` (an optional `RateTables` from `build_rate_tables`) swaps the per-iteration
analytic rate fits for a log–log table lookup; it helps the full network but is a wash-to-
slower here (this mode is dominated by the cooling/temperature transcendentals and its low
iteration count, not the rate fits), so the fits remain the default.
"""
@inline function evolve_cell_analytic(rho, e, HII_m, H2I_m, dt, z;
                                      hubble = 71.0, Om = 0.27, OL = 0.73,
                                      fh = FH_DEFAULT, hubble_expansion::Bool = false,
                                      adot_over_a = NaN, rate_tables = nothing,
                                      cool_tables = nothing,
                                      itcap::Int = _SUB_ITMAX, dtfrac::Real = 0.1)
    R    = typeof(e)
    mh   = R(MH); tiny = R(_SUB_TINY)
    # domain guard: f16 hydro can hand rho <= 0 / e <= 0 / X*rho < 0 at extreme
    # shock and void frontiers; unguarded they reach device sqrt/log/pow
    # (DomainError aborts the kernel).  Clamp to physical-positive; the T floor
    # in gas_temperature then makes the cell a benign ~1 K near-vacuum.
    rho  = ifelse(rho > R(1.0e-35), rho, R(1.0e-35))   # ifelse: NaN>x is false
    e    = ifelse(e > tiny, e, tiny)                    # (max() passes NaN through)
    HII_m = ifelse(HII_m > zero(R), HII_m, zero(R))
    H2I_m = ifelse(H2I_m > zero(R), H2I_m, zero(R))
    d    = rho / mh
    z0   = R(z)
    Hz0  = hubble_z_of(z0; hubble = hubble, Om = Om, OL = OL)
    Hz_ad = isnan(adot_over_a) ? Hz0 : R(adot_over_a)
    evolve_z = !isnan(adot_over_a)
    fhd  = R(fh) * d

    yHeI = (one(R) - R(fh)) * d
    yHII = HII_m / mh
    yH2I = H2I_m / mh
    yHI  = max(fhd - yHII - yH2I, tiny)
    yde  = yHII

    Tc0  = comp2_cmb(z0); c10 = comp1_cmb(z0)
    f    = R(dtfrac)
    ttot = zero(R); iter = 0
    @inbounds while ttot < dt && iter < itcap
        iter += 1
        rem = dt - ttot
        if evolve_z
            zt = (one(R) + z0) * exp(-Hz_ad * ttot) - one(R)
            Tc = comp2_cmb(zt); c1 = comp1_cmb(zt); Hz = hubble_z_of(zt; hubble=hubble, Om=Om, OL=OL)
        else
            zt = z0; Tc = Tc0; c1 = c10; Hz = Hz0
        end

        T = gas_temperature(rho, e, yHI, yHII, yHeI/R(4), tiny, tiny, yde,
                            tiny, yH2I/R(2), tiny; gamma = GAMMA_DEFAULT)

        # rate coefficients: lean analytic fits by default; the optional log–log rate
        # table (uniform across the warp ⇒ the branch compiles away) swaps the ~13
        # per-iteration transcendental fits for a branchless interpolation — the GPU hot
        # path.  Only the pure-T rates this mode needs are pulled from the NamedTuple.
        if rate_tables === nothing
            k2   = peebles_k2(T, yHI, Hz)
            kb1s = beta1s_freq(Tc) * k2 / (recfast_alpha(T) * R(1.0e6))
            k1v=k1(T); k7v=k7(T); k8v=k8(T); k9v=k9(T); k10v=k10(T); k15v=k15(T)
            k22v=k22(T); k57v=k57(T); k58v=k58(T); k27v=k27_cmb(Tc); k28v=k28_cmb(Tc)
        else
            K = table_rates(rate_tables, T, yHI, Hz, cmb_rates(Tc))
            k2=K.k2; kb1s=K.k_beta1s; k1v=K.k1; k7v=K.k7; k8v=K.k8; k9v=K.k9
            k10v=K.k10; k15v=K.k15; k22v=K.k22; k57v=K.k57; k58v=K.k58
            k27v=K.k27; k28v=K.k28
        end

        # cooling & the Compton/non-Compton split
        edot = cooling_edot(yHI, yHII, yHeI/R(4), yde, yH2I/R(2), zero(R), T, zt;
                            nH = fhd, cool_tables = cool_tables)
        if T <= R(1.01)*R(MIN_TEMPERATURE) && edot < zero(R); edot = zero(R); end
        edot_c    = -c1 * (T - Tc) * yde
        edot_rest = edot - edot_c
        hubble_expansion && (edot_rest -= R(2) * Hz_ad * e * rho)
        Kc      = c1 * yde * (T / e) / rho          # Compton specific frequency
        de_rest = edot_rest / rho

        # Step limited by (a) non-Compton cooling and (b) the Compton T-change — the Compton
        # is integrated analytically, but its ΔT must stay ≲f·T so the T-dependent rates (and
        # the x_e–T feedback) stay accurate.  This is LOOSE where T≈T_cmb (Compton-locked) or
        # Kc is small (decoupled) — big steps survive there; it only bites in the transition.
        dT     = abs(Tc - T)
        exmax  = dT > tiny ? f*T/dT : one(R)
        dtc_c  = (exmax < one(R) && Kc > tiny) ? -log(one(R) - exmax)/Kc : typemax(R)
        dtc = min(_step_f(e, de_rest, f), dtc_c, rem)

        # ── energy: analytic Compton relaxation + explicit non-Compton cooling over dtc ──
        ex  = -expm1(-Kc*dtc)                       # 1 − e^{−Kc·dtc}  (stable)
        g   = Kc > tiny ? ex/Kc : dtc               # ∫₀^{dtc} e^{−Kc t}dt → dtc as Kc→0
        e   = e + (e*(Tc/T) - e)*ex + de_rest*g     # exact for de/dt = −Kc(e−e_cmb) + de_rest
        e   = max(e, tiny)

        # ── x_HII: exact Riccati (recomb k2 vs the k57/k58/CMB ionization source) ──
        scH  = k57v*yHI*yHI + k58v*yHI*yHeI/R(4) + kb1s*yHI + k1v*yHI*yde
        yHII = _riccati(yHII, k2, scH, dtc)
        # ── x_H2: pure formation quadrature (H⁻ + H₂⁺ closed-form channels + 3-body) ──
        nHM   = k7v*yHI*yde  / ((k8v + k15v)*yHI + k27v + tiny)
        nH2II = k9v*yHI*yHII / (k10v*yHI + k28v + tiny)
        yH2I += R(2)*(k8v*nHM*yHI + k10v*nH2II*yHI + k22v*yHI*yHI*yHI) * dtc

        yH2I = yH2I < tiny ? tiny : (yH2I > fhd ? fhd : yH2I)
        yHII = yHII < tiny ? tiny : (yHII > fhd - yH2I ? fhd - yH2I : yHII)
        yHI  = max(fhd - yHII - yH2I, tiny)
        yde  = yHII
        ttot += dtc
    end
    return e, yHII*mh, yH2I*mh, ttot
end

# ── GPU / batched solve path for the analytic mode ───────────────────────────
# One thread per cell; no metals/dust/deuterium.  The analytic solver's low, uniform
# iteration count (the Compton stiffness is integrated in closed form) minimizes warp
# divergence — the reason this is the fast GPU chemistry path.
@kernel function _evolve_analytic_k!(e, HII, H2I, @Const(rho), du, vu2, tu, dt, z,
                                     hubble, Om, OL, fh, hexp, aoa, rtab, ctab, dtfrac, itcap)
    i = @index(Global)
    @inbounds begin
        en, hii, h2, _ = evolve_cell_analytic(rho[i]*du, e[i]*vu2, HII[i]*du, H2I[i]*du,
                                              dt*tu, z; hubble=hubble, Om=Om, OL=OL, fh=fh,
                                              hubble_expansion=hexp, adot_over_a=aoa,
                                              rate_tables=rtab, cool_tables=ctab,
                                              itcap=itcap, dtfrac=dtfrac)
        e[i]   = en  / vu2
        HII[i] = hii / du
        H2I[i] = h2  / du
    end
end

"""
    solve_chem_analytic!(rho, e_int, HII, H2I; a_value, dt, density_units, length_units,
                         time_units, hubble=71, Om=0.27, OL=0.73, fh=0.76,
                         hubble_expansion=false, adot_over_a=NaN, cool_tables=nothing,
                         dtfrac=0.1, itcap=_SUB_ITMAX, workgroup_size=0,
                         backend=:cpu, precision=Float64)

Batched [`evolve_cell_analytic`](@ref) over every cell (KA kernel; `:cpu`/`:cuda`/`:metal`).
Primordial H+H₂ only (no deuterium/dust/metals); `e_int`, `HII`, `H2I` updated in place.
"""
function solve_chem_analytic!(rho::AbstractVector, e_int::AbstractVector,
                              HII::AbstractVector, H2I::AbstractVector;
                              a_value::Real, dt::Real, density_units::Real,
                              length_units::Real, time_units::Real,
                              hubble::Real = 71.0, Om::Real = 0.27, OL::Real = 0.73,
                              fh::Real = 0.76, hubble_expansion::Bool = false,
                              adot_over_a::Real = NaN, rate_tables = nothing,
                              cool_tables = nothing,
                              dtfrac::Real = 0.1, itcap::Int = _SUB_ITMAX, workgroup_size::Int = 0,
                              backend::Symbol = :cpu, precision::Type = Float64)
    n  = length(rho)
    @assert length(e_int) == n && length(HII) == n && length(H2I) == n
    P  = precision
    be = ChemistryKernels.backend(backend)
    du = P(density_units); vu2 = P((length_units/time_units)^2); tu = P(time_units)
    z  = P(1.0/a_value - 1.0)
    d_rho = to_device(be, collect(rho),   P); d_e   = to_device(be, collect(e_int), P)
    d_HII = to_device(be, collect(HII),   P); d_H2I = to_device(be, collect(H2I),   P)
    k! = workgroup_size > 0 ? _evolve_analytic_k!(be, workgroup_size) : _evolve_analytic_k!(be)
    k!(d_e, d_HII, d_H2I, d_rho, du, vu2, tu, P(dt), z, P(hubble), P(Om), P(OL), P(fh),
       hubble_expansion, P(adot_over_a), rate_tables, cool_tables, P(dtfrac), itcap; ndrange = n)
    e_int .= to_host(d_e); HII .= to_host(d_HII); H2I .= to_host(d_H2I)
    return nothing
end
export solve_chem_analytic!

"""
    solve_chem_analytic_device!(rho, e_int, HII, H2I; …, backend=:cuda, precision=Float32)

Zero-copy variant of [`solve_chem_analytic!`](@ref): all arrays are ALREADY device
arrays of element type `precision`; the kernel runs in place, no host round-trip.
"""
function solve_chem_analytic_device!(rho, e_int, HII, H2I;
                                     a_value::Real, dt::Real, density_units::Real,
                                     length_units::Real, time_units::Real,
                                     hubble::Real = 71.0, Om::Real = 0.27, OL::Real = 0.73,
                                     fh::Real = 0.76, hubble_expansion::Bool = false,
                                     adot_over_a::Real = NaN, rate_tables = nothing,
                                     cool_tables = nothing,
                                     dtfrac::Real = 0.1, itcap::Int = _SUB_ITMAX,
                                     workgroup_size::Int = 0, backend::Symbol = :cuda,
                                     precision::Type = Float32)
    n  = length(rho); P = precision; be = ChemistryKernels.backend(backend)
    du = P(density_units); vu2 = P((length_units/time_units)^2); tu = P(time_units)
    z  = P(1.0/a_value - 1.0)
    k! = workgroup_size > 0 ? _evolve_analytic_k!(be, workgroup_size) : _evolve_analytic_k!(be)
    k!(e_int, HII, H2I, rho, du, vu2, tu, P(dt), z, P(hubble), P(Om), P(OL), P(fh),
       hubble_expansion, P(adot_over_a), rate_tables, cool_tables, P(dtfrac), itcap; ndrange = n)
    KA.synchronize(be)
    return nothing
end
export solve_chem_analytic_device!

# ── UInt16 log₂-encoded species variant of the analytic path ──────────────────
# HII/H2I are carried as UInt16 log₂-encoded mass FRACTIONS (X = ρ_species/ρ_total)
# via the `log2_species.jl` codec (2 B/cell vs 4 B f32; ~0.12 %/ULP over 33 dex).  The
# kernel decodes to physical CGS at load, runs `evolve_cell_analytic`, and re-encodes at
# store; energy and rho stay floating-point.  Primordial H+H₂ only (no deuterium/dust/metals).
@kernel function _evolve_analytic_k_u16!(e, HII_u16, H2I_u16, @Const(rho), du, vu2, tu, dt, z,
                                         hubble, Om, OL, fh, hexp, aoa, rtab, ctab, dtfrac, itcap)
    i = @index(Global)
    @inbounds begin
        T = eltype(e)
        # clamp: f16 hydro can hand rho ≤ 0; the re-encode below divides by r
        # and log2's the ratio (device DomainError without the guard)
        r0 = rho[i] * du
        r = ifelse(r0 > T(1.0e-35), r0, T(1.0e-35))        # NaN-safe (physical CGS)
        hii_m = decode_log2sp(T, HII_u16[i]) * r           # UInt16 fraction → CGS mass density
        h2i_m = decode_log2sp(T, H2I_u16[i]) * r
        en, hii, h2, _ = evolve_cell_analytic(r, e[i]*vu2, hii_m, h2i_m, dt*tu, z;
                                              hubble=hubble, Om=Om, OL=OL, fh=fh,
                                              hubble_expansion=hexp, adot_over_a=aoa,
                                              rate_tables=rtab, cool_tables=ctab,
                                              itcap=itcap, dtfrac=dtfrac)
        e[i]       = en / vu2
        HII_u16[i] = encode_log2sp(hii / r)                # CGS mass density → fraction → UInt16
        H2I_u16[i] = encode_log2sp(h2  / r)
    end
end

"""
    solve_chem_analytic_u16!(rho, e_int, HII_u16, H2I_u16; a_value, dt, density_units,
                             length_units, time_units, …, backend=:cpu, precision=Float32)

UInt16 log₂-encoded-species variant of [`solve_chem_analytic!`](@ref).  `HII_u16`,
`H2I_u16` are `AbstractVector{UInt16}` storing log₂-encoded **mass fractions**
X = ρ_species/ρ_total (codec: u=0 → X≈7.73e-34, u=65535 → X=1.0; ≈0.12 %/ULP).  Host
round-trip; prepare/recover with `encode_log2sp_vec(HII ./ rho)` / `decode_log2sp_vec`.
2 B/species vs 4 B f32; primordial H+H₂ only.
"""
function solve_chem_analytic_u16!(rho::AbstractVector, e_int::AbstractVector,
                                  HII_u16::AbstractVector{UInt16},
                                  H2I_u16::AbstractVector{UInt16};
                                  a_value::Real, dt::Real, density_units::Real,
                                  length_units::Real, time_units::Real,
                                  hubble::Real = 71.0, Om::Real = 0.27, OL::Real = 0.73,
                                  fh::Real = 0.76, hubble_expansion::Bool = false,
                                  adot_over_a::Real = NaN, rate_tables = nothing,
                                  cool_tables = nothing,
                                  dtfrac::Real = 0.1, itcap::Int = _SUB_ITMAX, workgroup_size::Int = 0,
                                  backend::Symbol = :cpu, precision::Type = Float32)
    n  = length(rho)
    @assert length(e_int) == n && length(HII_u16) == n && length(H2I_u16) == n
    P  = precision
    be = ChemistryKernels.backend(backend)
    du = P(density_units); vu2 = P((length_units/time_units)^2); tu = P(time_units)
    z  = P(1.0/a_value - 1.0)
    d_rho = to_device(be, collect(rho),   P); d_e = to_device(be, collect(e_int), P)
    d_HII = to_device(be, collect(HII_u16), UInt16); d_H2I = to_device(be, collect(H2I_u16), UInt16)
    k! = workgroup_size > 0 ? _evolve_analytic_k_u16!(be, workgroup_size) : _evolve_analytic_k_u16!(be)
    k!(d_e, d_HII, d_H2I, d_rho, du, vu2, tu, P(dt), z, P(hubble), P(Om), P(OL), P(fh),
       hubble_expansion, P(adot_over_a), rate_tables, cool_tables, P(dtfrac), itcap; ndrange = n)
    e_int .= to_host(d_e); HII_u16 .= to_host(d_HII); H2I_u16 .= to_host(d_H2I)
    return nothing
end
export solve_chem_analytic_u16!

"""
    solve_chem_analytic_device_u16!(rho, e_int, HII_u16, H2I_u16; …, backend=:cuda, precision=Float32)

Zero-copy variant of [`solve_chem_analytic_u16!`](@ref): `rho`/`e_int` are device arrays of
element type `precision`, `HII_u16`/`H2I_u16` are device `UInt16` arrays.  No host round-trip.
"""
function solve_chem_analytic_device_u16!(rho, e_int, HII_u16, H2I_u16;
                                         a_value::Real, dt::Real, density_units::Real,
                                         length_units::Real, time_units::Real,
                                         hubble::Real = 71.0, Om::Real = 0.27, OL::Real = 0.73,
                                         fh::Real = 0.76, hubble_expansion::Bool = false,
                                         adot_over_a::Real = NaN, rate_tables = nothing,
                                         cool_tables = nothing,
                                         dtfrac::Real = 0.1, itcap::Int = _SUB_ITMAX,
                                         workgroup_size::Int = 0, backend::Symbol = :cuda,
                                         precision::Type = Float32)
    n  = length(rho); P = precision; be = ChemistryKernels.backend(backend)
    du = P(density_units); vu2 = P((length_units/time_units)^2); tu = P(time_units)
    z  = P(1.0/a_value - 1.0)
    k! = workgroup_size > 0 ? _evolve_analytic_k_u16!(be, workgroup_size) : _evolve_analytic_k_u16!(be)
    k!(e_int, HII_u16, H2I_u16, rho, du, vu2, tu, P(dt), z, P(hubble), P(Om), P(OL), P(fh),
       hubble_expansion, P(adot_over_a), rate_tables, cool_tables, P(dtfrac), itcap; ndrange = n)
    KA.synchronize(be)
    return nothing
end
export solve_chem_analytic_device_u16!

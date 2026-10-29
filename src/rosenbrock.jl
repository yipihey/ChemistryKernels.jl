# rosenbrock.jl — adaptive, L-stable ROS2 integrator for the v2026 reduced
# PRIMORDIAL network (H + He-in-equilibrium + optional D), specialized: NO dust,
# NO metals.  Same call/return contract as the fraction-change subcycler
# `evolve_cell` (subcycle.jl); exposed as the standalone `evolve_cell_ros2`.
#
# WHY.  The subcycler's coupled-stiff tail (hot cells, x_e far from equilibrium,
# t_cool ≈ t_ion) has no error feedback in its fraction-change limiter, so it errs
# ~50% on the stiffest cells at the default dtfrac=0.1.  ROS2 solves the coupling
# implicitly with an embedded error estimate and is markedly MORE ACCURATE.  It is
# NOT faster, though: clean timing on a realistic z≈10 mix puts it 1.6–7× SLOWER
# than the (well-tuned) subcycler — ~6 RHS evals + a 4×4 solve per step outweigh its
# lower step count.  So this is an accuracy / reference option, not an accelerator.
# (An earlier "30× fewer steps" claim came from a strawman subcycler + GC-polluted
# single-shot timing; both were wrong.)
#
# METHOD (Verwer et al. 1999, 2-stage L-stable Rosenbrock-W, γ = 1+1/√2 ⇒ R(−∞)=0):
#     W  = I − γ·h·J
#     k1 = W \ (h·F(y))
#     k2 = W \ (h·F(y+k1) − 2·k1)
#     y_{n+1} = y + (3k1 + k2)/2,     err = (k1 + k2)/2   (embedded ROS1 estimate)
# Being a W-method it is consistent for ANY approximation of J, so a cheap
# finite-difference Jacobian (even in Float32) preserves 2nd order — only stability
# / step size feel an inexact J.  Pure & allocation-free (StaticArrays on the
# stack) ⇒ runs unchanged in a KA kernel and is GPU-ready.
#
# STATE (mass-equivalent y-convention, as network_step.jl):
#   N=4 (H-only)  : y = (yHI, yHII, yH2I, e)
#   N=6 (with D)  : y = (yHI, yHII, yH2I, yDI, yHDI, e)
# He (HeI/II/III), H⁻ and H₂⁺ are fast algebraic intermediaries recomputed from the
# state each RHS eval (exactly as network_step does); nₑ is charge conservation.
# The He ionisation balance is evaluated off a lagged electron density `ne_aux`
# (one Gauss–Seidel lag, matching evolve_cell's "He from OLD nₑ").

using StaticArrays: SVector, SMatrix, setindex

export evolve_cell_ros2

const _ROS2_ITMAX = 5_000          # watchdog (well-behaved cells take ≪100 steps)
const _ROS2_TINY  = 1.0e-20

# ── electron density from charge conservation (HM, H2II algebraic; He off ne_aux) ──
# Returns (ne, HM, H2II, yHeI) so callers can reuse the intermediaries.
@inline function _primordial_ne(yHI, yHII, yH2I, K, nHe4, ne_aux, tiny)
    R = typeof(yHI)
    she1 = K.she1; she2 = K.she2
    _, nHeII, nHeIII = helium_equilibrium(she1, she2, K.k3, K.k4, K.k5, K.k6,
                                          max(ne_aux, tiny), nHe4 / R(4))
    yHeII  = R(4) * nHeII                       # ×4 mass-equiv convention
    yHeIII = R(4) * nHeIII
    yHeI   = max(nHe4 - yHeII - yHeIII, tiny)
    # H⁻ and H₂⁺ equilibrium (use the lagged ne_aux for the H2II e⁻ destruction so
    # the pair is a pure function of the state + ne_aux; trace species, lag harmless)
    HM    = equilibrium_HM(yHI, yHII, ne_aux, zero(R),
                           K.k7, K.k8, K.k14, K.k15, K.k16, K.k17, K.k19, K.k27)
    H2II  = equilibrium_H2II(yHI, yHII, yH2I, ne_aux, HM,
                             K.k9, K.k10, K.k11, K.k17, K.k18, K.k19, K.k28)
    ne = yHII + yHeII / R(4) + yHeIII / R(2) - HM + H2II / R(2)   # charge conservation
    return max(ne, tiny), HM, H2II, yHeI
end

# temperature + rate coefficients of a state (the transcendental-heavy part).
# Hoisted so the FD Jacobian can REUSE one (K,Tg) across the species columns —
# perturbing HI/HII/H2I moves T negligibly (T is set by e + total density), and a
# W-method is consistent for any approximate Jacobian, so freezing the T-dependent
# rate coefficients there is exact in the limit and cheap out of it (3 rate-builds
# /step instead of 6).  The energy column and the k2 stage rebuild (T moved).
@inline function _state_KT(y::SVector{N,R}, p) where {N,R}
    tiny = R(_ROS2_TINY)
    yHI = max(y[1], tiny); yHII = max(y[2], tiny); yH2I = max(y[3], tiny)
    e   = max(y[N], tiny); deut = (N == 6)
    Tg = gas_temperature(p.rho, e, yHI, yHII, p.nHe4/R(4), tiny, tiny, p.ne_aux,
                         tiny, yH2I/R(2), tiny; gamma = GAMMA_DEFAULT)
    K = p.rate_tables === nothing ?
        _build_rates_cr(Tg, yHI, p.Hz, p.cr; deuterium = deut) :
        table_rates(p.rate_tables, Tg, yHI, p.Hz, p.cr; deuterium = deut)
    return K, Tg
end

# ── the network + cooling right-hand side: dy/dt for the ODE state ─────────────
# `p` carries the per-step frozen context (CMB/Trad terms, Hz, redshift, tables…).
# `K`,`Tg` are the (possibly reused) rate coefficients + temperature for this state.
@inline primordial_rhs(y::SVector{N,R}, p) where {N,R} =
    primordial_rhs(y, p, _state_KT(y, p)...)

@inline function primordial_rhs(y::SVector{N,R}, p, K, Tg) where {N,R}
    tiny = R(_ROS2_TINY)
    yHI  = max(y[1], tiny); yHII = max(y[2], tiny); yH2I = max(y[3], tiny)
    if N == 6
        yDI = max(y[4], tiny); yHDI = max(y[5], tiny); e = max(y[6], tiny)
    else
        yDI = zero(R); yHDI = zero(R); e = max(y[4], tiny)
    end
    d = p.d; fh = p.fh; rho = p.rho; nHe4 = p.nHe4

    ne, HM, H2II, yHeI = _primordial_ne(yHI, yHII, yH2I, K, nHe4, p.ne_aux, tiny)
    nH2II = H2II / R(2)

    k1=K.k1; k2=K.k2; k7=K.k7; k8=K.k8; k9=K.k9; k10=K.k10; k11=K.k11; k12=K.k12
    k13=K.k13; k14=K.k14; k15=K.k15; k16=K.k16; k17=K.k17; k18=K.k18; k19=K.k19
    k22=K.k22; k28=K.k28; k57=K.k57; k58=K.k58; kb=K.k_beta1s

    # HI: dyHI/dt = sc − ac·yHI  (network_step.jl block C-1, dust/Γ omitted)
    scHI = k2*yHII*ne + k13*yHI*yH2I + k11*yHII*yH2I/R(2) + k12*ne*yH2I +
           k14*HM*ne + k15*HM*yHI + R(2)*k16*HM*yHII + k18*H2II*ne +
           k19*H2II*HM/R(2) + k28*nH2II
    acHI = k1*ne + k7*ne + k8*HM + k9*yHII + k10*H2II/R(2) + R(2)*k22*yHI^2 +
           k57*yHI + k58*yHeI/R(4) + kb
    dHI = scHI - acHI*yHI

    # HII (block C-2)
    scHII = k1*yHI*ne + k10*H2II*yHI/R(2) + k57*yHI^2 + k58*yHI*yHeI/R(4) +
            kb*yHI + k28*nH2II
    acHII = k2*ne + k9*yHI + k11*yH2I/R(2) + k16*HM + k17*HM
    dHII = scHII - acHII*yHII

    # H2 (block C-7)
    scH2 = R(2)*(k8*HM*yHI + k10*H2II*yHI/R(2) + k19*H2II*HM/R(2) + k22*yHI*yHI^2)
    acH2 = k13*yHI + k11*yHII + k12*ne
    dH2 = scH2 - acH2*yH2I

    # cooling (volumetric, signed) → specific de/dt; PRIMORDIAL only (metals=nothing)
    nHD  = yHDI / R(3)
    edot = cooling_edot(yHI, yHII, yHeI/R(4), ne, yH2I/R(2), nHD, Tg, p.zt;
                        nH = R(fh)*d, metals = nothing, cool_tables = p.cool_tables)
    if Tg <= R(1.01)*R(MIN_TEMPERATURE) && edot < zero(R)
        edot = zero(R)
    end
    if p.hubble_expansion
        edot -= R(2) * p.Hz_ad * e * rho
    end
    de = edot / rho

    if N == 6
        k50=K.k50; k51=K.k51; k52=K.k52; k53=K.k53; k54=K.k54; k55=K.k55; k56=K.k56
        DII = equilibrium_DII(yDI, ne, yHI, yHII, yH2I, yHDI,
                              k1, k2, k50, k51, k52, k53)
        # DI (deuterium block 1)
        scDI = k2*DII*ne + k51*DII*yHI + R(2)*k55*yHDI*yHI/R(3)
        acDI = k1*ne + k50*yHII + k54*yH2I/R(2) + k56*HM
        dDI = scDI - acDI*yDI
        # HDI (deuterium block 3)
        scHD = R(3)*(k52*DII*yH2I/R(4) + k54*yDI*yH2I/R(4) + k56*yDI*HM)
        acHD = k53*yHII + k55*yHI
        dHD = scHD - acHD*yHDI
        return SVector{6,R}(dHI, dHII, dH2, dDI, dHD, de)
    else
        return SVector{4,R}(dHI, dHII, dH2, de)
    end
end

# ── finite-difference Jacobian (W-method ⇒ approximate J is fine) ──────────────
# Species columns reuse the base (K,Tg); the energy column (j=N) rebuilds rates
# since perturbing e moves T.
@inline function _fd_jacobian(y::SVector{N,R}, F0::SVector{N,R}, p, K, Tg) where {N,R}
    h = sqrt(eps(R))
    cols = ntuple(Val(N)) do j
        dyj = h * max(abs(y[j]), p.ref[j] * R(1.0e-6))
        yj  = setindex(y, y[j] + dyj, j)
        Fj  = j == N ? primordial_rhs(yj, p) : primordial_rhs(yj, p, K, Tg)
        (Fj - F0) / dyj
    end
    return hcat(cols...)                 # SMatrix{N,N,R}
end

# ── one ROS2 substep: returns (y_new, err_estimate) ───────────────────────────
@inline function _ros2_substep(y::SVector{N,R}, h::R, p, gamma::R) where {N,R}
    K, Tg = _state_KT(y, p)
    F0 = primordial_rhs(y, p, K, Tg)
    J  = _fd_jacobian(y, F0, p, K, Tg)
    W  = SMatrix{N,N,R}(I) - (gamma*h) * J
    k1 = W \ (h .* F0)
    y1 = y .+ k1
    F1 = primordial_rhs(y1, p)
    k2 = W \ (h .* F1 .- R(2) .* k1)
    ynew = y .+ R(1.5) .* k1 .+ R(0.5) .* k2
    err  = R(0.5) .* (k1 .+ k2)
    return ynew, err
end

@inline function _errnorm(err::SVector{N,R}, y::SVector{N,R}, ynew::SVector{N,R},
                          atol::SVector{N,R}, rtol::R) where {N,R}
    acc = zero(R)
    @inbounds for i in 1:N
        sc  = atol[i] + rtol * max(abs(y[i]), abs(ynew[i]))
        acc += (err[i] / sc)^2
    end
    return sqrt(acc / R(N))
end

@inline _floor_state(y::SVector{N,R}) where {N,R} =
    SVector{N,R}(ntuple(i -> max(y[i], R(_ROS2_TINY)), Val(N)))

# Lagged electron density off an accepted state (one Gauss–Seidel pass): build
# rates at the state's T (μ from the previous ne_aux), then charge conservation.
@inline function _refresh_ne(y::SVector{N,R}, p) where {N,R}
    deut = (N == 6)
    tiny = R(_ROS2_TINY)
    yHI = max(y[1], tiny); yHII = max(y[2], tiny); yH2I = max(y[3], tiny)
    e   = max(y[N], tiny)
    Tg  = gas_temperature(p.rho, e, yHI, yHII, p.nHe4/R(4), tiny, tiny, p.ne_aux,
                          tiny, yH2I/R(2), tiny; gamma = GAMMA_DEFAULT)
    K = p.rate_tables === nothing ?
        _build_rates_cr(Tg, yHI, p.Hz, p.cr; deuterium = deut) :
        table_rates(p.rate_tables, Tg, yHI, p.Hz, p.cr; deuterium = deut)
    ne, _, _, _ = _primordial_ne(yHI, yHII, yH2I, K, p.nHe4, p.ne_aux, tiny)
    return ne
end

"""
    evolve_cell_ros2(rho, e, HII_m, H2I_m, HDI_m, dt, z; …) -> (e, HII_m, H2I_m, HDI_m, ttot)

Adaptive ROS2 integration of one primordial cell over the macro-step `dt` [s].
Drop-in for [`evolve_cell`](@ref) (same args/return), but the stiff coupled tail is
solved implicitly with embedded error control instead of fraction-change
subcycling.  PRIMORDIAL ONLY — no `dust`/`metals` kwargs.  Pure.

This is an ACCURACY option, not a speed win.  On a realistic z≈10 cell mix (clean
allocation-free timing) it is 1.6–7× SLOWER than the well-tuned `evolve_cell`
subcycler — ROS2 does ~6 RHS evals + a 4×4 solve per step, and although it takes
fewer steps than the subcycler has sub-steps, that per-step overhead is not
recovered (and the optional rate/cooling tables make the subcycler faster still).
What it buys is accuracy: median x_HII error ~5e-5 (rtol=1e-2) vs the subcycler's
~4e-4, and it never errs ~50% on the stiffest cells the way the subcycler's default
`dtfrac=0.1` does.  Use it where a more accurate / reference primordial solve is
worth the cost, not as a drop-in accelerator.

`rtol` trades cost for stiff-cell accuracy (the cheap majority is accurate at any
`rtol`; only the stiff tail moves): 1e-2 (default) ≲1% worst-cell, 3e-2 ≲3%, 1e-1
≲12%.  `rtol ≳ 3e-1` under-resolves the initial transient and hot cells can diverge.
"""
@inline function evolve_cell_ros2(rho, e, HII_m, H2I_m, HDI_m, dt, z;
                                  hubble = 71.0, Om = 0.27, OL = 0.73,
                                  fh = FH_DEFAULT, deuterium::Bool = false,
                                  hubble_expansion::Bool = false,
                                  adot_over_a = NaN,
                                  rate_tables = nothing, cool_tables = nothing,
                                  itcap::Int = _ROS2_ITMAX,
                                  nsteps_out = nothing,
                                  rtol::Real = 1.0e-2)
    R    = typeof(e)
    mh   = R(MH); tiny = R(_ROS2_TINY)
    d    = rho / mh
    z0   = R(z)
    Hz0  = hubble_z_of(z0; hubble = hubble, Om = Om, OL = OL)
    Hz_ad = isnan(adot_over_a) ? Hz0 : R(adot_over_a)
    evolve_z = !isnan(adot_over_a)
    nHe4 = (one(R) - R(fh)) * d
    gamma = one(R) + one(R)/sqrt(R(2))

    # advected species → y-convention number densities (mirrors evolve_cell)
    yHII = HII_m / mh
    yH2I = H2I_m / mh
    yHDI = deuterium ? HDI_m / mh : zero(R)
    yHI  = max((R(fh)*rho - HII_m - H2I_m) / mh, tiny)
    yDI  = deuterium ? R(DTOH_SEED)*yHI : zero(R)
    ne_aux = max(yHII, tiny)

    Tc0 = comp2_cmb(z0); cr0 = cmb_rates(Tc0)

    # per-component reference scales (for FD perturbation + atol) and atol vector
    if deuterium
        ref  = SVector{6,R}(d, d, d, d, d, e)
        atol = SVector{6,R}(R(1e-12)*d, R(1e-12)*d, R(1e-12)*d,
                            R(1e-15)*d, R(1e-15)*d, R(1e-10)*e)
        y    = SVector{6,R}(yHI, yHII, yH2I, yDI, yHDI, e)
    else
        ref  = SVector{4,R}(d, d, d, e)
        atol = SVector{4,R}(R(1e-12)*d, R(1e-12)*d, R(1e-12)*d, R(1e-10)*e)
        y    = SVector{4,R}(yHI, yHII, yH2I, e)
    end

    ttot = zero(R); iter = 0; h = dt    # cheap near-equilibrium cells finish in 1 step
    while ttot < dt && iter < itcap
        iter += 1
        h = min(h, dt - ttot)
        if evolve_z
            zt = (one(R) + z0) * exp(-Hz_ad * ttot) - one(R)
            Tc = comp2_cmb(zt); Hz = hubble_z_of(zt; hubble=hubble, Om=Om, OL=OL); cr = cmb_rates(Tc)
        else
            zt = z0; Tc = Tc0; Hz = Hz0; cr = cr0
        end
        p = (; d, fh, rho, nHe4, ne_aux, Hz, zt, cr,
               rate_tables, cool_tables, hubble_expansion, Hz_ad, ref)

        ynew, err = _ros2_substep(y, h, p, gamma)
        en = _errnorm(err, y, ynew, atol, rtol)
        forced = h <= dt * R(1.0e-10)              # progress floor — accept to avoid stall
        if (isfinite(en) && en <= one(R)) || forced
            y = _floor_state(ynew)
            ttot += h
            ne_aux = _refresh_ne(y, p)                   # lag the e⁻ density forward
            fac = isfinite(en) && en > zero(R) ? R(0.9)*en^R(-0.5) : R(5)
        else
            fac = R(0.3)                            # reject: shrink, retry (no advance)
        end
        h = h * clamp(fac, R(0.2), R(5.0))
        h = max(h, dt * R(1.0e-12))
    end

    nsteps_out === nothing || (nsteps_out[] = iter)
    e_out   = y[length(y)]
    HII_out = y[2] * mh
    H2I_out = y[3] * mh
    HDI_out = deuterium ? y[5] * mh : HDI_m
    return e_out, HII_out, H2I_out, HDI_out, ttot
end

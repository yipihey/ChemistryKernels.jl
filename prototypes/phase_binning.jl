#!/usr/bin/env julia
# phase_binning.jl — prototype + evaluation of phase-space-binned chemistry.
#
# Idea (user): on a large 3D grid, every cell takes the SAME redshift z and the SAME
# outer dt in a chemistry step, so the per-cell update is one deterministic function
#     F : (rho, e, x_HII, x_H2I, x_HDI)  ->  (e', x_HII', x_H2I', x_HDI')
# of the cell's input state alone.  Instead of calling the stiff solver once per cell,
# bin cells in that input space, solve F ONCE per occupied bin, and map the result
# back.  At high z the gas is nearly uniform (Compton-locked T, tiny density contrast)
# so the occupied region collapses to a handful of bins -> few solves for ~all cells.
#
# This script:
#   1. synthesizes a physically-motivated per-cell state at a given z (mean adiabat +
#      lognormal density contrast; T–rho coupling that vanishes at high z = Compton
#      lock; small ionization scatter),
#   2. runs the REFERENCE per-cell solve (ChemistryKernels.evolve_cell, the real
#      network) and times it,
#   3. runs the BINNED solve with two map-back rules and times it,
#   4. reports speedup (call-count + wall) and accuracy — crucially the error in the
#      cell-to-cell STD (the variance P(k) measures), not just the mean,
#   5. sweeps redshift and bin tolerance to show where it pays off.
#
# Run:  julia --project=ChemistryKernels.jl/test ChemistryKernels.jl/prototypes/phase_binning.jl

using ChemistryKernels
using Printf, Random, Statistics

const EV = ChemistryKernels.evolve_cell

# ── physical constants (CGS) ────────────────────────────────────────────────
const MH    = 1.6726e-24            # g
const KB    = 1.380649e-16          # erg/K
const MPC   = 3.0857e24             # cm
const RHOC  = 1.8788e-29            # g/cm³ × h²  (critical density today)
const GAMMA = 5/3
const FH    = 0.76                  # hydrogen mass fraction
const H0    = 71.0e5 / MPC          # s⁻¹  (h=0.71)
const OMB   = 0.046
const OMM   = 0.27
const OML   = 0.73

Hofz(z) = H0 * sqrt(OMM*(1+z)^3 + OML)

# specific internal energy [erg/g] for a gas at temperature T with mean molecular
# weight mu (neutral primordial mu≈1.22): e = kB T / ((γ-1) μ mh)
e_of_T(T, mu) = KB*T / ((GAMMA-1)*mu*MH)
T_of_e(e, mu) = e * (GAMMA-1)*mu*MH / KB

# ── synthetic cosmological cell state at redshift z ─────────────────────────
# Returns CGS arrays (rho, e, HII_m, H2I_m, HDI_m) of length N. `sigd` is the
# density-contrast rms; `lock` ∈ [0,1] is the Compton T–rho decoupling (1 = T fully
# adiabatic ∝ρ^(γ-1); 0 = T locked to CMB, no scatter — the high-z limit).
function gen_state(z; N=200_000, sigd=nothing, lock=nothing, seed=1)
    rng = MersenneTwister(seed)
    # linear-ish growth of the density-contrast rms (small box, illustrative)
    sigd === nothing && (sigd = 0.02 * (201/(1+z)))          # ~1e-3 @z1000 … ~0.13 @z30
    sigd = clamp(sigd, 1e-4, 1.5)
    # Compton coupling: gas T locked to CMB for z≳150, decoupling below
    lock === nothing && (lock = clamp((150/(1+z)), 0.0, 1.0))   # 0 = locked (high z)

    rhobar = OMB * RHOC * 0.71^2 * (1+z)^3                   # g/cm³ (mean baryon)
    Tcmb   = 2.725*(1+z)
    Tbar   = z > 150 ? Tcmb : Tcmb*(151/(1+z))              # adiabatic drop after decoupling
    # mean residual ionization: ~1 before recombination, ~2e-4 after
    xHIIbar = 0.5*(1 - tanh((1100 - (1+z))/80)) * (1-2e-4) + 2e-4
    xH2bar  = 2e-6
    xHDbar  = 3.4e-5 * xH2bar                               # ~ (D/H)·x_H2

    mu = 1.22
    rho   = Vector{Float64}(undef, N); e = similar(rho)
    HIIm  = similar(rho); H2Im = similar(rho); HDIm = similar(rho)
    for i in 1:N
        g   = sigd*randn(rng) - 0.5*sigd^2                  # lognormal, mean-preserving
        r   = exp(g)                                        # ρ/ρ̄
        ρ   = rhobar*r
        # T–ρ relation: adiabatic part scales as ρ^(γ-1), weighted by `lock`
        T   = Tbar * r^((GAMMA-1)*lock)
        # ionization slightly enhanced where denser (faster recomb) — small effect
        xHII = clamp(xHIIbar * r^(-0.1*lock), 1e-8, 1.0)
        rho[i]  = ρ
        e[i]    = e_of_T(T, mu)
        HIIm[i] = xHII * ρ
        H2Im[i] = xH2bar * ρ
        HDIm[i] = xHDbar * ρ
    end
    return (; rho, e, HIIm, H2Im, HDIm, z, sigd, lock, Tbar, xHIIbar)
end

# ── reference: one evolve_cell per cell ─────────────────────────────────────
function solve_reference(s; dt, kw...)
    N = length(s.rho)
    e2 = similar(s.e); H2 = similar(s.HIIm); M2 = similar(s.H2Im); D2 = similar(s.HDIm)
    @inbounds for i in 1:N
        en, hii, h2, hd, _ = EV(s.rho[i], s.e[i], s.HIIm[i], s.H2Im[i], s.HDIm[i],
                                dt, s.z; kw...)
        e2[i]=en; H2[i]=hii; M2[i]=h2; D2[i]=hd
    end
    return (; e=e2, HIIm=H2, H2Im=M2, HDIm=D2)
end

# ── binned solve ────────────────────────────────────────────────────────────
# Bin in log-space of the actual evolve_cell inputs. `tol` is the bin width in dex
# (per axis). map ∈ (:broadcast, :ratio). Returns outputs + nbins.
function solve_binned(s; dt, tol=0.05, map=:ratio, kw...)
    N = length(s.rho)
    inv = 1.0/tol
    key(i) = (round(Int, log10(s.rho[i])*inv),
              round(Int, log10(s.e[i])*inv),
              round(Int, log10(max(s.HIIm[i]/s.rho[i],1e-30))*inv),
              round(Int, log10(max(s.H2Im[i]/s.rho[i],1e-30))*inv),
              round(Int, log10(max(s.HDIm[i]/s.rho[i],1e-30))*inv))
    bins = Dict{NTuple{5,Int}, Vector{Int}}()
    @inbounds for i in 1:N
        push!(get!(bins, key(i), Int[]), i)
    end
    e2 = similar(s.e); H2 = similar(s.HIIm); M2 = similar(s.H2Im); D2 = similar(s.HDIm)
    gmean(v) = exp(mean(log, v))
    @inbounds for (_, idx) in bins
        # representative input = geometric mean of members (consistent with log bins)
        ρr  = gmean(@view s.rho[idx]); er = gmean(@view s.e[idx])
        hr  = gmean(@view s.HIIm[idx]); mr = gmean(@view s.H2Im[idx]); dr = gmean(@view s.HDIm[idx])
        en, hii, h2, hd, _ = EV(ρr, er, hr, mr, dr, dt, s.z; kw...)
        if map === :broadcast
            # intensive outputs held constant across the bin (fractions + specific e)
            xe = en/er                       # e'/e of the representative (specific e is intensive)
            xh = hii/ρr; xm = h2/ρr; xd = hd/ρr
            for i in idx
                e2[i]=s.e[i]*xe; H2[i]=xh*s.rho[i]; M2[i]=xm*s.rho[i]; D2[i]=xd*s.rho[i]
            end
        else  # :ratio — transfer the bin's RESPONSE RATIO, preserving sub-bin spread
            re = en/er; rh = hii/hr; rm = h2/mr; rd = hd/dr
            for i in idx
                e2[i]=s.e[i]*re; H2[i]=s.HIIm[i]*rh; M2[i]=s.H2Im[i]*rm; D2[i]=s.HDIm[i]*rd
            end
        end
    end
    return (; e=e2, HIIm=H2, H2Im=M2, HDIm=D2, nbins=length(bins))
end

# ── metrics: accuracy of a field vs reference (mean bias, std preservation, rms) ──
function field_err(ref, approx)
    relerr = abs.(approx .- ref) ./ (abs.(ref) .+ 1e-30)
    (; rms = sqrt(mean(relerr.^2)), p99 = quantile(relerr, 0.99),
       meanbias = mean(approx)/mean(ref) - 1,
       stdratio = std(approx)/(std(ref)+1e-30))
end

xfrac(m, rho) = m ./ rho   # mass density -> fraction

function evaluate(z; N=200_000, tol=0.05, kw...)
    s = gen_state(z; N=N)
    dt = 0.05 / Hofz(z)                          # macro-step ≈ Δln a = 0.05
    cosmo = (; hubble=71.0, Om=OMM, OL=OML, fh=FH, deuterium=true,
               hubble_expansion=true, adot_over_a=Hofz(z))
    # time reference (1 rep run; @elapsed after a warmup on a small slice)
    EV(s.rho[1], s.e[1], s.HIIm[1], s.H2Im[1], s.HDIm[1], dt, z; cosmo...)  # warmup
    tref = @elapsed ref = solve_reference(s; dt=dt, cosmo...)
    tbin = @elapsed bin = solve_binned(s; dt=dt, tol=tol, map=:ratio, cosmo...)
    br = solve_binned(s; dt=dt, tol=tol, map=:broadcast, cosmo...)
    # accuracy on x_HII (feeds species P(k)) and T (from e)
    ex_ratio = field_err(xfrac(ref.HIIm,s.rho), xfrac(bin.HIIm,s.rho))
    eT_ratio = field_err(ref.e, bin.e)
    ex_bcast = field_err(xfrac(ref.HIIm,s.rho), xfrac(br.HIIm,s.rho))
    return (; z, sigd=s.sigd, lock=s.lock, N, nbins=bin.nbins,
              speedup_calls = N/bin.nbins, tref, tbin,
              wall_speedup = tref/tbin, ex_ratio, eT_ratio, ex_bcast)
end

# ── main sweep ──────────────────────────────────────────────────────────────
function main()
    N = parse(Int, get(ENV, "PROTO_N", "200000"))
    tol = parse(Float64, get(ENV, "PROTO_TOL", "0.05"))
    zs = [1000.0, 300.0, 100.0, 30.0, 20.0]
    # warm up both solvers (JIT) on a small state so the swept timings are clean
    let sw = gen_state(300.0; N=2000), dtw = 0.05/Hofz(300.0)
        cw = (; hubble=71.0, Om=OMM, OL=OML, fh=FH, deuterium=true,
                hubble_expansion=true, adot_over_a=Hofz(300.0))
        solve_reference(sw; dt=dtw, cw...); solve_binned(sw; dt=dtw, tol=tol, map=:ratio, cw...)
        solve_binned(sw; dt=dtw, tol=tol, map=:broadcast, cw...)
    end
    @printf("\nphase-space-binned chemistry — N=%d cells, bin tol=%.3f dex, primordial+D\n", N, tol)
    @printf("dt ≈ Δln a=0.05 / H(z); reference = evolve_cell per cell (the real network)\n\n")
    @printf("%6s %7s %8s %9s %8s %9s %9s | %9s %9s %9s | %9s\n",
            "z","σ_δ","nbins","calls↓","wall↑","tref[s]","tbin[s]",
            "xHII rms","xHII std","T rms","bcast rms")
    println("-"^108)
    for z in zs
        r = evaluate(z; N=N, tol=tol)
        @printf("%6.0f %7.4f %8d %8.0f× %7.1f× %9.3f %9.4f | %9.2e %9.3f %9.2e | %9.2e\n",
                r.z, r.sigd, r.nbins, r.speedup_calls, r.wall_speedup, r.tref, r.tbin,
                r.ex_ratio.rms, r.ex_ratio.stdratio, r.eT_ratio.rms, r.ex_bcast.rms)
    end
    println("\nlegend: calls↓ = N/nbins (solve-count reduction); wall↑ = tref/tbin (incl. binning);")
    println("        xHII std = std(approx)/std(ref), 1.000 = variance preserved; 'ratio' map-back.")

    # tolerance sweep at the hard case (recombination): variance vs cost knob
    zc = 1000.0
    @printf("\ntolerance sweep at z=%.0f (recombination — the variance-sensitive regime):\n", zc)
    @printf("%9s %8s %9s %9s %9s %9s\n","tol[dex]","nbins","calls↓","wall↑","xHII rms","xHII std")
    println("-"^58)
    for t in (0.05, 0.02, 0.01, 0.005, 0.002)
        r = evaluate(zc; N=N, tol=t)
        @printf("%9.3f %8d %8.0f× %7.1f× %9.2e %9.3f\n",
                t, r.nbins, r.speedup_calls, r.wall_speedup, r.ex_ratio.rms, r.ex_ratio.stdratio)
    end
end

main()

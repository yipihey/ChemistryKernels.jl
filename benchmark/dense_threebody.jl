#!/usr/bin/env julia

# Dense primordial H₂ benchmark.
#
# Measures the public zero-copy full and analytic H+H₂ kernels on states where
# three-body formation and H-impact dissociation compete. Run, for example:
#
#   julia -t auto --project=test benchmark/dense_threebody.jl 1048576
#
# Metal is used when available; the CPU backend always runs.

using ChemistryKernels
using Printf

const CK = ChemistryKernels
const FH = 0.76
const MH = CK.MH
const KB = 1.380649e-16
const GAMMA = 5 / 3

function pair_equilibrium_fraction(nH, T)
    kd = k13(T)
    kd == 0 && return 1.0
    a = 2 * k22(T) / kd
    xHI = 2 * nH / (1 + sqrt(1 + 4 * a * nH))
    return 1 - xHI / nH
end

function state_vectors(P, n, nH, T, fH2)
    rho0 = nH * MH / FH
    xHII = 1e-12
    nHI = max(1 - fH2 - xHII, 0.0) * nH
    nH2 = fH2 * nH / 2
    nHe = (1 - FH) / (4FH) * nH
    ne = xHII * nH
    e0 = (nHI + nH2 + nHe + ne) * KB * T / (rho0 * (GAMMA - 1))
    return (rho=fill(P(rho0), n), e=fill(P(e0), n),
            HII=fill(P(xHII * nH * MH), n), H2I=fill(P(fH2 * nH * MH), n))
end

function best_time!(run!, arrays, initial; samples=5)
    best = Inf
    for _ in 1:samples
        reset_state!(arrays, initial)
        best = min(best, @elapsed run!())
    end
    return best
end

function reset_state!(arrays, initial)
    copyto!(arrays.e, initial.e)
    copyto!(arrays.HII, initial.HII)
    copyto!(arrays.H2I, initial.H2I)
end

first_host(x) = first(Array(x))

function try_warmup(run!)
    try
        run!()
        return nothing
    catch err
        @warn "solver unavailable on this backend" exception_type=typeof(err)
        return err
    end
end

function benchmark_backend(name, to_backend, n; P=Float32)
    println("\nbackend=$name precision=$P cells=$n")
    @printf("%-10s %8s %8s %10s %10s %7s %10s %10s %8s\n",
            "state", "full ms", "anal ms", "full Mc/s", "anal Mc/s", "ratio",
            "full fH2", "anal fH2", "Ea/Ef")
    analytic_available = true
    for (label, nH, T, dt, compare_analytic) in (
            ("forming",   1e12, 1500.0, 1e5, true),
            ("3b-eq",     1e16, 3500.0, 1.0, true),
            ("diss-eq",   1e16, 5000.0, 1.0, true),
            # The reduced analytic solver is outside its intended regime here
            # and can exhaust its subcycle cap, so only benchmark the general
            # full network for the 10^18 cm^-3 asymptote.
            ("3b-eq18",   1e18, 3500.0, 1.0, false),
            ("diss-eq18", 1e18, 5000.0, 1.0, false))
        fH2 = pair_equilibrium_fraction(nH, T)
        label == "forming" && (fH2 = 1e-4)
        h = state_vectors(P, n, nH, T, fH2)
        a = (rho=to_backend(h.rho), e=to_backend(h.e),
             HII=to_backend(h.HII), H2I=to_backend(h.H2I))
        # Copies are required for the CPU identity backend; otherwise the reset
        # buffers alias the evolved arrays and repeated samples accumulate steps.
        initial = (e=to_backend(copy(h.e)), HII=to_backend(copy(h.HII)),
                   H2I=to_backend(copy(h.H2I)))
        common = (; a_value=1 / 21, dt, density_units=1.0,
                  length_units=1.0, time_units=1.0,
                  backend=name, precision=P, dtfrac=0.1)
        full! = () -> solve_chem_device!(a.rho, a.e, a.HII, a.H2I; common...)
        anal! = () -> solve_chem_analytic_device!(a.rho, a.e, a.HII, a.H2I; common...)
        full_err = try_warmup(full!)
        anal_err = analytic_available && compare_analytic ?
                   try_warmup(anal!) : ErrorException("skipped")
        compare_analytic && (analytic_available &= anal_err === nothing)
        tf = full_err === nothing ? best_time!(full!, a, initial) : NaN
        full_fh2 = isfinite(tf) ? first_host(a.H2I) / P(nH*MH) : NaN
        full_e = isfinite(tf) ? first_host(a.e) : NaN
        ta = anal_err === nothing ? best_time!(anal!, a, initial) : NaN
        anal_fh2 = isfinite(ta) ? first_host(a.H2I) / P(nH*MH) : NaN
        anal_e = isfinite(ta) ? first_host(a.e) : NaN
        @printf("%-10s %8s %8s %10s %10s %7s %10s %10s %8s\n", label,
                isfinite(tf) ? @sprintf("%.3f", 1e3tf) : "n/a",
                isfinite(ta) ? @sprintf("%.3f", 1e3ta) : "n/a",
                isfinite(tf) ? @sprintf("%.2f", n / tf / 1e6) : "n/a",
                isfinite(ta) ? @sprintf("%.2f", n / ta / 1e6) : "n/a",
                isfinite(tf) && isfinite(ta) ? @sprintf("%.3f", tf / ta) : "n/a",
                isfinite(full_fh2) ? @sprintf("%.4g", full_fh2) : "n/a",
                isfinite(anal_fh2) ? @sprintf("%.4g", anal_fh2) : "n/a",
                isfinite(anal_e) ? @sprintf("%.4g", anal_e/full_e) : "n/a")
    end
end

n = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 2^18
benchmark_backend(:cpu, identity, n)

try
    @eval using Metal
    if ChemistryKernels.has_backend(:metal)
        benchmark_backend(:metal, x -> Metal.MtlArray(x), n)
    end
catch err
    @warn "Metal benchmark unavailable" exception=(err, catch_backtrace())
end

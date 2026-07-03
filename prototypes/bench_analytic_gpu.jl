#!/usr/bin/env julia
# bench_analytic_gpu.jl — A6000 throughput of the analytic H+H2 mode vs the full network.
#
# Builds a 128³-scale batch of cells spanning IGM→halo conditions, runs both the
# zero-copy device solvers (Float32), and reports cells/sec.  The analytic path's
# closed-form Compton + Riccati x_HII + quadrature H₂ should give a low, uniform
# iteration count (little warp divergence) → the fast GPU chemistry path.
#
# Run (MultiCode test project has CUDA + ChemistryKernels path-dev'd):
#   LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/cuda/13.1/lib64:\
#     /opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/lib \
#   julia --project=$HOME/Projects/Vespa.jl/lib/MultiCode/test \
#         $HOME/Projects/ChemistryKernels.jl/prototypes/bench_analytic_gpu.jl

using ChemistryKernels, CUDA, Printf
const CK = ChemistryKernels

CUDA.functional() || error("CUDA not functional")
@info "GPU" name=CUDA.name(CUDA.device())

const MH=1.6726e-24; const KB=1.380649e-16; const FH=0.76; const GAMMA=5/3
const HH=0.71; const OMB=0.0456
nHbar(z)=FH*OMB*1.8788e-29*HH^2*(1+z)^3/MH

# ── a PHYSICAL single-redshift batch (as a real grid is): all cells share z=z_grid;
#    density spans mean IGM → first-halo overdensities (δ = 1 … 1e5), with temperature on
#    the adiabat above the CMB floor.  Mass-density (network) units, physical CGS.  (An
#    earlier version assigned each cell a DIFFERENT z's density while stepping at one z —
#    unphysical z-mismatched gas that stresses f32 intrinsics into overflow.) ──
const Z_GRID = 30.0
function make_batch(n)
    rho = Vector{Float64}(undef, n); e = similar(rho); HII = similar(rho); H2I = similar(rho)
    nbar = nHbar(Z_GRID); Tcmb = 2.725*(1+Z_GRID)
    for i in 1:n
        δ  = 10.0^(5.0*(i-1)/(n-1))                  # overdensity 1 … 1e5 (IGM → halo)
        nH = nbar*δ
        # adiabatic heating, floored at T_cmb and capped at ~1000 K — the H₂-cooling floor
        # of first minihalos (the mode's target regime; warm/hot gas is out of scope, and
        # there the H₂ mass-density field underflows f32 in BOTH this mode and the full
        # network — a shared GPU-f32 limitation of the H₂ representation, not this solver).
        T  = clamp(Tcmb*δ^(2/3)*0.5, Tcmb, 1000.0)
        xe = δ < 1e2 ? 2e-4 : 3e-5                   # IGM relic vs halo freeze-out
        xH2 = δ < 1e2 ? 1e-6 : 5e-4
        rho[i] = nH*MH/FH
        e[i]   = KB*T/((GAMMA-1)*1.22*MH)
        HII[i] = xe*nH*MH
        H2I[i] = 2*xH2*nH*MH
    end
    rho, e, HII, H2I
end

function bench(f, label, n; reps=5)
    f()                                              # warmup / compile
    CUDA.synchronize()
    t = Inf
    for _ in 1:reps
        dt = CUDA.@elapsed f()
        t = min(t, dt)
    end
    cps = n/t
    @printf("%-28s  %6.2f ms   %8.1f Mcell/s\n", label, t*1e3, cps/1e6)
    cps
end

function main()
    for N in (64, 128)
        n = N^3
        rho, e, HII, H2I = make_batch(n)
        a_value = 1.0/(1.0+Z_GRID)                     # the grid redshift
        # arrays are physical CGS ⇒ all unit scales = 1 (du=vu2=tu=1); dt in seconds.
        args = (; a_value, dt=1e13, density_units=1.0, length_units=1.0,
                  time_units=1.0, fh=FH, precision=Float32)

        # device arrays (Float32), fresh copies per solver
        println("\n=== N=$N ($(n) cells) ===")

        rt = build_rate_tables(; backend=:cuda, precision=Float32)

        dR=CuArray(Float32.(rho)); dE=CuArray(Float32.(e)); dH=CuArray(Float32.(HII)); dM=CuArray(Float32.(H2I))
        bench("analytic (fits)", n) do
            solve_chem_analytic_device!(dR, dE, dH, dM; args...)
        end

        dRt=CuArray(Float32.(rho)); dEt=CuArray(Float32.(e)); dHt=CuArray(Float32.(HII)); dMt=CuArray(Float32.(H2I))
        bench("analytic (rate table)", n) do
            solve_chem_analytic_device!(dRt, dEt, dHt, dMt; args..., rate_tables=rt)
        end

        dR2=CuArray(Float32.(rho)); dE2=CuArray(Float32.(e)); dH2=CuArray(Float32.(HII)); dM2=CuArray(Float32.(H2I))
        bench("full network (fits)", n) do
            solve_chem_device!(dR2, dE2, dH2, dM2; args...)
        end

        dR3=CuArray(Float32.(rho)); dE3=CuArray(Float32.(e)); dH3=CuArray(Float32.(HII)); dM3=CuArray(Float32.(H2I))
        bench("full network (rate table)", n) do
            solve_chem_device!(dR3, dE3, dH3, dM3; args..., rate_tables=rt)
        end

        # sanity: mean x_e / f_H2 after a SINGLE fresh step (the benched arrays above
        # were stepped 6× in place, which drives the out-of-equilibrium benchmark cells
        # to extreme f32 values — not representative of a real single host step).
        sR=CuArray(Float32.(rho)); sE=CuArray(Float32.(e)); sH=CuArray(Float32.(HII)); sM=CuArray(Float32.(H2I))
        solve_chem_analytic_device!(sR, sE, sH, sM; args...)
        xe = Array(sH ./ sR) ./ FH; fH2 = Array(sM ./ sR) ./ (2FH)
        @printf("   analytic 1-step  mean x_e=%.3e  mean f_H2=%.3e  non-finite=%d\n",
                sum(xe)/n, sum(fH2)/n, count(!isfinite, xe)+count(!isfinite, fH2))
    end
end

main()

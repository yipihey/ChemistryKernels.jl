# binned.jl — phase-space-binned chemistry.
#
# In one chemistry step every cell takes the SAME redshift z and the SAME outer dt,
# so the per-cell update is one deterministic function of the cell's input state:
#     F : (ρ, e, x_HII, x_H2I, x_HDI)  ->  (e', x_HII', x_H2I', x_HDI')      (ρ fixed)
# Instead of calling the stiff solver once per cell, bin cells in that input space,
# evolve ONE representative per occupied bin (reusing `solve_chem!`, so the rep solve
# inherits the units/backend/tables path), and map the per-bin response back to every
# member cell.  At high z the gas is nearly uniform (Compton-locked T, tiny density
# contrast) so the occupied region collapses to a handful of bins → few solves for
# ~all cells, with the per-cell solve being most expensive exactly there (recombination).
#
# Map-back (`mapback`):
#   :linear   (default) — FIRST-ORDER.  Per bin, the log-Jacobian s_{o,a} = ∂ln(y'_o)/∂ln(x_a)
#                         is measured by a batched finite-difference (one extra rep solve per
#                         input axis, all in ONE solve_chem! call), then
#                             y'_o,i = y'_o,rep · ∏_a (x_a,i / x_a,rep)^{s_{o,a}} .
#                         Transfers the cell-to-cell variance to first order, so coarse bins
#                         stay accurate even in the variance-sensitive recombination regime.
#   :ratio              — ZEROTH-ORDER self-response, y'_o,i = x_o,i·(y'_o,rep/x_o,rep)
#                         (the s_{o,o}=1, s_{o,a≠o}=0 special case; no perturbation solves).
#   :broadcast          — bin-constant intensive output (flattens sub-bin variance).
#
# `tol` is the bin width in dex (per log axis).  tol → 0 ⇒ one cell per bin ⇒ bit-identical
# to `solve_chem!`.  Dust/metals add per-cell parameters that are extra inputs to F, so those
# runs delegate to the per-cell `solve_chem!` (no binning).

export solve_chem_binned!, solve_chem_binned_device!

# log-safety floor ONLY — far below any physical CGS value (species mass densities are
# legitimately ~1e-35), so it never clobbers real data; it only guards log(0).
const _BIN_TINY = 1e-300

"""
    solve_chem_binned_device!(rho, e_int, HII, H2I, [HDI]; tol=0.02, mapback=:linear,
                              fd_step=0.05, <solve_chem_device! kwargs>) -> nbins

Fully on-device phase-space-binned solve for device-resident arrays (e.g. `CuArray`s):
the binning (key + sort + segmented geometric-mean representatives), the batched
representative solve, and the scatter all run on the GPU — no host O(N) round-trip per
step.  Same semantics and map-back modes as [`solve_chem_binned!`](@ref); primordial(+D)
only (no dust/metals).  Implemented in the CUDA package extension — `using CUDA`
activates it; this stub errors otherwise.
"""
function solve_chem_binned_device! end

"""
    solve_chem_binned!(rho, e_int, HII, H2I, [HDI]; tol=0.02, mapback=:linear,
                       fd_step=0.05, <all solve_chem! kwargs>) -> nbins

Variance-preserving acceleration of [`solve_chem!`](@ref): solve the reduced
primordial(+D) network once per occupied phase-space bin and map the response back to
every cell, updating `e_int`, `HII`, `H2I` (and `HDI`) in place.  Returns the number of
occupied bins (the solve-count, ≤ `length(rho)`; for `:linear` the total solver work is
`(1+naxes)×nbins` cells in ONE batched call).

`tol` = per-axis bin width in dex (smaller = more bins = more accurate; tol→0 reproduces
`solve_chem!`).  `mapback ∈ (:linear, :ratio, :broadcast)`; `:linear` (default) measures a
per-bin log-Jacobian with finite-difference step `fd_step` (in ln) and is the most
accurate.  All `solve_chem!` keywords are forwarded to the representative solve;
`dust=true`/`metals!==nothing` delegate to the per-cell `solve_chem!` (no binning).

**When it pays.**  Binning replaces N per-cell solves with `nbins` (≪ N) representative
solves.  It is a big win whenever the per-cell solve is the cost — i.e. the **CPU** chem
path, and any regime where chemistry dominates the step (high-z recombination is both the
costliest AND the most uniform → fewest bins).  It is *not* a win for the **GPU** chem
path at large N: the dense per-cell kernel already runs all cells in parallel cheaply, so
the binning's sort/scatter overhead exceeds the kernel it replaces.  Enable it for CPU
chem; leave GPU chem as the dense kernel.  Set `ondevice=true` (GPU-resident arrays) to
keep the bin→solve→scatter entirely on the device.
"""
function solve_chem_binned!(rho::AbstractVector, e_int::AbstractVector,
                            HII::AbstractVector, H2I::AbstractVector,
                            HDI::Union{Nothing,AbstractVector} = nothing;
                            a_value::Real, dt::Real, density_units::Real,
                            length_units::Real, time_units::Real,
                            hubble::Real = 71.0, Om::Real = 0.27, OL::Real = 0.73,
                            fh::Real = 0.76, deuterium::Bool = false,
                            hubble_expansion::Bool = false, adot_over_a::Real = NaN,
                            metals = nothing, rate_tables = nothing, cool_tables = nothing,
                            dtfrac::Real = 0.1, itcap::Int = _SUB_ITMAX, workgroup_size::Int = 0,
                            dust::Bool = false,
                            Z_rel = nothing, G0 = nothing, A_V = nothing,
                            N_H = nothing, N_H2 = nothing,
                            backend::Symbol = :cpu, precision::Type = Float64,
                            tol::Real = 0.02, mapback::Symbol = :linear, fd_step::Real = 0.05,
                            ondevice::Bool = false)
    n = length(rho)
    @assert length(e_int) == n && length(HII) == n && length(H2I) == n
    deut = deuterium && HDI !== nothing
    deut && @assert length(HDI) == n

    # on-device binning (GPU): upload via the registered backend, run the fully-on-device
    # bin→solve→scatter (no host O(N)), download.  Lets a host-array caller use the GPU
    # path without depending on CUDA directly.  Falls through to the host path for :cpu or
    # when the device binning is unavailable / dust|metals are active.
    if ondevice && backend !== :cpu && !(dust || metals !== nothing) &&
       !isempty(methods(solve_chem_binned_device!))
        be = ChemistryKernels.backend(backend)
        rg = to_device(be, collect(rho), precision); eg = to_device(be, collect(e_int), precision)
        hg = to_device(be, collect(HII), precision);  mg = to_device(be, collect(H2I), precision)
        dg = deut ? to_device(be, collect(HDI), precision) : nothing
        nb = solve_chem_binned_device!(rg, eg, hg, mg, dg; a_value, dt, density_units,
            length_units, time_units, hubble, Om, OL, fh, deuterium = deut,
            hubble_expansion, adot_over_a, rate_tables, cool_tables, dtfrac, itcap,
            workgroup_size, precision, tol, mapback, fd_step)
        e_int .= to_host(eg); HII .= to_host(hg); H2I .= to_host(mg); deut && (HDI .= to_host(dg))
        return nb
    end

    # binning is defined for the primordial(+D) state vector only; per-cell dust/metal
    # parameters are extra inputs to F → fall back to the exact per-cell solver.
    if dust || metals !== nothing
        solve_chem!(rho, e_int, HII, H2I, HDI; a_value, dt, density_units, length_units,
            time_units, hubble, Om, OL, fh, deuterium, hubble_expansion, adot_over_a,
            metals, rate_tables, cool_tables, dtfrac, itcap, workgroup_size,
            dust, Z_rel, G0, A_V, N_H, N_H2, backend, precision)
        return n
    end

    # forwarded solve_chem! options (same for the rep solve and any perturbation solves)
    solveopts = (; a_value, dt, density_units, length_units, time_units, hubble, Om, OL, fh,
                   deuterium = deut, hubble_expansion, adot_over_a, rate_tables, cool_tables,
                   dtfrac, itcap, workgroup_size, backend, precision)

    invtol = 1.0 / float(tol)
    flr(x) = x < _BIN_TINY ? _BIN_TINY : float(x)
    lk(x) = round(Int, log10(flr(x)) * invtol)

    # ── pass 1: assign each cell to a bin; accumulate Σlog(input) per bin ──
    binid = Vector{Int}(undef, n)
    lut = Dict{NTuple{5,Int},Int}()
    sR = Float64[]; sE = Float64[]; sH = Float64[]; sM = Float64[]; sD = Float64[]; cnt = Int[]
    @inbounds for i in 1:n
        ri = flr(rho[i]); ei = flr(e_int[i])
        hi = flr(HII[i]); mi = flr(H2I[i]); di = deut ? flr(HDI[i]) : _BIN_TINY
        key = (lk(ri), lk(ei), lk(hi/ri), lk(mi/ri), deut ? lk(di/ri) : 0)
        b = get(lut, key, 0)
        if b == 0
            push!(sR, log(ri)); push!(sE, log(ei)); push!(sH, log(hi))
            push!(sM, log(mi)); push!(sD, log(di)); push!(cnt, 1)
            b = length(cnt); lut[key] = b
        else
            sR[b]+=log(ri); sE[b]+=log(ei); sH[b]+=log(hi); sM[b]+=log(mi); sD[b]+=log(di); cnt[b]+=1
        end
        binid[i] = b
    end
    nb = length(cnt)

    # ── representative inputs = per-bin geometric mean (consistent with log bins) ──
    repR = Vector{Float64}(undef, nb); repE = similar(repR); repH = similar(repR)
    repM = similar(repR); repD = similar(repR)
    @inbounds for b in 1:nb
        c = cnt[b]
        repR[b]=exp(sR[b]/c); repE[b]=exp(sE[b]/c); repH[b]=exp(sH[b]/c)
        repM[b]=exp(sM[b]/c); repD[b]=exp(sD[b]/c)
    end

    linear = mapback === :linear
    h = float(fd_step)
    # input axes for the Jacobian: 1=ρ, 2=e, 3=HII, 4=H2I, (5=HDI)
    naxes = deut ? 5 : 4

    # ── evolve the representatives (+ per-axis perturbations for :linear) in ONE solve ──
    L = linear ? nb*(1+naxes) : nb
    aR = Vector{Float64}(undef, L); aE = similar(aR); aH = similar(aR); aM = similar(aR); aD = similar(aR)
    @inbounds for b in 1:nb                                   # block 0: unperturbed reps
        aR[b]=repR[b]; aE[b]=repE[b]; aH[b]=repH[b]; aM[b]=repM[b]; aD[b]=repD[b]
    end
    if linear
        eh = exp(h)
        @inbounds for ax in 1:naxes, b in 1:nb               # block ax: perturb axis `ax` by ×exp(h)
            o = ax*nb + b
            aR[o]=repR[b]*(ax==1 ? eh : 1.0); aE[o]=repE[b]*(ax==2 ? eh : 1.0)
            aH[o]=repH[b]*(ax==3 ? eh : 1.0); aM[o]=repM[b]*(ax==4 ? eh : 1.0)
            aD[o]=repD[b]*(ax==5 ? eh : 1.0)
        end
    end
    solve_chem!(aR, aE, aH, aM, deut ? aD : nothing; solveopts...)

    # rep OUTPUTS (block 0)
    oE = @view aE[1:nb]; oH = @view aH[1:nb]; oM = @view aM[1:nb]; oD = @view aD[1:nb]

    # per-bin log-sensitivities s_{o,ax}=Δln(out_o)/h (matrices nb×naxes); :ratio ⇒ identity
    SE = zeros(nb, naxes); SH = zeros(nb, naxes); SM = zeros(nb, naxes); SD = zeros(nb, naxes)
    if linear
        @inbounds for ax in 1:naxes, b in 1:nb
            o = ax*nb + b
            SE[b,ax] = (log(flr(aE[o])) - log(flr(oE[b])))/h
            SH[b,ax] = (log(flr(aH[o])) - log(flr(oH[b])))/h
            SM[b,ax] = (log(flr(aM[o])) - log(flr(oM[b])))/h
            deut && (SD[b,ax] = (log(flr(aD[o])) - log(flr(oD[b])))/h)
        end
    else  # :ratio — unit self-sensitivity (axis 2=e for E, 3=HII for H, 4=H2I for M, 5=HDI for D)
        @inbounds for b in 1:nb
            SE[b,2]=1.0; SH[b,3]=1.0; SM[b,4]=1.0; deut && (SD[b,5]=1.0)
        end
    end

    # ── pass 2: scatter the response back (read-before-write per cell) ──
    bcast = mapback === :broadcast
    fhf = float(fh)
    @inbounds for i in 1:n
        b = binid[i]; ri = float(rho[i]); cap = fhf*ri
        if bcast
            e_int[i] = oE[b]/repE[b] * e_int[i]
            HII[i] = oH[b]/repR[b] * ri; H2I[i] = oM[b]/repR[b] * ri
            deut && (HDI[i] = oD[b]/repR[b] * ri)
        else
            # deviations of this cell's inputs from the bin representative (in ln)
            lR = log(flr(ri)/repR[b]); lE = log(flr(e_int[i])/repE[b])
            lH = log(flr(HII[i])/repH[b]); lM = log(flr(H2I[i])/repM[b])
            lD = deut ? log(flr(HDI[i])/repD[b]) : 0.0
            ee = SE[b,1]*lR + SE[b,2]*lE + SE[b,3]*lH + SE[b,4]*lM + (deut ? SE[b,5]*lD : 0.0)
            eh = SH[b,1]*lR + SH[b,2]*lE + SH[b,3]*lH + SH[b,4]*lM + (deut ? SH[b,5]*lD : 0.0)
            em = SM[b,1]*lR + SM[b,2]*lE + SM[b,3]*lH + SM[b,4]*lM + (deut ? SM[b,5]*lD : 0.0)
            e_int[i] = oE[b]*exp(ee); HII[i] = oH[b]*exp(eh); H2I[i] = oM[b]*exp(em)
            if deut
                ed = SD[b,1]*lR + SD[b,2]*lE + SD[b,3]*lH + SD[b,4]*lM + SD[b,5]*lD
                HDI[i] = oD[b]*exp(ed)
            end
        end
        # keep species physical: 0 ≤ x, HII+H2I ≤ fh·ρ
        m = H2I[i]; m = m < 0 ? zero(m) : (m > cap ? oftype(m, cap) : m); H2I[i] = m
        hmax = cap - m
        hh = HII[i]; hh = hh < 0 ? zero(hh) : (hh > hmax ? oftype(hh, hmax) : hh); HII[i] = hh
        if deut
            dd = HDI[i]; dd = dd < 0 ? zero(dd) : (dd > cap ? oftype(dd, cap) : dd); HDI[i] = dd
        end
        e_int[i] = e_int[i] < _BIN_TINY ? oftype(e_int[i], _BIN_TINY) : e_int[i]
    end
    return nb
end

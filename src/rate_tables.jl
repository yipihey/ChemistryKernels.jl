# rate_tables.jl — OPTIONAL log–log rate lookup for the per-iteration hot path.
#
# The breakdown (GPU, real cosmological cells) shows `build_rates` is ~60–65% of the
# sub-cycle cost: ~25 temperature-dependent analytic fits, each an exp/pow/log, re-
# evaluated EVERY sub-step because T moves.  This module precomputes those fits on a
# log-spaced T grid and replaces them with a branchless linear interpolation in
# (log₁₀T, log₁₀ rate).  log–log linear was chosen deliberately:
#   • power-law rates (k = C·Tᵖ) are EXACT (a straight line in log–log);
#   • exp / exp(poly(logT)) rates are smooth and well-approximated at fine spacing;
#   • linear interpolation NEVER introduces a maximum or minimum between nodes, so the
#     monotonic structure of every rate is preserved (no spurious wiggles) — important
#     for the stiff solver and for differentiability (the interpolant is piecewise-
#     linear ⇒ a well-defined slope inside every cell, AD-friendly).
#
# The analytic `build_rates` remains the default and the bit-exact reference; the table
# is opt-in (`evolve_cell(...; rate_tables=…)`).  Only the pure-T pieces are tabulated:
# the Peebles C-factor (needs nHI, Hz) and the Trad terms (β₁s, k27, k28, He Saha,
# hoisted in `cmb_rates`) stay as cheap arithmetic on the interpolated aB/bet.

export RateTables, build_rate_tables, table_rates

# Tabulated columns, IN ORDER.  aB = recfast_alpha(T); bet = the Saha β factor inside
# peebles_k2 (so k2's C-factor is rebuilt per-cell from nHI/Hz).  The rest are the
# direct pure-T reaction rates that `build_rates` evaluates every sub-step.
const _RT_COLS = (:aB, :bet,
                  :k1, :k3, :k4, :k5, :k6,
                  :k7, :k8, :k9, :k10, :k11, :k12, :k13, :k14, :k15,
                  :k16, :k17, :k18, :k19, :k22, :k57, :k58,
                  :k50, :k51, :k52, :k53, :k54, :k55, :k56)
const _NRT = length(_RT_COLS)

"""
    RateTables(logk, x0, invdx, N)

A precomputed log–log rate table: `logk` is an `(N, $(_NRT))` array of `log₁₀(rate)` on a
uniform `log₁₀T` grid, `x0 = log₁₀(Tmin)`, `invdx = 1/Δlog₁₀T`.  Device-resident (adapts
to `CuDeviceArray`/Metal inside a kernel via `Adapt`).  Build with [`build_rate_tables`](@ref).
"""
struct RateTables{A}
    logk::A
    x0::Float64
    invdx::Float64
    N::Int
end
Adapt.@adapt_structure RateTables

"""
    build_rate_tables(; Tmin=1.0, Tmax=1.0e9, N=1024, precision=Float64, backend=:cpu)

Evaluate every tabulated rate ([`_RT_COLS`](@ref)) on a uniform `log₁₀T` grid of `N`
points over `[Tmin, Tmax]` K, store `log₁₀(rate)`, and upload to `backend`.  The default
`N=1024` (≈0.0088 dex over 1–10⁹ K, ~0.24 MB) keeps the curved (exp) rates to ≲1% — far
inside the ~10–20% physical uncertainty of the analytic fits themselves — while power-law
rates are exact at any spacing; the small table stays comfortably L2-resident.
"""
function build_rate_tables(; Tmin::Real = 1.0, Tmax::Real = 1.0e9, N::Int = 1024,
                           precision::Type = Float64, backend::Symbol = :cpu)
    R  = precision
    x0 = log10(Float64(Tmin)); x1 = log10(Float64(Tmax))
    dx = (x1 - x0) / (N - 1)
    M  = Array{R}(undef, N, _NRT)
    for j in 1:N
        Tj  = R(10.0)^R(x0 + (j - 1) * dx)
        aB  = recfast_alpha(Tj)
        bet = aB * (R(_REC_CR) * Tj)^R(1.5) * exp(-R(_REC_CDB) / Tj)
        vals = (aB, bet,
                k1(Tj), k3(Tj), k4(Tj), k5(Tj), k6(Tj),
                k7(Tj), k8(Tj), k9(Tj), k10(Tj), k11(Tj), k12(Tj), k13(Tj), k14(Tj), k15(Tj),
                k16(Tj), k17(Tj), k18(Tj), k19(Tj), k22(Tj), k57(Tj), k58(Tj),
                k50(Tj), k51(Tj), k52(Tj), k53(Tj), k54(Tj), k55(Tj), k56(Tj))
        @inbounds for c in 1:_NRT
            M[j, c] = log10(max(vals[c], R(1.0e-300)))   # all rates > 0 (floored at tiny)
        end
    end
    dev = to_device(ChemistryKernels.backend(backend), M, R)
    return RateTables(dev, x0, 1.0 / dx, N)
end

# One column's interpolated rate: linear in (log₁₀T, log₁₀ rate), exponentiated back.
@inline function _rt_lookup(L, N::Int, i::Int, f, c::Int)
    @inbounds lo = L[i + (c - 1) * N]
    @inbounds hi = L[i + 1 + (c - 1) * N]
    return exp10(lo + f * (hi - lo))
end

"""
    table_rates(rt::RateTables, T, nHI, Hz, cr; deuterium=false)

Drop-in replacement for the per-iteration `build_rates(T, nHI, Hz, cr; …)`: returns the
SAME NamedTuple of rate coefficients, but the pure-T rates come from `rt` by log–log
interpolation instead of analytic fits.  k2's Peebles C-factor and `k_beta1s` are rebuilt
from the interpolated `aB`/`bet` and the hoisted Trad terms `cr` (β₁s, k27, k28, He Saha).
"""
@inline function table_rates(rt::RateTables, T, nHI, Hz, cr; deuterium::Bool = false)
    R = typeof(T)
    # locate T on the log grid (clamp to the endpoints → constant extrapolation outside)
    s = (log10(Float64(T)) - rt.x0) * rt.invdx
    s = clamp(s, 0.0, Float64(rt.N) - 1.0 - 1.0e-9)
    b = unsafe_trunc(Int, s)            # 0-based bin
    f = R(s - b)
    i = b + 1                            # 1-based row of the lower node
    L = rt.logk; N = rt.N
    @inline rd(c) = _rt_lookup(L, N, i, f, c)

    aB  = rd(1); bet = rd(2)
    # Peebles C-factor (Sobolev escape), exactly as peebles_k2 but on tabulated aB/bet.
    n1s = nHI * R(1.0e6)
    Kf  = R(_REC_LAM)^3 / (R(8.0) * R(π) * Hz)
    KL  = Kf * R(_REC_A8) * n1s
    KB  = Kf * bet * n1s
    C   = (one(R) + KL) / (one(R) + KL + KB)
    k2_val = aB * R(1.0e6) * C
    k_b1s  = cr.b1s * k2_val / (aB * R(1.0e6))      # = cr.b1s·C
    she1, she2 = cr.she

    base = (; k1=rd(3), k2=k2_val, k3=rd(4), k4=rd(5), k5=rd(6), k6=rd(7),
            k7=rd(8), k8=rd(9), k9=rd(10), k10=rd(11), k11=rd(12), k12=rd(13),
            k13=rd(14), k14=rd(15), k15=rd(16), k16=rd(17), k17=rd(18), k18=rd(19),
            k19=rd(20), k22=rd(21), k57=rd(22), k58=rd(23),
            k27=cr.k27, k28=cr.k28, k_beta1s=k_b1s, she1=she1, she2=she2)
    deuterium || return base
    return merge(base, (; k50=rd(24), k51=rd(25), k52=rd(26), k53=rd(27),
                        k54=rd(28), k55=rd(29), k56=rd(30)))
end

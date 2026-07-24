# test_rate_tables.jl — the OPTIONAL log–log rate/cooling tables vs the analytic fits.
#
# build_rate_tables / build_cooling_tables replace the per-sub-step temperature fits with
# linear interpolation in (log₁₀T, log₁₀ k).  Per-coefficient RELATIVE error is a
# misleading metric — it explodes where a rate is exp-suppressed to ~1e-36 (dead) yet is
# ~machine-precision where the rate is a significant power-law.  So this suite guards what
# actually matters:
#   1. END-TO-END one-zone solves (cosmological cold gas AND a moderate-T ionized cell, both
#      well-conditioned) match the analytic network tightly;
#   2. the cooling table matches the analytic cooling sum tightly across the physical range;
#   3. smooth, always-significant rates (recombination k2/k4/k6) match tightly;
#   4. on a grid node the interpolant reproduces the analytic fit exactly.
# NOTE (documented, not a failure): in the ~3e4–1e5 K thermal-instability band (cooling-
# function peak) the post-step state is intrinsically sensitive — the table deviates from
# analytic no more than a 1e-6 perturbation of the input energy does (see the high-T
# conditioning study); that band is not asserted here.

using ChemistryKernels, EmissionKernels, Test
const _CK = ChemistryKernels
const _EK = EmissionKernels

# one-zone host-boundary solve for n identical cells; returns (e, HII, H2I, HDI) host arrays.
function _onezone(; rt, ct, T0, xHII, a_value, dt, n = 256)
    du = 2.566e-21; lu = 5.557e20; tu = 4.337e11; vu2 = (lu / tu)^2
    μ = 1.22; mh = 1.6605e-24; kb = 1.3806e-16
    eint = (kb * T0) / ((5/3 - 1) * μ * mh) / vu2
    rho = fill(0.17, n); e = fill(eint, n)
    HII = rho .* xHII; H2I = rho .* 1e-6; HDI = rho .* 6.8e-5 .* xHII
    _CK.solve_chem!(rho, e, HII, H2I, HDI; a_value = a_value, dt = dt,
                    density_units = du, length_units = lu, time_units = tu,
                    deuterium = true, backend = :cpu, precision = Float64,
                    rate_tables = rt, cool_tables = ct)
    return e, HII, H2I, HDI
end
_relmax(a, b) = maximum(abs.(a .- b)) / (maximum(abs.(b)) + eps())

@testset "rate/cooling tables (log–log)" begin
    # use the package defaults (N=1024) so the suite tracks the shipped table size.
    rt = _CK.build_rate_tables(;  precision = Float64, backend = :cpu)
    ct = _EK.build_cooling_tables(; precision = Float64, backend = :cpu)

    # 1. end-to-end: well-conditioned regimes match the analytic network -----------------
    for (T0, xHII, a_value, dt) in ((2727.0, 0.06, 1/1001, 2.0),   # cosmological cold gas
                                    (1.0e4,  0.50, 1/51,   0.5))   # warm, partially ionized
        ea, h1a, h2a, hda = _onezone(; rt = nothing, ct = nothing, T0 = T0, xHII = xHII,
                                       a_value = a_value, dt = dt)
        et, h1t, h2t, hdt = _onezone(; rt = rt, ct = ct, T0 = T0, xHII = xHII,
                                       a_value = a_value, dt = dt)
        @test _relmax(et, ea)  < 3e-3
        @test _relmax(h1t, h1a) < 3e-3
        @test _relmax(h2t, h2a) < 3e-3
        # HDI is the most interpolation-sensitive (trace D species fed by the k50–k56
        # D-network); at N=1024 it carries ~1% — far inside HD's physical uncertainty.
        @test _relmax(hda, hdt) < 2e-2
    end

    # 2. cooling table: tight across the physical range (≥100 K; below the CMB it crosses
    #    zero, where relative error is meaningless) --------------------------------------
    z = 50.0; nH = 1.0
    coolmax = 0.0
    for T in 10.0 .^ range(2.0, 9.0, length = 3001)
        a = _EK.cooling_rate_total(0.7, 0.3, 0.08, 0.3, 1e-3, 1e-7, T, z; nH = nH)
        b = _EK.cooling_rate_total_tab(ct, 0.7, 0.3, 0.08, 0.3, 1e-3, 1e-7, T, z; nH = nH)
        coolmax = max(coolmax, abs(b - a) / (abs(a) + 1e-300))
    end
    @test coolmax < 3e-3

    # 3. smooth always-significant rates (recombination) match tightly everywhere --------
    Trad = _EK.comp2_cmb(z); cr = _CK.cmb_rates(Trad); Hz = _CK.hubble_z_of(z)
    smoothmax = 0.0
    for T in 10.0 .^ range(0.0, 9.0, length = 3001)
        a = _CK.build_rates(T, Trad, 0.1, Hz; deuterium = true)
        b = _CK.table_rates(rt, T, 0.1, Hz, cr; deuterium = true)
        for f in (:k2, :k4, :k6)
            smoothmax = max(smoothmax, abs(getfield(b, f) - getfield(a, f)) /
                                       (abs(getfield(a, f)) + 1e-300))
        end
    end
    @test smoothmax < 2e-3

    # HeH⁺ is a cold-gas channel. Test its complete assembled rates over the
    # relevant 1–10⁴ K range; above this, radiative association is exponentially
    # dead and relative interpolation error has no physical meaning.
    hehmax = 0.0
    for T in 10.0 .^ range(0.0, 4.0, length = 2001)
        a = _CK.build_rates(T, Trad, 0.1, Hz)
        b = _CK.table_rates(rt, T, 0.1, Hz, cr)
        for f in (:kHeH_ra, :kHeH_H, :kHeH_e)
            hehmax = max(hehmax, abs(getfield(b, f) - getfield(a, f)) /
                                 (abs(getfield(a, f)) + 1e-300))
        end
    end
    @test hehmax < 2e-3

    # 4. interpolation identity on a grid node -------------------------------------------
    Tnode = 10.0 ^ (512 * 9.0 / (rt.N - 1))
    an = _CK.build_rates(Tnode, Trad, 0.1, Hz; deuterium = true)
    tn = _CK.table_rates(rt, Tnode, 0.1, Hz, cr; deuterium = true)
    @test isapprox(tn.k1, an.k1; rtol = 1e-6)
    @test isapprox(tn.k2, an.k2; rtol = 1e-6)

    # The table follows the unfloored atomic tails instead of imposing a second
    # 0.8-eV branch. At 3000 K k1 is small but still representable in Float64.
    Ttail = 3000.0
    atail = _CK.build_rates(Ttail, Trad, 0.1, Hz)
    ttail = _CK.table_rates(rt, Ttail, 0.1, Hz, cr)
    @test 0.0 < ttail.k1 < 1.0e-20
    @test isapprox(ttail.k1, atail.k1; rtol = 5e-3)

    # k55's low-T continuation is monotone and C¹-matched at 200 K rather
    # than held at the former 1.08e-22 sentinel.
    @test _CK.k55(0.0) == 0.0
    @test issorted(_CK.k55.(10.0 .^ range(0, log10(200.0); length=201)))
    @test _CK.k55(100.0) < 1.0e-22
    @test isapprox(_CK.k55(prevfloat(200.0)), _CK.k55(nextfloat(200.0)); rtol=1e-12)

    # Savin's k50 difference fit crosses slightly negative below its useful
    # range. Both analytic and table paths represent that inactive tail as zero.
    @test _CK.k50(0.0) == 0.0
    @test _CK.k50(1.0) == 0.0
    k50tab = _CK.table_rates(rt, 1.0, 0.1, Hz, cr; deuterium=true)
    @test k50tab.k50 == 0.0

    # 5. exact-zero nodes and empty cooling states remain finite in both precisions -----
    for R in (Float64, Float32)
        rtz = _CK.build_rate_tables(; precision = R, backend = :cpu, N = 128)
        ctz = _EK.build_cooling_tables(; precision = R, backend = :cpu, N = 128)
        Tz = R(100); crz = _CK.cmb_rates(R(2.725))
        kz = _CK.table_rates(rtz, Tz, R(1), R(1e-16), crz; deuterium = true)
        @test kz.k3 == zero(R)
        @test kz.k5 == zero(R)
        @test kz.k11 == zero(R)
        @test kz.k12 == zero(R)
        @test kz.k13 == zero(R)
        @test kz.k14 == zero(R)
        @test all(isfinite, values(kz))
        kmax = _CK.table_rates(rtz, R(1e9), R(1), R(1e-16), crz; deuterium = true)
        @test all(isfinite, values(kmax))
        empty_cooling = _EK.cooling_rate_total_tab(
            ctz, zero(R), zero(R), zero(R), zero(R), zero(R), zero(R), Tz, R(20))
        @test empty_cooling == zero(R)
        @test isfinite(empty_cooling)
        hot_cooling = _EK.cooling_rate_total_tab(
            ctz, R(0.7), R(0.3), R(0.08), R(0.3), R(1e-3), R(1e-7), R(1e9), R(20))
        @test isfinite(hot_cooling)
    end
end

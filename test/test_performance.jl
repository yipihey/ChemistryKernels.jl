# test_performance.jl — accuracy tests for performance-tuning knobs.
#
# Guards: dtfrac relaxation stays within tolerance; workgroup_size produces
# identical results; cell-sort permutation invariance; sort utility correctness.
# No GPU required; all tests run on :cpu.

using ChemistryKernels, Test
using ChemistryKernels: solve_chem!, sort_cells_by_temperature

# Shared one-zone setup — 256 identical cells at a given (T0, xHII).
const _DU_P = 2.566e-21; const _LU_P = 5.557e20; const _TU_P = 4.337e11
const _VU2_P = (_LU_P / _TU_P)^2

function _cells_p(; T0, xHII, n = 256)
    μ = 1.22; mh = 1.6605e-24; kb = 1.3806e-16
    eint = (kb * T0) / ((5/3 - 1) * μ * mh) / _VU2_P
    rho = fill(0.17, n)
    e   = fill(eint, n)
    HII = rho .* xHII
    H2I = rho .* 1e-6
    HDI = rho .* 6.8e-5 .* xHII
    return rho, e, HII, H2I, HDI
end

function _run_p(rho, e0, HII0, H2I0, HDI0; a_value, dt, dtfrac = 0.1, workgroup_size = 0)
    e   = copy(e0);   HII = copy(HII0)
    H2I = copy(H2I0); HDI = copy(HDI0)
    solve_chem!(rho, e, HII, H2I, HDI;
                a_value = a_value, dt = dt,
                density_units = _DU_P, length_units = _LU_P, time_units = _TU_P,
                deuterium = true, backend = :cpu, precision = Float64,
                dtfrac = dtfrac, workgroup_size = workgroup_size)
    return e, HII, H2I, HDI
end

_relmax(a, b) = maximum(abs.(a .- b) ./ (abs.(b) .+ eps()))

@testset "dtfrac relaxation accuracy" begin
    # dtfrac=0.2 doubles the allowed fraction-change per sub-step vs dtfrac=0.1.
    # Cold cosmological gas: slow reactions, few sub-steps → small error.
    # Warm partially-ionized gas: faster chemistry → larger truncation per sub-step.
    # These tolerances document the actual regime sensitivity, not physical accuracy:
    # in production hydro coupling chemistry error << hydro truncation error anyway.
    for (T0, xHII, a_value, dt, tol_e, tol_x) in (
            (2727.0, 0.06, 1/1001, 2.0, 5e-3, 1.1e-2), # cosmological cold; coupled H solve
            (1.0e4,  0.50, 1/51,   0.5, 1e-1, 2e-1))   # warm, faster chemistry
        rho, e0, HII0, H2I0, HDI0 = _cells_p(; T0 = T0, xHII = xHII)
        e_a, hii_a, h2i_a, _ = _run_p(rho, e0, HII0, H2I0, HDI0; a_value = a_value, dt = dt, dtfrac = 0.1)
        e_b, hii_b, h2i_b, _ = _run_p(rho, e0, HII0, H2I0, HDI0; a_value = a_value, dt = dt, dtfrac = 0.2)
        @test _relmax(e_b,   e_a)   < tol_e
        @test _relmax(hii_b, hii_a) < tol_x
        @test _relmax(h2i_b, h2i_a) < tol_x
    end

    # dtfrac=0.1 (default) is bit-identical to itself (sanity).
    rho, e0, HII0, H2I0, HDI0 = _cells_p(; T0 = 2727.0, xHII = 0.06)
    ea, _, _, _ = _run_p(rho, e0, HII0, H2I0, HDI0; a_value = 1/1001, dt = 2.0, dtfrac = 0.1)
    eb, _, _, _ = _run_p(rho, e0, HII0, H2I0, HDI0; a_value = 1/1001, dt = 2.0, dtfrac = 0.1)
    @test ea == eb
end

@testset "workgroup_size result invariance" begin
    # On CPU the workgroup_size parameter is ignored by KernelAbstractions, so any
    # value must give bit-identical results to the default (0 = backend default).
    rho, e0, HII0, H2I0, HDI0 = _cells_p(; T0 = 5000.0, xHII = 0.1)
    kw = (; a_value = 1/101, dt = 1.0)
    e0r, h0, h0_, hd0 = _run_p(rho, e0, HII0, H2I0, HDI0; kw..., workgroup_size = 0)
    for ws in (64, 128, 256)
        ew, hw, hw_, hdw = _run_p(rho, e0, HII0, H2I0, HDI0; kw..., workgroup_size = ws)
        @test ew   == e0r
        @test hw   == h0
        @test hw_  == h0_
        @test hdw  == hd0
    end
end

@testset "cell-sort permutation invariance" begin
    # Sorting cells by temperature before dispatch must not change per-cell results.
    n = 64
    rng_T = range(500.0, 1.5e4; length = n)
    μ = 1.22; mh = 1.6605e-24; kb = 1.3806e-16
    e0    = [(kb * T) / ((5/3 - 1) * μ * mh) / _VU2_P for T in rng_T]
    rho   = fill(0.17, n)
    HII0  = rho .* range(1e-5, 0.3; length = n)
    H2I0  = rho .* 1e-6
    HDI0  = rho .* 1e-8

    # Reference: original ordering.
    e_ref  = copy(e0);  hii_ref = copy(HII0)
    h2_ref = copy(H2I0); hdi_ref = copy(HDI0)
    solve_chem!(rho, e_ref, hii_ref, h2_ref, hdi_ref;
                a_value = 1/51, dt = 0.5,
                density_units = _DU_P, length_units = _LU_P, time_units = _TU_P,
                deuterium = true, backend = :cpu, precision = Float64)

    # Sorted run: permute inputs → solve → invert permutation.
    perm = sort_cells_by_temperature(e0)
    e_s   = e0[perm];   hii_s  = HII0[perm]
    h2_s  = H2I0[perm]; hdi_s  = HDI0[perm]
    rho_s = rho[perm]
    solve_chem!(rho_s, e_s, hii_s, h2_s, hdi_s;
                a_value = 1/51, dt = 0.5,
                density_units = _DU_P, length_units = _LU_P, time_units = _TU_P,
                deuterium = true, backend = :cpu, precision = Float64)
    inv_perm = invperm(perm)
    @test e_s[inv_perm]   == e_ref
    @test hii_s[inv_perm] == hii_ref
    @test h2_s[inv_perm]  == h2_ref
    @test hdi_s[inv_perm] == hdi_ref
end

@testset "sort_cells_by_temperature utility" begin
    # Ascending sort proxy: sorted e_int should be non-decreasing.
    e = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    p = sort_cells_by_temperature(e)
    @test issorted(e[p])
    # Identity on already-sorted input.
    e2 = collect(1.0:8.0)
    @test sort_cells_by_temperature(e2) == 1:8
    # Single cell.
    @test sort_cells_by_temperature([42.0]) == [1]
end

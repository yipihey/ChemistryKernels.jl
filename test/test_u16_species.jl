# test_u16_species.jl — UInt16 log₂-encoded species storage tests.
#
# Verifies: codec accuracy (encode/decode roundtrip ≈ 0.12 %/ULP), boundary
# saturation, monotonicity, solve_chem_u16! physical sanity, and agreement
# with the standard Float32 path within the storage quantization tolerance.

using ChemistryKernels, Test
using ChemistryKernels: encode_log2sp, decode_log2sp,
                        encode_log2sp_vec, decode_log2sp_vec,
                        solve_chem!, solve_chem_u16!

# ── Codec unit tests ──────────────────────────────────────────────────────────

@testset "encode_log2sp / decode_log2sp" begin
    # Boundary values
    @test encode_log2sp(1.0)           == UInt16(65535)
    @test encode_log2sp(0.0)           == UInt16(0)    # 0 → log2(0)=-Inf → clamped to u=0
    @test encode_log2sp(1e-300)        == UInt16(0)    # Float32 underflow → -Inf → u=0
    @test encode_log2sp(exp2(-110.0))  <= UInt16(2)    # exact 2^(-110) → u≈0

    # Decode boundary
    @test decode_log2sp(Float64, UInt16(65535)) ≈ 1.0   rtol=1e-4
    @test decode_log2sp(Float64, UInt16(0))     ≈ 7.73e-34 rtol=1e-2

    # Monotonicity: encode is non-decreasing in x
    xs = [1e-30, 1e-20, 1e-10, 1e-6, 1e-3, 0.1, 0.5, 1.0]
    us = encode_log2sp.(xs)
    @test issorted(us)

    # Roundtrip accuracy: decode(encode(x)) ≈ x within 0.12 % per ULP.
    # We test at 0.5 % (≈ 4 ULPs) to account for rounding in both directions.
    for x in [1e-25, 1e-15, 1e-10, 1e-6, 1e-3, 0.01, 0.1, 0.3, 0.76, 1.0]
        u  = encode_log2sp(x)
        xr = decode_log2sp(Float64, u)
        @test abs(xr - x) / x < 5e-3   # ≤ 0.5% roundtrip error
    end

    # Type stability: decode returns requested type
    @test decode_log2sp(Float32, UInt16(1000)) isa Float32
    @test decode_log2sp(Float64, UInt16(1000)) isa Float64

    # Vector helpers round-trip
    xs_vec = [0.5, 0.1, 1e-4, 1e-8, 1e-15]
    u_vec  = encode_log2sp_vec(xs_vec)
    @test u_vec isa Vector{UInt16}
    xr_vec = decode_log2sp_vec(Float64, u_vec)
    @test all(abs.(xr_vec .- xs_vec) ./ xs_vec .< 5e-3)
end

# ── Physics sanity: solve_chem_u16! produces valid output ────────────────────

const _DU_U = 2.566e-21; const _LU_U = 5.557e20; const _TU_U = 4.337e11
const _VU2_U = (_LU_U / _TU_U)^2

@testset "solve_chem_u16! physical sanity" begin
    n = 64; fh = 0.76
    μ = 1.22; mh = 1.6605e-24; kb = 1.3806e-16; T0 = 2727.0
    eint_cgs = (kb * T0) / ((5/3 - 1) * μ * mh)
    rho  = fill(0.17, n)
    e_int = fill(eint_cgs / _VU2_U, n)

    # Species as fractions → encode to UInt16
    xHII  = fill(0.06f0, n)
    xH2I  = fill(1f-6, n)
    xHDI  = fill(6.8f-5 * 0.06f0, n)
    HII_u16 = encode_log2sp_vec(xHII)
    H2I_u16 = encode_log2sp_vec(xH2I)
    HDI_u16 = encode_log2sp_vec(xHDI)

    solve_chem_u16!(rho, e_int, HII_u16, H2I_u16, HDI_u16;
                    a_value = 1/1001, dt = 2.0,
                    density_units = _DU_U, length_units = _LU_U, time_units = _TU_U,
                    deuterium = true, backend = :cpu, precision = Float32)

    # Energy positive and finite
    @test all(isfinite, e_int) && all(>(0), e_int)

    # Species fractions bounded in (0, 1] after decode
    xHII_out = decode_log2sp_vec(Float32, HII_u16)
    xH2I_out = decode_log2sp_vec(Float32, H2I_u16)
    @test all(>(0), xHII_out) && all(<=(1.01f0), xHII_out)
    @test all(>(0), xH2I_out) && all(<=(0.51f0), xH2I_out)
end

# ── Parity with Float32 path ──────────────────────────────────────────────────
#
# solve_chem_u16! and solve_chem!(precision=Float32) compute the same chemistry;
# the only difference is storage quantization (≈ 0.12 %/ULP).  After one step,
# both start from the same initial state (modulo encode/decode round-trip error);
# we expect results to agree within ~1% (a few ULPs of storage error).

@testset "solve_chem_u16! vs Float32 path" begin
    n = 256; fh = 0.76
    μ = 1.22; mh = 1.6605e-24; kb = 1.3806e-16

    for (T0, xHII0, a_val, dt) in (
            (2727.0,  0.06f0,  1/1001, 2.0),    # cosmological cold
            (8000.0,  0.20f0,  1/51,   0.5))    # warm ionized

        eint_cgs = (kb * T0) / ((5/3 - 1) * μ * mh)
        rho   = fill(0.17, n)
        e0    = fill(eint_cgs / _VU2_U, n)
        xHII  = fill(xHII0, n)
        xH2I  = fill(1f-6,  n)

        # Float32 path: species as mass densities (fraction × rho)
        e_f32  = copy(e0)
        HII_f32 = rho .* xHII
        H2I_f32 = rho .* xH2I
        solve_chem!(rho, e_f32, HII_f32, H2I_f32;
                    a_value = a_val, dt = dt,
                    density_units = _DU_U, length_units = _LU_U, time_units = _TU_U,
                    backend = :cpu, precision = Float32)

        # UInt16 path: encode fractions, solve, decode back to fractions
        e_u16    = copy(e0)
        HII_u16  = encode_log2sp_vec(xHII)
        H2I_u16  = encode_log2sp_vec(xH2I)
        solve_chem_u16!(rho, e_u16, HII_u16, H2I_u16;
                        a_value = a_val, dt = dt,
                        density_units = _DU_U, length_units = _LU_U, time_units = _TU_U,
                        backend = :cpu, precision = Float32)

        # Compare: allow up to ~1% difference (≈ 8 ULPs of storage quantization)
        HII_f32_frac = HII_f32 ./ rho
        HII_u16_frac = decode_log2sp_vec(Float32, HII_u16)
        H2I_f32_frac = H2I_f32 ./ rho
        H2I_u16_frac = decode_log2sp_vec(Float32, H2I_u16)

        relmax_e   = maximum(abs.(e_u16   .- e_f32)   ./ (abs.(e_f32)   .+ eps(Float32)))
        relmax_hii = maximum(abs.(HII_u16_frac .- HII_f32_frac) ./ (abs.(HII_f32_frac) .+ eps(Float32)))
        relmax_h2i = maximum(abs.(H2I_u16_frac .- H2I_f32_frac) ./ (abs.(H2I_f32_frac) .+ eps(Float32)))

        @test relmax_e   < 2e-2   # energy: ≤ 2% (quantization + f32 truncation)
        @test relmax_hii < 2e-2   # HII: ≤ 2%
        @test relmax_h2i < 2e-2   # H2I: ≤ 2%
    end
end

# ── No-deuterium path doesn't touch HDI ──────────────────────────────────────

@testset "solve_chem_u16! deuterium=false" begin
    n = 16; rho = fill(0.17, n)
    μ = 1.22; mh = 1.6605e-24; kb = 1.3806e-16
    e0 = fill((kb * 5000.0) / ((5/3-1)*μ*mh) / _VU2_U, n)
    HII_u16 = encode_log2sp_vec(fill(0.1f0, n))
    H2I_u16 = encode_log2sp_vec(fill(1f-5,  n))

    solve_chem_u16!(rho, e0, HII_u16, H2I_u16;
                    a_value = 1/51, dt = 0.5,
                    density_units = _DU_U, length_units = _LU_U, time_units = _TU_U,
                    deuterium = false, backend = :cpu, precision = Float32)

    @test all(isfinite, e0) && all(>(0), e0)
    @test all(decode_log2sp_vec(Float32, HII_u16) .> 0)
end

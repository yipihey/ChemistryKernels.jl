# log2_species.jl — UInt16 log₂-encoded species fraction storage.
#
# Direct port of the GLMMHDTurb march_bridge/cu/spike_25d.cu -DU16SP codec:
#
#   dec_log2(u) = −110 + u × (110/65535)                         → log₂(X)
#   enc_log2(l2) = clamp(round((l2+110) × (65535/110)), 0, 65535) → UInt16
#
# Mapping: u=0 → X=2^(−110) ≈ 7.73e-34 (below TINY);  u=65535 → X=1.0
# Range   : 33.1 decades  (sufficient for all primordial species + trace metals)
# Precision: 110/65535 ≈ 1.68e-3 log₂-units/ULP ≈ 5.0e-4 dex/ULP ≈ 0.12%/ULP
#
# Memory benefit: 2 B/cell vs 4 B (f32) — 2× bandwidth on species arrays.
# On an A6000 (768 GB/s) this saves ~12 GB/s per species array at 33 Mcell/s.
#
# Usage (host side):
#   HII_u16 = encode_log2sp_vec(HII ./ rho)   # mass-density → fraction → UInt16
#   solve_chem_u16!(rho, e_int, HII_u16, H2I_u16; ...)
#   HII .= decode_log2sp_vec(Float64, HII_u16) .* rho   # back to mass-density

export encode_log2sp, decode_log2sp, encode_log2sp_vec, decode_log2sp_vec

# Codec constants — all Float32 for GPU/Metal compatibility.
const _LOG2SP_LO    = -110f0               # log₂(X_min): lower end of the linear map
const _LOG2SP_RANGE = 110f0                # total span in log₂ units
const _LOG2SP_SCALE = 65535f0 / 110f0     # ULPs per log₂-unit (encode direction)
const _LOG2SP_INV   = 110f0 / 65535f0    # log₂-units per ULP  (decode direction)
const _LOG2SP_XMIN  = 7.73f-34            # 2^(−110) — minimum representable fraction

"""
    decode_log2sp(T, u::UInt16) -> T

Decode a UInt16 log₂-encoded species mass fraction to floating-point type `T`.

  u = 0     → T(2^−110) ≈ 7.73e-34   (minimum; sub-TINY)
  u = 65535 → T(1.0)                  (full species)

All arithmetic is Float32 internally, so this is safe on Metal and CUDA.
"""
@inline function decode_log2sp(::Type{T}, u::UInt16) where {T <: AbstractFloat}
    T(exp2(Float32(u) * _LOG2SP_INV + _LOG2SP_LO))
end

"""
    encode_log2sp(x) -> UInt16

Encode a species mass fraction `x ∈ (0, 1]` as a UInt16 log₂-encoded value.
Fractions below 7.73e-34 saturate to u=0; above 1 saturate to u=65535.
All arithmetic is Float32 internally (GPU/Metal-safe).
"""
@inline function encode_log2sp(x::Real)
    # Direct port of the CUDA enc_log2: compute log₂(x), map to [0,65535], clamp output.
    # Saturation: x ≤ 2^(−110) → t ≤ 0 → u=0; x ≥ 1 → t ≥ 65535 → u=65535.
    # log2(0) = −Inf in IEEE 754 → t = −Inf → clamped to 0 (sub-TINY fractions round to u=0).
    l2 = log2(Float32(x))                          # −Inf for x=0; passes NaN through
    t  = (l2 - _LOG2SP_LO) * _LOG2SP_SCALE        # CUDA: (l2 + 110) * (65535/110)
    tc = clamp(t + 0.5f0, 0f0, 65535f0)           # clamp+round-by-truncation (CUDA fmin/fmax)
    unsafe_trunc(UInt16, unsafe_trunc(Int32, tc))  # Int32 step avoids UInt16 wrap on truncation
end

"""
    encode_log2sp_vec(fracs::AbstractVector) -> Vector{UInt16}

Encode a vector of species mass fractions to UInt16 log₂ form.
Convenience wrapper for preparing host arrays before passing to `solve_chem_u16!`.

```julia
# Convert mass-density arrays (same units as rho) → fraction → UInt16
HII_u16 = encode_log2sp_vec(HII ./ rho)
H2I_u16 = encode_log2sp_vec(H2I ./ rho)
```
"""
encode_log2sp_vec(x::AbstractVector) = UInt16[encode_log2sp(xi) for xi in x]

"""
    decode_log2sp_vec(T, u::AbstractVector{UInt16}) -> Vector{T}

Decode a vector of UInt16 log₂-encoded fractions back to floating-point type `T`.

```julia
# Recover mass-density arrays after a solve_chem_u16! call
HII .= decode_log2sp_vec(Float64, HII_u16) .* rho
```
"""
decode_log2sp_vec(::Type{T}, u::AbstractVector{UInt16}) where {T} =
    T[decode_log2sp(T, ui) for ui in u]

"""
    ChemistryKernelsCUDAExt

Package extension that lights up the CUDA (NVIDIA GPU) backend for
`ChemistryKernels`. Loaded automatically when `CUDA` is present in the
environment. Registers the `:cuda` backend and specialises the device-array
helpers onto `CuArray`. CUDA supports both Float32 and Float64.
"""
module ChemistryKernelsCUDAExt

using ChemistryKernels
using CUDA

function __init__()
    if CUDA.functional()
        ChemistryKernels.register_backend!(:cuda, CUDABackend())
    end
end

ChemistryKernels.device_zeros(::CUDABackend, ::Type{T}, dims::Dims) where {T} =
    CUDA.zeros(T, dims)

# ── on-device phase-space binning ────────────────────────────────────────────
# Bin → batched representative solve → scatter, entirely on the GPU (no host O(N)
# round-trip).  All steps are high-level CUDA.jl array ops: a packed integer key per
# cell, CUDA `sortperm`, a `cumsum` segmented geometric mean for the representatives,
# the existing `solve_chem_device!` kernel for the (few) reps, and a broadcast scatter
# with gathered per-bin response (zeroth-order `:ratio` or first-order `:linear`).
const _TINY = 1e-300

function ChemistryKernels.solve_chem_binned_device!(
        rho::CuArray, e_int::CuArray, HII::CuArray, H2I::CuArray,
        HDI::Union{Nothing,CuArray} = nothing;
        a_value::Real, dt::Real, density_units::Real, length_units::Real, time_units::Real,
        hubble::Real = 71.0, Om::Real = 0.27, OL::Real = 0.73, fh::Real = 0.76,
        deuterium::Bool = false, hubble_expansion::Bool = false, adot_over_a::Real = NaN,
        rate_tables = nothing, cool_tables = nothing, dtfrac::Real = 0.1,
        itcap::Int = ChemistryKernels._SUB_ITMAX, workgroup_size::Int = 0,
        precision::Type = Float32, tol::Real = 0.02, mapback::Symbol = :linear,
        fd_step::Real = 0.05)
    n = length(rho)
    deut = deuterium && HDI !== nothing
    T = precision
    invtol = 1.0 / float(tol)
    B = 12                                    # bits per axis in the packed key

    # quantized log indices (Float64 math), offset per-axis to a 0-based local index
    qix(x) = round.(Int64, log10.(max.(Float64.(x), _TINY)) .* invtol)
    Mx = (Int64(1) << B) - 1
    loc(q) = clamp.(q .- minimum(q), Int64(0), Mx)
    kr = loc(qix(rho)); ke = loc(qix(e_int)); kh = loc(qix(HII ./ rho)); km = loc(qix(H2I ./ rho))
    key = kr .| (ke .<< B) .| (kh .<< 2B) .| (km .<< 3B)
    deut && (key = key .| (loc(qix(HDI ./ rho)) .<< 4B))

    perm = sortperm(key); sk = key[perm]
    bnd = CUDA.ones(Int32, n)
    n > 1 && (@views bnd[2:n] .= Int32.(sk[2:n] .!= sk[1:n-1]))
    binid_sorted = cumsum(bnd)
    nb = Int(CUDA.@allowscalar binid_sorted[n])
    starts = findall(!iszero, bnd)
    ends   = vcat(starts[2:nb] .- 1, CuArray([n]))
    cnt    = Float64.(ends .- starts .+ 1)

    # per-bin geometric-mean representative inputs (segmented Σlog via cumsum).  The few
    # representatives are solved in Float64 regardless of the cell precision T — the wide
    # CGS dynamic range underflows Float32 in the stiff rate products, and the reps are
    # cheap, so f64 buys accuracy for free.  Only the final scatter casts back to T.
    gmean(X) = begin
        sl = log.(max.(Float64.(X[perm]), _TINY)); pre = cumsum(sl)
        exp.((pre[ends] .- pre[starts] .+ sl[starts]) ./ cnt)
    end
    repR = gmean(rho); repE = gmean(e_int); repH = gmean(HII); repM = gmean(H2I)
    repD = deut ? gmean(HDI) : CUDA.zeros(Float64, nb)

    linear = mapback === :linear
    naxes  = deut ? 5 : 4
    reps = linear ? 1+naxes : 1
    aR = repeat(repR, reps); aE = repeat(repE, reps); aH = repeat(repH, reps)
    aM = repeat(repM, reps); aD = repeat(repD, reps)
    if linear                                 # block ax perturbs input axis ax by ×exp(h)
        eh = exp(float(fd_step))
        @views aR[nb+1:2nb]  .*= eh; @views aE[2nb+1:3nb] .*= eh; @views aH[3nb+1:4nb] .*= eh
        @views aM[4nb+1:5nb] .*= eh
        deut && (@views aD[5nb+1:6nb] .*= eh)
    end
    ChemistryKernels.solve_chem_device!(aR, aE, aH, aM, deut ? aD : nothing;
        a_value, dt, density_units, length_units, time_units, hubble, Om, OL, fh,
        deuterium = deut, hubble_expansion, adot_over_a, rate_tables, cool_tables,
        dtfrac, itcap, workgroup_size, backend = :cuda, precision = Float64)

    # per-cell bin id (original order) + gathered representative inputs/outputs
    bc = CUDA.zeros(Int32, n); bc[perm] = binid_sorted
    g(v) = v[bc]
    repRc=g(repR); repEc=g(repE); repHc=g(repH); repMc=g(repM); repDc = deut ? g(repD) : repRc
    oEc=g(@view aE[1:nb]); oHc=g(@view aH[1:nb]); oMc=g(@view aM[1:nb]); oDc = deut ? g(@view aD[1:nb]) : oHc

    # ln deviation of each cell's inputs from its bin representative (Float64 rep-side math)
    r64 = Float64.(rho); e64 = Float64.(e_int); h64 = Float64.(HII); m64 = Float64.(H2I)
    d64 = deut ? Float64.(HDI) : r64
    lR = log.(max.(r64,_TINY)./repRc); lE = log.(max.(e64,_TINY)./repEc)
    lH = log.(max.(h64,_TINY)./repHc); lM = log.(max.(m64,_TINY)./repMc)
    lD = deut ? log.(max.(d64,_TINY)./repDc) : lR

    # per-bin log-sensitivity s_{o,ax}=Δln(out_o)/h (block ax output − block 0 output)
    S(blk, ax) = g((log.(max.(blk[ax*nb+1:(ax+1)*nb],_TINY)) .-
                    log.(max.(blk[1:nb],_TINY))) ./ float(fd_step))
    acc(blk) = S(blk,1).*lR .+ S(blk,2).*lE .+ S(blk,3).*lH .+ S(blk,4).*lM .+ (deut ? S(blk,5).*lD : zero(lR))

    if mapback === :broadcast
        e_new = oEc .* (e64 ./ repEc); H_new = (oHc./repRc).*r64; M_new = (oMc./repRc).*r64
        D_new = deut ? (oDc./repRc).*r64 : d64
    elseif linear
        e_new = oEc .* exp.(acc(aE)); H_new = oHc .* exp.(acc(aH)); M_new = oMc .* exp.(acc(aM))
        D_new = deut ? oDc .* exp.(acc(aD)) : d64
    else  # :ratio — unit self-response
        e_new = oEc .* exp.(lE); H_new = oHc .* exp.(lH); M_new = oMc .* exp.(lM)
        D_new = deut ? oDc .* exp.(lD) : d64
    end

    cap = Float64(fh) .* r64
    M_new = clamp.(M_new, 0.0, cap)
    e_int .= T.(max.(e_new, _TINY)); HII .= T.(clamp.(H_new, 0.0, cap .- M_new)); H2I .= T.(M_new)
    deut && (HDI .= T.(clamp.(D_new, 0.0, cap)))
    return nb
end

end # module

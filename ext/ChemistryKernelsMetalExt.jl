"""
    ChemistryKernelsMetalExt

Package extension that lights up the Metal (Apple GPU) backend for
`ChemistryKernels`. Loaded automatically when `Metal` is present. Registers the
`:metal` backend and specialises the device-array helpers onto `MtlArray`.
Metal is Float32-only, so callers must request `Float32` element types.
"""
module ChemistryKernelsMetalExt

using ChemistryKernels
using Metal
import EmissionKernels
import ChemistryKernels: _default_rate_tables, _default_cool_tables

const _TABLE_LOCK = ReentrantLock()
const _RATE_TABLE = Ref{Any}(nothing)
const _COOL_TABLE = Ref{Any}(nothing)

function __init__()
    if Metal.functional()
        ChemistryKernels.register_backend!(:metal, Metal.MetalBackend())
    end
end

ChemistryKernels.device_zeros(::Metal.MetalBackend, ::Type{T}, dims::Dims) where {T} =
    Metal.zeros(T, dims)

# Apple GPU production kernels use persistent Float32 tables by default. Besides being
# substantially faster, this keeps the analytic fits out of the generated AIR program;
# recent AGX compilers can otherwise fail to legalize the very large inlined shader.
function _default_rate_tables(::Val{:metal}, ::Type{Float32})
    lock(_TABLE_LOCK) do
        _RATE_TABLE[] === nothing &&
            (_RATE_TABLE[] = ChemistryKernels.build_rate_tables(
                backend=:metal, precision=Float32))
        return _RATE_TABLE[]
    end
end

function _default_cool_tables(::Val{:metal}, ::Type{Float32})
    lock(_TABLE_LOCK) do
        _COOL_TABLE[] === nothing &&
            (_COOL_TABLE[] = EmissionKernels.build_cooling_tables(
                backend=:metal, precision=Float32))
        return _COOL_TABLE[]
    end
end

end # module

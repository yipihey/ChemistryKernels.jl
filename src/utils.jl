# utils.jl — host-side performance utilities.

export sort_cells_by_temperature

"""
    sort_cells_by_temperature(e_int::AbstractVector) -> Vector{Int}

Return a permutation that sorts cells by ascending specific internal energy
`e_int` (a proxy for temperature).  Dispatching `solve_chem!` on the permuted
arrays reduces warp divergence on the GPU: adjacent cells have similar
temperatures and hence similar sub-step counts, keeping all threads in a warp
active together.

```julia
perm = sort_cells_by_temperature(e_int)
solve_chem!(rho[perm], e_int[perm], HII[perm], H2I[perm]; …)
```

Apply the inverse permutation (`invperm(perm)`) after to restore the original
ordering, or write results back to the appropriate pre-allocated output arrays.
"""
sort_cells_by_temperature(e_int::AbstractVector) = sortperm(e_int)

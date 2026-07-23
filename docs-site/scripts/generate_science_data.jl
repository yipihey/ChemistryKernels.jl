using ChemistryKernels
using Printf

include(joinpath(@__DIR__, "..", "..", "test", "recomb_helpers.jl"))

loggrid(lo, hi, n) = 10.0 .^ range(log10(lo), log10(hi); length=n)

function js_string(s)
    return "\"" * replace(String(s), "\\" => "\\\\", "\"" => "\\\"") * "\""
end

num(x) = isfinite(x) ? @sprintf("%.9g", Float64(x)) : "null"
arr(xs) = "[" * join(num.(xs), ",") * "]"

function series(name, values; note="")
    return "{name:" * js_string(name) * ",values:" * arr(values) *
           (isempty(note) ? "" : ",note:" * js_string(note)) * "}"
end

function dataset(id, title, x_label, y_label, x, curves;
                 x_scale="log", y_scale="log", y_floor=nothing, note="")
    body = "{id:" * js_string(id) * ",title:" * js_string(title) *
           ",xLabel:" * js_string(x_label) * ",yLabel:" * js_string(y_label) *
           ",xScale:" * js_string(x_scale) * ",yScale:" * js_string(y_scale) *
           ",x:" * arr(x) * ",series:[" * join(curves, ",") * "]"
    y_floor !== nothing && (body *= ",yFloor:" * num(y_floor))
    !isempty(note) && (body *= ",note:" * js_string(note))
    return body * "}"
end

T = loggrid(1.0, 1.0e8, 181)

atomic_rates = [
    series("k1 · H coll. ion.", ChemistryKernels.k1.(T)),
    series("k2 · H recomb. fit", ChemistryKernels.k2.(T)),
    series("k3 · He I coll. ion.", ChemistryKernels.k3.(T)),
    series("k4 · He II recomb.", ChemistryKernels.k4.(T)),
    series("k5 · He II coll. ion.", ChemistryKernels.k5.(T)),
    series("k6 · He III recomb.", ChemistryKernels.k6.(T)),
    series("k57 · H–H ion.", ChemistryKernels.k57.(T)),
    series("k58 · H–He ion.", ChemistryKernels.k58.(T)),
]

molecular_fns = [
    ("k7 · H⁻ radiative attach.", k7), ("k8 · associative detach.", k8),
    ("k9 · H₂⁺ radiative assoc.", k9), ("k10 · H₂⁺ charge transfer", k10),
    ("k11 · H₂ charge exchange", k11), ("k12 · H₂ e⁻ dissoc.", k12),
    ("k13 · H₂–H dissoc.", k13), ("k14 · H⁻ e⁻ detach.", k14),
    ("k15 · H⁻–H detach.", k15), ("k16 · mutual neutral.", k16),
    ("k17 · H⁻ + H → H₂", k17), ("k18 · H₂ dissoc. recomb.", k18),
    ("k19 · H₂ + H⁻", k19), ("k22 · three-body H₂", k22),
    ("HeH⁺ radiative assoc.", kHeH_ra_spont),
    ("HeH⁺ + H → H₂⁺", kHeH_H), ("HeH⁺ + e⁻", kHeH_e),
]
molecular_rates = [series(n, f.(T)) for (n, f) in molecular_fns]

deuterium_fns = [("k50", k50), ("k51", k51), ("k52", k52), ("k53", k53),
                  ("k54", k54), ("k55", k55), ("k56", k56)]
deuterium_rates = [series(n, f.(T)) for (n, f) in deuterium_fns]

atomic_cooling_fns = [
    ("H I excitation", ceHI), ("He I excitation", ceHeI), ("He II excitation", ceHeII),
    ("H I ionization", ciHI), ("He I ionization", ciHeI), ("He II ionization", ciHeII),
    ("He I metastable ion.", ciHeIS), ("H II recombination", reHII),
    ("He II recomb. (rad.)", reHeII1), ("He II recomb. (diel.)", reHeII2),
    ("He III recombination", reHeIII), ("Bremsstrahlung", brem),
]
atomic_cooling = [series(n, f.(T)) for (n, f) in atomic_cooling_fns]

Tm = loggrid(10.0, 3.0e4, 161)
molecular_cooling_fns = [
    ("H₂–H", GAHI), ("H₂–H₂", GAH2), ("H₂–He", GAHe),
    ("H₂–H⁺", GAHp), ("H₂–e⁻", GAel), ("H₂ LTE", H2LTE),
    ("HD low-density", HDlow), ("HD LTE", HDlte),
]
molecular_cooling = [series(n, f.(Tm)) for (n, f) in molecular_cooling_fns]

z_cmb = collect(range(0.0, 8000.0; length=181))
Trad = 2.725 .* (1 .+ z_cmb)
cmb_rates = [
    series("k27 · H⁻ photodetach.", ChemistryKernels.k27_cmb.(Trad)),
    series("k28 · H₂⁺ photodissoc.", ChemistryKernels.k28_cmb.(Trad)),
    series("HeH⁺ photodissoc.", ChemistryKernels.gamma_HeH_cmb.(Trad)),
    series("β₁s · H photoion.", beta1s_freq.(Trad)),
]

uvb = fg20_uvb()
z_uvb = collect(range(0.0, 15.0; length=181))
uvb_rows = uvb_rates.(Ref(uvb), z_uvb)
uvb_ion = [
    series("Γ H I", getindex.(uvb_rows, 1)),
    series("Γ He I", getindex.(uvb_rows, 3)),
    series("Γ He II", getindex.(uvb_rows, 2)),
]
uvb_heat = [
    series("q̇ H I", getindex.(uvb_rows, 4)),
    series("q̇ He I", getindex.(uvb_rows, 5)),
    series("q̇ He II", getindex.(uvb_rows, 6)),
]

# Dust functions at a documented fiducial state; the column-dependent shielding
# curves are kept separate from the temperature-dependent grain microphysics.
Td = T_dust_eq.(ones(length(Tm)), zeros(length(Tm)), fill(2.725, length(Tm)))
dust_temp = [
    series("H₂ formation · Z⊙", k_H2_dust.(Tm, Td, ones(length(Tm)))),
    series("grain H II recomb. · nₑ=1", k_gr_recomb_HII.(Tm, ones(length(Tm)), ones(length(Tm)), ones(length(Tm)))),
    series("photoelectric heating / H", Gamma_PE.(Tm, ones(length(Tm)), ones(length(Tm)), ones(length(Tm)))),
]
Ncol = loggrid(1.0e12, 1.0e24, 161)
dust_shield = [
    series("H₂ self-shielding", f_shield_H2.(Ncol)),
    series("dust LW · Z⊙", f_dust_LW.(Ncol, ones(length(Ncol)))),
]

# Metal cooling is shown per element at solar abundance, holding the gas state fixed.
# This exposes the independent alpha/Fe behavior without hiding it in a summed curve.
Tmet = loggrid(10.0, 2.0e4, 161)
nH = 1.0; nHI = 0.99; nHII = 0.01; ne = 0.01; nH2 = 1.0e-4; zmet = 0.0
solar = metal_abund(solar=1.0)
element_abund = [
    ("Carbon", MetalAbundances(solar.C, 0.0, 0.0, 0.0)),
    ("Oxygen", MetalAbundances(0.0, solar.O, 0.0, 0.0)),
    ("Silicon", MetalAbundances(0.0, 0.0, solar.Si, 0.0)),
    ("Iron", MetalAbundances(0.0, 0.0, 0.0, solar.Fe)),
]
metal_curves = [series(name, [metal_cooling_rate(t, zmet, nHI, nHII, ne, nH2, nH, ab) for t in Tmet])
                for (name, ab) in element_abund]

# Recombination comparison uses the exact fixture and production solver exercised by
# test/test_recombination_mixing.jl. CAMB/RECFAST-v2 is also the HyRec comparison proxy
# used by the package's validation ladder.
fixture = joinpath(@__DIR__, "..", "..", "test", "fixtures", "recfast_v2_xe.csv")
rows = [parse.(Float64, split(ln, ",")) for ln in readlines(fixture)
        if !isempty(ln) && !startswith(ln, "#")]
zr = [r[1] for r in rows]; xer = [r[2] for r in rows]
lerp(zq, zs, xs) = begin
    i = searchsortedfirst(zs, zq)
    i <= 1 && return xs[1]
    i > length(zs) && return xs[end]
    t = (zq-zs[i-1])/(zs[i]-zs[i-1])
    xs[i-1]*(1-t) + xs[i]*t
end
z0 = 1200.0; xe0 = lerp(z0, zr, xer)
Tb0 = begin
    tbr = [r[3] for r in rows]
    lerp(z0, zr, tbr)
end
zv1, xev1, _ = integrate_onezone(z_start=z0, z_end=700.0, n_steps=600,
                                  x_e_init=xe0, T_init=Tb0)
zv2, xev2, _ = integrate_onezone(z_start=z0, z_end=700.0, n_steps=600,
                                  x_e_init=xe0, T_init=Tb0, recfast_hswitch=true)
zrec = collect(range(700.0, 1200.0; length=121))
rec_curves = [
    series("CAMB / RECFAST-v2 fixture", [lerp(z, zr, xer) for z in zrec]),
    series("ChemistryKernels · Peebles", [lerp(z, reverse(zv1), reverse(xev1)) for z in zrec]),
    series("ChemistryKernels · RECFAST-v2", [lerp(z, reverse(zv2), reverse(xev2)) for z in zrec]),
]

# High-z hydrogen+Saha-helium reconstruction compared against the same fixture.
zh, xh, _ = integrate_onezone(z_start=5000.0, z_end=2700.0, n_steps=450,
                               x_e_init=1.0, T_init=2.725*5001.0, recfast_hswitch=true)
zhigh = collect(range(2700.0, 5000.0; length=111))
xe_total = Float64[]
for z in zhigh
    xH = lerp(z, reverse(zh), reverse(xh))
    push!(xe_total, total_electron_fraction(xH, n_H_at_z(z), 2.725*(1+z)))
end
highz_curves = [
    series("CAMB / RECFAST-v2 fixture", [lerp(z, zr, xer) for z in zhigh]),
    series("ChemistryKernels · H + Saha He", xe_total),
]

datasets = [
    dataset("atomic-rates", "Atomic reaction rates", "gas temperature · K", "rate coefficient · cm³ s⁻¹", T, atomic_rates,
            y_floor=1.0e-30),
    dataset("molecular-rates", "H₂ / H⁻ / HeH⁺ reaction rates", "gas temperature · K", "rate coefficient · cm³ s⁻¹", T, molecular_rates,
            y_floor=1.0e-30,
            note="The plotted HeH⁺ association curve is spontaneous association; the solver also includes the CMB-stimulated term. k22 is a three-body coefficient (cm⁶ s⁻¹)."),
    dataset("deuterium-rates", "Deuterium reaction rates", "gas temperature · K", "rate coefficient · cm³ s⁻¹", T, deuterium_rates,
            y_floor=1.0e-30),
    dataset("atomic-cooling", "Atomic cooling coefficients", "gas temperature · K", "cooling coefficient · erg cm³ s⁻¹", T, atomic_cooling),
    dataset("molecular-cooling", "H₂ and HD cooling functions", "gas temperature · K", "coefficient / LTE rate", Tm, molecular_cooling,
            note="Low-density coefficients and per-molecule LTE rates share the panel; use the assembler for density-weighted cooling."),
    dataset("cmb-rates", "CMB photoprocesses", "redshift z", "rate · s⁻¹", z_cmb, cmb_rates,
            x_scale="linear", y_floor=1.0e-30),
    dataset("uvb-ion", "FG20 photoionisation", "redshift z", "rate · s⁻¹", z_uvb, uvb_ion,
            x_scale="linear", y_floor=1.0e-30),
    dataset("uvb-heat", "FG20 photoheating", "redshift z", "heating / absorber · erg s⁻¹", z_uvb, uvb_heat, x_scale="linear"),
    dataset("dust-temp", "Dust microphysics · fiducial field", "gas temperature · K", "rate / heating coefficient", Tm, dust_temp,
            note="G₀=1, Z=Z⊙, Aᵥ=0, nₑ=1 cm⁻³; Tdust is solved locally including the z=0 CMB floor."),
    dataset("dust-shield", "Lyman–Werner shielding", "column density · cm⁻²", "transmission", Ncol, dust_shield),
    dataset("metal-cooling", "Independent metal cooling", "gas temperature · K", "ΛX · erg cm⁻³ s⁻¹", Tmet, metal_curves,
            note="Solar element abundance, nH=1 cm⁻³, xHII=xe=0.01, xH₂=10⁻⁴, z=0; each element is isolated."),
    dataset("recombination", "Hydrogen recombination comparison", "redshift z", "xₑ = nₑ / nH", zrec, rec_curves, x_scale="linear"),
    dataset("highz", "High-redshift helium-aware comparison", "redshift z", "total xₑ", zhigh, highz_curves, x_scale="linear"),
]

out = joinpath(@__DIR__, "..", "app", "science-data.ts")
open(out, "w") do io
    println(io, "// Generated from the package's Julia formulas by scripts/generate_science_data.jl")
    println(io, "export type ScienceSeries = { name: string; values: (number | null)[]; note?: string };")
    println(io, "export type ScienceDataset = { id: string; title: string; xLabel: string; yLabel: string; xScale: string; yScale: string; x: number[]; series: ScienceSeries[]; yFloor?: number; note?: string };")
    println(io, "export const scienceDatasets: ScienceDataset[] = [")
    println(io, join(datasets, ",\n"))
    println(io, "];")
end

println(out)

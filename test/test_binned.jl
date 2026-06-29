# test_binned.jl — phase-space-binned solve vs the per-cell reference.
using ChemistryKernels, Test, Random, Statistics

# Build a synthetic cosmological cell distribution at redshift z, in CGS (so the
# solve_chem! unit factors are all 1).  Tight at high z, broader at low z.
function _synth(z; N=20_000, seed=7)
    rng = MersenneTwister(seed)
    MH=1.6726e-24; KB=1.380649e-16; GAMMA=5/3; FH=0.76
    rhobar = 0.046*1.8788e-29*0.71^2*(1+z)^3
    Tcmb = 2.725*(1+z); Tbar = z>150 ? Tcmb : Tcmb*(151/(1+z))
    sigd = clamp(0.02*(201/(1+z)), 1e-4, 1.0)
    lock = clamp(150/(1+z), 0.0, 1.0)
    xHIIbar = 0.5*(1-tanh((1100-(1+z))/80))*(1-2e-4)+2e-4
    mu=1.22
    rho=Vector{Float64}(undef,N); e=similar(rho); HII=similar(rho); H2I=similar(rho); HDI=similar(rho)
    for i in 1:N
        r = exp(sigd*randn(rng)-0.5*sigd^2); ρ=rhobar*r
        T = Tbar*r^((GAMMA-1)*lock)
        rho[i]=ρ; e[i]=KB*T/((GAMMA-1)*mu*MH)
        HII[i]=clamp(xHIIbar*r^(-0.1*lock),1e-8,1.0)*ρ
        H2I[i]=2e-6*ρ; HDI[i]=3.4e-5*2e-6*ρ
    end
    (; rho,e,HII,H2I,HDI)
end

_units = (; density_units=1.0, length_units=1.0, time_units=1.0)
_cosmo(z) = (; hubble=71.0, Om=0.27, OL=0.73, fh=0.76, deuterium=true,
               hubble_expansion=true, adot_over_a=(71.0e5/3.0857e24)*sqrt(0.27*(1+z)^3+0.73))
_dt(z) = 0.05/((71.0e5/3.0857e24)*sqrt(0.27*(1+z)^3+0.73))

@testset "solve_chem_binned!" begin
    @testset "tol→0 reproduces solve_chem!" begin
        z = 100.0; s = _synth(z; N=4000)
        # reference
        eR=copy(s.e); hR=copy(s.HII); mR=copy(s.H2I); dR=copy(s.HDI)
        solve_chem!(s.rho, eR, hR, mR, dR; a_value=1/(1+z), dt=_dt(z), _units..., _cosmo(z)...)
        # binned with a vanishing tolerance ⇒ every cell its own bin
        eB=copy(s.e); hB=copy(s.HII); mB=copy(s.H2I); dB=copy(s.HDI)
        nb = solve_chem_binned!(s.rho, eB, hB, mB, dB; a_value=1/(1+z), dt=_dt(z),
                                _units..., _cosmo(z)..., tol=1e-9)
        @test nb == length(s.rho)                          # all distinct bins
        @test maximum(abs.(eB.-eR)./abs.(eR))   < 1e-6
        @test maximum(abs.(hB.-hR)./abs.(hR))   < 1e-6
        @test maximum(abs.(mB.-mR)./abs.(mR))   < 1e-6
    end

    @testset "moderate tol: few bins, mean + variance preserved (z=30)" begin
        z = 30.0; s = _synth(z; N=20_000)
        eR=copy(s.e); hR=copy(s.HII); mR=copy(s.H2I); dR=copy(s.HDI)
        solve_chem!(s.rho, eR, hR, mR, dR; a_value=1/(1+z), dt=_dt(z), _units..., _cosmo(z)...)
        eB=copy(s.e); hB=copy(s.HII); mB=copy(s.H2I); dB=copy(s.HDI)
        nb = solve_chem_binned!(s.rho, eB, hB, mB, dB; a_value=1/(1+z), dt=_dt(z),
                                _units..., _cosmo(z)..., tol=0.05)
        @test nb < length(s.rho) ÷ 100                      # large reduction
        # fractions (the P(k)-relevant fields)
        xR = hR ./ s.rho; xB = hB ./ s.rho
        @test abs(mean(xB)/mean(xR) - 1) < 0.03             # mean bias
        @test 0.90 < std(xB)/std(xR) < 1.10                 # variance preserved
        @test abs(mean(eB)/mean(eR) - 1) < 0.03             # energy mean
    end

    @testset "first-order recovers variance at recombination (z=1000)" begin
        # z=1000 with coarse bins is the variance-sensitive regime: density-driven
        # recombination gives output scatter that a zeroth-order (:ratio) map flattens.
        z = 1000.0; s = _synth(z; N=20_000)
        xR = (begin h=copy(s.HII); solve_chem!(s.rho, copy(s.e), h, copy(s.H2I), copy(s.HDI);
                a_value=1/(1+z), dt=_dt(z), _units..., _cosmo(z)...); h end) ./ s.rho
        hL=copy(s.HII); solve_chem_binned!(s.rho, copy(s.e), hL, copy(s.H2I), copy(s.HDI);
                a_value=1/(1+z), dt=_dt(z), _units..., _cosmo(z)..., tol=0.05, mapback=:linear)
        hR=copy(s.HII); solve_chem_binned!(s.rho, copy(s.e), hR, copy(s.H2I), copy(s.HDI);
                a_value=1/(1+z), dt=_dt(z), _units..., _cosmo(z)..., tol=0.05, mapback=:ratio)
        sL = std(hL./s.rho)/std(xR); sR = std(hR./s.rho)/std(xR)
        @test abs(sL-1) < abs(sR-1)        # first-order closer to true variance than zeroth
        @test sL > 0.5                     # and recovers a substantial fraction of it
    end
end

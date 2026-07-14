# test_fast.jl — the reduced fast analytic H+H2 mode vs the full network.
using ChemistryKernels, Test, Printf

let CK = ChemistryKernels,
    MH=1.6726e-24, KB=1.380649e-16, FH=0.76, GAMMA=5/3,
    H0=71.0e5/3.0857e24, OMB=0.0456, OMM=0.27, OML=0.73, HH=0.71

    nHbar(z)=FH*OMB*1.8788e-29*HH^2*(1+z)^3/MH
    tcos(z)=(2/(3*H0*sqrt(OML)))*asinh(sqrt(OML/OMM)*(1+z)^(-1.5))
    gT(nH,nHII,nH2,e)=CK.gas_temperature(nH*MH/FH,e,max(nH-nHII-2nH2,1e-30),nHII,(1-FH)/(4FH)*nH,
                                          0.0,0.0,nHII,0.0,nH2,0.0)
    fast(nHII,nH2,e,nH,z,dt)=(r=CK.evolve_cell_fast(nH*MH/FH,e,nHII*MH,2nH2*MH,dt,z;fh=FH); (r[2]/MH,r[3]/(2MH),r[1]))
    anal(nHII,nH2,e,nH,z,dt)=(r=CK.evolve_cell_analytic(nH*MH/FH,e,nHII*MH,2nH2*MH,dt,z;fh=FH); (r[2]/MH,r[3]/(2MH),r[1]))
    full(nHII,nH2,e,nH,z,dt)=(r=CK.evolve_cell(nH*MH/FH,e,nHII*MH,2nH2*MH,0.0,dt,z;fh=FH,deuterium=false); (r[2]/MH,r[3]/(2MH),r[1]))
    igm(solver)=(N=120; zs=exp.(range(log(1200.0),log(20.0),length=N)); e0=KB*CK.comp2_cmb(zs[1])/((GAMMA-1)*1.22*MH);
        s=(nHII=0.9*nHbar(zs[1]),nH2=1e-8*nHbar(zs[1]),e=e0);
        for i in 2:N; z=zs[i];nH=nHbar(z);dt=tcos(z)-tcos(zs[i-1]);sc=nH/nHbar(zs[i-1]);ad=sc^(GAMMA-1);
            a=solver(s.nHII*sc,s.nH2*sc,s.e*ad,nH,z,dt); s=(nHII=a[1],nH2=a[2],e=a[3]); end; s)
    halo(solver)=(zc=25.0;N=250;ns=exp.(range(log(nHbar(zc)),log(1e4),length=N));
        e0=KB*max(CK.comp2_cmb(zc)*0.5,60.0)/((GAMMA-1)*1.22*MH); s=(nHII=2e-4*ns[1],nH2=2e-6*ns[1],e=e0);
        for i in 2:N; nH=ns[i];sc=nH/ns[i-1];dt=0.3/sqrt(6.67e-8*nH*MH/FH);ad=sc^(GAMMA-1);
            a=solver(s.nHII*sc,s.nH2*sc,s.e*ad,nH,zc,dt); s=(nHII=a[1],nH2=a[2],e=a[3]); end; s)

    @testset "evolve_cell_fast" begin
        @testset "IGM history z=1000→20 tracks the full network" begin
            N=80; zs=exp.(range(log(1000.0),log(20.0),length=N))
            e0=KB*CK.comp2_cmb(zs[1])/((GAMMA-1)*1.22*MH)
            sF=(nHII=0.05*nHbar(zs[1]),nH2=1e-8*nHbar(zs[1]),e=e0); sR=sF
            for i in 2:N
                z=zs[i]; nH=nHbar(z); dt=tcos(z)-tcos(zs[i-1]); sc=nH/nHbar(zs[i-1]); ad=sc^(GAMMA-1)
                a=fast(sF.nHII*sc,sF.nH2*sc,sF.e*ad,nH,z,dt); b=full(sR.nHII*sc,sR.nH2*sc,sR.e*ad,nH,z,dt)
                sF=(nHII=a[1],nH2=a[2],e=a[3]); sR=(nHII=b[1],nH2=b[2],e=b[3])
            end
            z=20.0; nH=nHbar(z)
            @test isapprox(sF.nHII/nH, sR.nHII/nH; rtol=0.05)   # x_e within 5%
            @test isapprox(sF.e, sR.e; rtol=0.05)               # energy within 5%
            @test 0.3*(sR.nH2/nH) < sF.nH2/nH < 3*(sR.nH2/nH)   # f_H2 order-unity
        end

        @testset "minihalo collapse: electrons recombine + H2 cooling" begin
            zc=25.0; N=250; ns=exp.(range(log(nHbar(zc)),log(1e4),length=N))
            e0=KB*max(CK.comp2_cmb(zc)*0.5,60.0)/((GAMMA-1)*1.22*MH)
            sF=(nHII=2e-4*ns[1],nH2=2e-6*ns[1],e=e0); sR=sF
            for i in 2:N
                nH=ns[i]; sc=nH/ns[i-1]; dt=0.3/sqrt(6.67e-8*nH*MH/FH); ad=sc^(GAMMA-1)
                a=fast(sF.nHII*sc,sF.nH2*sc,sF.e*ad,nH,zc,dt); b=full(sR.nHII*sc,sR.nH2*sc,sR.e*ad,nH,zc,dt)
                sF=(nHII=a[1],nH2=a[2],e=a[3]); sR=(nHII=b[1],nH2=b[2],e=b[3])
            end
            nH=1e4
            # relic x_e (2e-4 IC) RECOMBINES to ~7e-8 during collapse (case-B, no spurious
            # collisional floor); H⁻ channel self-halts ⇒ f_H2 saturates ~4-5e-4.  fast≈full.
            @test isapprox(sF.nHII/nH, sR.nHII/nH; rtol=0.05)   # recombined x_e (~7e-8), fast≈full
            @test isapprox(sF.nH2/nH,  sR.nH2/nH;  rtol=0.25)   # halo f_H2 ~4e-4, fast≈full ±25%
            @test sF.nH2/nH < 3e-3                              # SELF-HALTED (not the old 0.1 over-form)
            @test 60 < gT(nH,sF.nHII,sF.nH2,sF.e) < 200         # cooled to the H2 floor
        end

        @testset "recombination-only cell recombines away (no spurious collisional floor)" begin
            nH=1e4; T=100.0; e=KB*T/((GAMMA-1)*1.22*MH); zc=25.0
            r=CK.evolve_cell_fast(nH*MH/FH, e, 4e-5*nH*MH, 2e-12*nH*MH, 1e16, zc; fh=FH)
            # no ionizing source ⇒ x_e recombines FAR below the 4e-5 IC (case-B).  The former
            # k57/k58 1e-20 floor wrongly FROZE it at ~3.6e-5 (a fake collisional-ionisation
            # source at T where it should be ~0) → over-formed H2 to f_H2~0.1 in long collapses.
            @test 0.0 < r[2]/(nH*MH) < 1.0e-6                   # recombined away, not frozen; no NaN
        end
    end

    @testset "evolve_cell_analytic (closed-form, no BDF)" begin
        @testset "IGM recombination freeze-out matches the full network" begin
            a=igm(anal); b=igm(full); nH=nHbar(20.0)
            @test isapprox(a.nHII/nH, b.nHII/nH; rtol=0.05)     # relic x_e freeze to 5%
            @test isapprox(a.e, b.e; rtol=0.05)                 # thermal history to 5%
            @test 0.3*(b.nH2/nH) < a.nH2/nH < 3*(b.nH2/nH)      # f_H2 order-unity
        end
        @testset "minihalo collapse matches the full network" begin
            a=halo(anal); b=halo(full); nH=1e4
            @test isapprox(a.nHII/nH, b.nHII/nH; rtol=0.05)     # recombined x_e (~7e-8), anal≈full
            @test isapprox(a.nH2/nH,  b.nH2/nH;  rtol=0.25)     # halo f_H2 ~5e-4, anal≈full ±25%
            @test a.nH2/nH < 3e-3                               # H⁻ self-halts (no over-formation)
            @test 60 < gT(nH,a.nHII,a.nH2,a.e) < 200            # cooled to the H2 floor
        end

        @testset "H recombination history z=1500→900 agrees with HyRec/RECFAST-v2" begin
            # End-to-end H-recombination gate: integrate evolve_cell_analytic through the
            # z≈1100 recombination epoch and check x_e against the CAMB/RECFAST-v2 fixture
            # (calibrated to HyRec-2 to <0.2%).  The reduced rates (peebles_k2_mixing with
            # the fudge=1.125 α_B C-factor + the Hswitch Gaussian on the Lyα K-factor +
            # the closed H₂⁺ cycle) reproduce HyRec to <0.1% under a dedicated integrator
            # (see recfast_v2_comparison); here the WHOLE analytic path — Riccati x_HII,
            # analytic Compton, He fold — must reach the same history to <1% at a one-zone
            # resolution representative of an IC-generation run.  Guards k2/fudge/gauss and
            # the linear-source Riccati against any future shift of the recombination front.
            #
            # Fixture cosmology (test/fixtures/gen_recfast_v2.py): H0=71, Ωb=0.044, OL=0.73,
            # T_CMB=2.725; x_e is H-only (He neutral for z<1500).  Seed from the fixture at
            # z=1500 and march end-of-step to z=900 (comoving mass densities ∝(1+z)³).
            fixture = joinpath(@__DIR__, "fixtures", "recfast_v2_xe.csv")
            rows = [parse.(Float64, split(ln, ",")) for ln in readlines(fixture)
                    if !isempty(ln) && !startswith(ln, "#")]
            zref = [r[1] for r in rows]; xref = [r[2] for r in rows]; Tref = [r[3] for r in rows]
            lerp(zq, ys) = begin
                i = searchsortedfirst(zref, zq)
                i <= 1 ? ys[1] : i > length(zref) ? ys[end] :
                    ys[i-1] + (ys[i]-ys[i-1])*(zq-zref[i-1])/(zref[i]-zref[i-1])
            end
            nH_of(z) = FH*0.044*9.47e-30*(1+z)^3/MH      # 9.47e-30 = 1.8788e-29·0.71² [g/cm³]
            Hz(z)    = CK.hubble_z_of(z; hubble=71.0, Om=0.27, OL=0.73)

            zstart, zstop, N = 1500.0, 900.0, 800
            lz = range(log(1+zstart), log(1+zstop); length=N+1)
            xe0, Tg0 = lerp(zstart, xref), lerp(zstart, Tref)
            nH  = nH_of(zstart); rho = nH*MH/FH
            e   = (rho/MH)*(FH*(1+xe0)+(1-FH)/4)*KB*Tg0/(rho*(GAMMA-1))
            HIIm, H2m = xe0*nH*MH, 1e-40
            hist = Tuple{Float64,Float64}[]
            for k in 1:N
                zhi = exp(lz[k])-1; zlo = exp(lz[k+1])-1; zm = 0.5*(zhi+zlo)
                Hzm = Hz(zm); dt = (zhi-zlo)/((1+zm)*Hzm)
                nHlo = nH_of(zlo); sc = nHlo/(nH_of(zhi))   # comoving (1+z)³ dilution
                rho = nHlo*MH/FH; HIIm *= sc; H2m *= sc
                r = CK.evolve_cell_analytic(rho, e, HIIm, H2m, dt, zlo;
                        hubble=71.0, Om=0.27, OL=0.73, fh=FH,
                        hubble_expansion=true, adot_over_a=Hzm)
                e, HIIm, H2m = r[1], r[2], r[3]
                push!(hist, (zlo, (HIIm/MH)/nHlo))
            end
            zc = [z for (z,_) in hist]; xc = [x for (_,x) in hist]
            xe_at(z) = xc[argmin(abs.(zc .- z))]

            # Core recombination epoch (the x_e "knee") — agree with HyRec to <1%.
            for z in (1000.0, 1050.0, 1100.0)
                err = abs(xe_at(z) - lerp(z, xref)) / lerp(z, xref)
                @test err < 0.01
            end
            # Whole window (steep pre-knee to freeze-out shoulder) — <1.5%.
            for z in (900.0, 950.0, 1200.0, 1300.0)
                err = abs(xe_at(z) - lerp(z, xref)) / lerp(z, xref)
                @test err < 0.015
            end
            # Monotone recombination (x_e strictly falling through the epoch — no bounce
            # from the linear-source Riccati; the old frozen-source form oscillated here).
            @test all(diff(xc) .< 0)
        end
    end

    @testset "warm-gas H2 rates finite & positive (GPU fast-pow guard)" begin
        # k9/k11/k14/k15 are exp(poly in lnT); their lnT powers must be INTEGER LITERALS
        # so a negative lnT (T ≲ 1e4 K) stays exact.  A Float exponent NaNs on CUDA f32
        # (pow→exp(y·log(neg))).  Guard both precisions across the warm-gas H₂ regime.
        for P in (Float64, Float32), T in P.(200.0:200.0:10000.0)
            for kf in (CK.k9, CK.k11, CK.k14, CK.k15)
                v = kf(T)
                @test isfinite(v) && v ≥ zero(P)
            end
        end
    end

    @testset "batched solve_chem_analytic! (CPU KA) + rate-table parity" begin
        # a small physical single-z batch (z=30, δ=1…1e4, cold H₂-cooled halos)
        nb=nHbar(30.0); Tc=2.725*31.0; N=64
        δs=exp.(range(log(1.0),log(1e4),length=N))
        rho=[nb*δ*MH/FH for δ in δs]
        e=[KB*clamp(Tc*δ^(2/3)*0.5,Tc,1000.0)/((GAMMA-1)*1.22*MH) for δ in δs]
        HII=[(δ<1e2 ? 2e-4 : 3e-5)*nb*δ*MH for δ in δs]
        H2I=[2*(δ<1e2 ? 1e-6 : 5e-4)*nb*δ*MH for δ in δs]
        uu=(; a_value=1/31, dt=1e13, density_units=1.0, length_units=1.0, time_units=1.0, fh=FH)

        # batched == per-cell evolve_cell_analytic
        rb=copy(rho); eb=copy(e); hb=copy(HII); mb=copy(H2I)
        CK.solve_chem_analytic!(rb, eb, hb, mb; uu..., backend=:cpu, precision=Float64)
        @test all(isfinite, eb) && all(isfinite, hb) && all(isfinite, mb)
        i=48; r1=CK.evolve_cell_analytic(rho[i],e[i],HII[i],H2I[i],1e13,30.0; fh=FH)
        @test isapprox(eb[i], r1[1]; rtol=1e-10)
        @test isapprox(hb[i], r1[2]; rtol=1e-10)
        @test isapprox(mb[i], r1[3]; rtol=1e-10)

        # rate-table path tracks the analytic-fit path (log–log table ≲1% on the curved rates)
        rt=CK.build_rate_tables(; backend=:cpu, precision=Float64)
        rc=copy(rho); ec=copy(e); hc=copy(HII); mc=copy(H2I)
        CK.solve_chem_analytic!(rc, ec, hc, mc; uu..., rate_tables=rt, backend=:cpu, precision=Float64)
        @test isapprox(ec[i], eb[i]; rtol=0.02)
        @test isapprox(hc[i], hb[i]; rtol=0.05)
        @test isapprox(mc[i], mb[i]; rtol=0.05)

        # UInt16 log₂ species path tracks the f32 path (codec is ~0.12 %/ULP)
        ru=copy(rho); eu=copy(e)
        hu=CK.encode_log2sp_vec(HII ./ rho); mu=CK.encode_log2sp_vec(H2I ./ rho)
        CK.solve_chem_analytic_u16!(ru, eu, hu, mu; uu..., backend=:cpu, precision=Float32)
        @test all(isfinite, eu)
        hd=CK.decode_log2sp_vec(Float64, hu).*rho; md=CK.decode_log2sp_vec(Float64, mu).*rho
        # f32 reference for the same cells
        rf=copy(rho); ef=copy(e); hf=copy(HII); mf=copy(H2I)
        CK.solve_chem_analytic!(rf, ef, hf, mf; uu..., backend=:cpu, precision=Float32)
        @test isapprox(hd[i], hf[i]; rtol=0.03)             # x_HII fraction within codec ULP
        @test isapprox(md[i], mf[i]; rtol=0.03)             # f_H2 fraction within codec ULP
        @test all(isfinite, hd) && all(isfinite, md)
    end
end

@testset "fast analytic: negative rho/e domain guard (f16 hydro frontier)" begin
    # f16 dual-energy hydro can hand rho <= 0 / e <= 0 at extreme shock and void
    # frontiers (bamr256L6b: device DomainError at z~13, lmax=6 collapse).  The
    # entry guards must make these cells benign (~1 K near-vacuum), never throw.
    for (r, e) in ((-1.0e-24, 1.0e10), (1.0e-24, -1.0e10), (-1.0e-24, -1.0e10),
                   (0.0, 0.0), (1.0e-24, 1.0e10))
        for f in (CK.evolve_cell_analytic, CK.evolve_cell_fast)
            en, hii, h2, _ = f(Float32(r), Float32(e), Float32(1e-4 * abs(r)),
                               Float32(1e-6 * abs(r)), 1.0f13, 30.0f0; fh = 0.76)
            @test isfinite(en) && isfinite(hii) && isfinite(h2)
            @test en > 0
        end
    end
    # batched u16 device path (CPU backend), whole-vector pathological block
    n = 64
    rho = fill(1.0f-3, n); rho[1] = -1.0f-3; rho[2] = 0.0f0     # code units
    e   = fill(1.0f-7, n); e[3] = -1.0f-7; e[4] = 0.0f0
    hii = fill(CK.encode_log2sp(1.0f-4), n); h2 = fill(CK.encode_log2sp(1.0f-6), n)
    CK.solve_chem_analytic_u16!(rho, e, hii, h2;
        a_value = 1/31.0, dt = 1.0e-3, density_units = 1.0e-21,
        length_units = 3.0e22, time_units = 1.0e15, backend = :cpu,
        precision = Float32)
    @test all(isfinite, e)
end

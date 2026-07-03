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

        @testset "minihalo collapse: relic electron freeze + H2 cooling" begin
            zc=25.0; N=250; ns=exp.(range(log(nHbar(zc)),log(1e4),length=N))
            e0=KB*max(CK.comp2_cmb(zc)*0.5,60.0)/((GAMMA-1)*1.22*MH)
            sF=(nHII=2e-4*ns[1],nH2=2e-6*ns[1],e=e0); sR=sF
            for i in 2:N
                nH=ns[i]; sc=nH/ns[i-1]; dt=0.3/sqrt(6.67e-8*nH*MH/FH); ad=sc^(GAMMA-1)
                a=fast(sF.nHII*sc,sF.nH2*sc,sF.e*ad,nH,zc,dt); b=full(sR.nHII*sc,sR.nH2*sc,sR.e*ad,nH,zc,dt)
                sF=(nHII=a[1],nH2=a[2],e=a[3]); sR=(nHII=b[1],nH2=b[2],e=b[3])
            end
            nH=1e4
            @test isapprox(sF.nHII/nH, sR.nHII/nH; rtol=0.05)   # relic x_e freeze matches (~3.6e-5)
            @test isapprox(sF.nH2/nH,  sR.nH2/nH;  rtol=0.15)   # halo f_H2 within 15%
            @test 60 < gT(nH,sF.nHII,sF.nH2,sF.e) < 200         # cooled to the H2 floor
        end

        @testset "recombination-only cell freezes at the k57/k58 relic value" begin
            nH=1e4; T=100.0; e=KB*T/((GAMMA-1)*1.22*MH); zc=25.0
            r=CK.evolve_cell_fast(nH*MH/FH, e, 4e-5*nH*MH, 2e-12*nH*MH, 1e16, zc; fh=FH)
            @test 2e-5 < r[2]/(nH*MH) < 6e-5                    # frozen near relic, not → 0
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
            @test isapprox(a.nHII/nH, b.nHII/nH; rtol=0.05)     # relic x_e freeze
            @test isapprox(a.nH2/nH,  b.nH2/nH;  rtol=0.15)     # halo f_H2 within 15%
            @test 60 < gT(nH,a.nHII,a.nH2,a.e) < 200            # cooled to the H2 floor
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

"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { scienceDatasets, type ScienceDataset } from "./science-data";

const COLORS = [
  "#1757d7", "#d2452d", "#0a816c", "#7b4cc7", "#d39114", "#1683a3",
  "#a93680", "#557026", "#ef6a2c", "#334f8f", "#12939a", "#8b5b36",
  "#754668", "#4f7d62", "#a95f13", "#62666f",
];

function fmt(value: number) {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  const a = Math.abs(value);
  if (a >= 1e4 || a < 1e-2) return value.toExponential(2).replace("e+", "e");
  return value.toPrecision(3);
}

function ScienceChart({ data }: { data: ScienceDataset }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [hidden, setHidden] = useState<Set<number>>(new Set());
  const [hover, setHover] = useState<number | null>(null);
  const [width, setWidth] = useState(760);

  useEffect(() => {
    const node = wrapRef.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(320, entry.contentRect.width)));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const bounds = useMemo(() => {
    const xs = data.x.filter((v) => Number.isFinite(v) && (data.xScale !== "log" || v > 0));
    const ys: number[] = [];
    data.series.forEach((s, si) => {
      if (hidden.has(si)) return;
      s.values.forEach((v) => { if (v !== null && v > 0 && Number.isFinite(v)) ys.push(v); });
    });
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    let ymin = Math.min(...ys), ymax = Math.max(...ys);
    if (!Number.isFinite(ymin) || !Number.isFinite(ymax)) { ymin = 1e-30; ymax = 1; }
    if (data.yFloor !== undefined) {
      ymin = Math.max(ymin, data.yFloor);
      ymax = Math.max(ymax, data.yFloor * 10);
    }
    if (ymin === ymax) { ymin /= 10; ymax *= 10; }
    return { xmin, xmax, ymin, ymax };
  }, [data, hidden]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const cssHeight = width < 620 ? 350 : 430;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(cssHeight * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${cssHeight}px`;
    const c = canvas.getContext("2d");
    if (!c) return;
    c.scale(dpr, dpr);
    const m = { l: width < 520 ? 57 : 74, r: 18, t: 20, b: 54 };
    const w = width - m.l - m.r, h = cssHeight - m.t - m.b;
    const lx = (v: number) => data.xScale === "log" ? Math.log10(v) : v;
    const ly = (v: number) => Math.log10(v);
    const x0 = lx(bounds.xmin), x1 = lx(bounds.xmax);
    const y0 = ly(bounds.ymin), y1 = ly(bounds.ymax);
    const X = (v: number) => m.l + (lx(v) - x0) / (x1 - x0) * w;
    const Y = (v: number) => m.t + (y1 - ly(v)) / (y1 - y0) * h;

    c.fillStyle = "#fbfaf7";
    c.fillRect(0, 0, width, cssHeight);
    c.strokeStyle = "#d9d7d0";
    c.lineWidth = 1;
    c.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    c.fillStyle = "#6a6b68";
    for (let i = 0; i <= 5; i++) {
      const tx = x0 + (x1 - x0) * i / 5;
      const px = m.l + w * i / 5;
      c.beginPath(); c.moveTo(px, m.t); c.lineTo(px, m.t + h); c.stroke();
      const val = data.xScale === "log" ? 10 ** tx : tx;
      c.textAlign = i === 0 ? "left" : i === 5 ? "right" : "center";
      c.fillText(fmt(val), px, m.t + h + 20);
    }
    for (let i = 0; i <= 5; i++) {
      const ty = y0 + (y1 - y0) * i / 5;
      const py = m.t + h - h * i / 5;
      c.beginPath(); c.moveTo(m.l, py); c.lineTo(m.l + w, py); c.stroke();
      c.textAlign = "right";
      c.fillText(fmt(10 ** ty), m.l - 9, py + 4);
    }
    c.save();
    c.beginPath(); c.rect(m.l, m.t, w, h); c.clip();
    data.series.forEach((s, si) => {
      if (hidden.has(si)) return;
      c.strokeStyle = COLORS[si % COLORS.length];
      c.lineWidth = 2;
      c.beginPath();
      let started = false;
      s.values.forEach((v, i) => {
        const xv = data.x[i];
        if (v === null || v <= 0 || !Number.isFinite(v) || (data.xScale === "log" && xv <= 0)) {
          started = false; return;
        }
        const px = X(xv), py = Y(Math.max(v, bounds.ymin));
        if (!started) { c.moveTo(px, py); started = true; } else c.lineTo(px, py);
      });
      c.stroke();
    });
    if (hover !== null) {
      const px = X(data.x[hover]);
      c.strokeStyle = "rgba(22, 27, 34, .5)";
      c.setLineDash([4, 4]); c.beginPath(); c.moveTo(px, m.t); c.lineTo(px, m.t + h); c.stroke();
      c.setLineDash([]);
    }
    c.restore();
    c.fillStyle = "#343b42"; c.textAlign = "center"; c.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
    c.fillText(data.xLabel, m.l + w / 2, cssHeight - 10);
    c.save(); c.translate(13, m.t + h / 2); c.rotate(-Math.PI / 2); c.fillText(data.yLabel, 0, 0); c.restore();
  }, [data, hidden, hover, width, bounds]);

  function pointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const mleft = width < 520 ? 57 : 74;
    const usable = width - mleft - 18;
    const t = Math.max(0, Math.min(1, (e.clientX - rect.left - mleft) / usable));
    const target = data.xScale === "log"
      ? 10 ** (Math.log10(bounds.xmin) + t * (Math.log10(bounds.xmax) - Math.log10(bounds.xmin)))
      : bounds.xmin + t * (bounds.xmax - bounds.xmin);
    let best = 0, dist = Infinity;
    data.x.forEach((v, i) => { const d = Math.abs(v - target); if (d < dist) { dist = d; best = i; } });
    setHover(best);
  }

  return (
    <div className="plot-layout">
      <div className="plot-canvas-wrap" ref={wrapRef}>
        <canvas ref={canvasRef} onPointerMove={pointerMove} onPointerLeave={() => setHover(null)} aria-label={`${data.title} plot`} />
        {hover !== null && (
          <div className="plot-tooltip">
            <strong>{data.xLabel.split("·")[0].trim()} {fmt(data.x[hover])}</strong>
            {data.series.map((s, i) => hidden.has(i) ? null : (
              <span key={s.name}><i style={{ background: COLORS[i % COLORS.length] }} />{s.name}<b>{s.values[hover] === null ? "—" : fmt(s.values[hover] as number)}</b></span>
            ))}
          </div>
        )}
      </div>
      <div className="plot-legend" aria-label="Toggle chart series">
        {data.series.map((s, i) => (
          <button key={s.name} className={hidden.has(i) ? "is-hidden" : ""} onClick={() => {
            const next = new Set(hidden);
            if (next.has(i)) next.delete(i); else next.add(i);
            setHidden(next);
          }} title={s.note || `Toggle ${s.name}`}>
            <i style={{ background: COLORS[i % COLORS.length] }} />{s.name}
          </button>
        ))}
        <button className="legend-reset" onClick={() => setHidden(new Set())}>Show all</button>
      </div>
    </div>
  );
}

function CodeBlock({ children }: { children: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="code-block">
      <button onClick={() => { navigator.clipboard.writeText(children); setCopied(true); setTimeout(() => setCopied(false), 1200); }}>{copied ? "Copied" : "Copy"}</button>
      <pre><code>{children}</code></pre>
    </div>
  );
}

const quickStart = `using ChemistryKernels

rho = fill(1.0e-24, 1_000_000)  # total mass density [g cm⁻³]
e   = fill(1.0e12,  1_000_000)  # specific internal energy [erg g⁻¹]
HII = fill(1.2e-25, 1_000_000)  # species mass density
H2I = fill(1.0e-30, 1_000_000)  # 2 n(H₂) mH convention

solve_chem!(rho, e, HII, H2I;
    a_value=1.0, dt=3.0e13,
    density_units=1.0, length_units=1.0, time_units=1.0)`;

const gpuStart = `using ChemistryKernels, EmissionKernels, CUDA

rt = build_rate_tables(backend=:cuda, precision=Float32)
ct = EmissionKernels.build_cooling_tables(backend=:cuda, precision=Float32)

# Arrays already live on the GPU: no staging or host round-trip.
solve_chem_device!(d_rho, d_e, d_HII, d_H2I;
    a_value, dt, density_units, length_units, time_units,
    backend=:cuda, precision=Float32,
    rate_tables=rt, cool_tables=ct, dtfrac=0.2)`;

const mixingStart = `fa = FAlphaTable(z_nodes, f_alpha_nodes)

solve_chem_mixing!(rho, e, HII, H2I, rho_smoothed;
    a_value, dt, density_units, length_units, time_units,
    fa_table=fa, Xe_mean=x_e_bar,
    recfast_hswitch=true,
    smoothed_is_neutral=false)`;

const cAbiStart = `/* Illustrative stable ABI implemented by a thin Julia @ccallable wrapper. */
typedef struct {
    double density_units, length_units, time_units, a_value;
} ck_units;

int ck_solve_h_h2_f32(size_t n,
    const float *rho, float *e, float *HII, float *H2I,
    double dt, const ck_units *units);

/* Initialize the bundled Julia runtime once per process, then pass the host's
   contiguous field storage directly: the wrapper uses unsafe_wrap, without copying. */`;

const fortranStart = `use, intrinsic :: iso_c_binding
type, bind(C) :: ck_units
  real(c_double) :: density_units, length_units, time_units, a_value
end type
interface
  function ck_solve_h_h2_f32(n, rho, e, HII, H2I, dt, units) &
      bind(C, name="ck_solve_h_h2_f32") result(ierr)
    import :: c_size_t, c_float, c_double, c_int, ck_units
    integer(c_size_t), value :: n
    real(c_float), intent(in)    :: rho(*)
    real(c_float), intent(inout) :: e(*), HII(*), H2I(*)
    real(c_double), value :: dt
    type(ck_units), intent(in) :: units
    integer(c_int) :: ierr
  end function
end interface`;

export default function Home() {
  const [datasetId, setDatasetId] = useState("atomic-rates");
  const dataset = scienceDatasets.find((d) => d.id === datasetId) ?? scienceDatasets[0];

  return (
    <main>
      <header className="topbar">
        <a className="wordmark" href="#top"><span>CK</span>ChemistryKernels.jl</a>
        <nav aria-label="Primary navigation">
          <a href="#motivation">Motivation</a><a href="#physics">Physics</a><a href="#solver">Solver</a><a href="#rates">Rate atlas</a>
          <a href="#recombination">Recombination</a><a href="#usage">Usage</a><a href="#integration">C / Fortran</a><a href="#references">References</a>
        </nav>
        <a className="github-link" href="https://github.com/yipihey/ChemistryKernels.jl">Source ↗</a>
      </header>

      <section className="hero" id="top">
        <div className="eyebrow"><span>v0.1 · methods overview</span><span>Julia · CPU · CUDA · Metal</span></div>
        <div className="hero-grid">
          <div>
            <p className="kicker">More resolution for the same machine</p>
            <h1>From recombination<br />to the first stars.</h1>
            <p className="hero-lede">The Abel–Anninos primordial network, redesigned around a two-species 16-bit production state: about 16× less chemistry memory, dramatically wider redshift coverage, and CPU/GPU paths engineered for billions of cell updates per second.</p>
            <div className="hero-actions"><a className="primary" href="#usage">Run the first zone</a><a href="#rates">Explore every rate ↓</a></div>
          </div>
          <div className="hero-figure" aria-label="ChemistryKernels capability summary">
            <div className="orbit orbit-a" /><div className="orbit orbit-b" />
            <div className="core"><span>H</span><b>H₂</b><small>HD</small></div>
            <div className="callout callout-a"><b>≈16× smaller</b><span>chemistry-state memory</span></div>
            <div className="callout callout-b"><b>z = 0–8000</b><span>validated cosmology paths</span></div>
            <div className="callout callout-c"><b>≈4 Gcell s⁻¹</b><span>A6000 analytic reference</span></div>
          </div>
        </div>
        <div className="metric-strip">
          <div><strong>≈16×</strong><span>less chemistry memory<br />2×UInt16 vs 9×Float64</span></div>
          <div><strong>≈8×</strong><span>less total memory<br />ABN-style first-star runs</span></div>
          <div><strong>&lt;0.1%</strong><span>H recombination error<br />z = 700–1100</span></div>
          <div><strong>≈4 Gcell/s</strong><span>analytic reference<br />CPU and GPU kernels</span></div>
        </div>
      </section>

      <section className="section motivation-section" id="motivation">
        <div className="section-label">01 · Why rebuild the network?</div>
        <div className="split-heading">
          <h2>In a cosmological code, chemistry memory is resolution.</h2>
          <p>The original Abel et al. model—the lineage underlying Grackle’s primordial solver—carries nine species as 64-bit fields. ChemistryKernels’ minimal production path carries two evolved species as 16-bit logarithmic fields. That change is the central design motivation, not a secondary compression option.</p>
        </div>
        <div className="memory-compare" aria-label="Chemistry state memory comparison">
          <article className="legacy-state">
            <span className="overline">ABEL / GRACKLE LINEAGE</span>
            <h3>Nine advected species</h3>
            <div className="memory-fields" aria-hidden="true">{Array.from({length: 9}).map((_, i) => <i key={i} />)}</div>
            <p>64-bit abundance fields keep the complete primordial network in every hydrodynamic cell.</p>
          </article>
          <div className="memory-ratio"><strong>≈16×</strong><span>less chemistry-state<br />memory in production</span></div>
          <article className="compact-state">
            <span className="overline">CHEMISTRYKERNELS</span>
            <h3>Two evolved species</h3>
            <div className="memory-fields" aria-hidden="true"><i /><i /></div>
            <p>16-bit log-encoded H II and H₂; neutral H, electrons, and short-lived intermediates are reconstructed.</p>
          </article>
        </div>
        <div className="memory-outcome">
          <strong>≈8× lower total memory</strong>
          <p>For an Abel–Bryan–Norman-style first-star calculation, the compact chemistry state reduces the memory footprint of the complete simulation by roughly a factor of eight. On fixed hardware, that headroom can become more cells, deeper refinement, larger volumes, or more realizations.</p>
        </div>
        <div className="science-drivers">
          <article><span>FIRST-STAR COLLAPSE</span><h3>Abel, Bryan & Norman</h3><p>Resolving H₂-cooled minihalo gas across an enormous density range is memory hungry. Compact species fields return scarce memory to the mesh and refinement hierarchy.</p><a href="https://arxiv.org/abs/astro-ph/0112088">Science 295, 93 ↗</a></article>
          <article><span>PRIMORDIAL MAGNETIC FIELDS</span><h3>Jedamzik, Abel & collaborators</h3><p>Magnetically driven baryon clumping changes recombination and makes non-local Lyα escape relevant. Coverage to z=8000 connects the ionized initial state to the structured recombination era.</p><a href="https://arxiv.org/abs/2312.11448">JCAP 03, 012 ↗</a></article>
          <article><span>STREAMING VELOCITIES</span><h3>Tseliakhovich–Hirata to O’Leary–McQuinn</h3><p>Supersonic baryon–dark-matter motion alters gas accretion into the first minihalos. Chemistry valid from recombination through z≈15–200 supports the thermal closure such simulations require.</p><a href="https://arxiv.org/abs/1204.1344">ApJ 760, 4 ↗</a></article>
        </div>
        <p className="scope-note"><b>Scope.</b> ChemistryKernels evolves the chemical and thermal state used by these hydrodynamic and MHD experiments; the host code evolves gravity, magnetic fields, radiative transport, and baryon–dark-matter streaming dynamics.</p>
      </section>

      <section className="section intro" id="physics">
        <div className="section-label">02 · Physical model</div>
        <div className="split-heading">
          <h2>A reduced network that keeps the slow physics explicit.</h2>
          <p>The package follows the philosophy of Abel et al. and Anninos et al., but changes what must persist in memory: advect only the species whose slow timescales matter, close fast intermediates algebraically, and reconstruct conserved species inside each update. The minimal H + H₂ path therefore carries two fields rather than the original nine-species state vector.</p>
        </div>
        <div className="species-grid">
          <article><span className="tag blue">ADVECTED</span><h3>H II · H₂ · HD</h3><p>The base state evolves ionized hydrogen and molecular hydrogen; HD is opt-in. A dedicated early-Universe path can also advect He II through its freeze-out.</p></article>
          <article><span className="tag green">RECONSTRUCTED</span><h3>H I · e⁻</h3><p>Neutral hydrogen follows from H-nucleus conservation. Free electrons follow from charge conservation, including the helium stages active in the selected path.</p></article>
          <article><span className="tag ochre">EQUILIBRIUM</span><h3>H⁻ · H₂⁺ · D⁺</h3><p>Short-lived intermediates use algebraic quasi-equilibrium. Helium defaults to collisional-radiative/Saha equilibrium; He II may instead be integrated.</p></article>
        </div>
        <div className="network-card">
          <div className="network-copy">
            <span className="overline">State representation</span>
            <h3>Physical abundances inside.<br />Code units only at the boundary.</h3>
            <p>Each kernel converts mass-density fields to CGS number densities and carries dimensionless, order-unity abundances. This avoids Float32 underflow and decouples numerical chemistry from the host simulation’s unit system.</p>
            <dl><div><dt>H₂ field</dt><dd>2 n(H₂) m<sub>H</sub></dd></div><div><dt>HD field</dt><dd>3 n(HD) m<sub>H</sub></dd></div><div><dt>He field</dt><dd>4 n(He<sup>+</sup>) m<sub>H</sub></dd></div></dl>
          </div>
          <div className="network-diagram" aria-label="Reduced reaction network">
            <div className="node hi">H I</div><div className="arrow a1">⇄</div><div className="node hii">H II</div>
            <div className="branch b1">e⁻<br />↓ H⁻</div><div className="branch b2">H₂⁺<br />↓</div><div className="node h2">H₂</div>
            <div className="arrow a2">⇄ D exchange</div><div className="node hd">HD</div>
            <small>fast intermediates are solved algebraically</small>
          </div>
        </div>
        <div className="capability-table-wrap">
          <table className="capability-table"><thead><tr><th>Physics</th><th>Included treatment</th><th>Where it matters</th></tr></thead><tbody>
            <tr><td>Primordial atomic</td><td>H/He collisional ionization, recombination, excitation, bremsstrahlung</td><td>Ionized and atomic-cooling gas</td></tr>
            <tr><td>Molecules</td><td>H⁻ and H₂⁺ formation routes, three-body H₂, H₂ and HD cooling</td><td>Dark-age gas and first halos</td></tr>
            <tr><td>Radiation</td><td>CMB photoprocesses and Compton exchange; optional FG20 UV/X-ray background</td><td>z ≈ 0–8000</td></tr>
            <tr><td>Dust</td><td>Grain H₂, grain recombination, LW shielding, photoelectric heating, gas–grain exchange</td><td>Enriched, shielded gas</td></tr>
            <tr><td>Metals</td><td>C I/II, O I, Si I/II, Fe II fine-structure with independent Fe</td><td>T ≲ 10⁴ K; non-solar [α/Fe]</td></tr>
          </tbody></table>
        </div>
      </section>

      <section className="section solver-section" id="solver">
        <div className="section-label">03 · Time integration</div>
        <div className="split-heading"><h2>Stiff where it must be.<br />Closed-form where it can be.</h2><p>Two production solvers share the same rates and thermal physics. The full path is a robust backward-Euler network with adaptive per-cell subcycling. The analytic path replaces the stiffest high-redshift pieces with exact updates and sharply reduces iteration count and warp divergence.</p></div>
        <div className="solver-flow" aria-label="Solver sequence">
          <div><b>01</b><strong>Reconstruct</strong><span>T, H I, e⁻, fast species</span></div><i>→</i>
          <div><b>02</b><strong>Evaluate</strong><span>rates, cooling, radiation</span></div><i>→</i>
          <div><b>03</b><strong>Limit Δt</strong><span>≤10% in e⁻, H I, energy</span></div><i>→</i>
          <div><b>04</b><strong>Advance</strong><span>implicit energy + species</span></div><i>→</i>
          <div><b>05</b><strong>Conserve</strong><span>clamp and rebuild H I</span></div>
        </div>
        <div className="equation-grid">
          <article><span className="overline">Species sweep</span><div className="equation">nᵢⁿ⁺¹ = (nᵢⁿ + Cᵢ Δt) / (1 + Dᵢ Δt)</div><p>Each equation is written as a creation term Cᵢ and linear destruction frequency Dᵢ. The backward-Euler update remains stable when destruction is rapid; a single ordered sweep couples the reduced network.</p></article>
          <article><span className="overline">Step control</span><div className="equation">Δt<sub>chem</sub> ≤ f · X / |dX/dt|, &nbsp; f = 0.1</div><p>The default limits electron, neutral-H, and energy changes to ten percent. <code>dtfrac=0.2</code> trades roughly half the substeps for a documented ≲1% chemistry cost in hydro-coupled work.</p></article>
          <article><span className="overline">Compton split</span><div className="equation">de/dt = −K<sub>C</sub>(e − e<sub>CMB</sub>) + q̇<sub>rest</sub></div><p>When K<sub>C</sub>Δt &gt; 1, the stiff CMB exchange is integrated implicitly in the full solver. The analytic solver uses the exponential solution, so Compton locking does not dictate the step.</p></article>
        </div>
        <div className="fast-panel">
          <div className="fast-title"><span>FAST ANALYTIC PATH</span><h3>Closed forms inside a deliberately smaller physics envelope.</h3><p><code>evolve_cell_analytic</code> advances primordial H + H₂. It assumes cell density and external radiation are fixed during each chemistry call, reconstructs H I and electrons by conservation, and closes H⁻/H₂⁺ in instantaneous quasi-steady state.</p></div>
          <div className="fast-steps">
            <div><b>A</b><h4>Energy</h4><p>Compton exchange is an exact exponential. Non-Compton cooling is frozen over the substep and integrated as a source; only that source and the Compton transition limit the step.</p></div>
            <div><b>B</b><h4>Ionization</h4><p>An exact linear-source Riccati update balances case-B/Peebles recombination against CMB photoionization, electron impact, and H–H/H–He ionization.</p></div>
            <div><b>C</b><h4>Molecules</h4><p>H⁻ and H₂⁺ are algebraic. H₂ formation integrates H I depletion analytically, then a backward-Euler sink applies H/H⁺/e⁻ collisional dissociation.</p></div>
          </div>
          <div className="fast-foot"><span><b>Use it for</b> recombination-era volumes, primordial IGM, and the H₂-cooled minihalo envelope</span><span><b>Use the full path for</b> HD, dust, metals, UVB heating, and strongly non-equilibrium shocks or dense cores</span></div>
        </div>
        <div className="analytic-contract">
          <div className="analytic-assumptions"><span className="overline">MODEL ASSUMPTIONS</span><h3>What “analytic” does—and does not—mean</h3><ul><li>The rate coefficients are held at the substep state; the species updates themselves are closed form.</li><li>Charge neutrality supplies n<sub>e</sub>. Helium is neutral in cold collapse, stateless Saha in hot gas, or an explicitly carried He II abundance in the high-z path.</li><li>H⁻ and H₂⁺ must remain faster than the resolved H II/H₂ evolution. The path is not a non-equilibrium intermediate-species solver.</li><li><code>dtfrac</code> controls thermal/rate re-evaluation accuracy, not stability of the Riccati or Compton solutions.</li></ul><a href="https://github.com/yipihey/ChemistryKernels.jl/blob/main/src/fast.jl">Read the closed-form kernel ↗</a></div>
          <div className="analytic-table-wrap"><table className="analytic-table"><thead><tr><th>Path</th><th>Measured envelope</th><th>Interpretation</th></tr></thead><tbody>
            <tr><td><code>analytic!</code></td><td>Mean IGM, z=1200→20</td><td>At z=20: H II and energy within 5% of the full network; H₂ within a factor of three.</td></tr>
            <tr><td><code>analytic!</code></td><td>z=25 collapse, n<sub>H</sub> to 10⁴ cm⁻³</td><td>H II within 5%, H₂ within 25%, and a 60–200 K terminal-temperature gate.</td></tr>
            <tr><td><code>analytic!</code> + RECFAST-v2</td><td>z=900–1300</td><td>&lt;1% through the z=1000–1100 knee and &lt;1.5% across the tested wider window.</td></tr>
            <tr><td>carried He II</td><td>z=1900–8000</td><td>Saha at fully ionized epochs; HyRec-derived He I freeze-out below z≈4500, with the stated comparison gates in §05.</td></tr>
            <tr><td><code>analytic_mixing!</code></td><td>recombination-era closure</td><td>Same reduced state plus host-calibrated f<sub>α</sub> and smoothing scale; f<sub>α</sub>=0 is the canonical local kernel.</td></tr>
          </tbody></table><p><b>These are validation windows, not hard runtime guards.</b> Outside them, compare against <code>solve_chem!</code>. For collapse approaching three-body densities (n<sub>H</sub> ∼ 10⁸ cm⁻³), switch early with <code>solve_chem_hybrid_device_u16!</code> and validate the chosen <code>rho_switch</code> for the problem.</p></div>
        </div>
        <div className="mode-grid">
          <article><h3>Rate tables</h3><p>Optional monotonic log–log interpolation replaces ≈25 transcendental fits. It can cut rate-building cost 3–5× on GPU; direct fits remain the reference.</p><span>GPU hot path</span></article>
          <article><h3>Phase-space bins</h3><p>One stiff solve per occupied log-state bin, with a finite-difference log-Jacobian mapping variance back to member cells.</p><span>CPU / uniform high-z volumes</span></article>
          <article><h3>UInt16 species</h3><p>Log₂ encoding spans 2⁻¹¹⁰ to 1 with ≈0.12% per ULP, halving each species array from four to two bytes.</p><span>Bandwidth-bound runs</span></article>
          <article><h3>Zero-copy device</h3><p>Device-resident arrays are updated in place. No host staging, no field allocation, and one independent thread per cell.</p><span>CUDA / Metal integration</span></article>
        </div>
      </section>

      <section className="section rate-section" id="rates">
        <div className="section-label">04 · Rate & cooling atlas</div>
        <div className="split-heading"><h2>The implemented functions,<br />not a schematic.</h2><p>Every curve below is generated by loading ChemistryKernels.jl and evaluating its current Float64 formulas. Choose a family, hover for values, or switch individual channels off to isolate a process.</p></div>
        <div className="dataset-tabs" role="tablist" aria-label="Scientific plot dataset">
          {scienceDatasets.map((d) => <button key={d.id} role="tab" aria-selected={d.id === datasetId} onClick={() => setDatasetId(d.id)}>{d.title}</button>)}
        </div>
        <div className="plot-card">
          <div className="plot-title"><div><span>LIVE FORMULA ATLAS</span><h3>{dataset.title}</h3></div><small>{dataset.series.length} channels · {dataset.x.length} samples</small></div>
          <ScienceChart key={dataset.id} data={dataset} />
          {dataset.note && <p className="plot-note">Method note — {dataset.note}</p>}
        </div>
        <div className="atlas-notes">
          <div><b>Reference path</b><span>Analytic fits are evaluated directly and remain the parity anchor.</span></div>
          <div><b>Plot convention</b><span>Positive values on logarithmic y axes; zeros are omitted and rate panels are visually floored at 10⁻³⁰.</span></div>
          <div><b>Regeneration</b><span>The checked-in dataset is rebuilt from the package, keeping documentation tied to code.</span></div>
        </div>
      </section>

      <section className="section recomb-section" id="recombination">
        <div className="section-label">05 · Recombination & Lyα mixing</div>
        <div className="split-heading"><h2>From Peebles’ bottleneck to an inhomogeneous recombination volume.</h2><p>The early-Universe path updates the Peebles C factor with the RECFAST-v2 correction, validates hydrogen against a CAMB/RECFAST-v2 fixture cross-checked to HyRec, and optionally lets Lyα escape depend on a host-supplied density smoothed over the photon mixing scale.</p></div>
        <div className="recomb-layout">
          <div className="formula-card">
            <span className="overline">Local three-level atom</span>
            <div className="formula-lines"><p>K = g(z) λ<sub>α</sub>³ / 8πH</p><p>C = f<sub>u</sub>(1 + KΛ<sub>2γ</sub>n<sub>1s,eff</sub>) / (1 + KΛ<sub>2γ</sub>n<sub>1s,eff</sub> + f<sub>u</sub>Kβn<sub>1s,local</sub>)</p><p>k₂ = α<sub>B</sub>C</p></div>
            <p>RECFAST-v2 sets <b>f<sub>u</sub> = 1.125</b> on α<sub>B</sub> and applies two Gaussians in ln(1+z) to K. The correction is not a multiplier on the two-photon decay term.</p>
          </div>
          <div className="formula-card accent">
            <span className="overline">Lyα streaming approximation</span>
            <div className="formula-lines big"><p>n<sub>1s,eff</sub> = (1 − f<sub>α</sub>) n<sub>1s,local</sub> + f<sub>α</sub> n<sub>1s,smoothed</sub></p></div>
            <p>Only the escape term uses the mixed density; the photoionization term remains local. <b>f<sub>α</sub>(z)</b> and the smoothed field come from the host’s transport calibration. At f<sub>α</sub>=0 the mixing solver is bit-identical to the local path.</p>
          </div>
        </div>
        <div className="discussion-grid">
          <article><h3>What it captures</h3><p>Resonant photons produced in overdense regions can sample a larger neutral-density environment before redshifting out of Lyα. Replacing only the Sobolev escape density is a low-storage closure for that non-local communication.</p></article>
          <article><h3>What it assumes</h3><p>The host provides a meaningful smoothing scale and a calibrated f<sub>α</sub>(z). Converting smoothed total H to neutral H uses the global mean electron fraction, appropriate when x<sub>e</sub> varies slowly across the mixing volume.</p></article>
          <article><h3>What it is not</h3><p>It is not a Monte-Carlo Lyα transport solver and does not predict f<sub>α</sub>. Treat it as a controlled closure derived from transport experiments, with f<sub>α</sub>=0 as the safe local limit.</p></article>
        </div>
        <div className="validation-block">
          <div className="validation-copy"><span className="overline">CURRENT VALIDATION</span><h3>The large Peebles tail is removed.</h3><p>Seeded from the same state at z=1200, pure Peebles runs high by 8.35% at z=700. The corrected implementation stays within 0.08% at every sampled redshift in the primary hydrogen gate.</p></div>
          <div className="validation-table"><div className="vhead"><span>z</span><span>Peebles</span><span>RECFAST-v2</span></div>
            {[[700,"+8.35%","+0.08%"],[800,"+4.61%","−0.00%"],[900,"+0.52%","+0.05%"],[1000,"−0.35%","+0.08%"],[1100,"−0.01%","+0.03%"]].map(r => <div key={r[0]}><span>{r[0]}</span><span>{r[1]}</span><span className="good">{r[2]}</span></div>)}
          </div>
        </div>
        <div className="validity-band">
          <div><span>700–1100</span><b>H RECFAST-v2</b><p>&lt;0.1% measured error against the CAMB fixture.</p></div>
          <div><span>3000–4500</span><b>H + Saha helium</b><p>&lt;0.5% regression checks on total x<sub>e</sub>.</p></div>
          <div><span>1900–2500</span><b>Advected He II</b><p>&lt;1.5% end-to-end regression gate through He I freeze-out.</p></div>
          <div><span>to z = 8000</span><b>Reference coverage</b><p>Fixture and helium-aware solver path extend through fully ionized epochs.</p></div>
        </div>
      </section>

      <section className="section performance-section">
        <div className="section-label">06 · Execution & storage</div>
        <div className="split-heading"><h2>Throughput starts by moving less state.</h2><p>Per-cell independence maps directly to accelerators, but the larger gains come from doing less work and moving less memory per cell: two compact evolved species, closed-form stiff updates, uniform iteration counts, and optional rate tables.</p></div>
        <div className="perf-hero"><div><strong>≈4</strong><span>Gcell s⁻¹</span><p>analytic H + H₂ path<br />RTX A6000 · 128³ · Float32</p></div><div className="bars"><div><span>analytic</span><i style={{width:"100%"}} /><b>4×</b></div><div><span>full network</span><i style={{width:"25%"}} /><b>1×</b></div><small>Repository benchmark reference; hardware and state distribution determine realized throughput.</small></div></div>
        <div className="storage-card"><div><span className="overline">THE KEY MEMORY RESULT</span><h3>Nine Float64 fields become two UInt16 fields.</h3><p>The minimal primordial production state reports about 16× less chemistry memory than the original Abel-network layout, and about 8× less total memory in ABN-style first-star calculations.</p></div><div className="bytes"><div className="float legacy"><b>Original model</b>{Array.from({length: 9}).map((_, i) => <i key={i} />)}<span>9 × Float64</span></div><div className="u16 compact"><b>Production path</b><i/><i/><span>2 × UInt16</span></div></div><dl><div><dt>chemistry state</dt><dd>≈16× smaller</dd></div><div><dt>ABN total</dt><dd>≈8× smaller</dd></div><div><dt>encoded range</dt><dd>33.1 dex</dd></div><div><dt>quantization</dt><dd>≈0.12% / ULP</dd></div></dl></div>
      </section>

      <section className="section usage-section" id="usage">
        <div className="section-label">07 · Usage</div>
        <div className="split-heading"><h2>Start with one call.<br />Specialize only when it pays.</h2><p>Arrays may be any length and are updated in place. All density fields use the same host code units; <code>density_units</code>, <code>length_units</code>, and <code>time_units</code> define the conversion to CGS inside the kernel.</p></div>
        <div className="code-tabs">
          <article><div className="code-heading"><span>01</span><h3>CPU / portable baseline</h3></div><CodeBlock>{quickStart}</CodeBlock></article>
          <article><div className="code-heading"><span>02</span><h3>Zero-copy CUDA production</h3></div><CodeBlock>{gpuStart}</CodeBlock></article>
          <article><div className="code-heading"><span>03</span><h3>Lyα-mixing recombination</h3></div><CodeBlock>{mixingStart}</CodeBlock></article>
        </div>
        <div className="decision-card"><h3>Choose a solver</h3><div className="decision-grid">
          <div><span>Need all enabled physics?</span><b>solve_chem!</b><p>HD, dust, metals, general thermal evolution.</p></div>
          <div><span>Primordial H/H₂ and speed first?</span><b>solve_chem_analytic!</b><p>Closed-form chemistry and Compton exchange.</p></div>
          <div><span>Inhomogeneous cosmic recombination?</span><b>solve_chem_mixing!</b><p>Local/full solver with Lyα density mixing.</p></div>
          <div><span>Same, but throughput critical?</span><b>solve_chem_analytic_mixing!</b><p>Analytic primordial path plus mixed escape.</p></div>
        </div></div>
        <div className="api-grid">
          <article><h3>Units that must be right</h3><ul><li><code>e</code> is specific internal energy, not an energy density.</li><li><code>H2I</code> stores 2 n(H₂)m<sub>H</sub>; <code>HDI</code> stores 3 n(HD)m<sub>H</sub>.</li><li><code>a_value</code> sets z = 1/a − 1 and therefore T<sub>CMB</sub>.</li><li>Pass <code>adot_over_a</code> when the host owns the exact cosmological step.</li></ul></article>
          <article><h3>Production checks</h3><ul><li>Use Float64 CPU as the scientific reference before moving to Float32 GPU.</li><li>Sort cells by temperature if warp divergence is visible.</li><li>Treat <code>itcap</code> as a watchdog; resume unfinished stiff cells later.</li><li>Regenerate rate/cooling tables for the target backend and precision.</li></ul></article>
          <article><h3>Known boundaries</h3><ul><li>Fine-structure metal cooling tapers to zero from 10⁴–2×10⁴ K.</li><li>No full high-temperature tabulated metal cooling or RT heating.</li><li>Default helium equilibrium has a quantified z≈2000–2500 freeze-out error.</li><li>The analytic H₂ path is not the general shock/dissociation solver.</li></ul></article>
        </div>
      </section>

      <section className="section integration-section" id="integration">
        <div className="section-label">08 · C &amp; Fortran host codes</div>
        <div className="split-heading"><h2>One small C ABI.<br />Every host language.</h2><p>C, C++, and Fortran production codes should not embed Julia objects in their mesh state. Compile a thin <code>Base.@ccallable</code> wrapper into <code>libchemistrykernels</code>, expose flat pointer-and-length functions, and keep the package’s two-field memory advantage all the way to the hydro boundary.</p></div>
        <div className="host-principles">
          <article><span>01 · C / C++</span><h3>Direct, zero-copy arrays</h3><p>Pass contiguous <code>rho</code>, <code>e</code>, <code>HII</code>, and <code>H2I</code> buffers. The wrapper uses <code>unsafe_wrap</code> for the duration of the call and never retains host pointers.</p></article>
          <article><span>02 · Modern Fortran</span><h3><code>ISO_C_BINDING</code></h3><p>Bind to the same C symbols with exact-width kinds. A flattened multidimensional field is valid in native Fortran order because every chemistry cell is independent.</p></article>
          <article><span>03 · Legacy Fortran</span><h3>A tiny compiler shim</h3><p>Keep the scientific ABI in C, then provide a compiler-specific underscore/by-reference wrapper or one modern <code>bind(C)</code> bridge module. Hide symbol mangling in that single file.</p></article>
          <article><span>04 · Existing grackle call sites</span><h3>Source-compatible shim</h3><p>Mirror the grackle API and structs, relink the host, and reconstruct equilibrium-only legacy fields on return. Do not spoof grackle’s binary ABI or SONAME.</p></article>
        </div>
        <div className="integration-code">
          <article><div className="code-heading"><span>C ABI</span><h3>A narrow production boundary</h3></div><CodeBlock>{cAbiStart}</CodeBlock></article>
          <article><div className="code-heading"><span>FORTRAN 2003+</span><h3>The same symbol through <code>bind(C)</code></h3></div><CodeBlock>{fortranStart}</CodeBlock></article>
        </div>
        <div className="integration-checklist">
          <div><b>Build and initialization</b><p>Use <code>PackageCompiler.create_library</code> today; initialize the bundled Julia runtime once per MPI rank, warm the selected solver once, and finalize only at process shutdown.</p></div>
          <div><b>Ownership and threading</b><p>The host owns every array. Do not retain pointers after return. Serialize entry while bringing up the wrapper, then validate the intended Julia-thread/MPI policy under the real host scheduler.</p></div>
          <div><b>Units and layout</b><p>Pass the four unit scalars explicitly. Keep <code>e</code> as specific energy and H₂ as <code>2 n(H₂)mH</code>; expose separate Float32/Float64 symbols instead of a runtime type flag.</p></div>
          <div><b>Compatibility proof</b><p>Run a conformance grid over T, ρ, z, abundances, and timestep against the native Julia entry point and, during migration, the prior chemistry library.</p></div>
        </div>
        <p className="integration-note"><b>Status:</b> the repository documents the recommended wrapper contract; it does not yet ship a universal binary shim because grackle structure layouts and species-mapping policy vary by host. Start with the <a href="https://github.com/yipihey/ChemistryKernels.jl/blob/main/docs/integration.md">host integration guide ↗</a> and the <a href="https://github.com/yipihey/ChemistryKernels.jl/blob/main/docs/grackle_interface.md">C/Fortran shim recipe ↗</a>.</p>
      </section>

      <section className="section refs-section" id="references">
        <div className="section-label">09 · Provenance &amp; source map</div>
        <div className="split-heading"><h2>Primary literature is part of the interface.</h2><p>Rates are not anonymous constants. The package comments identify the paper behind each fit; this overview keeps the core method and validation lineage one click away.</p></div>
        <div className="refs-grid">
          <a href="https://doi.org/10.1016/S1384-1076(97)00010-9"><span>1997 · New Astronomy 2, 181</span><h3>Modeling primordial gas in numerical cosmology</h3><p>Abel, Anninos, Zhang & Norman · reaction network and rate fits</p></a>
          <a href="https://arxiv.org/abs/astro-ph/9608041"><span>1997 · New Astronomy 2, 209</span><h3>Cosmological hydrodynamics with multi-species chemistry</h3><p>Anninos, Zhang, Abel & Norman · backward differencing and equilibrium closures</p></a>
          <a href="https://arxiv.org/abs/astro-ph/9803315"><span>1998 · A&amp;A 335, 403</span><h3>The chemistry of the early Universe</h3><p>Galli & Palla · molecular network and cooling</p></a>
          <a href="https://arxiv.org/abs/astro-ph/0112088"><span>2002 · Science 295, 93</span><h3>The formation of the first star in the Universe</h3><p>Abel, Bryan & Norman · resolved primordial collapse and H₂-cooled core formation</p></a>
          <a href="https://arxiv.org/abs/1005.2416"><span>2010 · Phys. Rev. D 82, 083520</span><h3>Relative velocity of dark matter and baryonic fluids</h3><p>Tseliakhovich & Hirata · coherent supersonic streaming and first-structure formation</p></a>
          <a href="https://arxiv.org/abs/1204.1344"><span>2012 · ApJ 760, 4</span><h3>The first cosmic structures and the z≈20 Universe</h3><p>O’Leary & McQuinn · streaming-aware simulations across 15 ≲ z ≲ 200</p></a>
          <a href="https://arxiv.org/abs/1011.3758"><span>2011 · Phys. Rev. D 83</span><h3>HyRec</h3><p>Ali-Haïmoud & Hirata · high-accuracy H/He recombination reference</p></a>
          <a href="https://www.astro.ubc.ca/people/scott/recfast.html"><span>RECFAST · v1/v2</span><h3>Fast effective three-level atom</h3><p>Seager, Sasselov & Scott; Wong, Moss & Scott · calibrated recombination closure</p></a>
          <a href="https://arxiv.org/abs/2312.11448"><span>2025 · JCAP 03, 012</span><h3>Cosmic recombination with primordial magnetic fields</h3><p>Jedamzik, Abel & Ali-Haïmoud · baryon clumping and Lyα photon mixing</p></a>
          <a href="https://arxiv.org/abs/1903.08657"><span>2020 · MNRAS 493, 1614</span><h3>A cosmic UV/X-ray background model update</h3><p>Faucher-Giguère · FG20 photoionization and photoheating tables</p></a>
          <a href="https://doi.org/10.1086/519445"><span>2007 · ApJ 666, 1</span><h3>Star formation at very low metallicity</h3><p>Glover & Jappsen · C/O/Si fine-structure cooling lineage</p></a>
        </div>
        <div className="source-map"><div><span className="overline">READ THE IMPLEMENTATION</span><h3>The equations and their tests are one click away.</h3><p>Links point to <code>main</code>, so the site stays useful as the implementation evolves. For reproducible citation, replace <code>main</code> with the release tag or commit used by the simulation.</p></div><nav aria-label="Source code map">
          <a href="https://github.com/yipihey/ChemistryKernels.jl/blob/main/src/solve.jl"><b>Public drivers</b><span>src/solve.jl ↗</span></a>
          <a href="https://github.com/yipihey/ChemistryKernels.jl/blob/main/src/subcycle.jl"><b>Full ODE subcycling</b><span>src/subcycle.jl ↗</span></a>
          <a href="https://github.com/yipihey/ChemistryKernels.jl/blob/main/src/network_step.jl"><b>Backward-Euler sweep</b><span>src/network_step.jl ↗</span></a>
          <a href="https://github.com/yipihey/ChemistryKernels.jl/blob/main/src/fast.jl"><b>Analytic &amp; hybrid paths</b><span>src/fast.jl ↗</span></a>
          <a href="https://github.com/yipihey/ChemistryKernels.jl/blob/main/src/recombination_clumping.jl"><b>RECFAST &amp; Lyα mixing</b><span>src/recombination_clumping.jl ↗</span></a>
          <a href="https://github.com/yipihey/ChemistryKernels.jl/blob/main/src/log2_species.jl"><b>UInt16 codec</b><span>src/log2_species.jl ↗</span></a>
          <a href="https://github.com/yipihey/ChemistryKernels.jl/blob/main/src/rates_atomic.jl"><b>Atomic rates</b><span>src/rates_atomic.jl ↗</span></a>
          <a href="https://github.com/yipihey/ChemistryKernels.jl/blob/main/src/rates_h2.jl"><b>Molecular rates</b><span>src/rates_h2.jl ↗</span></a>
          <a href="https://github.com/yipihey/ChemistryKernels.jl/blob/main/src/rate_tables.jl"><b>GPU rate tables</b><span>src/rate_tables.jl ↗</span></a>
          <a href="https://github.com/yipihey/ChemistryKernels.jl/blob/main/test/test_fast.jl"><b>Analytic validation</b><span>test/test_fast.jl ↗</span></a>
          <a href="https://github.com/yipihey/ChemistryKernels.jl/blob/main/test/test_recombination_mixing.jl"><b>HyRec / RECFAST gates</b><span>test/test_recombination_mixing.jl ↗</span></a>
          <a href="https://github.com/yipihey/EmissionKernels.jl"><b>Cooling implementation</b><span>EmissionKernels.jl ↗</span></a>
        </nav></div>
        <div className="citation-note"><b>Scientific transparency</b><p>Plot data in this site are generated from the checked-out package. Reported performance numbers are repository benchmarks, not cross-platform guarantees. Reported accuracy is stated with its tested redshift window and comparison target.</p></div>
      </section>

      <footer><div><span className="footer-mark">CK</span><p><b>ChemistryKernels.jl</b><br />A modern numerical laboratory for primordial gas.</p></div><div><a href="#top">Back to top ↑</a><a href="https://github.com/yipihey/ChemistryKernels.jl">GitHub</a><a href="https://github.com/yipihey/VespaRegistry">Registry</a></div></footer>
    </main>
  );
}

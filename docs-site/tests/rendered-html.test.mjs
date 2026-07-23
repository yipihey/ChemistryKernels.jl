import assert from "node:assert/strict";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(new Request("http://localhost/", { headers: { accept: "text/html" } }), {
    ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) },
  }, { waitUntil() {}, passThroughOnException() {} });
}

test("renders the complete ChemistryKernels methods overview", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /ChemistryKernels\.jl/);
  assert.match(html, /From recombination/);
  assert.match(html, /≈16×/);
  assert.match(html, /≈8×/);
  assert.match(html, /Nine Float64 fields become two UInt16 fields/);
  assert.match(html, /Tseliakhovich/);
  assert.match(html, /O’Leary/);
  assert.match(html, /Abel, Bryan &amp; Norman/);
  assert.match(html, /ANALYTIC METHODS · REFERENCE ONLY/);
  assert.match(html, /One production path/);
  assert.match(html, /full network is[\s\S]*the fast path/i);
  assert.match(html, /not recommended for production/);
  assert.match(html, /What “analytic” does—and does not—mean/);
  assert.match(html, /z=25 collapse/);
  assert.match(html, /Rate &amp; cooling atlas/);
  assert.match(html, /visually floored at 10⁻³⁰/);
  assert.match(html, /Lyα streaming approximation/);
  assert.match(html, /C &amp; Fortran host codes/);
  assert.match(html, /ISO_C_BINDING/);
  assert.match(html, /Legacy Fortran/);
  assert.match(html, /src\/fast\.jl/);
  assert.match(html, /test\/test_recombination_mixing\.jl/);
  assert.match(html, /Primary literature/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Your site is taking shape/i);
});

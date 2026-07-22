import { cp, mkdir, rm, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const siteDir = resolve(scriptDir, "..");
const clientDir = resolve(siteDir, "dist/client");
const outputDir = resolve(siteDir, "dist/github-pages");
const workerPath = resolve(siteDir, "dist/server/index.js");

const repository = process.env.GITHUB_REPOSITORY ?? "yipihey/ChemistryKernels.jl";
const [owner = "yipihey", repositoryName = "ChemistryKernels.jl"] = repository.split("/");
const inferredBasePath = repositoryName.endsWith(".github.io") ? "" : `/${repositoryName}`;
const basePath = (process.env.GITHUB_PAGES_BASE_PATH ?? inferredBasePath).replace(/\/$/, "");
const origin = process.env.GITHUB_PAGES_ORIGIN ?? `https://${owner}.github.io`;

const workerUrl = pathToFileURL(workerPath);
workerUrl.searchParams.set("pages-export", `${process.pid}-${Date.now()}`);
const { default: worker } = await import(workerUrl.href);
const response = await worker.fetch(
  new Request(`${origin}/`, {
    headers: {
      accept: "text/html",
      "x-forwarded-host": new URL(origin).host,
      "x-forwarded-proto": new URL(origin).protocol.slice(0, -1),
    },
  }),
  { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
  { waitUntil() {}, passThroughOnException() {} },
);

if (!response.ok) {
  throw new Error(`Site renderer returned HTTP ${response.status}`);
}

let html = await response.text();
html = html.replaceAll(`${origin}/`, `${origin}${basePath}/`);
// Rewrite quoted root-relative URLs, but not the `"/>` sequence that closes
// self-closing HTML tags.
html = html.replace(/(["'])\/(?=[A-Za-z0-9_.~-])/g, `$1${basePath}/`);

await rm(outputDir, { recursive: true, force: true });
await mkdir(outputDir, { recursive: true });
await cp(clientDir, outputDir, { recursive: true });
await writeFile(resolve(outputDir, "index.html"), html);
await writeFile(resolve(outputDir, "404.html"), html);
await writeFile(resolve(outputDir, ".nojekyll"), "");

console.log(`Exported GitHub Pages site at ${origin}${basePath}/`);

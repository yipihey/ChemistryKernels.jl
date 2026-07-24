import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import test from "node:test";

const pagesDir = resolve(import.meta.dirname, "../dist/github-pages");

test("exports a repository-subpath-safe GitHub Pages site", async () => {
  const html = await readFile(resolve(pagesDir, "index.html"), "utf8");

  assert.match(html, /ChemistryKernels\.jl/);
  assert.match(html, /(?:href|src)="\/ChemistryKernels\.jl\/assets\//);
  assert.match(html, /https:\/\/yipihey\.github\.io\/ChemistryKernels\.jl\/og\.png/);
  assert.doesNotMatch(html, /(?:href|src)="\/assets\//);
  assert.doesNotMatch(html, /\/ChemistryKernels\.jl\/>/);

  const assetPaths = [...html.matchAll(/(?:href|src)="\/ChemistryKernels\.jl\/(assets\/[^"?#]+)/g)]
    .map((match) => match[1]);
  assert.ok(assetPaths.length > 0, "expected the exported HTML to reference built assets");
  await Promise.all([...new Set(assetPaths)].map((path) => access(resolve(pagesDir, path))));
  const cssPaths = [...new Set(assetPaths.filter((path) => path.endsWith(".css")))];
  assert.ok(cssPaths.length > 0, "expected at least one exported stylesheet");
  const css = (await Promise.all(cssPaths.map((path) => readFile(resolve(pagesDir, path), "utf8")))).join("\n");
  assert.match(css, /touch-action:pan-y/,
    "plots must leave native vertical touch scrolling enabled");
  await access(resolve(pagesDir, ".nojekyll"));
  await access(resolve(pagesDir, "404.html"));
});

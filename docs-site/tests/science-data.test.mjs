import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import test from "node:test";

test("reaction-rate datasets carry the 1e-30 display floor", async () => {
  const generated = await readFile(resolve(import.meta.dirname, "../app/science-data.ts"), "utf8");
  const rateIds = ["atomic-rates", "molecular-rates", "deuterium-rates", "cmb-rates", "uvb-ion"];

  for (const id of rateIds) {
    const start = generated.indexOf(`id:"${id}"`);
    assert.notEqual(start, -1, `missing generated dataset ${id}`);
    const next = generated.indexOf("\n{id:", start + 1);
    const record = generated.slice(start, next === -1 ? undefined : next);
    assert.match(record, /yFloor:1(?:\.0+)?e-30/, `${id} must be plotted no lower than 1e-30`);
  }

  const coolingStart = generated.indexOf('id:"atomic-cooling"');
  const coolingEnd = generated.indexOf("\n{id:", coolingStart + 1);
  assert.doesNotMatch(generated.slice(coolingStart, coolingEnd), /yFloor:/,
    "the rate display floor must not silently truncate cooling coefficients");
});

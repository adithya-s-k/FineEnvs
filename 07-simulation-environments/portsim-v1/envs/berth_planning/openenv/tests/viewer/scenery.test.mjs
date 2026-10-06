import assert from 'node:assert/strict';
import { test } from 'node:test';
import { surfaceAt } from '../../berth_openenv/web/twin.js';

// The rendered mesh splits a cell from north-east to south-west. A bilinear
// height lookup would hide parts of a road under this non-planar hillside.
const hill = {
  frame: { extent: [0, 0, 40, 40], grid: 40, terrain: { w: 2, h: 2 } },
  heights: new Float32Array([0, 20, 60, 140]),
};
test('roads sit on the north-west terrain triangle', () => {
  assert.equal(surfaceAt(hill, 8, 32), 16);
});
test('roads sit on the south-east terrain triangle', () => {
  assert.ok(Math.abs(surfaceAt(hill, 32, 12) - 88) < 1e-6);
});
test('the two road-height planes meet at the rendered diagonal', () => {
  assert.equal(surfaceAt(hill, 24, 24), 36);
  assert.ok(Math.abs(surfaceAt(hill, 24 + 1e-5, 24) - surfaceAt(hill, 24 - 1e-5, 24)) < 0.001);
});
test('flat harbour aprons cover submerged terrain samples', () => {
  const port = { ...hill, heights: new Float32Array([-8, -8, -8, -8]) };
  assert.equal(surfaceAt(port, 20, 20), 0);
});
test('compact-device roads follow the coarser rendered mesh', () => {
  const heights = new Float32Array(25).fill(999);
  heights[0] = 0; heights[2] = 20; heights[10] = 60; heights[12] = 140;
  const coarse = { frame: { extent: [0, 0, 160, 160], grid: 40, terrain: { w: 5, h: 5 } }, heights };
  assert.equal(surfaceAt(coarse, 16, 144, 2), 16);
});

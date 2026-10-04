import test from 'node:test';
import assert from 'node:assert/strict';
import { rewardDistribution } from '../web/js/reward-stats.js';

test('negative and unbounded rewards remain in the histogram', () => {
  const d = rewardDistribution([-100, -5, 0, 0.5, 1, 500]);
  assert.equal(d.min, -100);
  assert.equal(d.max, 500);
  assert.equal(d.hist.reduce((a, b) => a+b), 6);
  assert.equal(d.hist[0], 1);
  assert.equal(d.hist[9], 1);
});
test('empty and normalized rewards keep a useful finite range', () => {
  for (const xs of [[], [0, 0.5, 1], [NaN, Infinity]]) {
    const d = rewardDistribution(xs);
    assert.equal(d.min, 0);
    assert.equal(d.max, 1);
    assert.ok(d.hist.every(Number.isFinite));
  }
});

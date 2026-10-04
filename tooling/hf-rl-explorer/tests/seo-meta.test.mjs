import test from 'node:test';
import assert from 'node:assert/strict';
import { canonicalPath, imagePath } from '../web/js/seo-meta.js';

test('Space task canonical preserves identity and removes tracking', () => {
  const u = 'https://example.org/s/org/space?task=01&split=train+%26+test&env=game&utm_source=share#sec-tasks';
  assert.equal(canonicalPath(u), '/s/org/space?env=game&split=train+%26+test&task=1');
  assert.equal(imagePath(u), '/og/s/org/space.png?env=game&split=train+%26+test&task=1');
});
test('filters and rollout IDs do not become social preview endpoints', () => {
  assert.equal(canonicalPath('https://example.org/?owner=FineEnvs'), '/');
  assert.equal(imagePath('https://example.org/run/private-id'), '/social/rl-explorer.png');
  assert.equal(imagePath('https://example.org/t/org/dataset/train/1?x=1'), '/og/t/org/dataset/train/1.png');
  assert.equal(canonicalPath('https://example.org/s/org/space?task=-1&env=g&split=s'), '/s/org/space');
});

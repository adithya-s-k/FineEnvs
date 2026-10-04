import test from 'node:test';
import assert from 'node:assert/strict';
import { spaceMedia } from '../web/js/space-media.js';
import { taskContent } from '../web/js/space-task-view.js';

test('lazy Space media resolves against the serving Space without mutating observations', () => {
  const raw = { observation: { asset_path: '/assets/hash?task_id=a', mime: 'audio/wav' } };
  const out = spaceMedia(raw, 'https://owner-env.hf.space');
  assert.equal(out.observation.media_url, 'https://owner-env.hf.space/assets/hash?task_id=a');
  assert.equal(raw.observation.media_url, undefined);
});

test('media paths cannot change origin or use active URL schemes', () => {
  for (const path of ['//evil.example/assets/a', 'javascript:alert(1)', 'https://evil.example/a', '/other/a'])
    assert.equal(spaceMedia({ asset_path: path, mime: 'audio/wav' }, 'https://owner-env.hf.space').media_url, undefined);
  assert.equal(spaceMedia({ asset_path: '/assets/a', mime: 'audio/wav' }, 'http://localhost').media_url, undefined);
});

test('task inputs are escaped and lazy media remains an available episode', () => {
  const html = taskContent({ prompt: '<img src=x onerror=alert(1)>', task_id: 'a'.repeat(200), media_ready: false });
  assert.ok(!html.includes('<img src=x'));
  assert.ok(html.includes('Media loads when the episode starts'));
  assert.ok(html.includes('Task record'));
});

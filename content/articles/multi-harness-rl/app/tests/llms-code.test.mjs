import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import generateLlmsTxt from '../plugins/astro/generate-llms-txt.mjs';

const app = fileURLToPath(new URL('../', import.meta.url));

test('article llms.txt preserves model placeholders in runnable examples', async () => {
  const previous = process.cwd();
  const output = await mkdtemp(join(tmpdir(), 'article-llms-'));
  try {
    process.chdir(app);
    await generateLlmsTxt().hooks['astro:build:done']({ dir: pathToFileURL(output + '/') });
    const markdown = await readFile(join(output, 'llms.txt'), 'utf8');
    assert.ok(markdown.includes('vllm serve <model> --return-tokens-as-token-ids'));
    assert.ok(markdown.includes('--llm-url http://127.0.0.1:8000 --model <model>'));
    assert.ok(markdown.includes('# The ultimate guide to multi-harness RL'));
    assert.ok(!markdown.includes('<Introduction />'));
    assert.ok(!markdown.includes('<HtmlEmbed'));
  } finally {
    process.chdir(previous);
    await rm(output, { recursive: true, force: true });
  }
});

for (const [name, fenced] of [
  ['backticks', '```python\nif count < limit > 0:\n  print("<Note>literal</Note>")\n```'],
  ['tilde fence with longer closing', '~~~html\n<div>literal</div>\n~~~~'],
  ['nested shorter fence', '````md\n```html\n<Note>literal</Note>\n```\n````'],
  ['unclosed fence', '```bash\nvllm serve <model>'],
  ['placeholder-like source', '```text\nLLMS_CODE_BLOCK_0END\n```'],
]) {
  test(`llms generation preserves ${name} and still unwraps prose components`, async () => {
    const previous = process.cwd();
    const fixture = await mkdtemp(join(tmpdir(), 'llms-fixture-'));
    try {
      const { mkdir, writeFile } = await import('node:fs/promises');
      await mkdir(join(fixture, 'src/content'), { recursive: true });
      await writeFile(join(fixture, 'src/content/article.mdx'),
        `---\ntitle: "Fixture"\n---\n\n<Note>Prose stays readable</Note>\n\n${fenced}\n`);
      process.chdir(fixture);
      await generateLlmsTxt().hooks['astro:build:done']({ dir: pathToFileURL(fixture + '/') });
      const output = await readFile(join(fixture, 'llms.txt'), 'utf8');
      assert.ok(output.includes(fenced), output);
      assert.ok(output.includes('> Prose stays readable'));
      assert.ok(!output.includes('<Note>Prose stays readable</Note>'));
    } finally {
      process.chdir(previous);
      await rm(fixture, { recursive: true, force: true });
    }
  });
}

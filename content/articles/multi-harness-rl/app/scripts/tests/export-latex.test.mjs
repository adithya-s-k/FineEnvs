import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { promises as fs } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const app = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const exporter = process.env.TEST_EXPORT_LATEX_SCRIPT || resolve(app, 'scripts/export-latex.mjs');
const hasPandoc = spawnSync('pandoc', ['--version']).status === 0;

async function exportArticle(t, article, bibliography) {
  const cwd = await fs.mkdtemp(resolve(tmpdir(), 'latex-export-test-'));
  t.after(() => fs.rm(cwd, { recursive: true, force: true }));
  await fs.mkdir(resolve(cwd, 'src/content'), { recursive: true });
  await fs.mkdir(resolve(cwd, 'tmp'));
  if (article === null) {
    await fs.cp(resolve(app, 'src/content'), resolve(cwd, 'src/content'), { recursive: true });
  } else {
    await fs.writeFile(resolve(cwd, 'src/content/article.mdx'), article);
  }
  if (bibliography) await fs.writeFile(resolve(cwd, 'src/content/bibliography.bib'), bibliography);
  const result = spawnSync(process.execPath, [exporter, '--filename=exported'], {
    cwd, encoding: 'utf8', env: { ...process.env, TMPDIR: resolve(cwd, 'tmp') },
  });
  let tex = '';
  try { tex = await fs.readFile(resolve(cwd, 'dist/exported.tex'), 'utf8'); } catch { }
  return { result, tex, cwd };
}

// These execute the shipped exporter and real Pandoc, rather than a mocked writer.
test('the complete shipped article exports its title, eight authors and publication date', { skip: !hasPandoc }, async (t) => {
  const { result, tex } = await exportArticle(t, null);
  assert.equal(result.status, 0, result.stderr);
  assert.ok(tex.includes('\\title{The ultimate guide to multi-harness RL}'));
  for (const author of ['Adithya S Kolavi', 'Joel Niklaus', 'Sergio Paniego Blanco',
    'Leonie Monigatti', 'Amine Dirhoussi', 'Ben Burtenshaw', 'Lewis Tunstall', 'Leandro von Werra']) {
    assert.ok(tex.includes(author), author);
  }
  assert.ok(tex.includes('\\date{October 1, 2026}'));
  assert.ok(tex.includes('\\maketitle'));
  assert.ok(!result.stderr.includes('citation adithya_s_k not found'), result.stderr);
});

test('nested YAML author names and title characters are escaped by Pandoc', { skip: !hasPandoc }, async (t) => {
  const { result, tex } = await exportArticle(t, `---
title: 'Research & results: 100%'
authors:
  - name: 'Alice & Bob'
    url: 'https://example.org/alice'
  - name: 'Chloé_Test'
published: 'October 6, 2026'
twitter: '@unused_handle'
---
# Introduction
A short article.
`);
  assert.equal(result.status, 0, result.stderr);
  assert.ok(tex.includes('\\title{Research \\& results: 100\\%}'));
  assert.ok(tex.includes('\\author{Alice \\& Bob \\and Chloé\\_Test}'));
  assert.ok(tex.includes('\\date{October 6, 2026}'));
  assert.ok(!result.stderr.includes('citation unused_handle not found'), result.stderr);
});

test('explicit Pandoc author/date win and real bibliography citations still render', { skip: !hasPandoc }, async (t) => {
  const { result, tex } = await exportArticle(t, `---
title: 'Citation study'
author: 'Preferred Author'
authors:
  - name: 'Fallback Author'
date: 'Explicit date'
published: 'Fallback date'
---
# Introduction
A result [@smith2026].
`, '@article{smith2026, author={Smith, Jane}, title={Example result}, year={2026}, journal={Test Journal}}');
  assert.equal(result.status, 0, result.stderr);
  assert.ok(tex.includes('\\author{Preferred Author}'));
  assert.ok(tex.includes('\\date{Explicit date}'));
  assert.ok(!tex.includes('Fallback Author'));
  assert.ok(tex.includes('ref-smith2026'), tex);
  assert.ok(tex.includes('Smith 2026'), tex);
  assert.ok(!result.stderr.includes('citation smith2026 not found'), result.stderr);
});

test('articles without metadata retain normal standalone export', { skip: !hasPandoc }, async (t) => {
  const { result, tex, cwd } = await exportArticle(t, '# Introduction\nA plain article.\n');
  assert.equal(result.status, 0, result.stderr);
  assert.ok(tex.includes('A plain article.'));
  assert.ok(!tex.includes('\\maketitle'));
  assert.deepEqual(await fs.readdir(resolve(cwd, 'tmp')), []);
  await assert.rejects(fs.access(resolve(cwd, 'temp-article.md')));
});

test('malformed YAML fails and removes owned temporary files', { skip: !hasPandoc }, async (t) => {
  const { result, cwd } = await exportArticle(t, '---\ntitle: [broken\n---\n# Introduction\nBody.\n');
  assert.notEqual(result.status, 0);
  assert.deepEqual(await fs.readdir(resolve(cwd, 'tmp')), []);
  await assert.rejects(fs.access(resolve(cwd, 'temp-article.md')));
});

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
const imagePath = 'assets/image/joel-harness-pairing.png';

async function exportArticle(t, alter, compile = false) {
  const cwd = await fs.mkdtemp(resolve(tmpdir(), 'latex-image-export-'));
  t.after(() => fs.rm(cwd, { recursive: true, force: true }));
  const content = resolve(cwd, 'src/content');
  await fs.mkdir(resolve(content, 'assets/image'), { recursive: true });
  await fs.cp(resolve(app, 'src/content/chapters'), resolve(content, 'chapters'), { recursive: true });
  for (const file of ['article.mdx', 'bibliography.bib', imagePath]) {
    await fs.copyFile(resolve(app, 'src/content', file), resolve(content, file));
  }
  if (alter) await alter(content);
  const args = [exporter, '--filename=exported'];
  const env = { ...process.env };
  if (compile) {
    await fs.mkdir(resolve(cwd, 'bin'));
    await fs.writeFile(resolve(cwd, 'bin/pdflatex'), '#!/bin/sh\npwd > "$COMPILE_CWD_RECEIPT"\n', { mode: 0o755 });
    env.PATH = `${resolve(cwd, 'bin')}:${env.PATH}`;
    env.COMPILE_CWD_RECEIPT = resolve(cwd, 'compile-cwd.txt');
    args.push('--pdf');
  }
  const result = spawnSync(process.execPath, args, { cwd, encoding: 'utf8', env });
  assert.equal(result.status, 0, result.stderr);
  return { cwd, tex: await fs.readFile(resolve(cwd, 'dist/exported.tex'), 'utf8') };
}

async function assertFigure(cwd, tex) {
  const image = tex.match(/\\includegraphics[^\n]*\{([^{}]+\.png)\}/);
  assert.ok(image, 'the actual static article figure must be included in LaTeX');
  assert.ok(image[1].startsWith('exported-assets/'), 'figure assets belong to this named export');
  const exportedPng = await fs.readFile(resolve(cwd, 'dist', image[1]));
  assert.deepEqual(exportedPng, await fs.readFile(resolve(app, 'src/content', imagePath)), 'exported figure preserves the actual PNG bytes');
  assert.ok(tex.includes('\\caption{Pass@1 against cost per task for two models across'));
  assert.ok(tex.includes('Joel Niklaus'));
  assert.ok(tex.includes('2085725862142623875'), 'the original caption attribution link survives');
  assert.ok(tex.includes('each run under several coding-agent harnesses'), 'original alt text survives separately from the shorter caption');
}

test('the shipped article preserves its static PNG, caption and attribution', { skip: !hasPandoc }, async (t) => {
  const { cwd, tex } = await exportArticle(t);
  await assertFigure(cwd, tex);
  assert.ok(tex.includes('Interactive content not available in LaTeX'), 'interactive embeds keep their existing explicit placeholder');
  assert.ok(tex.includes('The difference can be measured'));
});

test('an aliased Image component resolves its image import from a nested chapter', { skip: !hasPandoc }, async (t) => {
  const { cwd, tex } = await exportArticle(t, async (content) => {
    const intro = (await fs.readFile(resolve(content, 'chapters/introduction.mdx'), 'utf8'))
      .replace(/\bImage\b/g, 'Picture').replace('components/Picture.astro', 'components/Image.astro')
      .replace('../../components/', '../../../components/')
      .replace(/\bjoelPlot\b/g, 'chartAsset')
      .replace('../assets/image/', '../../assets/image/');
    await fs.mkdir(resolve(content, 'chapters/nested'));
    await fs.writeFile(resolve(content, 'chapters/nested/introduction.mdx'), intro);
    const article = (await fs.readFile(resolve(content, 'article.mdx'), 'utf8'))
      .replace('./chapters/introduction.mdx', './chapters/nested/introduction.mdx');
    await fs.writeFile(resolve(content, 'article.mdx'), article);
  });
  await assertFigure(cwd, tex);
});

test('plain prose and bibliography still export without an Image component', { skip: !hasPandoc }, async (t) => {
  const { cwd, tex } = await exportArticle(t, async (content) => {
    await fs.writeFile(resolve(content, 'article.mdx'), '# Introduction\nA plain result [@smith2026].\n');
    await fs.writeFile(resolve(content, 'bibliography.bib'), '@article{smith2026, author={Smith, Jane}, title={Example result}, year={2026}, journal={Test Journal}}');
  });
  assert.ok(tex.includes('A plain result'));
  assert.ok(tex.includes('ref-smith2026'));
  assert.ok(tex.includes('Smith 2026'));
  assert.ok(!tex.includes('\\includegraphics['));
  await assert.rejects(fs.access(resolve(cwd, 'dist/exported-assets')));
});

test('optional compilation starts beside the exported figure assets', { skip: !hasPandoc }, async (t) => {
  // Only the external compiler is an owned shell receipt, not a TeX engine.
  // Pandoc and image export remain real; this asserts the process cwd contract.
  const { cwd, tex } = await exportArticle(t, undefined, true);
  await assertFigure(cwd, tex);
  const compileCwd = (await fs.readFile(resolve(cwd, 'compile-cwd.txt'), 'utf8')).trim();
  assert.equal(await fs.realpath(compileCwd), await fs.realpath(resolve(cwd, 'dist')));
});

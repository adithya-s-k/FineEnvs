import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, copyFile, symlink, readFile, rm } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scripts = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const content = resolve(scripts, '../src/content');

async function runCli(args) {
  const root = await mkdtemp(join(tmpdir(), 'article-assets-json-'));
  try {
    await mkdir(join(root, 'scripts'));
    await mkdir(join(root, 'src'));
    await copyFile(join(scripts, 'extract-used-assets.mjs'), join(root, 'scripts/extract-used-assets.mjs'));
    // Read the actual current chapters/assets; only generated output is owned.
    await symlink(content, join(root, 'src/content'), 'dir');
    const result = spawnSync(process.execPath, [join(root, 'scripts/extract-used-assets.mjs'), ...args], {
      encoding: 'utf8', timeout: 10000,
    });
    assert.equal(result.error, undefined);
    assert.equal(result.status, 0, result.stderr);
    const manifest = JSON.parse(await readFile(join(root, 'src/generated/used-assets.json'), 'utf8'));
    assert.ok(manifest.summary.totalMdxFiles > 0);
    assert.ok(manifest.images.used.includes('joel-harness-pairing.png'));
    assert.equal(manifest.images.missing.length, 0);
    return { ...result, manifest };
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

for (const args of [['--json'], ['--json', '--verbose']]) {
  test(`documented ${args.join(' ')} stdout is a JSON manifest`, async () => {
    const result = await runCli(args);
    assert.deepEqual(JSON.parse(result.stdout), result.manifest);
    assert.match(result.stderr, /Extracting used assets from MDX files/);
    if (args.includes("--verbose")) assert.match(result.stderr, /chapters\/introduction\.mdx/);
  });
}

test('human output retains progress, summary and a valid saved manifest', async () => {
  const result = await runCli(['--verbose']);
  assert.match(result.stdout, /Extracting used assets from MDX files/);
  assert.match(result.stdout, /ASSET EXTRACTION SUMMARY/);
  assert.match(result.stdout, /chapters\/introduction\.mdx/);
  assert.equal(result.stderr, '');
});

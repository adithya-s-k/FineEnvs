import { test, before } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkMdx from 'remark-mdx';
import remarkRehype from 'remark-rehype';
import rehypeCitation from 'rehype-citation';
import rehypeStringify from 'rehype-stringify';
import { chromium } from 'playwright';
import unwrap from '../../remark/unwrap-citation-links.mjs';
import ignore from '../../remark/ignore-citations-in-code.mjs';
import restore from '../restore-at-in-code.mjs';
import post from '../post-citation.mjs';

const app = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
let chapters, footerScript;
const processor = () => unified().use(remarkParse).use(remarkMdx).use(unwrap).use(ignore)
  .use(remarkRehype).use(rehypeCitation, {
    bibliography: 'src/content/bibliography.bib', linkCitations: true,
    csl: 'apa', noCite: false, suppressBibliography: false,
  }).use(post).use(restore).use(rehypeStringify);

before(async () => {
  const article = await readFile(join(app, 'src/content/article.mdx'), 'utf8');
  const names = [...article.matchAll(/from\s+["']\.\/chapters\/([^"']+\.mdx)["']/g)].map(m => m[1]);
  assert.equal(names.length, 6);
  chapters = await Promise.all(names.map(async name => {
    const path = join(app, 'src/content/chapters', name);
    return String(await processor().process({ cwd: app, path, value: await readFile(path, 'utf8') }));
  }));
  const footer = await readFile(join(app, 'src/components/Footer.astro'), 'utf8');
  footerScript = footer.match(/<script is:inline>([\s\S]*?)<\/script>/)[1];
});

async function withPage(fn) {
  const browser = await chromium.launch(process.env.PLAYWRIGHT_CHROME ? {
    executablePath: process.env.PLAYWRIGHT_CHROME,
  } : {});
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    await page.setContent(`<!doctype html><main>${chapters.join('\n')}</main>
      <footer class="footer"><section class="references-block"></section><script>${footerScript}</script></footer>`);
    await page.waitForSelector('footer[data-processed="true"]');
    return await fn(page);
  } finally {
    await browser.close();
  }
}

test('all current chapter citation anchors remain distinct after concatenation', async () => {
  const ids = chapters.flatMap(html => [...html.matchAll(/<a\b[^>]*\bid="([^"]+)"[^>]+data-ref-id=/g)].map(m => m[1]));
  assert.equal(ids.length, 35);
  assert.equal(new Set(ids).size, ids.length);
});

test('native merged bibliography navigates to each repeated citation occurrence', async () => {
  await withPage(async page => {
    // A unique citation is a native navigation positive control before the repeated case.
    await page.locator('footer li[id="bib-deng2025swebenchpro"] .backrefs a').click();
    assert.equal(await page.evaluate(() => document.querySelector(':target') ===
      document.querySelector('a[data-ref-id="bib-deng2025swebenchpro"]')), true);
    const refs = page.locator('footer li[id="bib-zhang2026stopcomparing"] .backrefs a');
    await refs.nth(1).click();
    assert.equal(await page.evaluate(() => document.querySelector(':target') ===
      document.querySelectorAll('a[data-ref-id="bib-zhang2026stopcomparing"]')[1]), true);
    assert.equal(await refs.count(), 3);
    for (let n = 0; n < 3; n++) {
      await refs.nth(n).click();
      assert.equal(await page.evaluate(n => document.querySelector(':target') ===
        document.querySelectorAll('a[data-ref-id="bib-zhang2026stopcomparing"]')[n], n), true);
    }
    assert.equal(await page.locator('footer ol.references > li').count(), 24);
    assert.equal(await page.locator('footer .backrefs a').count(), 35);
    assert.equal(await page.evaluate(() => [...document.querySelectorAll('footer .backrefs a')].every(a => {
      const target = document.getElementById(a.getAttribute('href').slice(1));
      return target?.dataset.refId === a.closest('li').id;
    })), true);
  });
});

test('current prose, canonical bibliography IDs, external URLs and stable source paths survive', async () => {
  const html = chapters.join('\n');
  assert.match(html, /The model chooses the tool and its arguments/);
  assert.match(html, /id="bib-zhang2026stopcomparing"/);
  assert.match(html, /href="https:\/\/huggingface.co\/papers\/2605\.23950"/);
  const name = 'basics.mdx';
  const value = await readFile(join(app, 'src/content/chapters', name), 'utf8');
  const copiedRoot = resolve(app, '../../../../../owned-copy');
  // Check stability through the real processor using its explicit bibliography root.
  const stable = unified().use(remarkParse).use(remarkMdx).use(unwrap).use(ignore).use(remarkRehype)
    .use(rehypeCitation, { path: app, bibliography: 'src/content/bibliography.bib', linkCitations: true, csl: 'apa' })
    .use(post).use(restore).use(rehypeStringify);
  const result = String(await stable.process({ cwd: copiedRoot,
    path: join(copiedRoot, 'src/content/chapters', name), value }));
  const ids = s => [...s.matchAll(/<a\b[^>]*\bid="([^"]+)"[^>]+data-ref-id=/g)].map(m => m[1]);
  assert.deepEqual(ids(result), ids(chapters.find(html => html.includes('The model chooses the tool and its arguments'))));
});

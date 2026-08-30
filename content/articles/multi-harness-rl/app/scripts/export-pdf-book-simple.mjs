#!/usr/bin/env node
/**
 * Export PDF Book – Clean "A Book Apart" style
 *
 * Uses Playwright native print (no Paged.js) for maximum reliability.
 * Features:
 *   - Auto-generated Table of Contents from H2/H3 headings
 *   - Page numbers "X / Y" via Playwright displayHeaderFooter
 *   - Banner/hero visualization preserved
 *   - Charts/embeds properly sized
 *   - Clean minimal typography
 *
 * Usage:
 *   npm run export:pdf:book:simple
 *   npm run export:pdf:book:simple -- --theme=light --format=A4
 */

import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';
import { chromium } from 'playwright';
import { resolve, dirname, basename, join } from 'node:path';
import { promises as fs } from 'node:fs';
import { fileURLToPath } from 'node:url';
import process from 'node:process';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ============================================================================
// Utilities
// ============================================================================

async function run(command, args = [], options = {}) {
  return new Promise((ok, fail) => {
    const child = spawn(command, args, { stdio: 'inherit', shell: false, ...options });
    child.on('error', fail);
    child.on('exit', (code) => {
      if (code === 0) ok(); else fail(new Error(`${command} ${args.join(' ')} exited ${code}`));
    });
  });
}

async function waitForServer(url, timeoutMs = 60_000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try { const r = await fetch(url); if (r.ok) return; } catch {}
    await delay(500);
  }
  throw new Error(`Server did not start within ${timeoutMs}ms: ${url}`);
}

function parseArgs(argv) {
  const out = {};
  for (const arg of argv.slice(2)) {
    if (!arg.startsWith('--')) continue;
    const [k, v] = arg.replace(/^--/, '').split('=');
    out[k] = v === undefined ? true : v;
  }
  return out;
}

function slugify(text) {
  return String(text || '')
    .normalize('NFKD').replace(/\p{Diacritic}+/gu, '')
    .toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '')
    .slice(0, 120) || 'article';
}

// ============================================================================
// Content-readiness helpers
// ============================================================================

async function waitForImages(page, timeoutMs = 15_000) {
  await page.evaluate(async (timeout) => {
    const deadline = Date.now() + timeout;
    const pending = Array.from(document.images || []).filter(i => !i.complete || i.naturalWidth === 0);
    await Promise.race([
      Promise.all(pending.map(i => new Promise(r => {
        if (i.complete && i.naturalWidth !== 0) return r();
        i.addEventListener('load', r, { once: true });
        i.addEventListener('error', r, { once: true });
      }))),
      new Promise(r => setTimeout(r, Math.max(0, deadline - Date.now())))
    ]);
  }, timeoutMs);
}

async function waitForD3(page, timeoutMs = 20_000) {
  try {
    await page.evaluate(async (timeout) => {
      const start = Date.now();
      const ready = () => {
        const hero = document.querySelector('.hero-banner');
        if (hero) return !!hero.querySelector('svg circle, svg path, svg rect, svg g');
        const c = [...document.querySelectorAll('.d3-line'), ...document.querySelectorAll('.d3-bar')];
        return !c.length || c.every(el => el.querySelector('svg circle, svg path, svg rect, svg g'));
      };
      while (!ready() && Date.now() - start < timeout) await new Promise(r => setTimeout(r, 200));
    }, timeoutMs);
  } catch {}
}

async function waitForPlotly(page, timeoutMs = 20_000) {
  try {
    await page.evaluate(async (timeout) => {
      const start = Date.now();
      while (!document.querySelectorAll('.js-plotly-plot').length && Date.now() - start < timeout)
        await new Promise(r => setTimeout(r, 200));
      const ok = () => Array.from(document.querySelectorAll('.js-plotly-plot')).every(e => e.querySelector('svg.main-svg'));
      while (!ok() && Date.now() - start < timeout) await new Promise(r => setTimeout(r, 200));
    }, timeoutMs);
  } catch {}
}

async function waitForHtmlEmbeds(page, timeoutMs = 15_000) {
  await page.evaluate(async (timeout) => {
    const start = Date.now();
    const ready = (e) => {
      try {
        if (!e.querySelector('svg, canvas, div[id^="frag-"]')) return false;
        for (const s of e.querySelectorAll('svg'))
          if (!s.querySelector('path, circle, rect, line, polygon, g')) return false;
        return true;
      } catch { return false; }
    };
    while (Date.now() - start < timeout) {
      const embeds = Array.from(document.querySelectorAll('.html-embed__card'));
      if (!embeds.length || embeds.every(ready)) break;
      await new Promise(r => setTimeout(r, 300));
    }
  }, timeoutMs);
}

async function waitForStableLayout(page, timeoutMs = 5_000) {
  const start = Date.now();
  let last = await page.evaluate(() => (document.scrollingElement || document.body).scrollHeight);
  let stable = 0;
  while (Date.now() - start < timeoutMs && stable < 3) {
    await page.waitForTimeout(250);
    const now = await page.evaluate(() => (document.scrollingElement || document.body).scrollHeight);
    if (now === last) stable++; else { stable = 0; last = now; }
  }
}

async function injectSvgViewBoxes(page) {
  return page.evaluate(() => {
    let fixed = 0, skipped = 0, errors = 0;
    document.querySelectorAll('.html-embed__card svg, [class^="d3-"] svg, [class*=" d3-"] svg').forEach(svg => {
      try {
        if (svg.getAttribute('viewBox')) { skipped++; return; }
        const r = svg.getBoundingClientRect();
        const w = r.width || parseFloat(svg.getAttribute('width')) || 0;
        const h = r.height || parseFloat(svg.getAttribute('height')) || 0;
        if (w > 0 && h > 0) {
          svg.setAttribute('viewBox', `0 0 ${Math.round(w)} ${Math.round(h)}`);
          svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
          svg.removeAttribute('width'); svg.removeAttribute('height');
          svg.style.width = '100%'; svg.style.height = 'auto'; svg.style.maxWidth = '100%';
          fixed++;
        } else skipped++;
      } catch { errors++; }
    });
    return { fixed, skipped, errors };
  });
}

async function openAllAccordions(page) {
  const count = await page.evaluate(() => {
    let opened = 0;
    document.querySelectorAll('details.accordion, details').forEach(d => {
      if (!d.hasAttribute('open')) {
        d.setAttribute('open', '');
        const w = d.querySelector('.accordion__content-wrapper');
        if (w) { w.style.height = 'auto'; w.style.overflow = 'visible'; }
        opened++;
      }
    });
    return opened;
  });
  if (count > 0) await waitForStableLayout(page, 2_000);
  return count;
}

// ============================================================================
// Table of Contents generation
// ============================================================================

async function injectTableOfContents(page) {
  return page.evaluate(() => {
    const headings = Array.from(document.querySelectorAll('main h2, main h3'));
    if (headings.length === 0) return 0;

    let chapterNum = 0;
    let sectionNum = 0;
    let tocHtml = '<nav class="book-toc" aria-label="Table of contents">';
    tocHtml += '<h2 class="book-toc__title">Contents</h2>';
    tocHtml += '<ol class="book-toc__list">';

    let currentChapterHtml = '';
    let sectionsHtml = '';

    const flushChapter = () => {
      if (currentChapterHtml) {
        if (sectionsHtml) {
          currentChapterHtml += `<ol class="book-toc__sections">${sectionsHtml}</ol>`;
          sectionsHtml = '';
        }
        tocHtml += currentChapterHtml + '</li>';
        currentChapterHtml = '';
      }
    };

    for (const h of headings) {
      const text = h.textContent.trim();
      if (!text) continue;

      if (h.tagName === 'H2') {
        flushChapter();
        chapterNum++;
        sectionNum = 0;
        currentChapterHtml = `<li class="book-toc__chapter">`;
        currentChapterHtml += `<span class="book-toc__number">${chapterNum}</span>`;
        currentChapterHtml += `<span class="book-toc__label">${text}</span>`;
      } else if (h.tagName === 'H3' && chapterNum > 0) {
        sectionNum++;
        sectionsHtml += `<li class="book-toc__section">`;
        sectionsHtml += `<span class="book-toc__section-number">${chapterNum}.${sectionNum}</span>`;
        sectionsHtml += `<span>${text}</span>`;
        sectionsHtml += `</li>`;
      }
    }

    flushChapter();
    tocHtml += '</ol></nav>';

    // Insert after meta header, before main content
    const meta = document.querySelector('header.meta');
    const main = document.querySelector('main');
    const insertTarget = meta || (main ? main.parentNode : document.body);

    if (meta && meta.nextSibling) {
      meta.parentNode.insertBefore(
        Object.assign(document.createElement('div'), { innerHTML: tocHtml }).firstChild,
        meta.nextSibling
      );
    } else if (main) {
      main.parentNode.insertBefore(
        Object.assign(document.createElement('div'), { innerHTML: tocHtml }).firstChild,
        main
      );
    }

    return chapterNum;
  });
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  const cwd = process.cwd();
  const port = Number(process.env.PREVIEW_PORT || 8080);
  const baseUrl = `http://127.0.0.1:${port}/`;
  const args = parseArgs(process.argv);

  const theme = (args.theme === 'dark' || args.theme === 'light') ? args.theme : 'light';
  const format = args.format || 'A4';
  const wait = args.wait || 'full';

  let outFileBase = (args.filename && String(args.filename).replace(/\.pdf$/i, '')) || '';

  // Build if needed
  const distDir = resolve(cwd, 'dist');
  let hasDist = false;
  try { const st = await fs.stat(distDir); hasDist = st?.isDirectory(); } catch {}
  if (!hasDist) {
    console.log('📦 Building Astro site…');
    await run('npm', ['run', 'build']);
  } else {
    console.log('✓ dist/ exists, skipping build');
  }

  // Start preview
  console.log('🚀 Starting preview server…');
  const preview = spawn('npm', ['run', 'preview'], { cwd, stdio: 'inherit', detached: true });
  const previewExit = new Promise(r => preview.on('close', (code, signal) => r({ code, signal })));

  try {
    await waitForServer(baseUrl, 60_000);
    console.log('✓ Server ready');

    const browser = await chromium.launch({ headless: true });
    try {
      const context = await browser.newContext();
      await context.addInitScript((desired) => {
        try {
          localStorage.setItem('theme', desired);
          if (document?.documentElement) document.documentElement.dataset.theme = desired;
        } catch {}
      }, theme);

      const page = await context.newPage();
      await page.setViewportSize({ width: 1200, height: 1600 });

      console.log('📄 Loading page…');
      await page.goto(baseUrl, { waitUntil: 'load', timeout: 60_000 });
      try { await page.waitForFunction(() => !!window.d3, { timeout: 8_000 }); } catch {}
      try { await page.waitForFunction(() => !!window.Plotly, { timeout: 5_000 }); } catch {}

      if (!outFileBase) {
        const fromBtn = await page.evaluate(() => {
          const btn = document.getElementById('download-pdf-btn');
          return btn?.getAttribute('data-pdf-filename') || '';
        });
        if (fromBtn) {
          outFileBase = String(fromBtn).replace(/\.pdf$/i, '') + '-book';
        } else {
          const title = await page.evaluate(() => {
            const h1 = document.querySelector('h1.hero-title');
            return (h1?.textContent || document.title || '').replace(/\s+/g, ' ').trim();
          });
          outFileBase = slugify(title) + '-book';
        }
      }

      // Scroll entire page to trigger lazy-loaded content (IntersectionObserver)
      console.log('📜 Scrolling page to trigger lazy-loaded content…');
      await page.evaluate(async () => {
        const step = window.innerHeight * 0.8;
        const max = document.body.scrollHeight;
        for (let y = 0; y < max; y += step) {
          window.scrollTo(0, y);
          await new Promise(r => setTimeout(r, 80));
        }
        window.scrollTo(0, 0);
      });
      await page.waitForTimeout(3000);

      // Wait for all content
      if (wait === 'images' || wait === 'full') { console.log('⏳ Images…'); await waitForImages(page); }
      if (wait === 'd3' || wait === 'full') { console.log('⏳ D3…'); await waitForD3(page); }
      if (wait === 'plotly' || wait === 'full') { console.log('⏳ Plotly…'); await waitForPlotly(page); }
      if (wait === 'full') {
        console.log('⏳ HTML embeds…');
        await waitForHtmlEmbeds(page);
        await waitForStableLayout(page);
      }

      // Prepare content
      console.log('📂 Opening accordions…');
      const acc = await openAllAccordions(page);
      console.log(`   ${acc} accordion(s) opened`);

      console.log('🔧 Fixing SVG viewBoxes…');
      const vb = await injectSvgViewBoxes(page);
      console.log(`   Fixed: ${vb.fixed}, Skipped: ${vb.skipped}`);

      // Replace embeds with pre-captured screenshots from screenshot-elements.mjs
      const screenshotsDir = resolve(cwd, 'screenshots');
      let hasScreenshots = false;
      try { const st = await fs.stat(screenshotsDir); hasScreenshots = st?.isDirectory(); } catch {}

      if (hasScreenshots) {
        console.log('📸 Loading pre-captured screenshots…');
        const pngFiles = (await fs.readdir(screenshotsDir)).filter(f => f.endsWith('.png'));
        const embedScreenshots = new Map();
        const MIN_SCREENSHOT_BYTES = 10_000; // Skip tiny screenshots (likely just UI controls)
        for (const f of pngFiles) {
          if (/--option-|--combo-|--btn-|--all-options|--open-select|--cb-/.test(f)) continue;
          const match = f.match(/^(\d+)-embed--(.+)\.png$/);
          if (match) {
            const filePath = join(screenshotsDir, f);
            const stat = await fs.stat(filePath);
            if (stat.size < MIN_SCREENSHOT_BYTES) {
              console.log(`   ⏭️  Skipping tiny screenshot: ${f} (${stat.size} bytes)`);
              continue;
            }
            embedScreenshots.set(match[2], filePath);
          }
        }
        console.log(`   ${embedScreenshots.size} embed screenshot(s) available`);

        const replaceStats = await page.evaluate(async (screenshotEntries) => {
          let replaced = 0, hidden = 0;
          const embeds = document.querySelectorAll('.html-embed');

          for (const embed of embeds) {
            const card = embed.querySelector('.html-embed__card');
            if (!card) continue;

            const titleEl = embed.querySelector('.html-embed__title');
            const titleText = titleEl?.textContent || '';
            const btn = embed.querySelector('.html-embed__download');
            const filename = btn?.getAttribute('data-filename') || '';
            const slugify = (t) => String(t || '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');

            // Try title first (screenshot-elements.mjs uses title), then data-filename
            const slugFromTitle = slugify(titleText);
            const slugFromFile = slugify(filename);
            const entry = screenshotEntries.find(([key]) => key === slugFromTitle || key === slugFromFile);

            if (entry) {
              const [, b64] = entry;
              card.innerHTML = `<img src="data:image/png;base64,${b64}" style="width:100%;height:auto;display:block;" />`;
              card.style.padding = '0';
              card.style.border = 'none';
              card.style.background = 'transparent';
              card.style.borderRadius = '0';
              replaced++;
            } else {
              // No screenshot — hide unless it's inside the hero/banner
              const inHero = embed.closest('.hero, .hero-banner');
              if (inHero) {
                // Keep hero/banner embeds even without screenshot
              } else {
                embed.style.display = 'none';
                hidden++;
              }
            }
          }
          return { replaced, hidden };
        }, await Promise.all(
          Array.from(embedScreenshots.entries()).map(async ([slug, filePath]) => {
            const buf = await fs.readFile(filePath);
            return [slug, buf.toString('base64')];
          })
        ));
        console.log(`   ${replaceStats.replaced} embed(s) replaced with screenshots, ${replaceStats.hidden} empty embed(s) hidden`);
      } else {
        console.log('⚠️  No screenshots/ folder found. Run "npm run export:images" first for best results.');
        console.log('   Falling back to live rendering…');
      }

      // Force all iframes and wide embeds to fit; hide broken images
      await page.evaluate(() => {
        document.querySelectorAll('iframe').forEach(el => el.style.display = 'none');
        document.querySelectorAll('.html-embed--wide').forEach(el => {
          el.style.width = '100%';
          el.style.marginLeft = '0';
          el.style.transform = 'none';
          el.style.padding = '0';
        });
        // Hide images that failed to load (gray placeholders)
        document.querySelectorAll('img').forEach(img => {
          if (!img.complete || img.naturalWidth === 0) {
            const wrapper = img.closest('.image-wrapper, figure');
            if (wrapper) wrapper.style.display = 'none';
            else img.style.display = 'none';
          }
        });
      });

      // Number all figures sequentially
      console.log('🔢 Numbering figures…');
      const figureStats = await page.evaluate(() => {
        const mainEl = document.querySelector('main');
        if (!mainEl) return { figures: 0 };

        let figNum = 0;
        const figures = mainEl.querySelectorAll('figure');

        for (const fig of figures) {
          if (fig.closest('.hero') || fig.closest('header.meta') || fig.closest('.book-toc')) continue;
          if (fig.style.display === 'none' || fig.closest('[style*="display: none"]')) continue;

          // Skip nested figures (only process the deepest figure)
          if (fig.querySelector('figure')) continue;

          const desc = fig.querySelector('.html-embed__desc') || fig.querySelector('figcaption:last-of-type');
          const titleCap = fig.querySelector('.html-embed__title');

          const hasVisual = fig.querySelector('img, svg, canvas, .html-embed__card');
          if (!hasVisual && !desc) continue;

          figNum++;

          if (desc && desc !== titleCap) {
            let text = desc.textContent || '';
            const figMatch = text.match(/^\s*Figure\s+\d+(\.\d+)?\s*/i);
            if (figMatch) {
              const afterFig = text.slice(figMatch[0].length);
              const cleaned = afterFig.replace(/^[^a-zA-Z0-9(]*/, '');
              desc.textContent = '';
              desc.innerHTML = `<strong>Figure ${figNum}</strong> \u00B7 ${cleaned}`;
            } else {
              const html = desc.innerHTML;
              desc.innerHTML = `<strong>Figure ${figNum}</strong> \u00B7 ${html}`;
            }
          }
        }

        return { figures: figNum };
      });
      console.log(`   ${figureStats.figures} figure(s) numbered`);

      // Inject chapter/section numbers into headings (CSS counters unreliable in Chromium print)
      console.log('🔢 Numbering chapters and sections…');
      const numberingStats = await page.evaluate(() => {
        const mainEl = document.querySelector('main');
        if (!mainEl) return { chapters: 0, sections: 0 };

        let chapterNum = 0;
        let sectionNum = 0;
        let chapters = 0;
        let sections = 0;

        // Get all headings inside main (skip hero, meta, etc.)
        const headings = mainEl.querySelectorAll('h2, h3');

        for (const h of headings) {
          // Skip headings inside the abstract or that are part of special sections
          if (h.closest('.hero') || h.closest('header.meta') || h.closest('.book-toc')) continue;

          if (h.tagName === 'H2') {
            chapterNum++;
            sectionNum = 0;
            h.setAttribute('data-chapter-num', String(chapterNum));
            chapters++;
          } else if (h.tagName === 'H3') {
            sectionNum++;
            h.setAttribute('data-section-num', `${chapterNum}.${sectionNum}`);
            sections++;
          }
        }

        return { chapters, sections };
      });
      console.log(`   ${numberingStats.chapters} chapter(s), ${numberingStats.sections} section(s) numbered`);

      // Wrap chapter headings in dedicated full-page openers
      console.log('📖 Creating chapter opening pages…');
      const chapterPages = await page.evaluate(() => {
        const mainEl = document.querySelector('main');
        if (!mainEl) return 0;

        let count = 0;
        const h2s = Array.from(mainEl.querySelectorAll('h2[data-chapter-num]'));

        for (const h2 of h2s) {
          const wrapper = document.createElement('div');
          wrapper.className = 'book-chapter-opener';
          h2.parentNode.insertBefore(wrapper, h2);
          wrapper.appendChild(h2);
          count++;
        }

        return count;
      });
      console.log(`   ${chapterPages} chapter opener(s) created`);

      // Generate Table of Contents
      console.log('📋 Generating Table of Contents…');
      const tocChapters = await injectTableOfContents(page);
      console.log(`   ${tocChapters} chapter(s) in ToC`);

      // Activate print mode and inject book CSS
      await page.emulateMedia({ media: 'print' });

      console.log('📚 Applying book styles…');
      const bookCssPath = resolve(__dirname, '..', 'src', 'styles', '_print-book.css');
      const bookCss = await fs.readFile(bookCssPath, 'utf-8');
      await page.addStyleTag({ content: bookCss });
      await page.waitForTimeout(1_000);

      // Header & footer templates for page numbers
      const footerTemplate = `
        <div style="width: 100%; text-align: center; font-size: 9pt; font-family: 'Source Sans Pro', 'Helvetica Neue', sans-serif; color: #999; padding: 0 22mm;">
          <span class="pageNumber"></span> / <span class="totalPages"></span>
        </div>
      `;

      const headerTemplate = `<div style="font-size: 1px;"></div>`;

      // Generate PDF
      const outPath = resolve(cwd, 'dist', `${outFileBase}.pdf`);
      console.log('🖨️  Generating PDF…');

      await page.pdf({
        path: outPath,
        format,
        printBackground: true,
        displayHeaderFooter: true,
        headerTemplate,
        footerTemplate,
        preferCSSPageSize: false,
        margin: { top: '22mm', right: '22mm', bottom: '28mm', left: '22mm' },
      });

      const stats = await fs.stat(outPath);
      const sizeKB = Math.round(stats.size / 1024);
      console.log(`✅ PDF generated: ${outPath} (${sizeKB} KB)`);

      if (sizeKB < 10) console.warn('⚠️  PDF very small – content might be missing');

      const publicPath = resolve(cwd, 'public', `${outFileBase}.pdf`);
      try {
        await fs.mkdir(resolve(cwd, 'public'), { recursive: true });
        await fs.copyFile(outPath, publicPath);
        console.log(`✅ Copied to: ${publicPath}`);
      } catch (e) { console.warn('⚠️  Copy failed:', e?.message); }

    } finally {
      await browser.close();
    }
  } finally {
    console.log('🛑 Stopping preview server…');
    try {
      if (process.platform !== 'win32') { try { process.kill(-preview.pid, 'SIGINT'); } catch {} }
      try { preview.kill('SIGINT'); } catch {}
      await Promise.race([previewExit, delay(3_000)]);
      if (!preview.killed) {
        if (process.platform !== 'win32') { try { process.kill(-preview.pid, 'SIGKILL'); } catch {} }
        try { preview.kill('SIGKILL'); } catch {}
        await Promise.race([previewExit, delay(1_000)]);
      }
    } catch {}
  }

  console.log('');
  console.log('╔═══════════════════════════════════════════════════════════════╗');
  console.log('║            📚 BOOK PDF GENERATED 📚                          ║');
  console.log('╚═══════════════════════════════════════════════════════════════╝');
  console.log('');
}

main().catch(err => { console.error('❌', err); process.exit(1); });

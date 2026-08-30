#!/usr/bin/env node
/**
 * Export PDF Book with Paged.js
 *
 * Generates a professional book-quality PDF using Playwright + Paged.js polyfill.
 * Paged.js implements W3C CSS Paged Media specs that browsers don't support natively:
 *   - @page :left / :right (alternating margins for binding)
 *   - Running headers via string-set
 *   - Footnotes via float: footnote
 *   - Named pages (@page chapter)
 *   - target-counter() for ToC page numbers
 *   - bookmark-level for PDF outline
 *
 * Usage:
 *   npm run export:pdf:book
 *   npm run export:pdf:book -- --theme=light --format=A4
 *
 * Options:
 *   --theme=light|dark   Color theme (default: light)
 *   --format=A4|Letter   Page format (default: A4)
 *   --filename=xxx       Output filename (default: <title>-book)
 *   --wait=full          Wait strategy (default: full)
 *   --debug              Keep browser open for inspection
 */

import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';
import { chromium } from 'playwright';
import { resolve, dirname } from 'node:path';
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
// Content-readiness helpers (ported from export-pdf.mjs)
// ============================================================================

async function waitForImages(page, timeoutMs = 15_000) {
  await page.evaluate(async (timeout) => {
    const deadline = Date.now() + timeout;
    const imgs = Array.from(document.images || []);
    const pending = imgs.filter(i => !i.complete || i.naturalWidth === 0);
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
        const containers = [
          ...document.querySelectorAll('.d3-line'),
          ...document.querySelectorAll('.d3-bar'),
          ...document.querySelectorAll('[class^="d3-"]'),
        ];
        return !containers.length || containers.every(c => c.querySelector('svg circle, svg path, svg rect, svg g'));
      };
      while (!ready() && Date.now() - start < timeout) await new Promise(r => setTimeout(r, 200));
    }, timeoutMs);
  } catch {}
}

async function waitForPlotly(page, timeoutMs = 20_000) {
  try {
    await page.evaluate(async (timeout) => {
      const start = Date.now();
      const has = () => document.querySelectorAll('.js-plotly-plot').length > 0;
      while (!has() && Date.now() - start < timeout) await new Promise(r => setTimeout(r, 200));
      const ok = () => Array.from(document.querySelectorAll('.js-plotly-plot')).every(e => e.querySelector('svg.main-svg'));
      while (!ok() && Date.now() - start < timeout) await new Promise(r => setTimeout(r, 200));
    }, timeoutMs);
  } catch {}
}

async function waitForHtmlEmbeds(page, timeoutMs = 15_000) {
  await page.evaluate(async (timeout) => {
    const start = Date.now();
    const isReady = (embed) => {
      try {
        const has = embed.querySelector('svg, canvas, div[id^="frag-"]');
        if (!has) return false;
        for (const svg of embed.querySelectorAll('svg')) {
          if (!svg.querySelector('path, circle, rect, line, polygon, g')) return false;
        }
        return true;
      } catch { return false; }
    };
    while (Date.now() - start < timeout) {
      const embeds = Array.from(document.querySelectorAll('.html-embed__card'));
      if (!embeds.length || embeds.every(isReady)) break;
      await new Promise(r => setTimeout(r, 300));
    }
  }, timeoutMs);
}

async function waitForStableLayout(page, timeoutMs = 5_000) {
  const start = Date.now();
  let last = await page.evaluate(() =>
    (document.scrollingElement || document.body).scrollHeight
  );
  let stable = 0;
  while (Date.now() - start < timeoutMs && stable < 3) {
    await page.waitForTimeout(250);
    const now = await page.evaluate(() =>
      (document.scrollingElement || document.body).scrollHeight
    );
    if (now === last) stable++; else { stable = 0; last = now; }
  }
}

// ============================================================================
// SVG viewBox injection (ensures SVGs scale properly after Paged.js reflows)
// ============================================================================

async function injectSvgViewBoxes(page) {
  return page.evaluate(() => {
    let fixed = 0, skipped = 0, errors = 0;
    const sel = [
      '.html-embed__card svg', '[class^="d3-"] svg', '[class*=" d3-"] svg',
    ].join(', ');
    document.querySelectorAll(sel).forEach(svg => {
      try {
        if (svg.getAttribute('viewBox')) { skipped++; return; }
        const r = svg.getBoundingClientRect();
        const w = r.width || svg.clientWidth || parseFloat(svg.getAttribute('width')) || 0;
        const h = r.height || svg.clientHeight || parseFloat(svg.getAttribute('height')) || 0;
        if (w > 0 && h > 0) {
          svg.setAttribute('viewBox', `0 0 ${Math.round(w)} ${Math.round(h)}`);
          svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
          svg.removeAttribute('width');
          svg.removeAttribute('height');
          svg.style.width = '100%';
          svg.style.height = 'auto';
          svg.style.maxWidth = '100%';
          fixed++;
        } else { skipped++; }
      } catch { errors++; }
    });
    return { fixed, skipped, errors };
  });
}

// ============================================================================
// Open all accordions so their content is visible in the book
// ============================================================================

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
  const debug = !!args.debug;

  let outFileBase = (args.filename && String(args.filename).replace(/\.pdf$/i, '')) || '';

  // -- Build if needed -------------------------------------------------------
  const distDir = resolve(cwd, 'dist');
  let hasDist = false;
  try { const st = await fs.stat(distDir); hasDist = st?.isDirectory(); } catch {}
  if (!hasDist) {
    console.log('📦 Building Astro site…');
    await run('npm', ['run', 'build']);
  } else {
    console.log('✓ dist/ exists, skipping build');
  }

  // -- Start preview server --------------------------------------------------
  console.log('🚀 Starting preview server…');
  const preview = spawn('npm', ['run', 'preview'], { cwd, stdio: 'inherit', detached: true });
  const previewExit = new Promise(r => preview.on('close', (code, signal) => r({ code, signal })));

  try {
    await waitForServer(baseUrl, 60_000);
    console.log('✓ Server ready');

    // -- Launch browser ------------------------------------------------------
    const browser = await chromium.launch({ headless: !debug });
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

      // -- Load page and wait for content ------------------------------------
      console.log('📄 Loading page…');
      await page.goto(baseUrl, { waitUntil: 'load', timeout: 60_000 });

      try { await page.waitForFunction(() => !!window.d3, { timeout: 8_000 }); } catch {}
      try { await page.waitForFunction(() => !!window.Plotly, { timeout: 5_000 }); } catch {}

      // Derive filename from page if not provided
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

      if (wait === 'images' || wait === 'full') {
        console.log('⏳ Waiting for images…');
        await waitForImages(page);
      }
      if (wait === 'd3' || wait === 'full') {
        console.log('⏳ Waiting for D3…');
        await waitForD3(page);
      }
      if (wait === 'plotly' || wait === 'full') {
        console.log('⏳ Waiting for Plotly…');
        await waitForPlotly(page);
      }
      if (wait === 'full') {
        console.log('⏳ Waiting for HTML embeds…');
        await waitForHtmlEmbeds(page);
        await waitForStableLayout(page);
      }

      // -- Prepare content for book ------------------------------------------
      console.log('📂 Opening accordions…');
      const accordionCount = await openAllAccordions(page);
      console.log(`   ${accordionCount} accordion(s) opened`);

      console.log('🔧 Fixing SVG viewBox attributes…');
      const vb = await injectSvgViewBoxes(page);
      console.log(`   Fixed: ${vb.fixed}, Skipped: ${vb.skipped}, Errors: ${vb.errors}`);

      // Hide web-only UI elements before Paged.js processes the DOM
      await page.evaluate(() => {
        const hide = [
          '#theme-toggle', '.table-of-contents', '.toc-mobile-toggle',
          '.toc-mobile-backdrop', '.toc-mobile-sidebar', '.right-aside',
          'nav', '.code-lang-chip', '.meta-container-cell--pdf', 'button',
          '.interactive-only',
        ];
        hide.forEach(sel => {
          document.querySelectorAll(sel).forEach(el => { el.style.display = 'none'; });
        });
        // Force single column
        const grid = document.querySelector('.content-grid');
        if (grid) grid.style.gridTemplateColumns = '1fr';
        const main = document.querySelector('main');
        if (main) { main.style.maxWidth = 'none'; main.style.padding = '0'; }
      });

      // -- Prepare book CSS ---------------------------------------------------
      console.log('📚 Preparing book styles…');
      const bookCssPath = resolve(__dirname, '..', 'src', 'styles', '_print-book.css');
      const bookCss = await fs.readFile(bookCssPath, 'utf-8');
      // Strip @media print wrapper – Paged.js works in screen mode
      const unwrappedCss = bookCss
        .replace(/^@media\s+print\s*\{/, '')
        .replace(/\}\s*$/, '');

      // Neutralize external stylesheet references that cause CORS/XHR errors
      // in Paged.js. Inline their content and remove @import rules.
      console.log('🔗 Neutralizing external stylesheets for Paged.js…');
      const neutralized = await page.evaluate(async () => {
        let count = 0;
        // 1. Replace <link rel="stylesheet"> with inline <style>
        for (const link of Array.from(document.querySelectorAll('link[rel="stylesheet"]'))) {
          try {
            const href = link.href;
            if (!href || href.startsWith('data:')) continue;
            const res = await fetch(href);
            if (!res.ok) { link.remove(); count++; continue; }
            let css = await res.text();
            // Also resolve @import inside fetched CSS
            const importRe = /@import\s+(?:url\(\s*['"]?([^'")\s]+)['"]?\s*\)|['"]([^'"]+)['"]);?/g;
            let m;
            while ((m = importRe.exec(css)) !== null) {
              const url = m[1] || m[2];
              if (!url) continue;
              try {
                const r2 = await fetch(url);
                if (r2.ok) css = css.replace(m[0], await r2.text());
                else css = css.replace(m[0], '');
              } catch { css = css.replace(m[0], ''); }
            }
            const style = document.createElement('style');
            style.textContent = css;
            link.parentNode.insertBefore(style, link);
            link.remove();
            count++;
          } catch { link.remove(); count++; }
        }
        // 2. Remove @import from existing <style> elements
        for (const style of Array.from(document.querySelectorAll('style'))) {
          const text = style.textContent || '';
          if (/@import\s/.test(text)) {
            const importRe = /@import\s+(?:url\(\s*['"]?([^'")\s]+)['"]?\s*\)|['"]([^'"]+)['"]);?/g;
            let css = text;
            let m;
            while ((m = importRe.exec(text)) !== null) {
              const url = m[1] || m[2];
              if (!url) continue;
              try {
                const r = await fetch(url);
                if (r.ok) css = css.replace(m[0], await r.text());
                else css = css.replace(m[0], '');
              } catch { css = css.replace(m[0], ''); }
            }
            if (css !== text) { style.textContent = css; count++; }
          }
        }
        return count;
      });
      console.log(`   ${neutralized} reference(s) neutralized`);

      // -- Inject and run Paged.js -------------------------------------------
      console.log('📖 Injecting Paged.js…');
      const pagedJsPath = resolve(cwd, 'node_modules', 'pagedjs', 'dist', 'paged.polyfill.js');
      try { await fs.access(pagedJsPath); } catch {
        throw new Error('pagedjs not found. Run: npm install --save-dev pagedjs');
      }

      // Capture page logs for debugging
      const pageErrors = [];
      page.on('console', msg => {
        const text = msg.text();
        if (msg.type() === 'error') {
          pageErrors.push(text);
          console.log(`   [page] ${text.slice(0, 150)}`);
        } else if (text.startsWith('[XHR]')) {
          console.log(`   ${text}`);
        }
      });
      page.on('pageerror', err => {
        pageErrors.push(err.message);
        console.log(`   [exception] ${err.message.slice(0, 150)}`);
      });

      // Isolate Paged.js: temporarily remove all page styles so Paged.js only
      // processes the book CSS. This avoids crashes on modern CSS features
      // (OKLCH, color-mix, @custom-media) that Paged.js 0.4 can't parse.
      console.log('🧹 Isolating styles for Paged.js…');
      const styleCount = await page.evaluate(() => {
        window.__savedStyles = [];
        const styles = Array.from(document.querySelectorAll('style, link[rel="stylesheet"]'));
        styles.forEach((el, i) => {
          window.__savedStyles.push({
            tag: el.tagName,
            html: el.outerHTML,
            parent: el.parentNode,
            next: el.nextSibling,
          });
          el.remove();
        });
        return styles.length;
      });
      console.log(`   ${styleCount} stylesheet(s) temporarily removed`);

      // Inject ONLY the book CSS
      await page.addStyleTag({ content: unwrappedCss });

      // Patch XHR to handle any remaining cross-origin fetches gracefully
      await page.evaluate(() => {
        const OrigXHR = window.XMLHttpRequest;
        const origProto = OrigXHR.prototype;
        window.XMLHttpRequest = function() {
          const xhr = new OrigXHR();
          const origOpen = xhr.open.bind(xhr);
          const origSend = xhr.send.bind(xhr);
          let url = '';
          xhr.open = function(m, u, ...a) { url = String(u); return origOpen(m, u, ...a); };
          xhr.send = function(...a) {
            if (/^https?:\/\//.test(url) && !url.startsWith(location.origin)) {
              Object.defineProperty(xhr, 'status', { value: 200, writable: false });
              Object.defineProperty(xhr, 'responseText', { value: '', writable: false });
              Object.defineProperty(xhr, 'response', { value: '', writable: false });
              Object.defineProperty(xhr, 'readyState', { value: 4, writable: false });
              setTimeout(() => { if (xhr.onload) xhr.onload(new Event('load')); }, 0);
              return;
            }
            return origSend(...a);
          };
          return xhr;
        };
        window.XMLHttpRequest.prototype = origProto;
      });

      // Let Paged.js auto-start with only the book CSS visible
      await page.evaluate(() => { window.PagedConfig = { auto: true }; });
      await page.addScriptTag({ path: pagedJsPath });

      console.log('⏳ Running Paged.js pagination…');

      const paginationResult = await page.waitForFunction(() => {
        const pages = document.querySelectorAll('.pagedjs_page');
        return pages && pages.length > 0;
      }, { timeout: 120_000 }).then(async () => {
        const count = await page.evaluate(() =>
          document.querySelectorAll('.pagedjs_page').length
        );
        return { ok: true, pages: count };
      }).catch(async (e) => {
        return { ok: false, error: e.message };
      });

      // Restore original stylesheets into the paginated DOM
      if (paginationResult.ok) {
        console.log('🔄 Restoring page styles…');
        await page.evaluate(() => {
          const head = document.head;
          for (const saved of (window.__savedStyles || [])) {
            const tmp = document.createElement('div');
            tmp.innerHTML = saved.html;
            const el = tmp.firstChild;
            if (el) head.appendChild(el);
          }
        });
        await page.waitForTimeout(500);
      }

      let pagedJsWorked = false;

      if (paginationResult.ok && paginationResult.pages > 0) {
        pagedJsWorked = true;
        console.log(`✓ Paged.js pagination: ${paginationResult.pages} pages`);
      } else {
        const reason = paginationResult.error || `${paginationResult.pages} pages produced`;
        console.warn(`⚠️  Paged.js failed: ${reason}`);
        console.log('   Reloading page for browser-native fallback…');

        // Paged.js may have destroyed the DOM – reload for a clean state
        await page.goto(baseUrl, { waitUntil: 'load', timeout: 60_000 });
        try { await page.waitForFunction(() => !!window.d3, { timeout: 8_000 }); } catch {}
        if (wait === 'full') {
          await waitForImages(page);
          await waitForD3(page);
          await waitForHtmlEmbeds(page);
          await waitForStableLayout(page);
        }
        await openAllAccordions(page);
        await injectSvgViewBoxes(page);
        await page.emulateMedia({ media: 'print' });
        await page.addStyleTag({ content: bookCss });
        await page.waitForTimeout(500);
      }

      // Extra stabilization
      await page.waitForTimeout(2_000);

      const pageInfo = { pagedJsWorked };

      // -- Generate PDF ------------------------------------------------------
      const outPath = resolve(cwd, 'dist', `${outFileBase}.pdf`);
      console.log('🖨️  Generating PDF…');

      if (pageInfo.pagedJsWorked) {
        // Paged.js handled pagination – it created its own page elements.
        // Use preferCSSPageSize + zero margins so Paged.js controls everything.
        await page.pdf({
          path: outPath,
          format,
          printBackground: true,
          preferCSSPageSize: true,
          margin: { top: '0mm', right: '0mm', bottom: '0mm', left: '0mm' },
        });
      } else {
        // Fallback: Paged.js didn't work – use browser print with book margins
        console.log('   Using browser-native pagination fallback');
        await page.emulateMedia({ media: 'print' });
        await page.waitForTimeout(500);
        await page.pdf({
          path: outPath,
          format,
          printBackground: true,
          preferCSSPageSize: false,
          margin: { top: '20mm', right: '20mm', bottom: '25mm', left: '25mm' },
        });
      }

      const stats = await fs.stat(outPath);
      const sizeKB = Math.round(stats.size / 1024);
      console.log(`✅ PDF generated: ${outPath} (${sizeKB} KB)`);

      if (sizeKB < 10) {
        console.warn('⚠️  PDF is very small – content might be missing');
      }

      // Copy to public/
      const publicPath = resolve(cwd, 'public', `${outFileBase}.pdf`);
      try {
        await fs.mkdir(resolve(cwd, 'public'), { recursive: true });
        await fs.copyFile(outPath, publicPath);
        console.log(`✅ Copied to: ${publicPath}`);
      } catch (e) {
        console.warn('⚠️  Unable to copy to public/:', e?.message);
      }

      if (debug) {
        console.log('🔍 Debug mode – browser stays open. Press Ctrl+C to exit.');
        await new Promise(() => {});
      }

    } finally {
      if (!debug) await browser.close();
    }
  } finally {
    // Shutdown preview server
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
  console.log('║                  📚 BOOK PDF GENERATED 📚                    ║');
  console.log('╚═══════════════════════════════════════════════════════════════╝');
  console.log('');
}

main().catch(err => { console.error('❌', err); process.exit(1); });

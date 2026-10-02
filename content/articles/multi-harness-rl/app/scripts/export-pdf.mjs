#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';
import { chromium } from 'playwright';
import { resolve } from 'node:path';
import { promises as fs } from 'node:fs';
import process from 'node:process';

async function run(command, args = [], options = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, { stdio: 'inherit', shell: false, ...options });
    child.on('error', reject);
    child.on('exit', (code) => {
      if (code === 0) resolvePromise(undefined);
      else reject(new Error(`${command} ${args.join(' ')} exited with code ${code}`));
    });
  });
}

async function waitForServer(url, timeoutMs = 60000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.ok) return;
    } catch { }
    await delay(500);
  }
  throw new Error(`Server did not start in time: ${url}`);
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
    .normalize('NFKD')
    .replace(/\p{Diacritic}+/gu, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 120) || 'article';
}

function parseMargin(margin) {
  if (!margin) return { top: '12mm', right: '12mm', bottom: '16mm', left: '12mm' };
  const parts = String(margin).split(',').map(s => s.trim()).filter(Boolean);
  if (parts.length === 1) {
    return { top: parts[0], right: parts[0], bottom: parts[0], left: parts[0] };
  }
  if (parts.length === 2) {
    return { top: parts[0], right: parts[1], bottom: parts[0], left: parts[1] };
  }
  if (parts.length === 3) {
    return { top: parts[0], right: parts[1], bottom: parts[2], left: parts[1] };
  }
  return { top: parts[0] || '12mm', right: parts[1] || '12mm', bottom: parts[2] || '16mm', left: parts[3] || '12mm' };
}

function cssLengthToMm(val) {
  if (!val) return 0;
  const s = String(val).trim();
  if (/mm$/i.test(s)) return parseFloat(s);
  if (/cm$/i.test(s)) return parseFloat(s) * 10;
  if (/in$/i.test(s)) return parseFloat(s) * 25.4;
  if (/px$/i.test(s)) return (parseFloat(s) / 96) * 25.4; // 96 CSS px per inch
  const num = parseFloat(s);
  return Number.isFinite(num) ? num : 0; // assume mm if unitless
}

function getFormatSizeMm(format) {
  const f = String(format || 'A4').toLowerCase();
  switch (f) {
    case 'letter': return { w: 215.9, h: 279.4 };
    case 'legal': return { w: 215.9, h: 355.6 };
    case 'a3': return { w: 297, h: 420 };
    case 'tabloid': return { w: 279.4, h: 431.8 };
    case 'a4':
    default: return { w: 210, h: 297 };
  }
}

async function waitForImages(page, timeoutMs = 15000) {
  await page.evaluate(async (timeout) => {
    const deadline = Date.now() + timeout;
    const imgs = Array.from(document.images || []);
    const unloaded = imgs.filter(img => !img.complete || (img.naturalWidth === 0));
    await Promise.race([
      Promise.all(unloaded.map(img => new Promise(res => {
        if (img.complete && img.naturalWidth !== 0) return res(undefined);
        img.addEventListener('load', () => res(undefined), { once: true });
        img.addEventListener('error', () => res(undefined), { once: true });
      }))),
      new Promise(res => setTimeout(res, Math.max(0, deadline - Date.now())))
    ]);
  }, timeoutMs);
}

async function waitForPlotly(page, timeoutMs = 20000) {
  try {
    await page.evaluate(async (timeout) => {
      const start = Date.now();
      const hasPlots = () => Array.from(document.querySelectorAll('.js-plotly-plot')).length > 0;
      // Wait until plots exist or timeout
      while (!hasPlots() && (Date.now() - start) < timeout) {
        await new Promise(r => setTimeout(r, 200));
      }
      const deadline = start + timeout;
      // Then wait until each plot contains the main svg
      const allReady = () => Array.from(document.querySelectorAll('.js-plotly-plot')).every(el => el.querySelector('svg.main-svg'));
      while (!allReady() && Date.now() < deadline) {
        await new Promise(r => setTimeout(r, 200));
      }
      console.log('Plotly ready or timeout');
    }, timeoutMs);
  } catch (e) {
    console.warn('waitForPlotly timeout or error:', e.message);
  }
}

async function waitForD3(page, timeoutMs = 20000) {
  try {
    await page.evaluate(async (timeout) => {
      const start = Date.now();
      const isReady = () => {
        // Prioritize hero banner if present (generic container)
        const hero = document.querySelector('.hero-banner');
        if (hero) {
          return !!hero.querySelector('svg circle, svg path, svg rect, svg g');
        }
        // Else require all D3 containers on page to have shapes
        const containers = [
          ...Array.from(document.querySelectorAll('.d3-line')),
          ...Array.from(document.querySelectorAll('.d3-bar'))
        ];
        if (!containers.length) return true;
        return containers.every(c => c.querySelector('svg circle, svg path, svg rect, svg g'));
      };
      while (!isReady() && (Date.now() - start) < timeout) {
        await new Promise(r => setTimeout(r, 200));
      }
      console.log('D3 ready or timeout');
    }, timeoutMs);
  } catch (e) {
    console.warn('waitForD3 timeout or error:', e.message);
  }
}

async function waitForStableLayout(page, timeoutMs = 5000) {
  const start = Date.now();
  let last = await page.evaluate(() => document.scrollingElement ? document.scrollingElement.scrollHeight : document.body.scrollHeight);
  let stableCount = 0;
  while ((Date.now() - start) < timeoutMs && stableCount < 3) {
    await page.waitForTimeout(250);
    const now = await page.evaluate(() => document.scrollingElement ? document.scrollingElement.scrollHeight : document.body.scrollHeight);
    if (now === last) stableCount += 1; else { stableCount = 0; last = now; }
  }
}

/**
 * Make all remaining SVGs, Mermaid diagrams, and iframes responsive for print.
 * Small icon SVGs are locked to their current pixel size to prevent breakage.
 */
async function makeMediaResponsive(page) {
  await page.evaluate(() => {
    function isSmallSvg(svg) {
      try {
        const vb = svg?.viewBox?.baseVal;
        if (vb && vb.width <= 50 && vb.height <= 50) return true;
        const r = svg.getBoundingClientRect?.();
        if (r && r.width <= 50 && r.height <= 50) return true;
      } catch { }
      return false;
    }
    function lockSmallSvgSize(svg) {
      try {
        const r = svg.getBoundingClientRect?.();
        if (r?.width) svg.style.setProperty('width', Math.round(r.width) + 'px', 'important');
        if (r?.height) svg.style.setProperty('height', Math.round(r.height) + 'px', 'important');
        svg.style.setProperty('max-width', 'none', 'important');
      } catch { }
    }
    function fixSvg(svg) {
      if (!svg) return;
      if (isSmallSvg(svg)) { lockSmallSvgSize(svg); return; }
      if (!svg.getAttribute('viewBox')) {
        const rect = svg.getBoundingClientRect();
        const w = rect.width || svg.clientWidth || parseFloat(svg.getAttribute('width')) || 0;
        const h = rect.height || svg.clientHeight || parseFloat(svg.getAttribute('height')) || 0;
        if (w > 0 && h > 0) svg.setAttribute('viewBox', `0 0 ${Math.round(w)} ${Math.round(h)}`);
      }
      try { svg.removeAttribute('width'); } catch { }
      try { svg.removeAttribute('height'); } catch { }
      svg.style.maxWidth = '100%';
      svg.style.width = '100%';
      svg.style.height = 'auto';
      if (!svg.getAttribute('preserveAspectRatio')) svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    }

    document.querySelectorAll('svg').forEach(fixSvg);
    document.querySelectorAll('.mermaid, .mermaid svg').forEach(el => {
      if (el.tagName?.toLowerCase() === 'svg') fixSvg(el);
      else { el.style.display = 'block'; el.style.width = '100%'; el.style.maxWidth = '100%'; }
    });
    document.querySelectorAll('iframe, embed, object').forEach(el => {
      el.style.width = '100%';
      el.style.maxWidth = '100%';
      try { el.removeAttribute('width'); } catch { }
      try {
        const doc = el.contentDocument;
        if (doc?.head) {
          const s = doc.createElement('style');
          s.textContent = 'html,body{overflow-x:hidden} svg,canvas,img,video{max-width:100%!important;height:auto!important} svg[width]{width:100%!important}';
          doc.head.appendChild(s);
          doc.querySelectorAll('svg').forEach(svg => { if (isSmallSvg(svg)) lockSmallSvgSize(svg); else fixSvg(svg); });
        }
      } catch { }
    });
  });
}

/**
 * Screenshot every visible .html-embed and replace its content with a static
 * <img>. This freezes embeds at their current (web-quality) render, preventing
 * D3/Plotly re-render issues when the viewport changes for PDF generation.
 */
async function screenshotAndReplaceEmbeds(page) {
  const handles = await page.$$('.html-embed');
  let replaced = 0;

  for (let i = 0; i < handles.length; i++) {
    const el = handles[i];
    try {
      const visible = await el.evaluate(node => {
        const r = node.getBoundingClientRect();
        return r.width > 10 && r.height > 10;
      });
      if (!visible) continue;

      await el.scrollIntoViewIfNeeded();
      await page.waitForTimeout(30);

      const buf = await el.screenshot({ type: 'png' });
      const b64 = buf.toString('base64');

      await el.evaluate((node, data) => {
        const r = node.getBoundingClientRect();
        const img = document.createElement('img');
        img.src = 'data:image/png;base64,' + data;
        img.style.cssText = 'width:100%;height:auto;display:block;max-width:100%;';
        img.setAttribute('width', String(Math.round(r.width)));
        img.setAttribute('height', String(Math.round(r.height)));
        node.innerHTML = '';
        node.appendChild(img);
      }, b64);

      replaced++;
    } catch (e) {
      // Leave the original embed in place if screenshotting fails
    }
  }

  return replaced;
}

async function main() {
  const cwd = process.cwd();
  const port = Number(process.env.PREVIEW_PORT || 8080);
  const baseUrl = `http://127.0.0.1:${port}/`;
  const args = parseArgs(process.argv);
  // Default: light (do not rely on env vars implicitly)
  const theme = (args.theme === 'dark' || args.theme === 'light') ? args.theme : 'light';
  const format = args.format || 'A4';
  const margin = parseMargin(args.margin);
  const wait = (args.wait || 'full'); // 'networkidle' | 'images' | 'plotly' | 'full'
  const bookMode = !!args.book;

  // filename can be provided, else computed from DOM (button) or page title later
  let outFileBase = (args.filename && String(args.filename).replace(/\.pdf$/i, '')) || 'article';

  // Build only if dist/ does not exist
  const distDir = resolve(cwd, 'dist');
  let hasDist = false;
  try {
    const st = await fs.stat(distDir);
    hasDist = st && st.isDirectory();
  } catch { }
  if (!hasDist) {
    console.log('> Building Astro site…');
    await run('npm', ['run', 'build']);
  } else {
    console.log('> Skipping build (dist/ exists)…');
  }

  console.log('> Starting Astro preview…');
  // Start preview in its own process group so we can terminate all children reliably
  const preview = spawn('npm', ['run', 'preview'], { cwd, stdio: 'inherit', detached: true });
  const previewExit = new Promise((resolvePreview) => {
    preview.on('close', (code, signal) => resolvePreview({ code, signal }));
  });

  try {
    await waitForServer(baseUrl, 60000);
    console.log('> Server ready, generating PDF…');

    const browser = await chromium.launch({ headless: true });
    try {
      const context = await browser.newContext();
      await context.addInitScript((desired) => {
        try {
          localStorage.setItem('theme', desired);
          // Apply theme immediately to avoid flashes
          if (document && document.documentElement) {
            document.documentElement.dataset.theme = desired;
          }
        } catch { }
      }, theme);
      const page = await context.newPage();
      // Wider viewport so D3/Plotly embeds render at full web size before screenshotting
      const webViewportWidth = 1200;
      await page.setViewportSize({ width: webViewportWidth, height: 1400 });
      await page.goto(baseUrl, { waitUntil: 'load', timeout: 60000 });
      // Wait for CDN scripts (Plotly/D3) in parallel to halve the timeout cost
      await Promise.allSettled([
        page.waitForFunction(() => !!window.Plotly, { timeout: 8000 }),
        page.waitForFunction(() => !!window.d3, { timeout: 8000 })
      ]);
      // Prefer explicit filename from the download button if present
      if (!args.filename) {
        const fromBtn = await page.evaluate(() => {
          const btn = document.getElementById('download-pdf-btn');
          const f = btn ? btn.getAttribute('data-pdf-filename') : null;
          return f || '';
        });
        if (fromBtn) {
          outFileBase = String(fromBtn).replace(/\.pdf$/i, '');
        } else {
          // Fallback: compute slug from hero title or document.title
          const title = await page.evaluate(() => {
            const h1 = document.querySelector('h1.hero-title');
            const t = h1 ? h1.textContent : document.title;
            return (t || '').replace(/\s+/g, ' ').trim();
          });
          outFileBase = slugify(title);
        }
        if (bookMode) {
          outFileBase += '-book';
        }
      }

      // Wait for render readiness (images + D3 + Plotly in parallel)
      {
        const waits = [];
        if (wait === 'images' || wait === 'full') waits.push(waitForImages(page));
        if (wait === 'd3' || wait === 'full') waits.push(waitForD3(page));
        if (wait === 'plotly' || wait === 'full') waits.push(waitForPlotly(page));
        if (waits.length) {
          console.log('⏳ Waiting for content readiness (parallel)…');
          await Promise.all(waits);
        }
        if (wait === 'full') {
          console.log('⏳ Waiting for stable layout…');
          await waitForStableLayout(page);
        }
      }

      // Generate OG thumbnail BEFORE any print modifications (1200x630, screen mode)
      try {
        const savedViewport = { width: webViewportWidth, height: 1400 };
        await page.setViewportSize({ width: 1200, height: 630 });
        await page.waitForTimeout(200);
        await page.evaluate(() => window.scrollTo(0, 0));

        const cssHandle = await page.addStyleTag({
          content: `.hero .points { mix-blend-mode: normal !important; }`
        });
        const thumbPath = resolve(cwd, 'dist', 'thumb.auto.jpg');
        await page.screenshot({ path: thumbPath, type: 'jpeg', quality: 85, fullPage: false });
        const thumbPngPath = resolve(cwd, 'dist', 'thumb.auto.png');
        await page.screenshot({ path: thumbPngPath, type: 'png', fullPage: false });
        await fs.mkdir(resolve(cwd, 'public'), { recursive: true });
        try { await fs.copyFile(thumbPath, resolve(cwd, 'public', 'thumb.auto.jpg')); } catch { }
        try { await fs.copyFile(thumbPngPath, resolve(cwd, 'public', 'thumb.auto.png')); } catch { }
        try { await cssHandle.evaluate((el) => el.remove()); } catch { }
        console.log(`✅ OG thumbnail generated: ${thumbPath}`);

        // Restore viewport
        await page.setViewportSize(savedViewport);
        await page.evaluate(() => window.scrollTo(0, 0));
        await page.waitForTimeout(100);
      } catch (e) {
        console.warn('Unable to generate OG thumbnail:', e?.message || e);
      }

      // If --thumbnail-only, skip everything else
      if (args['thumbnail-only']) {
        console.log('🏁 Thumbnail-only mode, skipping PDF generation');
        return;
      }

      // Screenshot all embeds and replace them with static <img> tags.
      // This freezes the render at 1200px, preventing D3/Plotly re-render
      // issues when the viewport shrinks to print width later.
      console.log('📸 Screenshotting embeds…');
      const embedCount = await screenshotAndReplaceEmbeds(page);
      console.log(`   ${embedCount} embed(s) replaced with screenshots`);

      // Make remaining SVGs/iframes responsive (Mermaid diagrams, icons, etc.)
      console.log('🔧 Fixing remaining SVGs for print…');
      try { await makeMediaResponsive(page); } catch { }

      if (bookMode) {
        console.log('📂 Opening all accordions for book mode…');
        await page.evaluate(() => {
          const accordions = document.querySelectorAll('details.accordion, details');
          accordions.forEach((accordion) => {
            if (!accordion.hasAttribute('open')) {
              accordion.setAttribute('open', '');
              const wrapper = accordion.querySelector('.accordion__content-wrapper');
              if (wrapper) {
                wrapper.style.height = 'auto';
                wrapper.style.overflow = 'visible';
              }
            }
          });
        });
        await waitForStableLayout(page, 2000);
      }

      // Inject a print-friendly Table of Contents built from actual headings
      console.log('📑 Injecting Table of Contents for PDF…');
      const tocCount = await page.evaluate(() => {
        const main = document.querySelector('main');
        if (!main) return 0;
        const headings = Array.from(main.querySelectorAll('h2, h3'));
        if (headings.length === 0) return 0;

        const tocEl = document.createElement('nav');
        tocEl.className = 'pdf-toc';

        const title = document.createElement('h2');
        title.className = 'pdf-toc__title';
        title.textContent = 'Table of Contents';
        tocEl.appendChild(title);

        const list = document.createElement('ol');
        list.className = 'pdf-toc__list';

        let sectionNum = 0;
        let subNum = 0;
        let currentSubList = null;

        for (const h of headings) {
          const text = h.textContent?.trim();
          if (!text) continue;
          const id = h.getAttribute('id') || '';

          if (h.tagName === 'H2') {
            sectionNum++;
            subNum = 0;

            const li = document.createElement('li');
            li.className = 'pdf-toc__section';

            const a = document.createElement('a');
            a.className = 'pdf-toc__link';
            if (id) a.href = '#' + id;

            const num = document.createElement('span');
            num.className = 'pdf-toc__num';
            num.textContent = String(sectionNum);

            const label = document.createElement('span');
            label.className = 'pdf-toc__label';
            label.textContent = text;

            a.appendChild(num);
            a.appendChild(label);
            li.appendChild(a);

            const subList = document.createElement('ol');
            subList.className = 'pdf-toc__sublist';
            li.appendChild(subList);
            currentSubList = subList;

            list.appendChild(li);
          } else if (h.tagName === 'H3' && currentSubList) {
            subNum++;

            const li = document.createElement('li');
            li.className = 'pdf-toc__subsection';

            const a = document.createElement('a');
            a.className = 'pdf-toc__link';
            if (id) a.href = '#' + id;

            const num = document.createElement('span');
            num.className = 'pdf-toc__num';
            num.textContent = sectionNum + '.' + subNum;

            const label = document.createElement('span');
            label.className = 'pdf-toc__label';
            label.textContent = text;

            a.appendChild(num);
            a.appendChild(label);
            li.appendChild(a);

            currentSubList.appendChild(li);
          }
        }
        tocEl.appendChild(list);

        const style = document.createElement('style');
        style.textContent = `
          .pdf-toc {
            page-break-after: always;
            break-after: page;
            max-width: 600px;
            margin: 0 auto;
            padding: 40px 0 0;
            font-family: Source Sans Pro, ui-sans-serif, system-ui, sans-serif;
          }
          .pdf-toc__title {
            font-size: 28px;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #111827;
            margin: 0 0 32px;
            padding-bottom: 16px;
            border-bottom: 2px solid #111827;
          }
          .pdf-toc__list {
            list-style: none;
            padding: 0;
            margin: 0;
            counter-reset: none;
          }
          .pdf-toc__sublist {
            list-style: none;
            padding: 0;
            margin: 0;
          }
          .pdf-toc__section {
            margin: 0;
            padding: 0;
          }
          .pdf-toc__section > .pdf-toc__link {
            display: flex;
            align-items: baseline;
            gap: 12px;
            padding: 10px 0;
            text-decoration: none;
            border-bottom: 1px solid rgba(0,0,0,0.08);
          }
          .pdf-toc__section > .pdf-toc__link .pdf-toc__num {
            flex-shrink: 0;
            width: 24px;
            font-size: 13px;
            font-weight: 700;
            color: #6b7280;
          }
          .pdf-toc__section > .pdf-toc__link .pdf-toc__label {
            font-size: 15px;
            font-weight: 600;
            color: #111827;
            line-height: 1.4;
          }
          .pdf-toc__subsection {
            margin: 0;
            padding: 0;
          }
          .pdf-toc__subsection > .pdf-toc__link {
            display: flex;
            align-items: baseline;
            gap: 10px;
            padding: 5px 0 5px 36px;
            text-decoration: none;
          }
          .pdf-toc__subsection > .pdf-toc__link .pdf-toc__num {
            flex-shrink: 0;
            width: 32px;
            font-size: 12px;
            font-weight: 500;
            color: #9ca3af;
          }
          .pdf-toc__subsection > .pdf-toc__link .pdf-toc__label {
            font-size: 13px;
            font-weight: 400;
            color: #6b7280;
            line-height: 1.4;
          }
        `;
        document.head.appendChild(style);

        const meta = document.querySelector('header.meta');
        if (meta && meta.nextElementSibling) {
          meta.parentNode.insertBefore(tocEl, meta.nextElementSibling);
        } else {
          const contentGrid = document.querySelector('.content-grid');
          if (contentGrid) {
            contentGrid.parentNode.insertBefore(tocEl, contentGrid);
          }
        }
        return headings.length;
      });
      console.log(`   ${tocCount} headings indexed`);

      await page.emulateMedia({ media: 'print' });

      // Set viewport to printable width for PDF generation
      const outPath = resolve(cwd, 'dist', `${outFileBase}.pdf`);
      try {
        const fmt2 = getFormatSizeMm(format);
        const mw2 = fmt2.w - cssLengthToMm(margin.left) - cssLengthToMm(margin.right);
        const printableWidthPx = Math.max(320, Math.round((mw2 / 25.4) * 96));
        console.log(`📐 Setting viewport to printable width: ${printableWidthPx}px`);
        await page.setViewportSize({ width: printableWidthPx, height: 1400 });
        await page.evaluate(() => window.scrollTo(0, 0));
        await page.waitForTimeout(300);
        await waitForStableLayout(page, 2000);
      } catch { }

      // Inject styles for PDF
      let pdfCssHandle = null;
      try {
        if (bookMode) {
          console.log('📚 Applying book styles…');
          const bookCssPath = resolve(cwd, 'src', 'styles', '_print-book.css');
          const bookCss = await fs.readFile(bookCssPath, 'utf-8');
          pdfCssHandle = await page.addStyleTag({ content: bookCss });
          await page.waitForTimeout(500);
        } else {
          pdfCssHandle = await page.addStyleTag({
            content: `
            /* General container safety */
            html, body { overflow-x: hidden !important; }

            /* Make all media responsive for print */
            svg, canvas, img, video { max-width: 100% !important; height: auto !important; }
            .mermaid, .mermaid svg { display: block; width: 100% !important; max-width: 100% !important; height: auto !important; }
            svg[width] { width: 100% !important; }
            iframe, embed, object { width: 100% !important; max-width: 100% !important; height: auto; }

            /* Embeds are screenshots - just ensure they scale */
            .html-embed, .html-embed__card { max-width: 100% !important; width: 100% !important; }
            .html-embed img { width: 100% !important; height: auto !important; display: block; }

            /* --- Cover page: hero + meta centered, then page break --- */
            .hero {
              display: flex;
              flex-direction: column;
              justify-content: center;
              align-items: center;
              min-height: 50vh;
              padding-top: 15vh !important;
            }

            /* Banner: 80% width, centered */
            .hero-banner {
              width: 80% !important;
              margin-left: auto !important;
              margin-right: auto !important;
              overflow: hidden;
            }
            .hero-banner > * {
              width: 100% !important;
              height: auto !important;
              max-width: 100% !important;
              display: block;
            }

            /* Force page break after meta (end of cover page) */
            header.meta {
              page-break-after: always !important;
              break-after: page !important;
            }
          ` });
        }
      } catch { }
      await page.pdf({
        path: outPath,
        format,
        printBackground: true,
        displayHeaderFooter: false,
        preferCSSPageSize: false,
        margin: bookMode ? {
          top: '20mm',
          right: '20mm',
          bottom: '25mm',
          left: '25mm'
        } : margin
      });
      try { if (pdfCssHandle) await pdfCssHandle.evaluate((el) => el.remove()); } catch { }
      console.log(`✅ PDF generated: ${outPath}`);

      // Copy into public only under the slugified name
      const publicSlugPath = resolve(cwd, 'public', `${outFileBase}.pdf`);
      try {
        await fs.mkdir(resolve(cwd, 'public'), { recursive: true });
        await fs.copyFile(outPath, publicSlugPath);
        console.log(`✅ PDF copied to: ${publicSlugPath}`);
      } catch (e) {
        console.warn('Unable to copy PDF to public/:', e?.message || e);
      }
    } finally {
      await browser.close();
    }
  } finally {
    // Try a clean shutdown of preview (entire process group first)
    try {
      if (process.platform !== 'win32') {
        try { process.kill(-preview.pid, 'SIGINT'); } catch { }
      }
      try { preview.kill('SIGINT'); } catch { }
      await Promise.race([previewExit, delay(3000)]);
      // Force kill if still alive
      // eslint-disable-next-line no-unsafe-optional-chaining
      if (!preview.killed) {
        try {
          if (process.platform !== 'win32') {
            try { process.kill(-preview.pid, 'SIGKILL'); } catch { }
          }
          try { preview.kill('SIGKILL'); } catch { }
        } catch { }
        await Promise.race([previewExit, delay(1000)]);
      }
    } catch { }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

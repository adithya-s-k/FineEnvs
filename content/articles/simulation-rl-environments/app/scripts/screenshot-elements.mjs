import { chromium } from 'playwright';
import { mkdir, readFile, writeFile, rm } from 'fs/promises';
import { join } from 'path';
import sharp from 'sharp';
import looksSame from 'looks-same';

const URL = process.env.SCREENSHOT_URL || 'http://localhost:8080/';
const OUTPUT_DIR = './screenshots';
const DEVICE_SCALE_FACTOR = 4; // 4x for high-quality print
const BASE_VIEWPORT = { width: 1200, height: 800 };
const TRIM_THRESHOLD = 10; // sharp.trim() color tolerance
const MARGIN_PX = 15 * DEVICE_SCALE_FACTOR; // 15px margin around every screenshot
const SCREENSHOT_TIMEOUT_MS = Number(process.env.SCREENSHOT_TIMEOUT_MS) || 10_000;

const slugify = (value) =>
  String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');

// ─── Helper: clone an embed element into an isolated wrapper ────────────────
// This avoids visual contamination from overlapping DOM elements.
// Returns { wrapperId, cloneId } for locating/cleaning up.
async function cloneEmbed(page, element, idx) {
  return page.evaluate(([el, idx]) => {
    const wrapperId = `__embed-clone-wrapper-${idx}`;
    const cloneId = `__embed-clone-${idx}`;

    // Remove any previous clone
    const prev = document.getElementById(wrapperId);
    if (prev) prev.remove();

    // Create isolated wrapper at top-left of page
    const wrapper = document.createElement('div');
    wrapper.id = wrapperId;
    wrapper.style.cssText =
      'position:absolute;left:0;top:0;background:white;z-index:99999;isolation:isolate;';

    // Clone the inner card (or the whole element if no card)
    const inner = el.querySelector('.html-embed__card') || el;
    const clone = inner.cloneNode(true);
    clone.id = cloneId;
    clone.style.cssText = `background:white;border:none;border-radius:0;box-shadow:none;width:${el.getBoundingClientRect().width}px;`;

    wrapper.appendChild(clone);
    document.body.appendChild(wrapper);
    return { wrapperId, cloneId };
  }, [await element.evaluateHandle((el) => el), idx]);
}

// ─── Helper: remove a clone wrapper ─────────────────────────────────────────
async function removeClone(page, wrapperId) {
  await page.evaluate((id) => {
    const el = document.getElementById(id);
    if (el) el.remove();
  }, wrapperId);
}

// ─── Helper: screenshot + auto-trim whitespace ──────────────────────────────
async function screenshotAndTrim(locator, filepath) {
  await locator.screenshot({ path: filepath, type: 'png' });

  // Auto-trim uniform borders (whitespace)
  try {
    const trimmed = await sharp(filepath)
      .trim({ threshold: TRIM_THRESHOLD })
      .toBuffer({ resolveWithObject: true });

    if (trimmed.info.width > 0 && trimmed.info.height > 0) {
      await writeFile(filepath, trimmed.data);
    }
  } catch {
    // trim() can fail if the image is entirely uniform; keep original
  }

  // Add uniform margin around the image
  if (MARGIN_PX > 0) {
    const padded = await sharp(filepath)
      .extend({
        top: MARGIN_PX,
        bottom: MARGIN_PX,
        left: MARGIN_PX,
        right: MARGIN_PX,
        background: { r: 255, g: 255, b: 255, alpha: 1 },
      })
      .toBuffer();
    await writeFile(filepath, padded);
  }
}

// ─── Helper: set a <select> to a given option index ─────────────────────────
async function setSelectOption(selectHandle, idx) {
  await selectHandle.evaluate((el, idx) => {
    el.selectedIndex = idx;
    Array.from(el.options).forEach((opt, j) => {
      if (j === idx) opt.setAttribute('selected', '');
      else opt.removeAttribute('selected');
    });
    el.dispatchEvent(new Event('change', { bubbles: true }));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, idx);
}

// ─── Helper: set a checkbox to a specific checked state ─────────────────────
async function setCheckbox(cbHandle, checked) {
  await cbHandle.evaluate((el, val) => {
    if (el.checked !== val) {
      el.checked = val;
      el.dispatchEvent(new Event('change', { bubbles: true }));
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('click', { bubbles: true }));
    }
  }, checked);
}

// ─── Helper: open a <select> visually (show all options) ────────────────────
async function openSelect(selectHandle) {
  await selectHandle.evaluate((el) => {
    el.dataset.__prevSize = el.getAttribute('size') ?? '';
    el.dataset.__prevStyle = el.getAttribute('style') ?? '';
    el.dataset.__prevMultiple = el.multiple ? '1' : '0';
    const count = el.querySelectorAll('option').length;
    el.setAttribute('size', String(Math.min(count || 1, 8)));
    el.multiple = true;
    el.style.position = 'relative';
    el.style.zIndex = '9999';
    el.style.height = 'auto';
    el.style.maxHeight = 'none';
    el.style.background = 'white';
  });
}

// ─── Helper: restore a <select> after openSelect ────────────────────────────
async function restoreSelect(selectHandle) {
  await selectHandle.evaluate((el) => {
    const prevSize = el.dataset.__prevSize;
    const prevStyle = el.dataset.__prevStyle;
    const prevMultiple = el.dataset.__prevMultiple;
    if (prevSize) el.setAttribute('size', prevSize);
    else el.removeAttribute('size');
    el.multiple = prevMultiple === '1';
    el.setAttribute('style', prevStyle || '');
    delete el.dataset.__prevSize;
    delete el.dataset.__prevStyle;
    delete el.dataset.__prevMultiple;
  });
}

// ─── Helper: render text as an image via SVG ────────────────────────────────
async function renderLabel(text, maxWidth, fontSize = 48) {
  const escaped = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const h = Math.round(fontSize * 1.6);
  const svg = Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${maxWidth}" height="${h}">
      <text x="${maxWidth / 2}" y="${h / 2}" text-anchor="middle" dominant-baseline="central"
            font-family="system-ui, -apple-system, sans-serif" font-size="${fontSize}"
            font-weight="600" fill="#333">${escaped}</text>
    </svg>`,
  );
  return sharp(svg).png().toBuffer();
}

// ─── Helper: build 1D composite (single select / button-group) ──────────────
async function buildComposite1D(capturedOptionPaths, compositeFilepath) {
  const refPath = capturedOptionPaths[0];
  const refMeta = await sharp(refPath).metadata();
  let unionTop = refMeta.height, unionBottom = 0;
  for (let k = 1; k < capturedOptionPaths.length; k++) {
    const db = await getDiffBounds(refPath, capturedOptionPaths[k]);
    if (db) { unionTop = Math.min(unionTop, db.top); unionBottom = Math.max(unionBottom, db.bottom); }
  }
  const innerW = refMeta.width - MARGIN_PX * 2;
  const pad = 20;
  const safetyPad = Math.round(refMeta.height * 0.02);
  const splitY = Math.max(MARGIN_PX, unionTop - safetyPad);
  const hasCommonHeader = unionBottom > unionTop && (splitY - MARGIN_PX) > refMeta.height * 0.10;

  if (hasCommonHeader) {
    const headerH = splitY - MARGIN_PX;
    const uniqueH = refMeta.height - splitY - MARGIN_PX;
    console.log(`  📌 Common header: ${headerH}px (${Math.round(headerH / refMeta.height * 100)}%), unique: ${uniqueH}px`);
    const commonHeader = await sharp(refPath).extract({ left: MARGIN_PX, top: MARGIN_PX, width: innerW, height: headerH }).toBuffer();
    const uniqueParts = await Promise.all(capturedOptionPaths.map(async (p) => ({
      buffer: await sharp(p).extract({ left: MARGIN_PX, top: splitY, width: innerW, height: uniqueH }).toBuffer(),
      width: innerW, height: uniqueH,
    })));
    const cols = uniqueParts.length <= 2 ? uniqueParts.length : 2;
    const rows = Math.ceil(uniqueParts.length / cols);
    const gridW = innerW * cols + pad * (cols + 1);
    const gridH = uniqueH * rows + pad * (rows + 1);
    const gap = 30;
    const totalW = Math.max(innerW, gridW) + MARGIN_PX * 2;
    const totalH = MARGIN_PX + headerH + gap + gridH + MARGIN_PX;
    const isLastRowIncomplete = uniqueParts.length % cols !== 0;
    const gridImg = await sharp({ create: { width: gridW, height: gridH, channels: 4, background: { r: 255, g: 255, b: 255, alpha: 1 } } })
      .composite(uniqueParts.map((t, i) => {
        const row = Math.floor(i / cols);
        const col = i % cols;
        const isOnLastRow = row === rows - 1;
        const itemsOnLastRow = uniqueParts.length - (rows - 1) * cols;
        let left;
        if (isOnLastRow && isLastRowIncomplete) {
          const usedW = itemsOnLastRow * innerW + (itemsOnLastRow - 1) * pad;
          left = Math.round((gridW - usedW) / 2) + col * (innerW + pad);
        } else {
          left = pad + col * (innerW + pad);
        }
        return { input: t.buffer, left, top: pad + row * (uniqueH + pad) };
      })).png().toBuffer();
    await sharp({ create: { width: totalW, height: totalH, channels: 4, background: { r: 255, g: 255, b: 255, alpha: 1 } } })
      .composite([
        { input: commonHeader, left: Math.round((totalW - innerW) / 2), top: MARGIN_PX },
        { input: gridImg, left: Math.round((totalW - gridW) / 2), top: MARGIN_PX + headerH + gap },
      ]).toFile(compositeFilepath);
    console.log(`  ✅ ${compositeFilepath.split('/').pop()} (header + ${cols}×${rows} grid)`);
  } else {
    const contextPad = Math.round(refMeta.height * 0.15);
    const cropY = Math.max(0, unionTop - contextPad);
    const cropBot = Math.min(refMeta.height, unionBottom + contextPad);
    const cropH = cropBot - cropY;
    const useCrop = cropH > 0 && cropH < refMeta.height * 0.75 && unionBottom > unionTop;
    if (useCrop) console.log(`  🎯 Vertical crop: ${refMeta.width}×${cropH}px (${Math.round((1 - cropH / refMeta.height) * 100)}% shorter)`);
    const tiles = await Promise.all(capturedOptionPaths.map(async (p) => {
      if (useCrop) return { buffer: await sharp(p).extract({ left: 0, top: cropY, width: refMeta.width, height: cropH }).toBuffer(), width: refMeta.width, height: cropH };
      const m = await sharp(p).metadata();
      return { buffer: await readFile(p), width: m.width, height: m.height };
    }));
    const cols = tiles.length <= 2 ? tiles.length : 2;
    const rows = Math.ceil(tiles.length / cols);
    const cellW = tiles[0].width, cellH = tiles[0].height;
    const gridW = cellW * cols + pad * (cols + 1);
    const gridH = cellH * rows + pad * (rows + 1);
    const isLastRowIncomplete2 = tiles.length % cols !== 0;
    await sharp({ create: { width: gridW, height: gridH, channels: 4, background: { r: 255, g: 255, b: 255, alpha: 1 } } })
      .composite(tiles.map((t, i) => {
        const row = Math.floor(i / cols);
        const col = i % cols;
        const isOnLastRow = row === rows - 1;
        const itemsOnLastRow = tiles.length - (rows - 1) * cols;
        let left;
        if (isOnLastRow && isLastRowIncomplete2) {
          const usedW = itemsOnLastRow * cellW + (itemsOnLastRow - 1) * pad;
          left = Math.round((gridW - usedW) / 2) + col * (cellW + pad);
        } else {
          left = pad + col * (cellW + pad);
        }
        return { input: t.buffer, left, top: pad + row * (cellH + pad) };
      }))
      .toFile(compositeFilepath);
    console.log(`  ✅ ${compositeFilepath.split('/').pop()} (${cols}×${rows} grid)`);
  }
}

// ─── Helper: build 2D composite with row/col labels + common header ─────────
async function buildComposite2D(capturedGrid, compositeFilepath, rowSelect, colSelect) {
  const nRows = capturedGrid.length, nCols = capturedGrid[0].length;
  const refPath = capturedGrid.flat().find(Boolean);
  const refMeta = await sharp(refPath).metadata();
  const innerW = refMeta.width - MARGIN_PX * 2;
  const allPaths = capturedGrid.flat().filter(Boolean);
  let unionTop = refMeta.height, unionBottom = 0;
  for (let k = 1; k < allPaths.length; k++) {
    const db = await getDiffBounds(allPaths[0], allPaths[k]);
    if (db) { unionTop = Math.min(unionTop, db.top); unionBottom = Math.max(unionBottom, db.bottom); }
  }
  const safetyPad = Math.round(refMeta.height * 0.02);
  const splitY = Math.max(MARGIN_PX, unionTop - safetyPad);
  const hasCommonHeader = unionBottom > unionTop && (splitY - MARGIN_PX) > refMeta.height * 0.10;
  const headerH = hasCommonHeader ? splitY - MARGIN_PX : 0;
  const tileTopY = hasCommonHeader ? splitY : 0;
  const tileH = hasCommonHeader ? refMeta.height - splitY - MARGIN_PX : refMeta.height;
  if (hasCommonHeader) console.log(`  📌 Common header: ${headerH}px (${Math.round(headerH / refMeta.height * 100)}%)`);

  const tiles = [];
  for (let r = 0; r < nRows; r++) {
    tiles[r] = [];
    for (let c = 0; c < nCols; c++) {
      const p = capturedGrid[r][c];
      if (p) {
        tiles[r][c] = hasCommonHeader
          ? await sharp(p).extract({ left: MARGIN_PX, top: tileTopY, width: innerW, height: tileH }).toBuffer()
          : await readFile(p);
      } else {
        const blankW = hasCommonHeader ? innerW : refMeta.width;
        tiles[r][c] = await sharp({ create: { width: blankW, height: tileH, channels: 4, background: { r: 240, g: 240, b: 240, alpha: 1 } } }).png().toBuffer();
      }
    }
  }
  const pad = 20, labelFontSize = 44, labelH = Math.round(labelFontSize * 1.6), rowLabelW = 300;
  const cellW = hasCommonHeader ? innerW : refMeta.width, cellH = tileH;
  const gridW = rowLabelW + nCols * (cellW + pad) + pad;
  const gridH = labelH + pad + nRows * (cellH + pad) + pad;
  const gap = 30;
  const totalW = (hasCommonHeader ? Math.max(innerW, gridW) : gridW) + MARGIN_PX * 2;
  const totalH = MARGIN_PX + (hasCommonHeader ? headerH + gap : 0) + gridH + MARGIN_PX;
  const compositeInputs = [];
  const gridOffsetY = MARGIN_PX + (hasCommonHeader ? headerH + gap : 0);
  const gridX = Math.round((totalW - gridW) / 2);
  if (hasCommonHeader) {
    const hdr = await sharp(refPath).extract({ left: MARGIN_PX, top: MARGIN_PX, width: innerW, height: headerH }).toBuffer();
    compositeInputs.push({ input: hdr, left: Math.round((totalW - innerW) / 2), top: MARGIN_PX });
  }
  for (let c = 0; c < nCols; c++) {
    compositeInputs.push({ input: await renderLabel(colSelect.options[c].text, cellW, labelFontSize), left: gridX + rowLabelW + pad + c * (cellW + pad), top: gridOffsetY });
  }
  for (let r = 0; r < nRows; r++) {
    const rowY = gridOffsetY + labelH + pad + r * (cellH + pad);
    const rl = await renderLabel(rowSelect.options[r].text, rowLabelW, labelFontSize);
    const rlMeta = await sharp(rl).metadata();
    compositeInputs.push({ input: rl, left: gridX, top: rowY + Math.round((cellH - rlMeta.height) / 2) });
    for (let c = 0; c < nCols; c++) {
      compositeInputs.push({ input: tiles[r][c], left: gridX + rowLabelW + pad + c * (cellW + pad), top: rowY });
    }
  }
  await sharp({ create: { width: totalW, height: totalH, channels: 4, background: { r: 255, g: 255, b: 255, alpha: 1 } } })
    .composite(compositeInputs).toFile(compositeFilepath);
  const outMeta = await sharp(compositeFilepath).metadata();
  console.log(`  ✅ ${compositeFilepath.split('/').pop()} (2D ${nRows}×${nCols} + labels, ${outMeta.width}x${outMeta.height})`);
}

// ─── Helper: looks-same diffBounds between two images ───────────────────────
async function getDiffBounds(img1Path, img2Path) {
  const result = await looksSame(img1Path, img2Path, {
    shouldCluster: false,
    ignoreAntialiasing: true,
    ignoreCaret: true,
    tolerance: 3,
  });

  if (result.equal) return null;
  const db = result.diffBounds;
  if (!db || db.left >= db.right || db.top >= db.bottom) return null;
  return db; // { left, top, right, bottom }
}

// ─── Main ───────────────────────────────────────────────────────────────────
async function main() {
  // Clean previous screenshots to avoid stale files in the archive
  await rm(OUTPUT_DIR, { recursive: true, force: true });
  await mkdir(OUTPUT_DIR, { recursive: true });

  console.log('🚀 Launching browser...');
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    deviceScaleFactor: DEVICE_SCALE_FACTOR,
    viewport: BASE_VIEWPORT,
  });
  const page = await context.newPage();

  console.log(`📄 Navigating to ${URL}...`);
  await page.goto(URL, { waitUntil: 'load', timeout: 60000 });
  await page.waitForTimeout(2000);

  // Scroll through entire page to trigger lazy-loaded embeds (IntersectionObserver)
  console.log('📜 Scrolling page to trigger lazy-loaded content…');
  await page.evaluate(async () => {
    const scrollStep = window.innerHeight * 0.8;
    const maxScroll = document.body.scrollHeight;
    for (let y = 0; y < maxScroll; y += scrollStep) {
      window.scrollTo(0, y);
      await new Promise(r => setTimeout(r, 100));
    }
    window.scrollTo(0, 0);
  });
  // Wait for all lazy content to initialize and render
  await page.waitForTimeout(5000);

  let totalCount = 0;

  const allElements = await page.$$(
    '.html-embed, .table-scroll > table, .image-wrapper, .katex-display',
  );
  console.log(`\n🔍 Found ${allElements.length} elements (DOM order)`);

  for (let i = 0; i < allElements.length; i++) {
    const element = allElements[i];

    const type = await element.evaluate((el) => {
      if (el.matches('.html-embed')) return 'embed';
      if (el.matches('.table-scroll > table')) return 'table';
      if (el.matches('.image-wrapper')) return 'image';
      if (el.matches('.katex-display')) return 'katex';
      return 'unknown';
    });

    {
      const visible = await element.evaluate((el) => {
        const cs = getComputedStyle(el);
        if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0') return false;
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0;
      });
      if (!visible) {
        console.log(`  ⏭️  Skipping hidden ${type} ${i + 1}`);
        continue;
      }
    }

    const label = await element.evaluate((el) => {
      if (el.classList.contains('html-embed')) {
        const btn = el.querySelector('.html-embed__download');
        const filename = btn?.getAttribute('data-filename') || '';
        if (filename) return filename;
        const title = el.querySelector('.html-embed__title');
        if (title?.textContent) return title.textContent;
      }

      const getAttr = (name) => el.getAttribute(name) || '';
      const direct =
        getAttr('data-title') ||
        getAttr('data-name') ||
        getAttr('data-label') ||
        getAttr('data-slug') ||
        getAttr('aria-label') ||
        getAttr('title') ||
        getAttr('id');
      if (direct) return direct;

      if (el.tagName.toLowerCase() === 'table') {
        const caption = el.querySelector('caption');
        if (caption) return caption.textContent || '';
      }

      const img = el.querySelector('img');
      if (img) return img.getAttribute('alt') || img.getAttribute('title') || '';

      const heading = el.querySelector('h1,h2,h3,h4,h5,h6');
      if (heading) return heading.textContent || '';

      return '';
    });

    const slug = slugify(label);
    const baseName = `${i + 1}-${type}${slug ? `--${slug}` : ''}`;
    const filename = `${baseName}.png`;
    const filepath = join(OUTPUT_DIR, filename);

    try {
      if (type !== 'katex') {
        await element.scrollIntoViewIfNeeded();
        await page.waitForTimeout(200);
      }

      // ── TABLE: clone into isolated wrapper (full-width, unclipped) ──────
      if (type === 'table') {
        const cloneId = await element.evaluate((el, idx) => {
          const existing = document.getElementById(`__table-clone-wrapper-${idx}`);
          if (existing) existing.remove();

          const wrapper = document.createElement('div');
          wrapper.id = `__table-clone-wrapper-${idx}`;
          wrapper.style.cssText =
            'position:absolute;left:0;top:0;background:transparent;z-index:99999;width:max-content;';

          const contentGrid = document.createElement('section');
          contentGrid.className = 'content-grid';
          const main = document.createElement('main');
          const tableScroll = document.createElement('div');
          tableScroll.className = 'table-scroll';
          tableScroll.style.cssText =
            'background:transparent;border:none;border-radius:0;box-shadow:none;';

          const clone = el.cloneNode(true);
          clone.id = `__table-clone-${idx}`;
          clone.style.width = 'max-content';
          clone.style.minWidth = '0';
          clone.style.maxWidth = 'none';
          clone.style.tableLayout = 'auto';

          clone.querySelectorAll('th, td').forEach((cell) => {
            cell.style.width = 'auto';
            cell.style.minWidth = '0';
            cell.style.maxWidth = 'none';
          });

          tableScroll.appendChild(clone);
          main.appendChild(tableScroll);
          contentGrid.appendChild(main);
          wrapper.appendChild(contentGrid);
          document.body.appendChild(wrapper);
          return clone.id;
        }, i);

        const wrapperSelector = `#__table-clone-wrapper-${i}`;
        const cloneSelector = `#${cloneId}`;

        const cloneWidth = await page.evaluate(
          (sel) => document.querySelector(sel)?.getBoundingClientRect().width ?? 0,
          wrapperSelector,
        );

        const currentVP = page.viewportSize();
        if (cloneWidth > currentVP.width) {
          await page.setViewportSize({
            width: Math.ceil(cloneWidth + 200),
            height: currentVP.height,
          });
          await page.waitForTimeout(200);
        }

        await screenshotAndTrim(page.locator(cloneSelector), filepath);
        await page.evaluate((sel) => document.querySelector(sel)?.remove(), wrapperSelector);
      }

      // ── KATEX: clone into isolated wrapper ──────────────────────────────
      else if (type === 'katex') {
        const cloneId = await element.evaluate((el, idx) => {
          const existing = document.getElementById(`__katex-clone-wrapper-${idx}`);
          if (existing) existing.remove();

          const wrapper = document.createElement('div');
          wrapper.id = `__katex-clone-wrapper-${idx}`;
          wrapper.style.cssText =
            'position:absolute;left:0;top:0;background:transparent;z-index:99999;width:max-content;';

          const clone = el.cloneNode(true);
          clone.id = `__katex-clone-${idx}`;
          clone.style.cssText = 'display:inline-block;width:max-content;max-width:none;margin:0;';

          wrapper.appendChild(clone);
          document.body.appendChild(wrapper);
          return clone.id;
        }, i);

        const wrapperSelector = `#__katex-clone-wrapper-${i}`;
        const cloneSelector = `#${cloneId}`;

        const cloneWidth = await page.evaluate(
          (sel) => document.querySelector(sel)?.getBoundingClientRect().width ?? 0,
          wrapperSelector,
        );

        const currentVP = page.viewportSize();
        if (cloneWidth > currentVP.width) {
          await page.setViewportSize({
            width: Math.ceil(cloneWidth + 200),
            height: currentVP.height,
          });
          await page.waitForTimeout(200);
        }

        await screenshotAndTrim(page.locator(cloneSelector), filepath);
        await page.evaluate((sel) => document.querySelector(sel)?.remove(), wrapperSelector);
      }

      // ── EMBED: clone into isolated wrapper ──────────────────────────────
      else if (type === 'embed') {
        const { wrapperId, cloneId } = await cloneEmbed(page, element, i);
        await page.waitForTimeout(200);
        await screenshotAndTrim(page.locator(`#${cloneId}`), filepath);
        await removeClone(page, wrapperId);
      }

      // ── IMAGE: screenshot in-place with sibling isolation ──────────────
      // (cloning doesn't work because Astro-optimized images won't re-fetch)
      else {
        // Ensure all images in the element are fully loaded
        await element.evaluate(async (el) => {
          const imgs = el.querySelectorAll('img');
          for (const img of imgs) {
            img.loading = 'eager';
            img.decoding = 'sync';
          }
          await Promise.all(Array.from(imgs).map((img) => {
            if (img.complete && img.naturalWidth > 0) return Promise.resolve();
            return new Promise((res) => {
              img.onload = res;
              img.onerror = res;
              setTimeout(res, 5000);
            });
          }));
        });
        await page.waitForTimeout(300);

        // Hide all sibling elements at every ancestor level up to <main>
        // to avoid visual contamination from neighboring content
        await element.evaluate((el) => {
          let current = el;
          while (current && current.tagName !== 'MAIN' && current.tagName !== 'BODY') {
            const parent = current.parentElement;
            if (!parent) break;
            for (const sibling of parent.children) {
              if (sibling !== current) {
                sibling.setAttribute('data-img-iso', sibling.style.visibility || '');
                sibling.style.visibility = 'hidden';
              }
            }
            current = parent;
          }
        });

        await page.waitForTimeout(100);
        await screenshotAndTrim(element, filepath);

        // Restore all hidden siblings
        await page.evaluate(() => {
          document.querySelectorAll('[data-img-iso]').forEach((el) => {
            el.style.visibility = el.getAttribute('data-img-iso');
            el.removeAttribute('data-img-iso');
          });
        });
      }

      const meta = await sharp(filepath).metadata();
      console.log(`  ✅ ${filename} (${meta.width}x${meta.height}px)`);
      totalCount++;

      // ── EMBED with <select>, checkbox, button-group: capture variants ─
      if (type === 'embed') {
        const allSelects = await element.$$('select');

        // ── Detect checkboxes ───────────────────────────────────────────
        const allCheckboxes = await element.$$('input[type="checkbox"]');
        const checkboxesInfo = await Promise.all(
          allCheckboxes.map(async (cb) =>
            cb.evaluate((el) => {
              const label = el.labels?.[0]?.textContent?.trim()
                || el.closest('label')?.textContent?.trim()
                || el.id || 'checkbox';
              return { label, checked: el.checked };
            }),
          ),
        );

        // If checkboxes found, wrap select logic in a loop over checkbox states
        const cbStates = checkboxesInfo.length > 0
          ? [false, true]
          : [null]; // null = no checkbox to toggle

        for (const cbState of cbStates) {
          // Set checkbox state if applicable
          if (cbState !== null && allCheckboxes.length > 0) {
            const cbLabel = checkboxesInfo[0]?.label || 'checkbox';
            console.log(`  ☑️  Checkbox "${cbLabel}" → ${cbState ? 'ON' : 'OFF'}`);
            for (const cbHandle of allCheckboxes) {
              await setCheckbox(cbHandle, cbState);
            }
            await page.waitForTimeout(300);
          }

          const cbSuffix = cbState !== null
            ? `--cb-${cbState ? 'on' : 'off'}--${slugify(checkboxesInfo[0]?.label || 'toggle').slice(0, 30)}`
            : '';
          const cbBaseName = `${baseName}${cbSuffix}`;

        // ── Single select → 1D grid ────────────────────────────────────
        if (allSelects.length === 1) {
          const selectHandle = allSelects[0];
          try {
            const options = await selectHandle.evaluate((el) =>
              Array.from(el.querySelectorAll('option')).map((opt, idx) => ({
                value: opt.value,
                text: opt.textContent || opt.value || `option-${idx}`,
                index: idx,
              })),
            );

            console.log(`  📸 Capturing ${options.length} select options...`);
            const capturedOptionPaths = [];

            for (const option of options) {
              const optionSlug = slugify(option.text).slice(0, 50);
              const optionFilename = `${cbBaseName}--option-${option.index}${optionSlug ? `--${optionSlug}` : ''}.png`;
              const optionFilepath = join(OUTPUT_DIR, optionFilename);
              try {
                await setSelectOption(selectHandle, option.index);
                await page.waitForTimeout(400);
                const { wrapperId, cloneId } = await cloneEmbed(page, element, `opt-${i}-${option.index}`);
                await page.waitForTimeout(200);
                await screenshotAndTrim(page.locator(`#${cloneId}`), optionFilepath);
                await removeClone(page, wrapperId);
                console.log(`    ✅ ${optionFilename}`);
                capturedOptionPaths.push(optionFilepath);
                totalCount++;
              } catch (err) {
                console.log(`    ❌ Failed: ${optionFilename}: ${err.message}`);
              }
            }

            // Capture with select OPEN
            const openFilename = `${cbBaseName}--open-select.png`;
            const openFilepath = join(OUTPUT_DIR, openFilename);
            try {
              await setSelectOption(selectHandle, 0);
              await page.waitForTimeout(200);
              await openSelect(selectHandle);
              await page.waitForTimeout(150);
              const { wrapperId, cloneId } = await cloneEmbed(page, element, `open-${i}`);
              await page.waitForTimeout(200);
              await screenshotAndTrim(page.locator(`#${cloneId}`), openFilepath);
              await removeClone(page, wrapperId);
              await restoreSelect(selectHandle);
              console.log(`  ✅ ${openFilename}`);
              totalCount++;
            } catch (err) {
              console.log(`  ❌ Failed: ${openFilename}: ${err.message}`);
            }

            // Composite (1D grid with optional common header)
            if (capturedOptionPaths.length > 1) {
              try {
                const compositeFilename = `${cbBaseName}--all-options.png`;
                const compositeFilepath = join(OUTPUT_DIR, compositeFilename);
                console.log(`  🖼️  Creating composite grid image...`);
                await buildComposite1D(capturedOptionPaths, compositeFilepath);
                totalCount++;
              } catch (err) {
                console.log(`  ❌ Failed composite: ${err.message}`);
              }
            }
          } catch (err) {
            console.log(`  ❌ Failed select processing: ${err.message}`);
          }
        }

        // ── Multiple selects → 2D grid of all combinations ─────────────
        else if (allSelects.length >= 2) {
          try {
            const selectsInfo = await Promise.all(
              allSelects.map(async (sel, sIdx) =>
                sel.evaluate((el, sIdx) => ({
                  sIdx,
                  name: el.name || el.id || `select-${sIdx}`,
                  options: Array.from(el.options).map((o, j) => ({
                    index: j,
                    text: o.textContent || o.value || `opt-${j}`,
                  })),
                }), sIdx),
              ),
            );

            // More options → rows (vertical), fewer options → columns (horizontal)
            let rowSelect, colSelect, rowHandle, colHandle;
            if (selectsInfo[0].options.length >= selectsInfo[1].options.length) {
              [rowSelect, colSelect] = [selectsInfo[0], selectsInfo[1]];
              [rowHandle, colHandle] = [allSelects[0], allSelects[1]];
            } else {
              [rowSelect, colSelect] = [selectsInfo[1], selectsInfo[0]];
              [rowHandle, colHandle] = [allSelects[1], allSelects[0]];
            }

            const nRows = rowSelect.options.length;
            const nCols = colSelect.options.length;
            console.log(`  📸 Capturing ${nRows * nCols} combinations (${nRows} × ${nCols}) from ${allSelects.length} selects...`);

            const capturedGrid = Array.from({ length: nRows }, () => Array(nCols).fill(null));

            for (let r = 0; r < nRows; r++) {
              for (let c = 0; c < nCols; c++) {
                const rowSlug = slugify(rowSelect.options[r].text).slice(0, 30);
                const colSlug = slugify(colSelect.options[c].text).slice(0, 30);
                const comboFilename = `${cbBaseName}--combo-${r}-${c}--${rowSlug}--${colSlug}.png`;
                const comboFilepath = join(OUTPUT_DIR, comboFilename);
                try {
                  await setSelectOption(rowHandle, rowSelect.options[r].index);
                  await setSelectOption(colHandle, colSelect.options[c].index);
                  await page.waitForTimeout(400);
                  const { wrapperId, cloneId } = await cloneEmbed(page, element, `combo-${i}-${r}-${c}`);
                  await page.waitForTimeout(200);
                  await screenshotAndTrim(page.locator(`#${cloneId}`), comboFilepath);
                  await removeClone(page, wrapperId);
                  console.log(`    ✅ [${r},${c}] ${comboFilename}`);
                  capturedGrid[r][c] = comboFilepath;
                  totalCount++;
                } catch (err) {
                  console.log(`    ❌ Failed [${r},${c}]: ${err.message}`);
                }
              }
            }

            // Open-select screenshots (one per select)
            for (let sIdx = 0; sIdx < allSelects.length; sIdx++) {
              const openFilename = `${cbBaseName}--open-select-${sIdx}.png`;
              const openFilepath = join(OUTPUT_DIR, openFilename);
              try {
                await setSelectOption(allSelects[sIdx], 0);
                await page.waitForTimeout(100);
                await openSelect(allSelects[sIdx]);
                await page.waitForTimeout(150);
                const { wrapperId, cloneId } = await cloneEmbed(page, element, `open-${i}-${sIdx}`);
                await page.waitForTimeout(200);
                await screenshotAndTrim(page.locator(`#${cloneId}`), openFilepath);
                await removeClone(page, wrapperId);
                await restoreSelect(allSelects[sIdx]);
                console.log(`  ✅ ${openFilename}`);
                totalCount++;
              } catch (err) {
                console.log(`  ❌ Failed: ${openFilename}: ${err.message}`);
              }
            }

            // 2D Composite with row/col labels
            const allPaths = capturedGrid.flat().filter(Boolean);
            if (allPaths.length > 1) {
              try {
                const compositeFilename = `${cbBaseName}--all-options.png`;
                const compositeFilepath = join(OUTPUT_DIR, compositeFilename);
                console.log(`  🖼️  Creating 2D composite grid (${nRows}×${nCols}) with labels...`);
                await buildComposite2D(capturedGrid, compositeFilepath, rowSelect, colSelect);
                totalCount++;
              } catch (err) {
                console.log(`  ❌ Failed 2D composite: ${err.message}`);
              }
            }
          } catch (err) {
            console.log(`  ❌ Failed multi-select processing: ${err.message}`);
          }
        }

        // ── No selects: check for button-group or toggle-group ────────
        else {
          const buttonGroup = await element.$('.button-group, .toggle-group');
          if (buttonGroup) {
            try {
              const buttons = await buttonGroup.$$eval(
                'button[data-model], button[data-value], button.active, button.toggle-btn',
                (btns) =>
                  btns.map((b, idx) => {
                    // Determine the best selector for this button
                    const parentClass = b.parentElement?.classList.contains('button-group')
                      ? '.button-group' : '.toggle-group';
                    return {
                      index: idx,
                      text: b.textContent.trim(),
                      selector: b.dataset.model
                        ? `button[data-model="${b.dataset.model}"]`
                        : b.dataset.value
                          ? `button[data-value="${b.dataset.value}"]`
                          : `${parentClass} button:nth-child(${idx + 1})`,
                    };
                  }),
              );

              if (buttons.length > 1) {
                console.log(`  🔘 Capturing ${buttons.length} button states...`);
                const capturedBtnPaths = [];

                for (const btn of buttons) {
                  const btnSlug = slugify(btn.text).slice(0, 50);
                  const btnFilename = `${cbBaseName}--btn-${btn.index}${btnSlug ? `--${btnSlug}` : ''}.png`;
                  const btnFilepath = join(OUTPUT_DIR, btnFilename);
                  try {
                    const btnHandle = await element.$(btn.selector);
                    if (btnHandle) {
                      await btnHandle.click();
                      await page.waitForTimeout(400);
                      const { wrapperId, cloneId } = await cloneEmbed(page, element, `btn-${i}-${btn.index}`);
                      await page.waitForTimeout(200);
                      await screenshotAndTrim(page.locator(`#${cloneId}`), btnFilepath);
                      await removeClone(page, wrapperId);
                      console.log(`    ✅ ${btnFilename}`);
                      capturedBtnPaths.push(btnFilepath);
                      totalCount++;
                    }
                  } catch (err) {
                    console.log(`    ❌ Failed: ${btnFilename}: ${err.message}`);
                  }
                }

                // Composite (1D grid with optional common header)
                if (capturedBtnPaths.length > 1) {
                  try {
                    const compositeFilename = `${cbBaseName}--all-options.png`;
                    const compositeFilepath = join(OUTPUT_DIR, compositeFilename);
                    console.log(`  🖼️  Creating composite grid image...`);
                    await buildComposite1D(capturedBtnPaths, compositeFilepath);
                    totalCount++;
                  } catch (err) {
                    console.log(`  ❌ Failed composite: ${err.message}`);
                  }
                }

                // Reset to first button
                try {
                  const firstBtn = await element.$(buttons[0].selector);
                  if (firstBtn) await firstBtn.click();
                  await page.waitForTimeout(200);
                } catch {}
              }
            } catch (err) {
              console.log(`  ❌ Failed button-group processing: ${err.message}`);
            }
          }
        }

        } // end for (cbState)

        // Reset checkboxes to original state
        if (allCheckboxes.length > 0) {
          for (let ci = 0; ci < allCheckboxes.length; ci++) {
            await setCheckbox(allCheckboxes[ci], checkboxesInfo[ci].checked);
          }
        }
      }
    } catch (err) {
      console.log(`  ❌ Failed to capture ${filename}: ${err.message}`);
    }
  }

  await browser.close();
  console.log(`\n🎉 Done! Captured ${totalCount} screenshots in ${OUTPUT_DIR}/`);
}

main().catch(console.error);

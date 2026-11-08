// Mixed-style text: a text element's runs ({text, color?, bold?, italic?}) as editor DOM and back.
//
// Every editor this one is modelled on styles a range inside a text box: select some characters while
// editing and the colour, Bold and Italic controls apply to them only. The text node holds one <span> per
// run, each carrying its own values in data attributes, so typing inside a span inherits its style (as in
// the real apps) and the runs can be read back exactly. Offsets count characters of the element's text.

import { cssColor } from "./util.js";

const KEYS = ["color", "bold", "italic"];

function styleOf(node, root) {
  const span = node.parentElement?.closest("span[data-run]");
  if (!span || !root.contains(span)) return {};
  const d = span.dataset;
  const out = {};
  if (d.color) out.color = d.color;
  if (d.bold) out.bold = d.bold === "1";
  if (d.italic) out.italic = d.italic === "1";
  return out;
}

const same = (a, b) => KEYS.every((k) => a[k] === b[k]);

/** Fill a text node: plain text, or one span per run. */
export function paintRuns(t, text, runs) {
  t.textContent = "";
  if (!runs || !runs.length) {
    t.textContent = text;
    return;
  }
  for (const r of runs) {
    const span = document.createElement("span");
    span.textContent = r.text;
    span.dataset.run = "1";
    if (r.color) { span.dataset.color = r.color; span.style.color = cssColor(r.color); }
    if (r.bold !== undefined && r.bold !== null) { span.dataset.bold = r.bold ? "1" : "0"; span.style.fontWeight = r.bold ? "700" : "400"; }
    if (r.italic !== undefined && r.italic !== null) { span.dataset.italic = r.italic ? "1" : "0"; span.style.fontStyle = r.italic ? "italic" : "normal"; }
    t.append(span);
  }
}

/** The runs an edited text node holds, read structurally (Chromium may add <br> or <div> on its own). */
export function readRuns(root) {
  const out = [];
  const push = (text, style) => {
    if (!text) return;
    const last = out[out.length - 1];
    if (last && same(last, style)) last.text += text;
    else out.push({ text, ...style });
  };
  const ends = () => !out.length || out[out.length - 1].text.endsWith("\n");
  const walk = (node) => {
    for (const c of node.childNodes) {
      if (c.nodeType === 3) push(c.data.replace(/​/g, "").replace(/\r/g, ""), styleOf(c, root));
      else if (c.nodeName === "BR") push("\n", styleOf(c, root));
      else {
        if ((c.nodeName === "DIV" || c.nodeName === "P") && !ends()) push("\n", {});
        walk(c);
      }
    }
  };
  walk(root);
  return out;
}

/** Merged runs with every value the element already has dropped; null when nothing differs. */
export function normRuns(runs, el) {
  const base = { color: (el.color || "#000000").toLowerCase(), bold: !!el.bold, italic: !!el.italic };
  const out = [];
  for (const r of runs || []) {
    if (!r.text) continue;
    const s = {};
    if (r.color && r.color.toLowerCase() !== base.color) s.color = r.color.toLowerCase();
    if (r.bold !== undefined && r.bold !== null && r.bold !== base.bold) s.bold = r.bold;
    if (r.italic !== undefined && r.italic !== null && r.italic !== base.italic) s.italic = r.italic;
    const last = out[out.length - 1];
    if (last && same(last, s)) last.text += r.text;
    else out.push({ text: r.text, ...s });
  }
  return out.some((r) => KEYS.some((k) => k in r)) ? out : null;
}

/** Runs with `props` applied to characters [start, end). */
export function styleRange(text, runs, start, end, props) {
  const src = runs && runs.length ? runs : [{ text }];
  const out = [];
  let pos = 0;
  for (const r of src) {
    const a = pos;
    const b = pos + r.text.length;
    pos = b;
    const cuts = [a, Math.max(a, Math.min(b, start)), Math.max(a, Math.min(b, end)), b];
    for (let i = 0; i < 3; i++) {
      if (cuts[i + 1] <= cuts[i]) continue;
      const piece = { ...r, text: r.text.slice(cuts[i] - a, cuts[i + 1] - a) };
      if (i === 1) Object.assign(piece, props);
      out.push(piece);
    }
  }
  return out;
}

/** Whether every character in [start, end) has `key` on (runs inherit the element's value). */
export function rangeAll(el, runs, start, end, key) {
  const src = runs && runs.length ? runs : [{ text: el.text }];
  let pos = 0;
  for (const r of src) {
    const a = pos;
    pos += r.text.length;
    if (pos <= start || a >= end) continue;
    const v = r[key] === undefined || r[key] === null ? !!el[key] : r[key];
    if (!v) return false;
  }
  return true;
}

function locate(root, target) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let n = 0;
  let node;
  let last = null;
  while ((node = walker.nextNode())) {
    last = node;
    const data = node.data;
    let clean = 0;
    for (let i = 0; i <= data.length; i++) {
      if (n + clean === target && (i === data.length ? true : data[i] !== "​")) return [node, i];
      if (i < data.length && data[i] !== "​") clean++;
    }
    n += clean;
  }
  return last ? [last, last.data.length] : [root, 0];
}

/** The selection inside `root` as character offsets, or null. */
export function selectionOffsets(root) {
  const sel = window.getSelection();
  if (!sel || !sel.rangeCount) return null;
  const r = sel.getRangeAt(0);
  if (!root.contains(r.startContainer) || !root.contains(r.endContainer)) return null;
  const at = (node, off) => {
    const pre = document.createRange();
    pre.selectNodeContents(root);
    pre.setEnd(node, off);
    const tmp = document.createElement("div");
    tmp.append(pre.cloneContents());
    return readRuns(tmp).reduce((n, x) => n + x.text.length, 0);
  };
  const a = at(r.startContainer, r.startOffset);
  const b = at(r.endContainer, r.endOffset);
  return { start: Math.min(a, b), end: Math.max(a, b) };
}

/** Select characters [start, end) inside `root`. */
export function selectOffsets(root, start, end) {
  const range = document.createRange();
  range.setStart(...locate(root, start));
  range.setEnd(...locate(root, end));
  const sel = window.getSelection();
  sel.removeAllRanges();
  sel.addRange(range);
}

/** Screen rectangles of characters [start, end) of a painted text node. */
export function rangeRects(root, start, end) {
  const range = document.createRange();
  range.setStart(...locate(root, start));
  range.setEnd(...locate(root, end));
  return [...range.getClientRects()].filter((r) => r.width > 0.5 && r.height > 0.5);
}

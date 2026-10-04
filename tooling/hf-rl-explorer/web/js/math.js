// Math in prompts, titles and task lists ($$…$$, \[…\], \(…\), $…$), typeset with KaTeX, which loads on the first page
// that has some. A "$…$" pair is math only when what's between looks like TeX (a backslash, ^, _ or braces, or a lone
// variable), so "spent $300 and set aside $200" stays prose. Code, preformatted text and anything marked .no-math are
// left alone; KaTeX runs untrusted (no \href, no \html*), and a formula it can't read stays as its source.

const KATEX = "https://cdn.jsdelivr.net/npm/katex@0.19.0/dist/";
const SRI = {
  js: "sha384-QFFtAGzvvj+bfgCGxXJlNZZR1nXEZgvG8tDLCCY1F19xl20WlfTYgguB4VcNdxYk",
  css: "sha384-3rdsX6e5mueWyoweR9NIVmtEsUkokpBT/0ALqKKIBMr9j4qhHkaIkAcGgsE6uVlp",
};
const RX = /\$\$([\s\S]{1,4000}?)\$\$|\\\[([\s\S]{1,4000}?)\\\]|\\\(([\s\S]{1,1000}?)\\\)|\$([^$\n]{1,400}?)\$/g;
const texy = (s) => /[\\^_{}]/.test(s) || /^\s*[A-Za-z]\s*$/.test(s);
const SKIP = "code, pre, textarea, script, style, .katex, .no-math, .fv, .runbox";
let loading = null;

function load() {
  if (window.katex) return Promise.resolve();
  if (loading) return loading;
  const css = Object.assign(document.createElement("link"), { rel: "stylesheet", href: KATEX + "katex.min.css", integrity: SRI.css, crossOrigin: "anonymous" });
  document.head.append(css);
  loading = new Promise((resolve, reject) => {
    const s = Object.assign(document.createElement("script"), { src: KATEX + "katex.min.js", integrity: SRI.js, crossOrigin: "anonymous", async: true });
    s.onload = () => resolve();
    s.onerror = () => { loading = null; reject(new Error("KaTeX didn't load")); };
    document.head.append(s);
  });
  return loading;
}

function* formulas(text) {
  RX.lastIndex = 0;
  for (const m of text.matchAll(RX)) {
    const [whole, dd, br, pa, d] = m;
    if (d != null && !texy(d)) continue;
    yield { at: m.index, whole, tex: dd ?? br ?? pa ?? d, display: dd != null || br != null };
  }
}

export function hasMath(text) {
  return !formulas(String(text || "")).next().done;
}

/** Typeset every formula under `root` (or each of several roots); a no-op, and no download, when there's none. */
export async function typeset(...roots) {
  roots = roots.flat().filter(Boolean);
  if (!roots.some((r) => hasMath(r.textContent))) return;
  try { await load(); } catch { return; }   // offline, or blocked: the formulas stay as their source
  for (const root of roots) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => (n.parentElement?.closest(SKIP) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const n of nodes) {
      const t = n.nodeValue;
      if (!t.includes("$") && !t.includes("\\")) continue;
      let last = 0, frag = null;
      for (const f of formulas(t)) {
        frag ??= document.createDocumentFragment();
        frag.append(t.slice(last, f.at));
        const span = document.createElement("span");
        span.className = f.display ? "math display" : "math";
        try {
          span.innerHTML = window.katex.renderToString(f.tex, { displayMode: f.display, throwOnError: false, trust: false, maxSize: 20, maxExpand: 300, output: "html" });
        } catch {
          span.textContent = f.whole;
        }
        frag.append(span);
        last = f.at + f.whole.length;
      }
      if (frag) { frag.append(t.slice(last)); n.replaceWith(frag); }
    }
  }
}

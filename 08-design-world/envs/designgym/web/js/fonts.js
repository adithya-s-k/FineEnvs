// Font loading. The editor uses exactly the files the server renders with,
// served from /fonts, so a glyph in the browser is the glyph that gets scored.

const loaded = new Set();
const pending = new Map();
let inflight = 0;
let catalog = null;

// True while a font file is still loading: text will re-measure and re-render
// when it lands, so the editor is not idle yet (see Store.whenIdle).
export function fontsLoading() {
  return inflight > 0;
}

export async function fontCatalog() {
  if (!catalog) {
    const res = await fetch("/api/fonts");
    catalog = (await res.json()).families;
  }
  return catalog;
}

export function familyInfo(family) {
  return (catalog || []).find((f) => f.family === family) || null;
}

export function hasStyle(family, style) {
  const info = familyInfo(family);
  return !!info && info.styles.includes(style);
}

export function ensureFonts(families) {
  const wanted = [...new Set(families.filter(Boolean))].filter((f) => !loaded.has(f));
  if (!wanted.length) return Promise.resolve();
  const key = wanted.join(",");
  if (pending.has(key)) return pending.get(key);
  inflight += 1;
  const p = new Promise((done) => {
    const resolve = () => { inflight -= 1; done(); };
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = `/fonts.css?families=${encodeURIComponent(key)}`;
    link.onload = async () => {
      const loads = [];
      for (const fam of wanted) {
        loaded.add(fam);
        for (const w of ["400", "700"]) for (const s of ["normal", "italic"]) {
          loads.push(document.fonts.load(`${s} ${w} 32px '${fam}'`).catch(() => null));
        }
      }
      await Promise.all(loads);
      resolve();
    };
    link.onerror = () => resolve();
    document.head.append(link);
  });
  pending.set(key, p);
  return p;
}

export function fontsInDoc(doc) {
  return doc.elements.filter((e) => e.kind === "text").map((e) => e.font_family);
}

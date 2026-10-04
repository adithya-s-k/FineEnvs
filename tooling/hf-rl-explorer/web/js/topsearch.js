// The search in the header: trending environments before you type, matching ones as you type (yours first, when
// signed in), and any dataset id opened directly, including a private one you can read.
import { $, esc, fmt, nav } from "./util.js";
import { icon } from "./icons.js";
import { api } from "./util.js";
import { myDatasets } from "./data.js";

const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });
const enc = (s) => s.split("/").map(encodeURIComponent).join("/");
const ID = /^[A-Za-z0-9][\w.-]*\/[\w.-]+$/;

export function wireTopSearch() {
  const box = $("#top-search"), input = $("#top-q"), pop = $("#top-sug");
  let rows = [], active = -1, seq = 0;

  const row = (d, i) => `<a class="ts-row${i === active ? " act" : ""}" role="option" href="/${d.kind === "space" ? "s" : "d"}/${enc(d.id)}" data-i="${i}">
      ${icon(d.private ? "shield" : d.kind === "space" ? "globe" : "database", 14)}<span class="ts-id"><span class="faint">${esc(d.id.split("/")[0])}/</span>${esc(d.id.split("/")[1])}</span>
      <span class="ts-meta">${d.kind === "space" ? `Space · ${d.stage === "RUNNING" ? "running" : "asleep"} · ${compact.format(d.likes || 0)} likes`
        : `${d.indexed?.tasks != null ? `${fmt.format(d.indexed.tasks)} tasks · ` : ""}${d.private ? "private" : `${compact.format(d.downloads || 0)} downloads`}`}</span></a>`;

  let ctrl = null;
  async function render() {
    const my = ++seq;
    const q = input.value.trim(), ql = q.toLowerCase();
    ctrl?.abort();
    ctrl = new AbortController();
    let found, mine;
    try {   // the server searches every environment (/api/search): eight cards, not the whole listing
      [found, mine] = await Promise.all([api(`/api/search?size=8&facets=0${q ? `&q=${encodeURIComponent(q)}` : ""}`, { signal: ctrl.signal }), myDatasets()]);
    } catch { return; }
    if (my !== seq) return;
    const sections = [];
    if (!q) {
      sections.push(["Trending environments", found.rows]);
      if (mine.length) sections.push(["Yours", mine.slice(0, 4)]);
    } else {
      const ours = mine.filter((d) => d.id.toLowerCase().includes(ql));
      const rest = found.rows.filter((d) => !ours.some((o) => o.id === d.id));
      if (ours.length) sections.push(["Yours", ours.slice(0, 4)]);
      sections.push([rest.length ? "Environments, trending first" : "", rest]);
    }
    const all = [...mine, ...found.rows];
    rows = sections.flatMap(([, list]) => list);
    let i = 0;
    const open = ID.test(q) && !all.some((d) => d.id.toLowerCase() === ql);   // any dataset by id, a private one too
    pop.innerHTML = sections.filter(([, list]) => list.length).map(([title, list]) =>
      `${title ? `<div class="ts-h">${esc(title)}</div>` : ""}${list.map((d) => row(d, i++)).join("")}`).join("")
      + (open ? `<a class="ts-row ts-open" href="/d/${enc(q)}">${icon("external", 14)}<span>Open <b>${esc(q)}</b></span><span class="ts-meta">any Harbor dataset, private ones when you're signed in</span></a>` : "")
      + (q ? `<a class="ts-row ts-all" href="/?q=${encodeURIComponent(q)}">${icon("search", 14)}<span>All results for “${esc(q)}”</span></a>`
           : `<a class="ts-row ts-all" href="/">${icon("grid", 14)}<span>Browse every environment</span></a>`);
    pop.hidden = false;
    box.setAttribute("aria-expanded", "true");
  }
  const close = () => { pop.hidden = true; active = -1; box.setAttribute("aria-expanded", "false"); };
  const go = (href) => { close(); input.blur(); nav(href); };

  input.addEventListener("focus", render);
  input.addEventListener("input", () => { active = -1; render(); });
  input.addEventListener("keydown", (e) => {
    const links = [...pop.querySelectorAll(".ts-row")];
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      active = Math.max(0, Math.min(links.length - 1, active + (e.key === "ArrowDown" ? 1 : -1)));
      links.forEach((a, j) => a.classList.toggle("act", j === active));
      links[active]?.scrollIntoView({ block: "nearest" });
    } else if (e.key === "Enter") {
      e.preventDefault();
      const q = input.value.trim();
      const pick = links[active] || (ID.test(q) ? { getAttribute: () => `/d/${enc(q)}` } : links.find((a) => a.classList.contains("ts-all")));
      if (pick) go(pick.getAttribute("href"));
    } else if (e.key === "Escape") { close(); input.blur(); }
  });
  pop.addEventListener("click", (e) => { const a = e.target.closest("a"); if (a) { e.preventDefault(); go(a.getAttribute("href")); input.value = ""; } });
  document.addEventListener("click", (e) => { if (!pop.hidden && !e.composedPath().includes(box)) close(); });
  addEventListener("popstate", close);
}

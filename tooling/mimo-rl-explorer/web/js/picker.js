// A searchable model picker: frontier models first, each row with provider, price, speed and context.
import { esc } from "./util.js";
import { icon } from "./icons.js";

const short = (id) => id.split("/")[1] || id;
const org = (id) => id.split("/")[0];
const ctx = (n) => (n ? (n >= 1e6 ? `${(n / 1e6).toFixed(n % 1e6 ? 1 : 0)}M` : `${Math.round(n / 1000)}k`) : "");

export function picker(host, { models, value, onChange, note, groupLabel }) {
  let current = models.find((m) => m.id === value) || models[0];
  let open = false, q = "", active = 0;
  host.classList.add("picker");
  const render = () => {
    const list = filtered();
    host.innerHTML = `
      <button type="button" class="pk-btn" aria-haspopup="listbox" aria-expanded="${open}">
        <span class="pk-name">${esc(short(current.id))}</span><span class="pk-org">${esc(org(current.id))}</span>
        <span class="pk-price">$${current.input} / $${current.output}</span>${icon("chevronDown", 15)}</button>
      ${open ? `<div class="pk-pop" role="listbox">
        <div class="pk-search">${icon("search", 15)}<input class="pk-q" placeholder="Search ${models.length} models or providers…" value="${esc(q)}" aria-label="Search models"></div>
        <div class="pk-list">${list.length ? list.map((m, i) => `${i === 0 || m.featured !== list[i - 1].featured
            ? `<div class="pk-group">${groupLabel || (m.featured ? "Frontier" : "All with tool calling")}</div>` : ""}
          <button type="button" class="pk-row${m.id === current.id ? " sel" : ""}${i === active ? " act" : ""}" data-id="${esc(m.id)}" role="option">
            <span class="pk-name">${esc(short(m.id))}${m.id === current.id ? icon("check", 13, "tick") : ""}</span><span class="pk-org">${esc(org(m.id))}${m.vision ? ' · <b>vision</b>' : ""}</span>
            <span class="pk-meta">${esc(m.provider)}${m.speed ? ` · ${m.speed} tok/s` : ""}${m.context ? ` · ${ctx(m.context)} context` : ""}${m.note ? ` · ${esc(m.note)}` : ""}</span>
            <span class="pk-price">$${m.input} / $${m.output}</span></button>`).join("") : `<p class="muted sm" style="padding:14px 8px">No model matches “${esc(q)}”.</p>`}</div>
        ${note ? `<p class="pk-note">${note}</p>` : ""}</div>` : ""}`;
    if (open) {
      const inp = host.querySelector(".pk-q");
      inp.focus(); inp.setSelectionRange(q.length, q.length);
      host.querySelector(".pk-row.act")?.scrollIntoView({ block: "nearest" });
    }
  };
  const filtered = () => {
    const t = q.trim().toLowerCase();
    return t ? models.filter((m) => (m.id + " " + m.provider).toLowerCase().includes(t)) : models;
  };
  const choose = (id) => {
    current = models.find((m) => m.id === id) || current;
    open = false; q = ""; render(); onChange(current);
  };
  host.addEventListener("click", (e) => {
    if (e.target.closest(".pk-btn")) { open = !open; active = 0; render(); return; }
    const row = e.target.closest(".pk-row");
    if (row) choose(row.dataset.id);
  });
  host.addEventListener("input", (e) => { if (e.target.classList.contains("pk-q")) { q = e.target.value; active = 0; render(); } });
  host.addEventListener("keydown", (e) => {
    if (!open) return;
    const list = filtered();
    if (e.key === "ArrowDown") { active = Math.min(list.length - 1, active + 1); render(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { active = Math.max(0, active - 1); render(); e.preventDefault(); }
    else if (e.key === "Enter" && list[active]) { choose(list[active].id); e.preventDefault(); }
    else if (e.key === "Escape") { open = false; render(); e.stopPropagation(); }
  });
  // composedPath is fixed when the click happens; host.contains(target) is not, because render() replaces the target
  document.addEventListener("click", (e) => { if (open && !e.composedPath().includes(host)) { open = false; render(); } });
  render();
  return { get value() { return current.id; }, get model() { return current; } };
}

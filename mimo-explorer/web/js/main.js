// App shell: hash router, account, theme, and the count of rollouts still running.
import { $, api, esc, storage, closeModal } from "./util.js";
import * as explore from "./explore.js";
import { getSession, setSession, refreshActive } from "./session.js";

let current = null;

const VIEWS = { task: () => import("./task.js"), run: () => import("./run.js"), runs: () => import("./runs.js") };

function parse(hash) {
  const [path, qs] = hash.replace(/^#/, "").split("?");
  const m = path.match(/^\/(task|run)\/(.+)$/);
  if (m) return { name: m[1], arg: decodeURIComponent(m[2]), qs };
  if (/^\/runs\/?$/.test(path)) return { name: "runs", qs };
  return { name: "explore", qs };
}

async function route() {
  const { name, arg, qs } = parse(location.hash || "#/");
  if (current?.unmount) current.unmount();
  current = null;
  closeModal();
  const inRuns = name === "run" || name === "runs";
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("on", (a.dataset.nav === "runs") === inRuns));
  window.scrollTo(0, 0);
  const mod = name === "explore" ? explore : await VIEWS[name]();
  current = mod;
  await mod.mount($("#view"), name === "explore" ? (qs ? new URLSearchParams(qs) : null) : arg);
}

async function loadSession() {
  try { setSession(await api("/api/me")); } catch { setSession({ user: null }); }
  renderAccount();
}
function renderAccount() {
  const el = $("#account");
  const u = getSession().user;
  if (!u) {
    // Inside the huggingface.co iframe, sign-in must open at the top level (third-party cookies).
    const framed = window.top !== window.self;
    el.innerHTML = `<a class="btn primary sm" href="/login" ${framed ? 'target="_blank" rel="noopener"' : ""}>
      <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="" width="16" height="16"> Sign in with Hugging Face</a>`;
    return;
  }
  el.innerHTML = `<div class="acct"><img src="${esc(u.avatar || "")}" alt="" width="26" height="26"><span>${esc(u.name)}</span>
    ${u.local ? '<span class="chip">local</span>' : '<a class="link" href="/logout">Sign out</a>'}</div>`;
}

function initTheme() {
  const saved = storage.get("theme");
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    storage.set("theme", document.documentElement.dataset.theme);
    explore.renderMap();
  });
  if (storage.get("costbar") === "hidden") $("#costbar").hidden = true;
  $("#costbar-close").addEventListener("click", () => { $("#costbar").hidden = true; storage.set("costbar", "hidden"); });
}

function wireGlobal() {
  $("#modal-close").addEventListener("click", closeModal);
  $("#scrim").addEventListener("click", closeModal);
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") closeModal();
    if (ev.key === "/" && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) { ev.preventDefault(); explore.focusSearch(); }
  });
  let rt;
  addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => explore.renderMap(), 150); });
  addEventListener("hashchange", route);
}

// No top-level await: views import session.js, not this file, but keep the shell non-blocking anyway.
initTheme();
wireGlobal();
loadSession().then(route).then(() => { refreshActive(); setInterval(refreshActive, 8000); });

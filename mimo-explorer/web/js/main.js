// App shell: hash router, account menu and sign-in, theme, and the count of rollouts still running.
import { $, api, esc, storage, closeModal, openDialog, progress, spinner, toast } from "./util.js";
import { icon } from "./icons.js";
import * as explore from "./explore.js";
import { getSession, setSession, refreshActive } from "./session.js";

let current = null;
const VIEWS = { task: () => import("./task.js"), run: () => import("./run.js"), runs: () => import("./runs.js"), community: () => import("./community.js") };
const framed = window.top !== window.self;

function parse(hash) {
  const [path, qs] = hash.replace(/^#/, "").split("?");
  const m = path.match(/^\/(task|run)\/(.+)$/);
  if (m) return { name: m[1], arg: decodeURIComponent(m[2]), qs };
  if (/^\/runs\/?$/.test(path)) return { name: "runs", qs };
  if (/^\/community\/?$/.test(path)) return { name: "community", qs };
  return { name: "explore", qs };
}

let routing = 0;
async function route() {
  const token = ++routing;
  const { name, arg, qs } = parse(location.hash || "#/");
  if (current?.unmount) current.unmount();
  current = null;
  closeModal();
  const tab = name === "runs" || name === "run" ? "runs" : name === "community" ? "community" : "explore";
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("on", a.dataset.nav === tab));
  window.scrollTo(0, 0);
  progress.start();
  try {
    const mod = name === "explore" ? explore : await VIEWS[name]();
    if (token !== routing) return;
    current = mod;
    await mod.mount($("#view"), name === "explore" ? (qs ? new URLSearchParams(qs) : null) : name === "community" ? qs : arg);
  } finally { progress.done(); }
}

// ── account ──────────────────────────────────────────────────────────────────
async function loadSession() {
  try { setSession(await api("/api/me")); } catch { setSession({ user: null }); }
  const v = getSession();
  if (v.version) $("#foot-version").innerHTML = `v${esc(v.version)} · <code title="source hash">${esc(v.source || "")}</code>`;
  renderAccount();
}

function renderAccount() {
  const el = $("#account");
  const s = getSession(), u = s.user;
  if (!u) {
    el.innerHTML = `<button class="btn primary signin" type="button" data-signin>${icon("user", 15)}Sign in<span class="long"> to run rollouts</span></button>`;
    return;
  }
  const via = { oauth: "Signed in with Hugging Face", token: "Signed in with an access token", local: "Using this machine's HF token" }[u.via] || "";
  el.innerHTML = `<div class="acct">
    <button class="acct-btn" type="button" aria-haspopup="menu" aria-expanded="false">
      ${u.avatar ? `<img src="${esc(u.avatar)}" alt="">` : `<span class="avatar"></span>`}<span class="name">${esc(u.name)}</span>${icon("chevronDown", 14)}</button>
    <div class="menu" role="menu" hidden>
      <div class="menu-h"><b>${esc(u.name)}</b><span>${esc(via)}</span></div>
      <a href="#/runs" role="menuitem">${icon("list")}Your rollouts</a>
      <a href="https://huggingface.co/settings/billing" target="_blank" rel="noopener" role="menuitem">${icon("coins")}Billing on Hugging Face</a>
      <a href="https://huggingface.co/spaces/FineEnvs/MiMo-RL-Envs-Explorer/discussions" target="_blank" rel="noopener" role="menuitem">${icon("flag")}Report an issue</a>
      <button type="button" data-signin role="menuitem">${icon("key")}Use a different token</button>
      ${u.via === "local" ? "" : `<button type="button" data-signout role="menuitem">${icon("logout")}Sign out</button>`}
    </div></div>`;
  const btn = $(".acct-btn", el), menu = $(".menu", el);
  btn.addEventListener("click", (e) => { e.stopPropagation(); menu.hidden = !menu.hidden; btn.setAttribute("aria-expanded", String(!menu.hidden)); });
  document.addEventListener("click", (e) => { if (!menu.hidden && !e.composedPath().includes(el)) { menu.hidden = true; btn.setAttribute("aria-expanded", "false"); } });
}

export function openSignIn() {
  const s = getSession();
  const dlg = openDialog(`
    <div class="dialog-h"><div><h2>${s.user ? "Use a different token" : "Sign in to run rollouts"}</h2>
      <p>Rollouts run on your Hugging Face account: an HF Sandbox (CPU, about $0.01 an hour) plus the model's tokens at the provider's price.</p></div>
      <button class="icon-btn" type="button" data-close aria-label="Close">${icon("x", 18)}</button></div>
    <div class="dialog-b">
      ${s.oauth ? `<a class="btn primary lg block" href="/login" ${framed ? 'target="_blank" rel="noopener"' : ""}>
          <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="" width="18" height="18">Continue with Hugging Face</a>
        ${framed ? `<p class="fine">Opens in a new tab. If you come back here and still aren't signed in, use the Space directly at
          <a href="${esc(location.origin)}" target="_blank" rel="noopener">${esc(location.host)}</a>.</p>` : ""}
        <div class="or">or use an access token</div>` : ""}
      <form class="field" id="tok-form" autocomplete="off">
        <span>Hugging Face access token <em>write, or fine-grained with Inference Providers and Jobs</em></span>
        <input class="input mono" id="tok" type="password" placeholder="hf_…" spellcheck="false" autocomplete="off" aria-label="Hugging Face access token" required>
        <div id="tok-err"></div>
        <button class="btn ${s.oauth ? "" : "primary"} block" type="submit" id="tok-go">Sign in with token</button>
      </form>
      <p class="fine">The token is checked with Hugging Face, then kept in an encrypted cookie on this browser for 7 days. It is never
        written to disk or into a trace. <a href="https://huggingface.co/settings/tokens/new?tokenType=write" target="_blank" rel="noopener">Create a token</a></p>
    </div>`);
  const form = $("#tok-form", dlg), input = $("#tok", dlg), go = $("#tok-go", dlg);
  if (!s.oauth) input.focus();
  dlg.addEventListener("click", (e) => { if (e.target.closest("[data-close]")) closeModal(); });
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    $("#tok-err", dlg).innerHTML = "";
    go.disabled = true; go.innerHTML = `${spinner()}Checking with Hugging Face…`;
    try {
      const r = await api("/api/login/token", { method: "POST", body: { token: input.value } });
      input.value = "";
      closeModal();
      await loadSession();
      toast(r.missing_scopes?.length ? `Signed in as ${r.user.name}. This token may be missing: ${r.missing_scopes.join(", ")}.` : `Signed in as ${r.user.name}`, 4000);
      refreshActive();
      route();
    } catch (err) {
      $("#tok-err", dlg).innerHTML = `<div class="note-box err">${icon("alert")}<span>${esc(err.message)}</span></div>`;
      go.disabled = false; go.textContent = "Sign in with token";
    }
  });
}

async function signOut() {
  try { await api("/api/logout", { method: "POST" }); } catch { /* the cookie is gone either way */ }
  await loadSession();
  toast("Signed out");
  route();
}

// ── theme ────────────────────────────────────────────────────────────────────
const isDark = () => (document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches);
function paintTheme() { $("#theme").innerHTML = icon(isDark() ? "sun" : "moon", 17); }
function initTheme() {
  paintTheme();
  $("#theme").addEventListener("click", () => {
    document.documentElement.dataset.theme = isDark() ? "light" : "dark";
    storage.set("theme", document.documentElement.dataset.theme);
    paintTheme();
    explore.renderMap();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { paintTheme(); explore.renderMap(); });
}

function wireGlobal() {
  $("#modal-close").innerHTML = icon("x", 18);
  $("#modal-close").addEventListener("click", closeModal);
  $("#scrim").addEventListener("click", closeModal);
  document.addEventListener("click", (e) => {
    if (e.target.closest(".menu a, .menu button")) { const m = $(".menu"); if (m) m.hidden = true; }
    if (e.target.closest("[data-signin]")) { e.preventDefault(); openSignIn(); }
    else if (e.target.closest("[data-signout]")) { e.preventDefault(); signOut(); }
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") closeModal();
    if (ev.key === "/" && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) { ev.preventDefault(); explore.focusSearch(); }
  });
  let rt;
  addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => explore.renderMap(), 150); });
  addEventListener("hashchange", route);
}

// No top-level await: views import session.js, not this file, and a module that waits at load time deadlocks.
initTheme();
wireGlobal();
loadSession().then(route).then(() => { refreshActive(); setInterval(refreshActive, 8000); });

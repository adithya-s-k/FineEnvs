// App shell: router, account menu and sign-in, theme. Every page has a real path (the server answers each with the
// page and its own title, description and structured data, app/seo.py); in-app links move with the History API.
//   /                           explore: every RL environment on the Hub
//   /d/<org>/<name>             an environment dataset, any format (app/envs): how it's read and graded, and its tasks
//   /t/<org>/<name>/<ref>       a task, any format: what the agent gets, how it is graded, what it runs in, its files;
//                               run it (ref is the adapter's: a Harbor task's folder, a row's [config/]split/i)
//   /s/<org>/<name>             an environment Space: live status, its app, a playground, its tasks, MCP
//   /compare/<org>/<name>/<ref>?r=a,b   rollouts of one task side by side
//   /run/<id>                   a rollout: its trajectory as it runs, then its reward
//   /runs                       your rollouts
//   /community                  everyone's public rollouts, by model
// Older addresses still open: #/… (the hash router this app had), and the MiMo RL Environment Explorer's #/task/<id>,
// #/compare/<id> and #/rewards.
import { $, api, esc, storage, closeModal, openDialog, progress, spinner, toast, nav, setMeta, emptyState } from "./util.js";
import { icon } from "./icons.js";
import { getSession, setSession, refreshActive } from "./session.js";
import { wireTopSearch } from "./topsearch.js";
import { forgetMine } from "./data.js";

let current = null;
const VIEWS = { home: () => import("./home.js"), dataset: () => import("./env.js"), task: () => import("./task.js"),
                run: () => import("./run.js"), runs: () => import("./runs.js"), community: () => import("./community.js"),
                space: () => import("./space.js"), compare: () => import("./compare.js") };
const MIMO = "XiaomiMiMo/MiMo-V2.6-RL-oss";
const framed = window.top !== window.self;

const decodePath = (s) => { try { return decodeURIComponent(s); } catch { return s; } };

function parse(url) {
  const [path, qs] = url.split("?");
  let m = path.match(/^\/run\/([\w-]+)$/);
  if (m) return { name: "run", arg: { id: m[1] }, qs };
  if (/^\/runs\/?$/.test(path)) return { name: "runs", arg: {}, qs };
  if (/^\/community\/?$/.test(path)) return { name: "community", arg: {}, qs };
  m = path.match(/^\/s\/([^/]+)\/([^/]+)\/?$/);
  if (m) return { name: "space", arg: { spec: `${decodePath(m[1])}/${decodePath(m[2])}` }, qs };
  m = path.match(/^\/compare\/([^/]+)\/([^/]+)\/(.+)$/);
  if (m) return { name: "compare", arg: { spec: `${decodePath(m[1])}/${decodePath(m[2])}`, ref: decodePath(m[3]) }, qs };
  m = path.match(/^\/(task|compare)\/([^/]+)$/);   // the MiMo explorer's addresses
  if (m) return { redirect: `/${m[1] === "task" ? "t" : "compare"}/${MIMO}/${m[2]}${qs ? `?${qs}` : ""}` };
  if (/^\/rewards\/?$/.test(path)) return { redirect: `/d/${MIMO}?rewards=1` };
  m = path.match(/^\/r\/([^/]+)\/([^/]+)\/?$/);
  if (m) {   // a row's old address: its task page now, like every other task
    const q = new URLSearchParams(qs || ""), c = q.get("c") || "default", sp = q.get("s") || "train", i = Math.max(0, +q.get("i") || 0);
    ["c", "s", "i"].forEach((k) => q.delete(k));
    return { redirect: `/t/${m[1]}/${m[2]}/${c === "default" ? "" : `${encodeURIComponent(c)}/`}${encodeURIComponent(sp)}/${i}${q.toString() ? `?${q}` : ""}` };
  }
  m = path.match(/^\/d\/([^/]+)\/([^/]+)\/?$/);
  if (m) return { name: "dataset", arg: { spec: `${decodePath(m[1])}/${decodePath(m[2])}` }, qs };
  m = path.match(/^\/t\/([^/]+)\/([^/]+)\/?(.*)$/);
  if (m) return { name: "task", arg: { spec: `${decodePath(m[1])}/${decodePath(m[2])}`, path: decodePath(m[3] || "") }, qs };
  return { name: "home", arg: {}, qs };
}

let routing = 0;
async function route() {
  const token = ++routing;
  if (location.hash.startsWith("#/")) { history.replaceState(null, "", location.hash.slice(1)); }   // an address from the hash router
  const { name, arg, qs, redirect } = parse(location.pathname + location.search);
  if (redirect) { history.replaceState(null, "", redirect); return route(); }
  if (current?.unmount) current.unmount();
  current = null;
  closeModal();
  const tip = document.getElementById("tip");
  if (tip) tip.hidden = true;
  document.body.dataset.view = name;   // the Explore page has its own search, so the header's hides there
  setMeta({ title: { runs: "My rollouts", community: "Community rollouts", run: "Rollout", compare: "Compare rollouts" }[name] });
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("on", a.dataset.nav === ({ runs: "runs", run: "runs", community: "community", compare: "community" }[name] || "explore")));
  window.scrollTo(0, 0);
  progress.start();
  try {
    const mod = await VIEWS[name]();
    if (token !== routing) return;
    current = mod;
    // a fresh container per page: views attach their listeners to it, so none survive into the next page
    const old = $("#view"), view = old.cloneNode(false);
    old.replaceWith(view);
    await mod.mount(view, { ...arg, qs: qs ? new URLSearchParams(qs) : new URLSearchParams() });
  } catch (error) {
    if (token !== routing) return;
    current?.unmount?.();
    current = null;
    $("#view").innerHTML = `<div class="wrap page">${emptyState("alert", "Couldn't load this page", esc(error.message || "Please try again."),
      '<button class="btn" type="button" id="page-retry">Try again</button><a class="btn" href="/">Explore environments</a>')}</div>`;
    $("#page-retry").addEventListener("click", route);
  } finally { progress.done(); }
}

// ── account ──────────────────────────────────────────────────────────────────
async function loadSession() {
  try { setSession(await api("/api/me")); } catch { setSession({ user: null }); }
  forgetMine();   // whose datasets "yours" means may have changed
  const v = getSession();
  const note = $("#announce");
  note.hidden = !v.announcement;
  note.querySelector("span").textContent = v.announcement || "";
  if (v.version) $("#foot-version").innerHTML = `v${esc(v.version)} · <code title="source hash">${esc(v.source || "")}</code>`;
  renderAccount();
}

function renderAccount() {
  const el = $("#account");
  const s = getSession(), u = s.user;
  if (!u) {
    el.innerHTML = `<button class="btn primary signin" type="button" data-signin aria-label="Sign in">${icon("user", 15)}<span class="long">Sign in</span></button>`;
    return;
  }
  const via = { oauth: "Signed in with Hugging Face", token: "Signed in with an access token", local: "Using this machine's HF token" }[u.via] || "";
  el.innerHTML = `<div class="acct">
    <button class="acct-btn" type="button" aria-haspopup="menu" aria-expanded="false">
      ${u.avatar ? `<img src="${esc(u.avatar)}" alt="">` : `<span class="avatar"></span>`}<span class="name">${esc(u.name)}</span>${icon("chevronDown", 14)}</button>
    <div class="menu" role="menu" hidden>
      <div class="menu-h"><b>${esc(u.name)}</b><span>${esc(via)}</span></div>
      <a href="https://huggingface.co/settings/billing" target="_blank" rel="noopener" role="menuitem">${icon("coins")}Billing on Hugging Face</a>
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
    <div class="dialog-h"><div><h2>${s.user ? "Use a different token" : "Sign in"}</h2>
      <p>Rollouts run on your Hugging Face account: an HF Sandbox plus the model's tokens at the provider's price. Signing in also
        opens your private Harbor datasets.</p></div>
      <button class="icon-btn" type="button" data-close aria-label="Close">${icon("x", 18)}</button></div>
    <div class="dialog-b">
      ${s.oauth ? `<a class="btn primary lg block" href="/login" ${framed ? 'target="_blank" rel="noopener"' : ""}>
          <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="" width="18" height="18">Continue with Hugging Face</a>
        <div class="or">or use an access token</div>` : ""}
      <form class="field" id="tok-form" autocomplete="off">
        <span>Hugging Face access token <em>write, or fine-grained with Inference Providers and Jobs</em></span>
        <input class="input mono" id="tok" type="password" placeholder="hf_…" spellcheck="false" autocomplete="off" aria-label="Hugging Face access token" required>
        <div id="tok-err"></div>
        <button class="btn ${s.oauth ? "" : "primary"} block" type="submit" id="tok-go">Sign in with token</button>
      </form>
      <p class="fine">The token is checked with Hugging Face, then kept in an encrypted cookie on this browser for 7 days. It is never
        written to disk or into a trace.</p>
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
      toast(`Signed in as ${r.user.name}`, 3000);
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
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", paintTheme);
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
    if (ev.key === "/" && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) {
      const s = document.querySelector("[data-search]") || $("#top-q");
      if (s) { ev.preventDefault(); s.focus(); }
    }
  });
  addEventListener("popstate", route);
  addEventListener("hashchange", () => { if (location.hash.startsWith("#/")) route(); });   // an old #/ link pasted in
  // in-app links: the History API, not a page load (new tab, modified clicks, downloads and other origins as usual)
  document.addEventListener("click", (e) => {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const a = e.target.closest("a[href]");
    if (!a || a.target || a.hasAttribute("download") || a.origin !== location.origin) return;
    const href = a.getAttribute("href");
    if (!href.startsWith("/") || /^\/(api|mcp|capture|login|logout|oauth)\b/.test(href)) return;
    e.preventDefault();
    if (href === location.pathname + location.search) return;
    nav(href);
  });
  wireTopSearch();
  // a Space preview that won't load (no thumbnail yet) leaves, rather than showing a broken image (no inline handlers: CSP)
  document.addEventListener("error", (e) => { if (e.target.matches?.(".sp-thumb, .fe-thumb, .sp-shot")) e.target.remove(); }, true);
}

// the FineEnvs banner: shown until it's closed, then not again on this browser for a week
function wireBanner() {
  const b = $("#fe-banner");
  const closed = Number(storage.get("fe-banner-closed") || 0);
  if (Date.now() - closed < 7 * 864e5) return;
  b.hidden = false;
  $("#fe-banner-close").addEventListener("click", () => { b.classList.add("closing"); storage.set("fe-banner-closed", String(Date.now())); setTimeout(() => (b.hidden = true), 220); });
}

initTheme();
wireBanner();
wireGlobal();
loadSession().then(() => { route(); refreshActive(); });
setInterval(refreshActive, 8000);

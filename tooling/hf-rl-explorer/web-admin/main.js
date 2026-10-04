// The admin Space's shell: who you are, the theme, and the dashboard (web-admin/admin.js).
import { $, api, esc, storage } from "/js/util.js";
import { icon } from "/js/icons.js";
import { getSession, setSession } from "/js/session.js";
import * as dash from "/admin/admin.js";

async function boot() {
  try { setSession(await api("/api/me")); } catch { setSession({ user: null }); }
  const s = getSession();
  $("#to-explorer").href = s.explorer || "/";
  if (s.version) $("#foot-version").innerHTML = `v${esc(s.version)} · <code>${esc(s.source || "")}</code>`;
  $("#account").innerHTML = s.user ? `<span class="acct-btn" title="${esc(s.user.name)}">${s.user.avatar ? `<img src="${esc(s.user.avatar)}" alt="">` : ""}<span class="name">${esc(s.user.name)}</span></span>
    ${s.local ? "" : `<button class="btn sm ghost" id="signout" type="button">${icon("logout", 13)}Sign out</button>`}` : "";
  $("#signout")?.addEventListener("click", async () => { await api("/api/logout", { method: "POST" }).catch(() => {}); location.reload(); });
  const route = () => {
    dash.unmount();
    const qs = new URLSearchParams((location.hash.split("?")[1]) || "");
    dash.mount($("#view"), { qs, explorer: s.explorer });
  };
  addEventListener("hashchange", route);
  route();
}

const isDark = () => (document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches);
const paint = () => { $("#theme").innerHTML = icon(isDark() ? "sun" : "moon", 17); };
paint();
$("#theme").addEventListener("click", () => { document.documentElement.dataset.theme = isDark() ? "light" : "dark"; storage.set("theme", document.documentElement.dataset.theme); paint(); });
boot();

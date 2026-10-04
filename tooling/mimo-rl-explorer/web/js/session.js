// Session state and the "N running" badge, shared by every view. No top-level await here: views import
// this, and a module that waits at load time while a view imports it deadlocks the page.
import { $, api, LIVE_STATUSES } from "./util.js";

let session = { user: null, local: false, oauth: false };
export const getSession = () => session;
export const setSession = (s) => { session = s; };

export async function refreshActive() {
  const b = $("#nav-active");
  if (!session.user) { b.hidden = true; return; }
  try {
    const { runs } = await api("/api/runs");
    const n = runs.filter((r) => LIVE_STATUSES.includes(r.status)).length;
    b.hidden = !n;
    b.innerHTML = n ? `<i class="pulse"></i>${n}` : "";
    b.title = n ? `${n} running` : "";
  } catch { /* offline: try again next tick */ }
}

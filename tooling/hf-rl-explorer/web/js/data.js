// Data more than one view reads: every Harbor dataset on the Hub, and the signed-in visitor's own. Fetched once.
import { api } from "./util.js";
import { getSession } from "./session.js";

let envs = null, mine = null;

export function environments() {
  if (!envs) {
    envs = api("/api/environments").then((d) => {
      d.datasets.forEach((x) => { x._blob = `${x.id} ${x.heading || ""} ${x.brief} ${x.tags.join(" ")}`.toLowerCase(); });
      d.byKey = Object.fromEntries(d.datasets.map((x) => [x.key, x]));
      return d;
    }).catch((e) => { envs = null; throw e; });
  }
  return envs;
}

// the visitor's own Harbor datasets and their organizations', private ones included; [] when signed out
export function myDatasets() {
  if (!getSession().user) return Promise.resolve([]);
  if (!mine) mine = api("/api/environments/mine").then((d) => d.datasets).catch(() => { mine = null; return []; });
  return mine;
}
export function forgetMine() { mine = null; }

// Ranking is centralized in /api/search so list, filters and individual ranks agree.

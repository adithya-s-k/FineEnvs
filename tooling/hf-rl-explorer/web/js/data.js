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

// trending first (the Hub's score, the same scale for datasets and Spaces), then likes, then downloads
export const trendingKey = (d) => d.trending * 1e9 + d.likes * 1e4 + Math.min(d.downloads || 0, 9999);

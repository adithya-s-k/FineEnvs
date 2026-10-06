// Data access. Paths resolve against the origin (the viewer may be served at /viewer/ or embedded).
const cache = new Map();

function get(path) {
  const url = new URL(path, location.origin).toString();
  if (!cache.has(url)) {
    const p = fetch(url, { headers: { Accept: "application/json" } }).then((r) => {
      if (!r.ok) throw new Error(`${r.status} ${r.statusText || ""} · ${path}`.trim());
      return r.json();
    });
    p.catch(() => cache.delete(url));
    cache.set(url, p);
  }
  return cache.get(url);
}

const enc = encodeURIComponent;
export const getTasks = () => get("/api/tasks");
export const getTask = (id) => get(`/api/tasks/${enc(id)}`);
export const getReference = (id) => get(`/api/tasks/${enc(id)}/reference`);
export const getRuns = () => get("/api/runs").catch(() => []);
export const getRun = (run) => get(`/api/runs/${enc(run)}`);
export const getEpisode = (run, model, taskId) => get(`/api/runs/${enc(run)}/episode?model=${enc(model)}&task_id=${enc(taskId)}`);

/** Every episode of every run, each tagged with its run name. */
export async function getAllEpisodes() {
  const runs = (await getRuns()) || [];
  const details = await Promise.all(runs.map((r) => getRun(r.run).catch(() => null)));
  const out = [];
  details.forEach((d, i) => {
    if (!d) return;
    for (const e of d.episodes || []) out.push({ ...e, run: d.run || runs[i].run });
  });
  return { runs, episodes: out };
}

/** Live endpoints are polled, so they bypass the cache. */
async function fresh(path) {
  const r = await fetch(new URL(path, location.origin).toString(), { headers: { Accept: "application/json" }, cache: "no-store" });
  if (!r.ok) {
    const err = new Error(`${r.status} ${r.statusText || ""} · ${path}`.trim());
    err.status = r.status;
    throw err;
  }
  return r.json();
}
export const getLiveEpisodes = () => fresh("/api/episodes");
export const getLiveEpisode = (id) => fresh(`/api/episodes/${enc(id)}`);

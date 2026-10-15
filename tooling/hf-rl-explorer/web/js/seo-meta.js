export const SITE = "HF RL Explorer";
export const DESCRIPTION = "Explore reinforcement learning environments and tasks on the Hugging Face Hub. Browse OpenEnv, Harbor, MiMo, NeMo Gym and Verifiers, inspect rewards, and run supported agent rollouts.";

// A Space task is a distinct resource. Ignore filters and tracking parameters,
// but preserve its environment, split and numeric task index in a stable order.
export function canonicalPath(url) {
  const u = new URL(url), q = u.searchParams;
  if (/^\/s\/[^/]+\/[^/]+$/.test(u.pathname) && /^\d{1,10}$/.test(q.get("task") || "") && q.get("env") && q.get("split")) {
    return u.pathname + "?" + new URLSearchParams({ env: q.get("env"), split: q.get("split"), task: String(Number(q.get("task"))) });
  }
  return u.pathname;
}

export function imagePath(url) {
  const path = canonicalPath(url), [name, query] = path.split("?");
  return /^\/(d|t|s)\//.test(name) ? `/og${name}.png${query ? `?${query}` : ""}` : "/social/rl-explorer.png";
}

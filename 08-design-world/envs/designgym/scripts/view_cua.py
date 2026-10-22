"""Browse a recorded CUA dataset: stats, filters, and a step-through player.

    python scripts/view_cua.py artifacts/cua          # writes artifacts/cua/index.html
    python -m http.server -d artifacts/cua 8097       # then open http://localhost:8097

The page reads `cua.jsonl` (inlined) and the dataset's own image files (by
relative path), so what you step through is exactly what a trainer would load.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

NAMES = {"figma": "Studio", "canva": "Template", "photoshop": "Pixel", "slides": "Deck", "powerpoint": "Present",
         "inkscape": "Vector"}


def _calls(msg):
    out = []
    for c in msg.get("tool_calls", []):
        args = json.loads(c["function"]["arguments"])
        parts = [f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items() if v is not None]
        out.append(f"{c['function']['name']}({', '.join(parts)})")
    return out


def summarise(traj: dict) -> dict:
    msgs = traj["messages"]
    user_imgs = [c["image"] for c in msgs[1]["content"] if c["type"] == "image"]
    steps, frames = [], [user_imgs[-1]]
    i = 2
    while i < len(msgs):
        a = msgs[i]
        n = len(a.get("tool_calls", []))
        shot, extra = None, []
        for r in msgs[i + 1:i + 1 + n]:
            if isinstance(r["content"], list):
                ims = [c["image"] for c in r["content"] if c["type"] == "image"]
                if ims:
                    shot = ims[-1]
                    extra += ims[:-1]
        steps.append({"t": a.get("content") or "", "c": _calls(a), "m": a.get("train") is False,
                      "f": len(frames) if shot else None, "x": extra})
        if shot:
            frames.append(shot)
        i += 1 + n
    sub = next((k for k, s in enumerate(steps) if any(w in s["t"] for w in ("Export", "export", "Submit", "submit"))),
               len(steps) - 1)
    done = next((s["f"] for s in reversed(steps[:sub]) if s["f"] is not None), len(frames) - 1)
    return {"meta": traj["meta"], "target": user_imgs[0] if traj["meta"]["mode"] == "reference" else None,
            "frames": frames, "steps": steps, "done": done}


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>CUA dataset viewer</title>
<style>
:root { --bg:#f7f7f5; --card:#fff; --fg:#1d1d1f; --muted:#6b6b70; --line:#e3e3e0; --accent:#4f46e5; --bad:#b91c1c; --code:#f1f1ee; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#141416; --card:#1d1d20; --fg:#ececef; --muted:#9a9aa2; --line:#2e2e33; --accent:#8b85ff; --bad:#f87171; --code:#26262a; } }
* { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 -apple-system,system-ui,Segoe UI,sans-serif }
main { max-width:1240px; margin:0 auto; padding:26px 16px 60px } h1 { font-size:24px; margin:0 0 4px } h2 { font-size:18px; margin:28px 0 10px }
.sub { color:var(--muted); margin:0 0 16px } .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px }
.stats { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:10px } .stat b { display:block; font-size:22px } .stat span { color:var(--muted); font-size:13px }
.chips { display:flex; flex-wrap:wrap; gap:6px; margin:12px 0 } .chips button { border:1px solid var(--line); background:var(--card); color:var(--fg); padding:5px 11px; border-radius:999px; cursor:pointer; font:inherit; font-size:13px }
.chips button.on { background:var(--accent); border-color:var(--accent); color:#fff } .chips .lab { color:var(--muted); font-size:13px; align-self:center; margin-right:2px }
.list { display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:10px }
.tr { border:1px solid var(--line); border-radius:10px; background:var(--card); padding:8px; cursor:pointer; text-align:left; color:var(--fg); font:inherit }
.tr.on { outline:2px solid var(--accent) } .tr img { width:100%; aspect-ratio:16/10; object-fit:cover; border-radius:6px; display:block; background:var(--code) }
.tr .k { font-size:13px; margin-top:6px } .tr .m { font-size:12px; color:var(--muted) }
.player { display:grid; grid-template-columns:minmax(0,1.6fr) minmax(0,1fr); gap:14px } @media (max-width:880px) { .player { grid-template-columns:1fr } }
.shot img { width:100%; border-radius:8px; border:1px solid var(--line); display:block } .ctrl { display:flex; gap:8px; align-items:center; margin-top:8px } .ctrl input { flex:1 }
.ctrl button { border:1px solid var(--line); background:var(--card); color:var(--fg); border-radius:8px; padding:4px 10px; cursor:pointer }
.side { display:flex; flex-direction:column; gap:10px } .side .ref { display:flex; gap:10px; align-items:flex-start } .side .ref img { width:120px; border-radius:6px; border:1px solid var(--line) }
.steps { position:relative; max-height:470px; overflow:auto; border:1px solid var(--line); border-radius:8px }
.step { padding:7px 10px; border-bottom:1px solid var(--line); cursor:pointer; font-size:13px } .step.cur { background:color-mix(in srgb, var(--accent) 13%, transparent) }
.n { color:var(--muted); font-size:12px } .calls { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px }
.calls div { background:var(--code); border-radius:5px; padding:2px 6px; margin-top:3px; overflow-wrap:anywhere } .slip { color:var(--bad); font-weight:600; font-size:11px; margin-left:6px }
.now { margin-top:10px; font-size:14px } .muted { color:var(--muted) } pre { background:var(--code); border-radius:8px; padding:10px; overflow:auto; font-size:12px; max-height:240px; white-space:pre-wrap }
</style></head><body><main>
<h1>CUA dataset viewer</h1><p class="sub" id="src"></p>
<div class="card"><div class="stats" id="stats"></div></div>
<h2>Trajectories</h2>
<div id="filters"></div><div class="list" id="list"></div>
<h2 id="ptitle">Player</h2>
<div class="card"><div class="player">
 <div><div class="shot"><img id="frame" alt="Editor screenshot"></div>
  <div class="ctrl"><button id="prev" aria-label="Previous">&#8592;</button><button id="play">Play</button><input type="range" id="slider" min="0" value="0" aria-label="Turn"><button id="next" aria-label="Next">&#8594;</button></div>
  <div class="now" id="now"></div></div>
 <div class="side"><div class="ref" id="ref"></div><div class="steps" id="steps"></div></div>
</div></div>
</main><script>
const D = __DATA__;
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;"})[c]);
const T = D.trajs;
document.getElementById("src").textContent = `${D.source} · recorder v${T[0]?.meta.recorder_version ?? "?"} · every export scored ≥ 0.98 or was dropped`;
const stat = (v, l) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`;
const S = D.stats;
document.getElementById("stats").innerHTML = [
  stat(S.n, "trajectories"), stat(Object.entries(S.skins).map(([k, v]) => `${D.names[k]} ${v}`).join(" · "), "per layout"),
  stat(Object.entries(S.modes).map(([k, v]) => `${k} ${v}`).join(" · "), "per mode"),
  stat(Object.entries(S.themes).map(([k, v]) => `${k} ${v}`).join(" · "), "per theme"),
  stat((T.reduce((a, t) => a + (t.meta.reward || 0), 0) / Math.max(1, T.length)).toFixed(4), "mean reward"), stat(S.mean_turns, "mean turns"), stat(S.mean_actions, "mean actions"),
  stat(S.images, "screenshots"), stat(S.slips, "masked slips"), stat(S.failed, "failed (dropped)"),
  stat(Object.entries(S.skipped).map(([k, v]) => `${v}`).reduce((a, b) => (+a) + (+b), 0), "skipped (not GUI-expressible)")].join("");

const F = { skin: null, mode: null, theme: null, layout: null };
const facets = [["skin", "Layout", (v) => D.names[v]], ["mode", "Mode", (v) => v], ["theme", "Theme", (v) => v], ["layout", "Panels", (v) => v === "swap" ? "mirrored" : "default"]];
const fEl = document.getElementById("filters");
function drawFilters() {
  fEl.innerHTML = facets.map(([k, lab, name]) => {
    const vals = [...new Set(T.map((t) => t.meta[k]))].sort();
    if (vals.length < 2) return "";  // nothing to filter on
    return `<div class="chips"><span class="lab">${lab}</span><button data-k="${k}" data-v="" class="${F[k] === null ? "on" : ""}">all</button>` +
      vals.map((v) => `<button data-k="${k}" data-v="${v}" class="${F[k] === v ? "on" : ""}">${name(v)}</button>`).join("") + "</div>";
  }).join("");
  fEl.querySelectorAll("button").forEach((b) => b.onclick = () => { F[b.dataset.k] = b.dataset.v || null; drawFilters(); drawList(); });
}
let curT = null, i = 0, timer = null;
function drawList() {
  const shown = T.map((t, k) => [t, k]).filter(([t]) => Object.entries(F).every(([k, v]) => v === null || t.meta[k] === v));
  document.getElementById("list").innerHTML = shown.map(([t, k]) => {
    const m = t.meta;
    return `<button class="tr ${k === curT ? "on" : ""}" data-k="${k}"><img loading="lazy" src="${t.frames[t.done]}" alt="finished canvas">
      <div class="k"><b>${D.names[m.skin]}</b> · ${m.theme} · ${m.layout === "swap" ? "mirrored" : "default"} · ${m.mode}</div>
      <div class="m">${m.task_id} · ${m.n_elements} layers · ${m.turns} turns · ${m.actions} actions${m.masked_turns ? ` · ${m.mistakes.length} slip(s)` : ""}</div></button>`;
  }).join("") || '<p class="muted">No trajectories match.</p>';
  document.querySelectorAll(".tr").forEach((b) => b.onclick = () => load(+b.dataset.k));
}
const frame = document.getElementById("frame"), slider = document.getElementById("slider"), stepsEl = document.getElementById("steps"), now = document.getElementById("now");
function show(k) {
  const t = T[curT]; i = Math.max(0, Math.min(t.steps.length - 1, k)); slider.value = i;
  const s = t.steps[i]; let f = s.f; for (let j = i; f == null && j >= 0; j--) f = t.steps[j].f;
  frame.src = t.frames[f ?? 0];
  now.innerHTML = `<b>Turn ${i + 1} / ${t.steps.length}</b>${s.m ? '<span class="slip">SLIP · masked from the loss</span>' : ""}
    <div>${esc(s.t) || '<span class="muted">(no thought: continues the previous edit)</span>'}</div><div class="calls">${s.c.map((c) => `<div>${esc(c)}</div>`).join("")}</div>
    ${s.x.length ? `<div class="muted" style="margin-top:6px">returned: ${s.x.map((x) => `<a href="${x}" target="_blank">image</a>`).join(", ")}</div>` : ""}`;
  [...stepsEl.children].forEach((el, j) => el.classList.toggle("cur", j === i));
  const el = stepsEl.children[i];
  if (el && (el.offsetTop < stepsEl.scrollTop || el.offsetTop + el.offsetHeight > stepsEl.scrollTop + stepsEl.clientHeight)) stepsEl.scrollTop = el.offsetTop - stepsEl.clientHeight / 3;
}
function load(k) {
  curT = k; const t = T[k], m = t.meta; slider.max = t.steps.length - 1;
  document.getElementById("ptitle").textContent = `Player · ${m.task_id} · ${D.names[m.skin]} · ${m.theme} · ${m.layout === "swap" ? "mirrored" : "default"} · ${m.mode} mode`;
  document.getElementById("ref").innerHTML = t.target ? `<img src="${t.target}" alt="target"><div class="muted" style="font-size:13px">Target the agent was shown (reference mode). Reward after submitting: <b>${m.reward.toFixed(3)}</b></div>`
    : `<div class="muted" style="font-size:13px">Description mode: the agent works from a written brief in its instructions, no target image. Reward: <b>${m.reward.toFixed(3)}</b></div>`;
  stepsEl.innerHTML = t.steps.map((s, j) => `<div class="step"><span class="n">${j + 1}</span>${s.m ? '<span class="slip">SLIP</span>' : ""}<div>${esc(s.t) || "…"}</div><div class="calls">${s.c.map((c) => `<div>${esc(c)}</div>`).join("")}</div></div>`).join("");
  [...stepsEl.children].forEach((el, j) => el.onclick = () => show(j));
  drawList(); show(0);
}
slider.oninput = () => show(+slider.value);
document.getElementById("prev").onclick = () => show(i - 1);
document.getElementById("next").onclick = () => show(i + 1);
document.getElementById("play").onclick = (e) => {
  if (timer) { clearInterval(timer); timer = null; e.target.textContent = "Play"; return; }
  e.target.textContent = "Pause";
  timer = setInterval(() => { if (i >= T[curT].steps.length - 1) { clearInterval(timer); timer = null; e.target.textContent = "Play"; } else show(i + 1); }, 650);
};
drawFilters(); if (T.length) load(0); else drawList();
</script></body></html>"""


def build(root: Path) -> Path:
    trajs = [summarise(json.loads(line)) for line in (root / "cua.jsonl").read_text().splitlines() if line.strip()]
    metas = [t["meta"] for t in trajs]
    summary = json.loads((root / "summary.json").read_text()) if (root / "summary.json").exists() else {}
    failed = len((root / "failures.jsonl").read_text().splitlines()) if (root / "failures.jsonl").exists() else 0
    stats = {
        "n": len(trajs), "skins": dict(Counter(m["skin"] for m in metas)), "modes": dict(Counter(m["mode"] for m in metas)),
        "themes": dict(Counter(m["theme"] for m in metas)), "mirrored": sum(m["layout"] == "swap" for m in metas),
        "mean_turns": round(sum(m["turns"] for m in metas) / max(1, len(metas)), 1),
        "mean_actions": round(sum(m["actions"] for m in metas) / max(1, len(metas)), 1),
        "images": sum(len(t["frames"]) for t in trajs), "slips": sum(len(m["mistakes"]) for m in metas),
        "failed": failed, "skipped": summary.get("skipped", {}),
    }
    data = {"trajs": trajs, "stats": stats, "names": NAMES, "source": str(root / "cua.jsonl")}
    out = root / "index.html"
    out.write_text(PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False)))
    return out


if __name__ == "__main__":
    print(build(Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/cua")))

// Scene + time scrubber + berth chart, kept in sync (hour, hover, selection). Used by the task and rollout pages.
import { BerthChart, legendHtml } from "./chart.js";
import { clockAt, escapeHtml, hasCranes } from "./model.js";

const ICON_PLAY = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M4 2.5v11l9-5.5z" fill="currentColor"/></svg>';
const ICON_PAUSE = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M3.5 2.5h3v11h-3zM9.5 2.5h3v11h-3z" fill="currentColor"/></svg>';

export function sceneMarkup() {
  return `
  <div class="stage">
    <div class="scene-host"><div class="scene-loading">Loading 3D scene…</div></div>
    <div class="ov ov-tl"></div>
    <div class="ov ov-tr">
      <button type="button" class="ovb" data-act="focus" hidden title="Move the camera to the selected ship">Focus</button>
      <button type="button" class="ovb" data-act="cargo" title="Watch the next container transfer at 10× real time">Cargo</button>
      <select class="ovs" data-act="light" aria-label="Scene lighting" title="Presentation lighting, independent of the simulation clock"><option value="afternoon">Mediterranean</option><option value="golden">Golden hour</option></select>
      <select class="ovs" data-act="view" aria-label="Camera view"><option value="overview">Overview</option><option value="harbour">Harbour</option><option value="quayside">Quayside</option><option value="overhead">Overhead</option></select>
      <button type="button" class="ovb" data-act="names" aria-pressed="true" title="Show ship names in the scene">Names</button>
      <button type="button" class="ovb" data-act="reset" title="Back to the overview">Reset</button>
      <button type="button" class="ovb" data-act="cinema" aria-pressed="false" title="Expand the harbour view · Escape to return">Cinema</button>
    </div>
    <div class="ov-attr">Port of Barcelona twin: <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">© OpenStreetMap contributors</a> · terrain: Terrain Tiles (AWS)</div>
    <div class="ov ov-bottom scrub">
      <button type="button" class="ovb play" data-act="play" aria-label="Play">${ICON_PLAY}</button>
      <select class="ovs speed" aria-label="Playback speed" title="Playback speed (hours per second)">
        <option value="0.0002777777777777778">Real time</option><option value="0.002777777777777778">10× real time</option><option value="0.016666666666666666">1 min/s</option>
        <option value="0.05">3 min/s</option><option value="0.25">15 min/s</option><option value="1">1 h/s</option><option value="2">2 h/s</option><option value="4" selected>4 h/s</option><option value="8">8 h/s</option><option value="16">16 h/s</option>
      </select>
      <input class="range" type="range" min="0" max="100" step="0.0002777777777777778" value="0" aria-label="Hour of the week">
      <div class="clock"><b class="c-main">–</b><span class="c-sub"></span></div>
    </div>
  </div>`;
}

export function chartMarkup({ chartTitle = "Dock chart", extra = "" } = {}) {
  return `
  <section class="chart-sec">
    <div class="sec-head"><h2>${escapeHtml(chartTitle)}</h2>${extra}<div class="legend">${legendHtml()}</div></div>
    <div class="chart-host"></div>
  </section>`;
}

export function stageMarkup(opts = {}) {
  return sceneMarkup() + chartMarkup(opts);
}

export function createStage(root, { task, horizon, sceneMod, chartMaxHeight, onSelect, onHover, onTime, editable = false, onMove, onCranes, chartFill = false }) {
  const host = root.querySelector(".scene-host");
  const tl = root.querySelector(".ov-tl");
  const tr = root.querySelector(".ov-tr");
  const btnFocus = tr.querySelector('[data-act="focus"]');
  const btnNames = tr.querySelector('[data-act="names"]');
  const btnReset = tr.querySelector('[data-act="reset"]');
  const btnPlay = root.querySelector('[data-act="play"]');
  const speedSel = root.querySelector(".speed");
  const range = root.querySelector(".range");
  const cMain = root.querySelector(".c-main");
  const cSub = root.querySelector(".c-sub");
  const chartHost = root.querySelector(".chart-host");

  const st = { t: 0, H: horizon, playing: false, speed: 4, hover: null, selected: null, ev: null, names: true };
  let names = true;
  try {
    names = localStorage.getItem("berth.names") !== "0";
  } catch {}
  st.names = names;
  btnNames.setAttribute("aria-pressed", String(names));

  root.querySelector(".stage").classList.add("ready");
  const scene = sceneMod.getScene();
  const loading = host.querySelector(".scene-loading");
  if (loading) loading.remove();
  scene.mount(host);
  scene.fmt = (h) => {
    const c = clockAt(task, h);
    return `${c.day} ${c.hm}`;
  };
  scene.setTask(task);
  const lightSel = tr.querySelector('[data-act="light"]');
  try { scene.setAtmosphere(localStorage.getItem("berth.atmosphere") || "afternoon"); } catch {}
  lightSel.value = scene.atmosphere;
  lightSel.addEventListener("change", () => {
    scene.setAtmosphere(lightSel.value);
    try { localStorage.setItem("berth.atmosphere", lightSel.value); } catch {}
  });
  const stageEl = root.querySelector(".stage");
  const btnCinema = tr.querySelector('[data-act="cinema"]');
  const setCinema = on => {
    stageEl.classList.toggle("cinema", on);
    btnCinema.setAttribute("aria-pressed", String(on));
    btnCinema.textContent = on ? "Exit cinema" : "Cinema";
  };
  btnCinema.addEventListener("click", () => setCinema(!stageEl.classList.contains("cinema")));
  scene.setNames(names);
  scene.onHover = (id) => setHover(id, "scene");
  scene.onSelect = (id) => select(id, { focus: false, from: "scene" });

  const chart = new BerthChart(chartHost, {
    maxHeight: chartMaxHeight,
    editable,
    onMove,
    onCranes,
    fill: chartFill,
    onHover: (id) => setHover(id, "chart"),
    onPick: (h, id) => {
      if (h != null && (id == null || !editable)) setTime(h);
      if (id != null) select(id, { focus: !editable, from: "chart" });
      else if (editable && h != null) select(null, { from: "chart" });
    },
    onScrub: (h) => setTime(h),
  });

  range.max = String(horizon);
  const legendEl = root.querySelector(".chart-sec .legend, .pa-chart-bar .legend");
  if (legendEl) legendEl.innerHTML = legendHtml({ cranes: hasCranes(task) });

  function updateClock() {
    const c = clockAt(task, st.t);
    cMain.textContent = `${c.day} ${c.date} · ${c.hm}`;
    cSub.textContent = `UTC · h ${Math.floor(st.t)}`;
  }

  let lastEmit = -1;
  function setTime(t, { fromRange = false } = {}) {
    st.t = Math.max(0, Math.min(st.H, t));
    if (!fromRange) range.value = String(st.t);
    scene.setTime(st.t);
    chart.setTime(st.t);
    updateClock();
    const q = Math.floor(st.t * 4);
    if (q !== lastEmit) {
      lastEmit = q;
      if (onTime) onTime(st.t);
    }
  }

  function setHover(id, from) {
    if (st.hover === id) return;
    st.hover = id;
    scene.setHover(id);
    chart.setHover(typeof id === "number" ? id : null);
    if (onHover) onHover(typeof id === "number" ? id : null, from);
  }

  function select(id, { focus = false, from = "" } = {}) {
    st.selected = id;
    scene.setSelected(id);
    chart.setSelected(typeof id === "number" ? id : null);
    const name = id == null ? "" : typeof id === "number" ? (task.ships.find((s) => s.id === id) || {}).name : (task.blocks || [])[Number(String(id).slice(1))]?.label;
    btnFocus.hidden = id == null;
    btnFocus.textContent = id == null ? "Focus" : `Focus ${name || ""}`.trim();
    if (focus && id != null) scene.focus(id);
    if (onSelect) onSelect(typeof id === "number" ? id : null, from);
  }

  function setPlaying(on) {
    if (on && st.t >= st.H - 0.01) setTime(0);
    st.playing = on;
    btnPlay.innerHTML = on ? ICON_PAUSE : ICON_PLAY;
    btnPlay.setAttribute("aria-label", on ? "Pause" : "Play");
  }

  btnPlay.addEventListener("click", () => setPlaying(!st.playing));
  tr.querySelector('[data-act="cargo"]').addEventListener('click', () => {
    const transfer = scene.nextCargoTransfer();
    if (!transfer) return;
    setTime(transfer.start);
    st.cargoUntil = transfer.end;
    st.speed = 10 / 3600;
    speedSel.value = String(st.speed);
    setPlaying(true);
  });
  speedSel.addEventListener("change", () => {
    st.cargoUntil = null;
    st.speed = Number(speedSel.value);
  });
  range.addEventListener("input", () => { st.cargoUntil = null; setTime(Number(range.value), { fromRange: true }); });
  btnNames.addEventListener("click", () => {
    st.names = !st.names;
    btnNames.setAttribute("aria-pressed", String(st.names));
    scene.setNames(st.names);
    try {
      localStorage.setItem("berth.names", st.names ? "1" : "0");
    } catch {}
  });
  tr.querySelector('[data-act="view"]').addEventListener("change", e => scene.setView(e.target.value));
  btnReset.addEventListener("click", () => {
    tr.querySelector('[data-act="view"]').value = "overview";
    scene._camDone = true;
    scene.resetView(true);
  });
  btnFocus.addEventListener("click", () => st.selected != null && scene.focus(st.selected));

  const onKey = (e) => {
    if (e.key === "Escape" && stageEl.classList.contains("cinema")) { setCinema(false); return; }
    if (e.target.closest && e.target.closest("input, select, textarea, button, summary, [data-keys]")) return;
    if (e.key === " ") {
      e.preventDefault();
      setPlaying(!st.playing);
    } else if (e.key === "ArrowRight") setTime(st.t + (e.shiftKey ? 6 : 1));
    else if (e.key === "ArrowLeft") setTime(st.t - (e.shiftKey ? 6 : 1));
    else if (e.key === "Escape") select(null);
  };
  window.addEventListener("keydown", onKey);

  let raf = 0;
  let last = performance.now();
  const loop = (now) => {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (st.playing) {
      const nt = st.t + dt * st.speed;
      if (st.cargoUntil != null && nt >= st.cargoUntil) {
        setTime(st.cargoUntil);
        st.cargoUntil = null;
        setPlaying(false);
      } else if (nt >= st.H) {
        setTime(st.H);
        setPlaying(false);
      } else setTime(nt);
    }
    raf = requestAnimationFrame(loop);
  };
  raf = requestAnimationFrame(loop);

  return {
    tl,
    scene,
    chart,
    get time() {
      return st.t;
    },
    get selected() {
      return st.selected;
    },
    setEvaluation(ev, H) {
      st.cargoUntil = null;
      st.ev = ev;
      if (H != null) {
        st.H = H;
        range.max = String(H);
      }
      scene.setPlan(ev, st.H);
      tr.querySelector('[data-act="cargo"]').disabled = !scene.cargo?.jobs.length;
      st.H = Math.max(st.H, scene.visualHorizon);
      range.max = String(st.H);
      chart.setData(task, ev, st.H);
      setTime(st.t);
    },
    /** Update the 3D scene only (live drag preview); the chart shows its own preview. */
    preview(ev) {
      scene.setPlan(ev, st.H, { snap: false });
    },
    setTime,
    select,
    hover: (id) => setHover(id, "ext"),
    setPlaying,
    setChartMaxHeight: (h) => chart.setMaxHeight(h),
    destroy() {
      cancelAnimationFrame(raf);
      window.removeEventListener("keydown", onKey);
      scene.onHover = null;
      scene.onSelect = null;
      scene.setHover(null);
      scene.setSelected(null);
      scene.unmount();
      chart.destroy();
    },
  };
}

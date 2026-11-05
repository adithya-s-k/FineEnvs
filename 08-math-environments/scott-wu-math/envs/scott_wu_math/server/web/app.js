// The play page. It holds one OpenEnv WebSocket session (/ws): "reset" deals, "step" submits the
// final expression. Combining cards on the table only builds that expression; the server grades it.
(function () {
  const $ = (id) => document.getElementById(id);
  const LEVEL_TITLES = { "24": "24", "three-digit": "Three-digit", year: "Year", "four-digit": "Four-digit", custom: "Your number", video: "From the video" };
  const SYM = { "+": "+", "-": "−", "*": "×", "/": "÷" };
  const PREC = { "+": 1, "-": 1, "*": 2, "/": 2 };

  // ---- exact fractions ----------------------------------------------------
  const gcd = (a, b) => { a = Math.abs(a); b = Math.abs(b); while (b) [a, b] = [b, a % b]; return a || 1; };
  const frac = (n, d = 1) => { if (d < 0) { n = -n; d = -d; } const g = gcd(n, d); return { n: n / g, d: d / g }; };
  const apply = (op, a, b) => {
    if (op === "+") return frac(a.n * b.d + b.n * a.d, a.d * b.d);
    if (op === "-") return frac(a.n * b.d - b.n * a.d, a.d * b.d);
    if (op === "*") return frac(a.n * b.n, a.d * b.d);
    if (b.n === 0) return null;
    return frac(a.n * b.d, a.d * b.n);
  };
  const show = (f) => (f.d === 1 ? String(f.n) : `${f.n}/${f.d}`);
  const num = (n) => (Math.abs(n) < 10000 ? String(n) : n.toLocaleString("en-US"));
  const pretty = (f) => (f.d === 1 ? num(f.n) : `${f.n}/${f.d}`);

  // Same parenthesisation as the server's to_text.
  function text(tree, symbols) {
    const walk = (t) => {
      if (typeof t === "number") return [String(t), 3];
      const [op, l, r] = t;
      let [lt, lp] = walk(l), [rt, rp] = walk(r);
      if (lp < PREC[op]) lt = `(${lt})`;
      if (rp < PREC[op] || (rp === PREC[op] && (op === "-" || op === "/"))) rt = `(${rt})`;
      return [`${lt} ${symbols ? SYM[op] : op} ${rt}`, PREC[op]];
    };
    return walk(tree)[0];
  }

  // ---- state ----------------------------------------------------------------
  const S = {
    ws: null, ready: null, pending: null,
    level: "year", videoIndex: 0,
    task: null, piles: [], history: [], steps: [],
    selected: null, op: null, done: false, busy: false,
    startedAt: 0, timer: null, nextId: 0,
  };

  // ---- OpenEnv session --------------------------------------------------------
  function connect() {
    if (S.ws && S.ws.readyState <= 1) return S.ready;
    const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
    S.ws = ws;
    S.ready = new Promise((resolve, reject) => {
      ws.onopen = () => resolve();
      ws.onerror = () => reject(new Error("Could not reach the environment server."));
    });
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      const p = S.pending; S.pending = null;
      if (!p) return;
      if (msg.type === "error") p.reject(new Error((msg.data && msg.data.message) || "Server error"));
      else p.resolve(msg.data);
    };
    ws.onclose = () => {
      if (S.pending) { S.pending.reject(new Error("The session closed. Deal again.")); S.pending = null; }
      S.ws = null;
    };
    return S.ready;
  }

  async function send(type, data) {
    await connect();
    return new Promise((resolve, reject) => {
      S.pending = { resolve, reject };
      S.ws.send(JSON.stringify({ type, data }));
    });
  }

  // ---- dealing -----------------------------------------------------------------
  async function deal() {
    if (S.busy) return;
    let data;
    if (S.level === "video") {
      data = { split: "video", index: S.videoIndex };
      S.videoIndex = (S.videoIndex + 1) % 2;
    } else if (S.level === "custom") {
      const raw = $("custom-target").value.replace(/[,\s_]/g, "");
      if (!/^\d{1,6}$/.test(raw)) { setStatus("Type a whole number for the target first.", "warn"); $("custom-target").focus(); return; }
      data = { target: Number(raw) };
    } else if (S.level === "year") {
      data = { level: "year", target: new Date().getFullYear() };
    } else {
      data = { level: S.level };
    }
    S.busy = true;
    setStatus("Dealing…");
    try {
      const res = await send("reset", data);
      start(res.observation);
    } catch (e) {
      setStatus(e.message, "warn");
    } finally {
      S.busy = false;
    }
  }

  function start(obs) {
    S.task = obs;
    S.piles = obs.cards.map((c) => ({ id: S.nextId++, value: frac(c.rank), tree: c.rank, card: c, count: 1 }));
    S.history = []; S.steps = []; S.selected = null; S.op = null; S.done = false;
    $("target").textContent = num(obs.target);
    $("task-id").textContent = obs.task_id.startsWith("video-") ? "As dealt in the video" : `${obs.cards.length} cards`;
    $("prompt").textContent = obs.system_prompt + "\n\n" + obs.prompt;
    $("expr").value = "";
    $("result").hidden = true;
    $("table").classList.remove("over");
    S.startedAt = performance.now();
    clearInterval(S.timer);
    S.timer = setInterval(tick, 250);
    tick();
    render();
    setStatus("");
  }

  function tick() {
    const s = Math.floor((performance.now() - S.startedAt) / 1000);
    $("timer").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }

  // ---- combining on the table ---------------------------------------------------
  function pick(id) {
    if (S.done || !S.task) return;
    if (S.selected === null) { S.selected = id; S.op = null; }
    else if (S.selected === id) { S.selected = null; S.op = null; }
    else if (S.op) { combine(S.selected, S.op, id); return; }
    else S.selected = id;
    render();
  }

  function chooseOp(op) {
    if (S.done || S.selected === null) { setStatus("Pick a card first.", "hint"); return; }
    S.op = S.op === op ? null : op;
    render();
  }

  function combine(aId, op, bId) {
    const a = S.piles.find((p) => p.id === aId), b = S.piles.find((p) => p.id === bId);
    const value = apply(op, a.value, b.value);
    if (!value) { setStatus("Can't divide by zero.", "warn"); S.op = null; render(); return; }
    S.history.push({ piles: S.piles.slice(), steps: S.steps.slice() });
    const made = { id: S.nextId++, value, tree: [op, a.tree, b.tree], card: null, count: a.count + b.count, fresh: true };
    S.steps.push(`${show(a.value)} ${SYM[op]} ${show(b.value)} = ${show(value)}`);
    const at = Math.min(S.piles.indexOf(a), S.piles.indexOf(b));
    S.piles = S.piles.filter((p) => p !== a && p !== b);
    S.piles.splice(Math.min(at, S.piles.length), 0, made);
    S.selected = S.piles.length > 1 ? made.id : null;
    S.op = null;
    render();
    if (S.piles.length === 1) {
      const last = S.piles[0];
      if (last.value.d === 1 && last.value.n === S.task.target) submit(text(last.tree, false));
      else setStatus(`All cards used: ${pretty(last.value)}, not ${num(S.task.target)}. Undo, or submit it as it is.`, "warn", true);
    } else setStatus("");
  }

  function undo() {
    if (S.done || !S.history.length) return;
    const prev = S.history.pop();
    S.piles = prev.piles; S.steps = prev.steps; S.selected = null; S.op = null;
    render(); setStatus("");
  }

  function restart() {
    if (S.done || !S.history.length) return;
    const first = S.history[0];
    S.history = []; S.piles = first.piles; S.steps = []; S.selected = null; S.op = null;
    render(); setStatus("");
  }

  // ---- submitting ---------------------------------------------------------------
  async function submit(expression) {
    if (S.done || S.busy || !S.task) return;
    S.busy = true;
    setStatus("Checking…");
    try {
      const res = await send("step", { expression });
      finish(res, expression);
    } catch (e) {
      setStatus(e.message, "warn");
    } finally {
      S.busy = false;
    }
  }

  function finish(res, expression) {
    S.done = true;
    clearInterval(S.timer);
    const obs = res.observation, r = obs.result || {}, check = r.check || {};
    const secs = obs.elapsed_s != null ? obs.elapsed_s : (performance.now() - S.startedAt) / 1000;
    $("table").classList.add("over");
    let head, cls;
    if (!expression) { head = "Here is one way"; cls = "neutral"; }
    else if (r.exact) { head = `Exact — ${num(obs.target)} in ${fmtTime(secs)}`; cls = "good"; }
    else if (r.valid) { head = `${check.value} — off by ${offBy(check.value, obs.target)}`; cls = "near"; }
    else { head = r.error ? capital(r.error) : "Not a valid answer"; cls = "bad"; }
    const out = $("result");
    out.className = "result " + cls;
    out.replaceChildren();
    const top = document.createElement("div");
    top.className = "result-head";
    top.append(span("result-title", head), span("result-reward", `reward ${Number(res.reward ?? 0).toFixed(2)}`));
    out.append(top);
    if (expression && check.canonical) out.append(kv("You", prettyExpr(check.canonical)));
    if (obs.solution) {
      out.append(kv(r.exact ? "Another way" : "A solution", prettyExpr(obs.solution)));
      const ol = document.createElement("ol");
      ol.className = "steps compact";
      for (const s of obs.solution_steps) ol.append(span("", s, "li"));
      out.append(ol);
    } else if (!r.exact) {
      out.append(kv("A solution", "None found for this hand."));
    }
    const again = document.createElement("button");
    again.className = "btn primary";
    again.textContent = "Deal again";
    again.onclick = deal;
    out.append(again);
    out.hidden = false;
    record(S.task, !!r.exact, secs);
    render();
    setStatus("");
    out.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  // ---- rendering ----------------------------------------------------------------
  function render() {
    const piles = $("piles");
    piles.replaceChildren();
    piles.dataset.count = S.piles.length;
    if (S.task) piles.style.setProperty("--n", S.task.cards.length);
    for (const p of S.piles) {
      const node = p.card ? Cards.playingCard(p.card) : Cards.resultCard(pretty(p.value), text(p.tree, true), p.count);
      const btn = document.createElement("button");
      btn.className = "pile" + (p.id === S.selected ? " selected" : "") + (p.fresh ? " fresh" : "");
      btn.type = "button";
      btn.disabled = S.done;
      btn.append(node);
      btn.setAttribute("aria-pressed", p.id === S.selected ? "true" : "false");
      btn.onclick = () => pick(p.id);
      piles.append(btn);
      p.fresh = false;
    }
    for (const b of document.querySelectorAll("#ops button")) {
      b.disabled = S.done || S.selected === null || S.piles.length < 2;
      b.classList.toggle("on", b.dataset.op === S.op);
    }
    $("undo").disabled = S.done || !S.history.length;
    $("restart").disabled = S.done || !S.history.length;
    $("giveup").disabled = S.done || !S.task;
    $("typed").querySelector("button").disabled = S.done || !S.task;
    $("expr").disabled = S.done || !S.task;
    const steps = $("steps");
    steps.replaceChildren(...S.steps.map((s) => span("", s, "li")));
    $("steps-empty").hidden = S.steps.length > 0;
    const groups = S.piles.filter((p) => !p.card);
    $("expression").textContent = S.piles.length === 1 ? text(S.piles[0].tree, true) : groups.length ? groups.map((p) => text(p.tree, true)).join("   ·   ") : "—";
    for (const b of document.querySelectorAll(".modes button")) b.setAttribute("aria-selected", b.dataset.level === S.level ? "true" : "false");
    $("custom").hidden = S.level !== "custom";
    renderStats();
  }

  function setStatus(msg, kind, withSubmit) {
    const node = $("status");
    node.className = "status" + (kind ? " " + kind : "");
    node.replaceChildren(document.createTextNode(msg));
    if (withSubmit) {
      const b = document.createElement("button");
      b.className = "btn small";
      b.textContent = "Submit anyway";
      b.onclick = () => submit(text(S.piles[0].tree, false));
      node.append(" ", b);
    }
  }

  // ---- per-browser record --------------------------------------------------------
  const STORE = "scott-wu-math-stats-v1";
  function load() { try { return JSON.parse(localStorage.getItem(STORE)) || {}; } catch { return {}; } }
  function record(task, solved, secs) {
    const all = load(), key = task.task_id.startsWith("video-") ? "video" : task.level;
    const s = all[key] || { played: 0, solved: 0, best: null };
    s.played += 1;
    if (solved) { s.solved += 1; s.best = s.best == null ? secs : Math.min(s.best, secs); }
    all[key] = s;
    try { localStorage.setItem(STORE, JSON.stringify(all)); } catch {}
  }
  function renderStats() {
    const all = load(), body = $("stats").querySelector("tbody");
    body.replaceChildren();
    for (const key of Object.keys(LEVEL_TITLES)) {
      const s = all[key];
      if (!s) continue;
      const tr = document.createElement("tr");
      tr.append(span("", LEVEL_TITLES[key], "td"), span("num", `${s.solved}/${s.played}`, "td"), span("num", s.best == null ? "—" : fmtTime(s.best), "td"));
      body.append(tr);
    }
    $("stats").hidden = !body.children.length;
    $("stats").closest("section").hidden = !body.children.length;
  }

  // ---- helpers ------------------------------------------------------------------
  function span(cls, t, tag = "span") { const n = document.createElement(tag); if (cls) n.className = cls; n.textContent = t; return n; }
  function kv(k, v) { const d = document.createElement("div"); d.className = "kv"; d.append(span("k", k), span("v", v)); return d; }
  function prettyExpr(s) { return s.replace(/\*/g, "×").replace(/\//g, "÷").replace(/ - /g, " − "); }
  function capital(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
  function fmtTime(secs) { const s = Math.round(secs); return s < 60 ? `${s}s` : `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; }
  function offBy(valueText, target) {
    const [n, d] = valueText.split("/").map(Number);
    const diff = Math.abs(n / (d || 1) - target);
    return Number.isInteger(diff) ? num(diff) : diff.toFixed(2);
  }

  // ---- wiring -------------------------------------------------------------------
  for (const b of document.querySelectorAll(".modes button")) {
    b.onclick = () => {
      S.level = b.dataset.level;
      render();
      if (S.level === "custom") { $("custom-target").focus(); return; }
      deal();
    };
  }
  for (const b of document.querySelectorAll("#ops button")) b.onclick = () => chooseOp(b.dataset.op);
  $("custom").onsubmit = (e) => { e.preventDefault(); deal(); };
  $("typed").onsubmit = (e) => { e.preventDefault(); const v = $("expr").value.trim(); if (v) submit(v); };
  $("undo").onclick = undo;
  $("restart").onclick = restart;
  $("giveup").onclick = () => submit("");
  $("deal").onclick = deal;
  $("rules-link").onclick = (e) => { e.preventDefault(); $("rules").showModal(); };

  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea, dialog") || e.metaKey || e.ctrlKey || e.altKey) return;
    const ops = { "+": "+", "-": "-", "*": "*", x: "*", X: "*", "/": "/" };
    if (ops[e.key]) { e.preventDefault(); chooseOp(ops[e.key]); }
    else if (e.key === "Backspace") { e.preventDefault(); undo(); }
    else if (e.key === "Escape") { S.selected = null; S.op = null; render(); }
    else if (e.key === "Enter" && S.done) { e.preventDefault(); deal(); }
  });

  // Start on the level in the URL (#24, #four-digit, …), else the year, as in the video.
  const fromHash = decodeURIComponent(location.hash.slice(1));
  if (LEVEL_TITLES[fromHash]) S.level = fromHash;
  render();
  if (S.level !== "custom") deal();
})();

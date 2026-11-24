// The editor's model: the episode's document, kept in sync with the server.
//
// The server is the source of truth. Every edit is sent as an operation over
// the episode's WebSocket; the server validates it, applies it, and pushes the
// new document to every open editor (and records it in the trajectory). For
// responsiveness, property updates are applied locally the moment they are
// sent and then replaced by the server's copy; drags never touch the store at
// all until the pointer is released.

import { fontsLoading } from "./fonts.js";
import { Emitter, assetNames } from "./util.js";

export class Store extends Emitter {
  constructor(episodeId, { source = "editor" } = {}) {
    super();
    this.episodeId = episodeId;
    this.source = source;
    this.doc = null;
    this.task = null;
    this.assets = [];
    this.assetById = new Map();
    this.workspace = null;  // where the images are on "this computer" (core.workspace), when the episode has one
    this.version = -1;
    this.done = false;
    this.result = null;
    this.canUndo = false;
    this.canRedo = false;
    this.selection = [];
    this.tool = "move";
    this.connected = false;
    this._pending = new Map();
    this._seq = 0;
    this._ws = null;
    this._busy = 0; // gestures in progress (drag, text edit)
    this._backoff = 500;
  }

  // ---------------------------------------------------------------- sync

  connect() {
    return new Promise((resolve) => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const url = `${proto}://${location.host}/api/episodes/${this.episodeId}/ws?source=${this.source}`;
      const ws = new WebSocket(url);
      this._ws = ws;
      let first = true;
      ws.onopen = () => {
        this.connected = true;
        this._backoff = 500;
        this.emit("connection", true);
      };
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === "state") {
          this._applyState(msg);
          if (first) { first = false; resolve(); }
        } else if (msg.type === "ack" || msg.type === "pong") {
          const p = this._pending.get(msg.id);
          if (p) {
            this._pending.delete(msg.id);
            msg.ok === false ? p.reject(new Error(msg.error || "rejected")) : p.resolve(msg.result);
          }
          if (!this._pending.size) this.emit("idle");
        } else if (msg.type === "error") {
          this.emit("fatal", msg.message);
        }
      };
      ws.onclose = () => {
        this.connected = false;
        this.emit("connection", false);
        for (const p of this._pending.values()) p.reject(new Error("disconnected"));
        this._pending.clear();
        setTimeout(() => this.connect(), this._backoff);
        this._backoff = Math.min(8000, this._backoff * 2);
      };
    });
  }

  _applyState(msg) {
    if (msg.version < this.version) return;
    const firstLoad = this.doc === null;
    this.version = msg.version;
    this.doc = msg.doc;
    this.task = msg.task;
    if (firstLoad || this.assets.length !== msg.assets.length) {
      this.assets = msg.assets;
      this.assetById = new Map(msg.assets.map((a) => [a.id, a]));
    }
    this.workspace = msg.workspace || null;
    if (msg.facts) for (const a of msg.assets) Object.assign(a, msg.facts[a.id] || {});
    assetNames.clear();
    for (const f of this.workspace?.files || []) assetNames.set(f.asset, f.name);
    this.canUndo = msg.can_undo;
    this.canRedo = msg.can_redo;
    const wasDone = this.done;
    this.done = msg.done;
    this.result = msg.result;
    this.stepCount = msg.step_count;
    this.maxSteps = msg.max_steps;
    const ids = new Set(this.doc.elements.map((e) => e.id));
    const kept = this.selection.filter((id) => ids.has(id));
    if (kept.length !== this.selection.length) {
      this.selection = kept;
      this.emit("selection", this.selection);
    }
    this.emit("doc", { source: msg.source, firstLoad });
    if (msg.source && msg.source !== this.source && !firstLoad) this.emit("remote", msg.source);
    if (!wasDone && this.done) this.emit("done", this.result);
  }

  _send(payload) {
    if (!this._ws || this._ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("not connected"));
    }
    const id = ++this._seq;
    const promise = new Promise((resolve, reject) => this._pending.set(id, { resolve, reject }));
    this._ws.send(JSON.stringify({ ...payload, id }));
    return promise;
  }

  // ---------------------------------------------------------------- edits

  async op(name, args = {}) {
    if (this.done) {
      this.emit("toast", "This episode is finished.");
      return null;
    }
    if (name === "update_element") this._optimistic(args);
    try {
      return await this._send({ type: "op", op: name, args });
    } catch (e) {
      this.emit("toast", e.message);
      this.emit("doc", { source: "rollback" });
      throw e;
    }
  }

  _optimistic(args) {
    const el = this.find(args.element_id);
    if (!el) return;
    const { element_id, ...props } = args;
    Object.assign(el, props);
    this.emit("doc", { source: "local" });
  }

  update(id, props) {
    return this.op("update_element", { element_id: id, ...props });
  }

  async updateMany(changes) {
    // Sequential so the undo history gets one entry per element in order.
    for (const [id, props] of changes) await this.update(id, props);
  }

  undo() { return this._send({ type: "undo" }).catch((e) => this.emit("toast", e.message)); }
  redo() { return this._send({ type: "redo" }).catch((e) => this.emit("toast", e.message)); }
  submit() { return this._send({ type: "submit" }); }

  // ---------------------------------------------------------------- queries

  find(id) { return this.doc?.elements.find((e) => e.id === id) || null; }
  selected() { return this.selection.map((id) => this.find(id)).filter(Boolean); }
  index(id) { return this.doc.elements.findIndex((e) => e.id === id); }

  select(ids, { add = false, toggle = false } = {}) {
    ids = Array.isArray(ids) ? ids : ids ? [ids] : [];
    let next;
    if (toggle) {
      const set = new Set(this.selection);
      for (const id of ids) set.has(id) ? set.delete(id) : set.add(id);
      next = [...set];
    } else if (add) {
      next = [...new Set([...this.selection, ...ids])];
    } else next = ids;
    const same = next.length === this.selection.length && next.every((id, i) => id === this.selection[i]);
    if (same) return;
    this.selection = next;
    this.emit("selection", next);
  }

  setTool(tool) {
    if (this.tool === tool) return;
    this.tool = tool;
    this.emit("tool", tool);
  }

  // ---------------------------------------------------------------- gestures

  busy(on) {
    this._busy = Math.max(0, this._busy + (on ? 1 : -1));
    if (!this._busy && !this._pending.size) this.emit("idle");
  }

  // Resolves once no operation is in flight and no gesture is running. The
  // computer-use driver awaits this before every screenshot.
  whenIdle(timeout = 2000) {
    return new Promise((resolve) => {
      const start = performance.now();
      const check = () => {
        if ((!this._pending.size && !this._busy && !fontsLoading()) || performance.now() - start > timeout) {
          requestAnimationFrame(() => requestAnimationFrame(() => resolve(true)));
        } else setTimeout(check, 16);
      };
      check();
    });
  }
}

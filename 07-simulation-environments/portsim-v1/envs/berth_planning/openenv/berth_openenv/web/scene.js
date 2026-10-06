// The 3D port: one WebGL renderer reused across pages. The app feeds it a task, a plan evaluation and an hour; ships,
// cranes and closures are pure functions of that hour. The port around them is a digital twin of the Port of
// Barcelona built from open data (see twin.js and tools/twin/build_twin.py).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { callKinds, cranePoolAt, escapeHtml, hasCranes } from "./model.js";
import { C, Parts, clamp, disposeTree, lerp, smooth, stripeTexture, vcMat } from "./scene-kit.js";
import { ATMOSPHERES, HAZE, NO_REFLECT, SUN_COLOR, buildSky, buildWater, setSkyAtmosphere, skyEnvironment, sunFor } from "./scene-env.js";
import { WATER_Y, buildWorld, layoutFor } from "./scene-world.js";
import { buildShipModel, shipColors } from "./scene-ships.js";
import { buildTraffic, createWaterMask, escortPoses, hullsOverlap, poseAt, trafficBodies, trafficVessels } from "./navigation.js";
import { buildCrane, craneStyle, craneTargets } from "./scene-cranes.js";
import { dischargeOrder, planDischarges } from './cargo-handling.js';
import { buildCargoReplay } from './scene-cargo.js';
import { loadTwin } from "./twin.js";

const twin = await loadTwin();

const LITE = typeof window !== "undefined" && (window.matchMedia("(pointer: coarse)").matches || window.innerWidth < 760);

export class PortScene {
  constructor() {
    this.el = document.createElement("div");
    this.el.className = "scene3d";
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance", logarithmicDepthBuffer: true });
    this.renderer.setPixelRatio(Math.min(Math.max(window.devicePixelRatio || 1, LITE ? 1 : 1.25), LITE ? 1.5 : 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.98;
    this.renderer.shadowMap.enabled = !LITE;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.canvas = this.renderer.domElement;
    this.el.appendChild(this.canvas);
    this.windEl = document.createElement("div");
    this.windEl.className = "scene-wind";
    this.windEl.hidden = true;
    this.el.appendChild(this.windEl);
    this.trafficEl = document.createElement("div");
    this.trafficEl.className = "scene-traffic";
    this.trafficEl.setAttribute("aria-label", "Harbour traffic");
    this.trafficEl.title = "Kinematic traffic replay. Manoeuvres wait for safe clearance; the chart and reward retain the submitted schedule.";
    this.el.appendChild(this.trafficEl);
    this.labelLayer = document.createElement("div");
    this.labelLayer.className = "scene-labels";
    this.el.appendChild(this.labelLayer);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(HAZE);
    this.scene.fog = new THREE.FogExp2(HAZE, 0.000062); // whitish coastal haze: Collserola (12 km) reads at ~60 %
    this.camera = new THREE.PerspectiveCamera(30, 1.6, 2, 40000);
    this.camera.layers.enable(NO_REFLECT);
    this.sky = buildSky(30000);
    this.scene.add(this.sky);
    this.water = buildWater({ lite: LITE });
    this.scene.add(this.water);
    this.worlds = new Map();
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.09;
    this.controls.maxPolarAngle = 1.48;
    this.controls.minDistance = 40;
    this.controls.screenSpacePanning = false;
    this.controls.zoomToCursor = true;

    this.hemi = new THREE.HemisphereLight("#DCE6F2", "#B5AC9C", 0.55);
    this.sun = new THREE.DirectionalLight(SUN_COLOR, 2.45);
    this.sun.castShadow = !LITE;
    this.sun.shadow.mapSize.set(4096, 4096);
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.6;
    this.scene.add(this.hemi, this.sun, this.sun.target);

    this.atmosphere = "afternoon";
    this.time = 0;
    this.names = true;
    this.hover = null;
    this.selected = null;
    this.actors = new Map();
    this.cranes = [];
    this.closures = [];
    this.labels = new Map();
    this.fmt = (h) => `h ${Math.round(h)}`;
    this.onHover = null;
    this.onSelect = null;
    this.raycaster = new THREE.Raycaster();
    this.pointer = null;
    this.pointerDirty = false;
    this._hoverPick = null;
    this.anim = null;
    this.running = false;
    this.clock = new THREE.Clock();
    this.realTime = 0;
    this.visible = true;
    this._bindPointer();
    this.composer = null;
    this._setupPost();
    this.ro = new ResizeObserver(() => this._resize());
    this.io = new IntersectionObserver((es) => {
      for (const e of es) this.visible = e.isIntersecting;
    });
  }

  mount(container) {
    container.appendChild(this.el);
    this.ro.observe(this.el);
    this.io.observe(this.el);
    this._resize();
    if (!this.running) {
      this.running = true;
      this.clock.getDelta();
      this.renderer.setAnimationLoop(() => this._frame());
    }
  }

  unmount() {
    this.running = false;
    this.renderer.setAnimationLoop(null);
    this.ro.disconnect();
    this.io.disconnect();
    if (this.el.parentNode) this.el.parentNode.removeChild(this.el);
  }

  /** Ambient occlusion (N8AO) for contact shadows under stacks, cranes and buildings, then tone mapping and SMAA. */
  async _setupPost() {
    if (LITE) return;
    try {
      const [{ EffectComposer }, { OutputPass }, { SMAAPass }, { N8AOPass }] = await Promise.all([
        import("three/addons/postprocessing/EffectComposer.js"),
        import("three/addons/postprocessing/OutputPass.js"),
        import("three/addons/postprocessing/SMAAPass.js"),
        import("n8ao"),
      ]);
      const w = Math.max(1, this.el.clientWidth);
      const h = Math.max(1, this.el.clientHeight);
      const composer = new EffectComposer(this.renderer);
      const ao = new N8AOPass(this.scene, this.camera, w, h);
      Object.assign(ao.configuration, { aoRadius: 5, distanceFalloff: 0.8, intensity: 2.0, aoSamples: 16, denoiseSamples: 8, denoiseRadius: 10, halfRes: true, gammaCorrection: false });
      ao.configuration.color = new THREE.Color(0x1a1612);
      composer.addPass(ao);
      composer.addPass(new OutputPass());
      composer.addPass(new SMAAPass(w, h));
      composer.setPixelRatio(this.renderer.getPixelRatio());
      composer.setSize(w, h);
      this.composer = composer;
      this.aoPass = ao;
      this.el.dataset.post = "ao";
    } catch (e) {
      this.el.dataset.post = "direct";
      console.warn("post-processing unavailable, rendering directly", e);
    }
  }

  _render() {
    if (this.composer) this.composer.render();
    else this.renderer.render(this.scene, this.camera);
  }

  _resize() {
    const w = Math.max(1, this.el.clientWidth);
    const h = Math.max(1, this.el.clientHeight);
    this.renderer.setSize(w, h, false);
    if (this.composer) {
      this.composer.setSize(w, h);
      if (this.aoPass) this.aoPass.setSize(w, h);
    }
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    if (this.lay && !this.userMoved) this.resetView(false);
  }

  /* ---------------- task / plan ---------------- */

  setTask(task) {
    if (this.task === task) return;
    this.task = task;
    const lay = layoutFor(task, twin);
    lay.railX0 = lay.quayX0;
    lay.railX1 = lay.quayX1;
    const nCranes = hasCranes(task) ? clamp(Number(task.rules.crane_pool), 3, 16) : clamp(Math.round(lay.L / 135), 5, 12);
    const key = `${lay.quay}|${lay.first}|${lay.last}|${lay.secM}`;
    if (key !== this.layKey) {
      if (this.world) this.scene.remove(this.world.root);
      let w = this.worlds.get(key);
      if (!w) {
        w = buildWorld(lay, twin, { lite: LITE });
        this.worlds.set(key, w);
      }
      this.world = w;
      this.scene.add(w.root);
      this.water.userData.setFrame(lay, twin.frame.extent);
      this.waterMask = createWaterMask(lay, twin);
      this._applyAtmosphere(lay);
    }
    if (key !== this.layKey || nCranes !== this.nCranes) {
      for (const c of this.cranes) {
        this.scene.remove(c.root);
        disposeTree(c.root);
      }
      const N = nCranes;
      this.cranes = Array.from({ length: N }, (_, i) => {
        const c = buildCrane(i, craneStyle(lay.quay, i, N), lay);
        c.st.x = lay.X0 + ((i + 0.5) * lay.L) / N;
        c.root.position.x = c.st.x;
        this.scene.add(c.root);
        return c;
      });
      this.nCranes = nCranes;
    }
    if (key !== this.layKey) {
      this.lay = lay;
      this.layKey = key;
      this._setupShadow();
    }
    this.lay = lay;
    for (const a of this.actors.values()) {
      this.scene.remove(a.model.group);
      disposeTree(a.model.group);
    }
    this.actors.clear();
    const kinds = callKinds(task);
    for (const s of task.ships) {
      const kind = kinds.get(s.id) || null;
      const unscheduled = !!kind;
      const col = shipColors(s, unscheduled);
      const beam = s.beam_m || Math.max(16, s.length_m / 7.2);
      const band = kind ? (kind.kind === "divert" ? "#3D6A9A" : C.signal) : null;
      const model = buildShipModel({ length: s.length_m, beam, seed: col.seed, livery: col.livery, band, lite: LITE, name: s.name });
      model.group.visible = false;
      this.scene.add(model.group);
      this.actors.set(`s${s.id}`, { key: `s${s.id}`, id: s.id, ship: s, name: s.name, unscheduled, kind, model, sched: { segs: [] } });
    }
    (task.blocks || []).forEach((b, i) => {
      if (b.kind !== "alongside") return;
      const len = (b.last - b.first + 1) * lay.secM - 14;
      const col = shipColors({ name: b.label, imo: `blk${i}` }, false);
      const model = buildShipModel({ length: len, beam: Math.max(18, len / 7.2), seed: col.seed, livery: col.livery, lite: LITE, name: b.label });
      model.group.visible = false;
      this.scene.add(model.group);
      this.actors.set(`b${i}`, { key: `b${i}`, id: `b${i}`, block: b, name: b.label, model, sched: { segs: [] } });
    });
    this._buildClosures();
    this.userMoved = false;
    this.resetView(false);
  }

  setAtmosphere(name) {
    if (!ATMOSPHERES[name] || name === this.atmosphere) return;
    this.atmosphere = name;
    if (this.lay) {
      this._applyAtmosphere(this.lay);
      this._setupShadow();
    }
  }

  _applyAtmosphere(lay) {
    const a = ATMOSPHERES[this.atmosphere];
    const sun = sunFor(lay, a);
    this.sunDir = sun;
    setSkyAtmosphere(this.sky, sun, a);
    this.sun.color.set(a.sun);
    this.sun.intensity = a.sunIntensity;
    this.hemi.color.set(a.sky.mid);
    this.hemi.groundColor.set("#8C8171");
    this.hemi.intensity = a.hemi;
    this.scene.background.set(a.haze);
    this.scene.fog.color.set(a.haze);
    this.scene.fog.density = a.fog;
    this.renderer.toneMappingExposure = a.exposure;
    const u = this.water.material.uniforms;
    u.sunDirection.value.copy(sun);
    u.sunColor.value.set(a.sun);
    u.waterColor.value.set(a.water);
    if (this.environmentTarget) this.environmentTarget.dispose();
    this.environmentTarget = skyEnvironment(this.renderer, sun, a);
    this.scene.environment = this.environmentTarget.texture;
    this.scene.environmentIntensity = a.environment;
    this.el.dataset.atmosphere = this.atmosphere;
  }

  _setupShadow() {
    const lay = this.lay;
    const s = this.sun;
    const cx = (lay.quayX0 + lay.quayX1) / 2;
    const cz = -180;
    const dir = (this.sunDir || new THREE.Vector3(0.3, 0.8, 0.5)).clone();
    s.target.position.set(cx, 0, cz);
    s.position.copy(dir.multiplyScalar(2600)).add(new THREE.Vector3(cx, 0, cz));
    const W = (lay.quayX1 - lay.quayX0) / 2 + 420;
    const cam = s.shadow.camera;
    cam.left = -W;
    cam.right = W;
    cam.top = 1050;
    cam.bottom = -1050;
    cam.near = 200;
    cam.far = 5600;
    cam.updateProjectionMatrix();
    s.shadow.mapSize.set(LITE ? 2048 : 4096, LITE ? 2048 : 4096);
    s.shadow.bias = -0.0003;
    s.shadow.normalBias = 0.3;
    cam.layers.enable(NO_REFLECT);
    if (s.shadow.map) {
      s.shadow.map.dispose();
      s.shadow.map = null;
    }
  }

  _buildClosures() {
    for (const c of this.closures) {
      this.scene.remove(c.group);
      disposeTree(c.group);
    }
    this.closures = [];
    const lay = this.lay;
    (this.task.blocks || []).forEach((b, i) => {
      if (b.kind !== "closed") return;
      const g = new THREE.Group();
      const x0 = lay.secX(b.first);
      const x1 = lay.secX(b.last + 1);
      const w = x1 - x0;
      const cx = (x0 + x1) / 2;
      const apronTex = stripeTexture("rgba(226,181,74,0.95)", "rgba(0,0,0,0)", { ratio: 0.45 });
      apronTex.repeat.set(w / 14, 60 / 14);
      const apron = new THREE.Mesh(new THREE.PlaneGeometry(w, 60), new THREE.MeshStandardMaterial({ map: apronTex, transparent: true, roughness: 1, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -3 }));
      apron.rotation.x = -Math.PI / 2;
      apron.position.set(cx, 0.3, -31);
      apron.receiveShadow = true;
      apron.renderOrder = 2;
      g.add(apron);
      const waterTex = stripeTexture("rgba(217,58,43,0.55)", "rgba(217,58,43,0.08)", { ratio: 0.35 });
      waterTex.repeat.set(w / 18, 80 / 18);
      const water = new THREE.Mesh(new THREE.PlaneGeometry(w, 80), new THREE.MeshBasicMaterial({ map: waterTex, transparent: true, depthWrite: false }));
      water.rotation.x = -Math.PI / 2;
      water.position.set(cx, WATER_Y + 0.15, 42);
      water.renderOrder = 2;
      g.add(water);
      const p = new Parts();
      let k = 0;
      for (let x = x0 + 2; x < x1 - 2; x += 4.4, k++) p.box(x + 2, 0.75, -3.4, 4, 1.5, 1.2, k % 2 ? "#F2EFE8" : C.red);
      for (const xe of [x0 + 1, x1 - 1]) {
        k = 0;
        for (let z = -6; z > -60; z -= 4.4, k++) p.box(xe, 0.75, z - 2, 1.2, 1.5, 4, k % 2 ? "#F2EFE8" : C.red);
      }
      // Workboats wait until any pre-existing ship has cleared the pocket.
      const boatParts = new Parts();
      const by = WATER_Y;
      boatParts.box(cx, by + 1, 34, 46, 4, 14, "#D9A33A");
      boatParts.box(cx, by - 0.6, 34, 46.4, 1.2, 14.4, "#2C2F33");
      boatParts.box(cx - 14, by + 5, 34, 9, 6, 9, "#F2EFE8");
      boatParts.box(cx - 14, by + 6.2, 34, 9.1, 1.2, 9.1, C.window);
      boatParts.beam([cx + 4, by + 3, 34], [cx + 20, by + 22, 22], 1.4, "#D9A33A");
      boatParts.box(cx + 8, by + 4.5, 34, 6, 4, 6, "#3A3F47");
      const solid = p.mesh(vcMat);
      const workboat = boatParts.mesh(vcMat);
      g.add(solid, workboat);
      g.visible = false;
      this.scene.add(g);
      this.closures.push({ group: g, workboat, block: b, key: `c${i}`, cx, x0, x1 });
    });
  }

  /** evaluation: model.evaluatePlan result; horizon: last hour on the scrubber. */
  setPlan(evaluation, horizon, { snap = true } = {}) {
    this.evaluation = evaluation;
    if (snap) this._snapCranes = true;
    this.horizon = horizon;
    const lay = this.lay;
    for (const row of evaluation.rows) {
      const actor = this.actors.get(`s${row.id}`);
      if (actor) actor.row = row;
    }
    const vessels = trafficVessels(lay, this.task, evaluation);
    const schedules = buildTraffic(lay, vessels, this.task, horizon, this.waterMask);
    const holds = [...schedules.values()].flatMap(s => s.segs.filter(p => p.kind === 'hold'));
    this.navBounds = { x0: Math.min(lay.X0 - 3500, ...holds.map(p => p.x - 700)), x1: Math.max(lay.X1 + 3500, ...holds.map(p => p.x + 700)), z0: -2500, z1: Math.max(lay.zWater + 3500, ...holds.map(p => p.z + 700)) };
    this.visualHorizon = horizon;
    for (const [key, schedule] of schedules) {
      this.actors.get(key).sched = schedule;
      if (schedule.visualDep != null) this.visualHorizon = Math.max(this.visualHorizon, Math.ceil(schedule.segs.at(-1).t1));
    }
    this.cargoSuspended = !snap;
    if (snap) this._planCargo();
    else {
      // Drag previews can arrive every pointer frame. Rebuild the cargo manifest
      // when the edit is accepted, and keep old transfer positions out of the preview.
      if (this.cargo) this.cargo.root.visible = false;
      for (const a of this.actors.values()) a.model.setDischarged(new Set());
    }
  }

  _planCargo() {
    if (this.cargo) { this.scene.remove(this.cargo.root); disposeTree(this.cargo.root); }
    const lay = this.lay, services = [];
    for (const a of this.actors.values()) {
      a.model.setDischarged(new Set());
      const hold = a.sched.segs.find(s => s.phase === 'alongside');
      if (!hold || a.row?.conflicts.length) continue;
      const m = a.model, side = Math.cos(hold.th) >= 0 ? 1 : -1;
      const work = a.row?.work ?? ((a.block?.end ?? 1) - (a.block?.start ?? 0));
      const start = Math.max(0, hold.t0), end = Math.min(hold.t1, hold.t0 + work);
      const want = hasCranes(this.task) ? (a.row?.cranes ?? a.block?.cranes ?? 0) : Math.max(1, Math.round(m.length / 95));
      const bays = m.cargoBays.map(b => ({ x: hold.x + side * b.x, top: b.top, localX: b.x }));
      const selected = craneTargets([{ id: a.key, x1: hold.x - m.length / 2 + 6, x2: hold.x + m.length / 2 - 6, bays, want }], this.cranes.length, lay).filter(c => c.working);
      const lanes = selected.map(c => {
        const bay = bays.find(b => Math.abs(b.x - c.x) < 0.01);
        return { x: c.x, clearY: WATER_Y + bay.top + 2.59 / 2 + 4, slots: dischargeOrder(m.cargoSlots, bay.localX, side) };
      });
      services.push({ ship: a.key, start: start * 3600, end: end * 3600, z: hold.z, side, waterY: WATER_Y, lanes });
    }
    // Do not start a representative job through a wind shutdown or a crane outage.
    const windows = [...(this.task.rules?.no_moves || []).filter(w => !(w.min_length > 0)), ...(this.task.rules?.crane_outages || [])].map(w => ({ start: w.start * 3600, end: w.end * 3600 }));
    const plan = planDischarges(services, this.world.transferSlots, { laneZ: lay.craneLane, roadZ: this.world.cargoRoadZ, windows });
    this.cargo = buildCargoReplay(plan, this.actors, lay.quay === '24B');
    this.scene.add(this.cargo.root);
  }

  setTime(t) {
    this.time = t;
  }

  setHover(id) {
    this.hover = id == null ? null : id;
  }

  setSelected(id) {
    this.selected = id == null ? null : id;
  }

  setNames(on) {
    this.names = !!on;
  }

  nextCargoTransfer() {
    const jobs = this.cargo?.jobs || [];
    const selected = this.selected == null ? null : typeof this.selected === 'number' ? `s${this.selected}` : this.selected;
    const eligible = selected && jobs.some(j => j.ship === selected) ? jobs.filter(j => j.ship === selected) : jobs;
    const upcoming = [...eligible].sort((a, b) => a.mark.start - b.mark.start);
    const job = upcoming.find(j => j.mark.clear > this.time * 3600) || upcoming[0];
    if (!job) return null;
    const target = new THREE.Vector3(job.source.x, 22, 10);
    this.userMoved = true;
    this._goto(target.clone().add(new THREE.Vector3(-160, 125, 100)), target, true);
    return { start: Math.max(job.mark.start, job.mark.pickup - 12) / 3600, end: job.mark.home / 3600 };
  }

  /* ---------------- camera ---------------- */

  _fitDistance(target, dir) {
    const lay = this.lay;
    const pts = [];
    for (const x of [lay.X0 - 10, lay.X1 + 10]) for (const z of [-120, lay.quay === "24B" ? 220 : 260]) for (const y of [0, 40]) pts.push(new THREE.Vector3(x, y, z));
    const cam = this.camera.clone();
    let lo = 200;
    let hi = 12000;
    for (let i = 0; i < 26; i++) {
      const d = (lo + hi) / 2;
      cam.position.copy(dir).multiplyScalar(d).add(target);
      cam.lookAt(target);
      cam.updateMatrixWorld();
      cam.updateProjectionMatrix();
      let ok = true;
      for (const p of pts) {
        const v = p.clone().project(cam);
        // keep the bottom clear of the scrubber bar
        if (Math.abs(v.x) > 0.97 || v.y > 0.9 || v.y < -0.8 || v.z > 1) {
          ok = false;
          break;
        }
      }
      if (ok) hi = d;
      else lo = d;
    }
    return hi;
  }

  homeView() {
    const lay = this.lay;
    // from over the basin, looking along the quay towards the north-east: the city, Montjuic and Collserola behind
    const az = this.camera.aspect < 1.2 ? -0.42 : -0.34;
    const el = this.camera.aspect < 1.2 ? 0.62 : 0.42;
    const dir = new THREE.Vector3(Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el));
    const target = new THREE.Vector3(lay.L * 0.03, 0, 30);
    const d = this._fitDistance(target, dir);
    return { pos: dir.clone().multiplyScalar(d).add(target), target, d };
  }

  setView(view) {
    if (view === "overview") return this.resetView(true);
    const lay = this.lay;
    const target = new THREE.Vector3(0, 5, 0);
    if (view === "harbour") {
      target.set(0, 20, -500);
      this._goto(target.clone().add(new THREE.Vector3(-1000, 1800, 3400)), target, true);
    } else if (view === "quayside") {
      // A waterfront view, clear of the cruise terminal across APM's basin.
      const visible = [...this.actors.values()].find(a => a.pose?.phase === "alongside");
      if (visible) target.set(visible.pose.x, 14, visible.pose.z);
      this._goto(target.clone().add(new THREE.Vector3(-300, 95, 330)), target, true);
    } else {
      target.set(0, 0, 90);
      this._goto(target.clone().add(new THREE.Vector3(0, lay.L * 1.8, 1)), target, true);
    }
    this.userMoved = true;
  }

  resetView(animate = true) {
    if (!this.lay) return;
    const v = this.homeView();
    // a shared view: #/...?cam=x,y,z,tx,ty,tz (scene metres)
    const cam = new URLSearchParams(location.hash.split("?")[1] || "").get("cam");
    const c = cam ? cam.split(",").map(Number) : null;
    if (c && c.length === 6 && c.every(Number.isFinite) && !this._camDone && !animate) {
      v.pos = new THREE.Vector3(c[0], c[1], c[2]);
      v.target = new THREE.Vector3(c[3], c[4], c[5]);
    }
    this.controls.maxDistance = Math.max(6000, v.d * 4);
    this.userMoved = false;
    this._goto(v.pos, v.target, animate);
  }

  _goto(pos, target, animate) {
    if (!animate) {
      this.camera.position.copy(pos);
      this.controls.target.copy(target);
      this.controls.update();
      this.anim = null;
      return;
    }
    this.anim = { p0: this.camera.position.clone(), t0: this.controls.target.clone(), p1: pos.clone(), t1: target.clone(), u: 0, started: performance.now() };
  }

  focus(id) {
    const actor = this.actors.get(typeof id === "number" ? `s${id}` : id);
    if (!actor) return;
    let pose = poseAt(actor.sched, this.time);
    if (!pose) {
      const segs = actor.sched.segs;
      const hold = segs.find((s) => s.kind === "hold" && s.phase === "alongside") || segs.find((s) => s.kind === "hold");
      if (!hold) return;
      pose = { x: hold.x, z: hold.z };
    }
    const target = new THREE.Vector3(pose.x, 8, pose.z - 20);
    const off = this.camera.position.clone().sub(this.controls.target).normalize();
    const d = Math.max(200, actor.model.length * 1.35);
    this.userMoved = true;
    this._goto(off.multiplyScalar(d).add(target), target, true);
  }

  /* ---------------- picking ---------------- */

  _bindPointer() {
    let down = null;
    this.canvas.addEventListener("pointermove", (e) => {
      const r = this.canvas.getBoundingClientRect();
      this.pointer = { x: ((e.clientX - r.left) / r.width) * 2 - 1, y: -((e.clientY - r.top) / r.height) * 2 + 1 };
      this.pointerDirty = true;
    });
    this.canvas.addEventListener("pointerleave", () => {
      this.pointer = null;
      this.pointerDirty = false;
      this.canvas.style.cursor = "";
      if (this._hoverPick != null) {
        this._hoverPick = null;
        if (this.onHover) this.onHover(null);
      }
    });
    this.canvas.addEventListener("pointerdown", (e) => {
      down = { x: e.clientX, y: e.clientY, t: performance.now() };
    });
    this.canvas.addEventListener("pointerup", (e) => {
      if (!down) return;
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
      const quick = performance.now() - down.t < 500;
      down = null;
      if (moved > 6 || !quick) return;
      const r = this.canvas.getBoundingClientRect();
      this.pointer = { x: ((e.clientX - r.left) / r.width) * 2 - 1, y: -((e.clientY - r.top) / r.height) * 2 + 1 };
      const hit = this._pick();
      if (this.onSelect) this.onSelect(hit ? hit.id : null);
    });
    this.controls.addEventListener("start", () => {
      this.userMoved = true;
      this._camDone = true;
      this.anim = null;
    });
  }

  _pick() {
    if (!this.pointer) return null;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const picks = [];
    for (const a of this.actors.values()) if (a.model.group.visible) picks.push(a.model.pick);
    const hits = this.raycaster.intersectObjects(picks, false);
    if (!hits.length) return null;
    for (const a of this.actors.values()) if (a.model.pick === hits[0].object) return a;
    return null;
  }

  /* ---------------- frame ---------------- */

  _frame() {
    const dt = Math.min(0.1, this.clock.getDelta());
    this.realTime += dt;
    if (!this.visible || !this.lay) return;
    const t = this.time;
    this._wind(t, dt);
    this.water.userData.tick(this.realTime);

    // camera animation
    if (this.anim) {
      this.anim.u = Math.min(1, (performance.now() - this.anim.started) / 900);
      const e = smooth(this.anim.u);
      this.camera.position.lerpVectors(this.anim.p0, this.anim.p1, e);
      this.controls.target.lerpVectors(this.anim.t0, this.anim.t1, e);
      if (this.anim.u >= 1) this.anim = null;
    }
    const lay = this.lay;
    const tg = this.controls.target;
    tg.x = clamp(tg.x, this.navBounds?.x0 ?? lay.X0 - 3500, this.navBounds?.x1 ?? lay.X1 + 3500);
    tg.z = clamp(tg.z, this.navBounds?.z0 ?? -2500, this.navBounds?.z1 ?? lay.zWater + 3500);
    tg.y = clamp(tg.y, 0, 120);
    this.controls.update();
    if (this.camera.position.y < 4) this.camera.position.y = 4;
    this.sky.position.copy(this.camera.position);

    if (this.pointerDirty) {
      this.pointerDirty = false;
      const hit = this._pick();
      const id = hit ? hit.id : null;
      this.canvas.style.cursor = hit ? "pointer" : "";
      if (id !== this._hoverPick) {
        const prev = this._hoverPick;
        this._hoverPick = id;
        if (this.onHover && (id != null || prev != null)) this.onHover(id);
      }
    }

    // ships
    const moored = [];
    const cr = hasCranes(this.task);
    for (const a of this.actors.values()) {
      const m = a.model;
      const pose = poseAt(a.sched, t);
      a.pose = pose;
      if (!pose) {
        m.group.visible = false;
        continue;
      }
      m.group.visible = true;
      const mooring = pose.phase === "alongside";
      const rough = this.water.material.uniforms.uRough.value;
      const motion = mooring ? 0 : 0.45 + rough * 0.65;
      m.group.position.set(pose.x, WATER_Y + Math.sin(this.realTime * 0.48 + m.length) * motion * 0.22, pose.z);
      m.group.rotation.set(Math.sin(this.realTime * 0.38 + m.length) * motion * 0.006, pose.th, Math.cos(this.realTime * 0.31 + m.length) * motion * 0.002, "YXZ");
      m.wakeMat.opacity = pose.lateral || pose.reverse ? 0 : clamp(pose.speed / 3.2, 0, 1) * 0.55;
      m.wake.scale.x = 0.35 + clamp(pose.speed / 3.2, 0, 1) * 0.65;
      if (m.wakeMat.userData.time) m.wakeMat.userData.time.value = this.realTime;
      m.wake.visible = m.wakeMat.opacity > 0.01;
      if (pose.phase === "alongside") {

        const half = m.length / 2;
        // on crane-rule tasks cranes leave a ship once it is finished (it may still wait for the wind)
        const working = a.row ? t < a.sched.visualBerth + a.row.work : t < (a.block?.end ?? Infinity);
        const want = cr ? (a.row ? a.row.cranes : a.block ? a.block.cranes || 0 : null) : null;
        if (working) moored.push({ id: a.key, x1: pose.x - half + 6, x2: pose.x + half - 6, zNear: pose.z - m.beam / 2, zFar: pose.z + m.beam / 2, deckY: m.deckY, bays: m.cargoBays.map(b => ({ x: pose.x + Math.cos(pose.th) * b.x, top: b.top })), want });
      }
      const tugsOn = !!pose.tugs;
      m.setMooring?.(mooring, pose.th, pose.z);
      const escorts = escortPoses(pose, { len: m.length, beam: m.beam });
      m.setTowlines?.(escorts);
      m.tugs.forEach((tug, i) => {
        tug.visible = tugsOn;
        if (tugsOn) {
          tug.position.set(escorts[i].localX, 0, escorts[i].localZ);
          tug.rotation.y = escorts[i].localHeading;
        }
      });
      const isSel = this.selected != null && this.selected === a.id;
      const isHov = this.hover != null && this.hover === a.id;
      const conflict = a.row && a.row.conflicts.length > 0;
      if (isSel || isHov) {
        m.ring.visible = true;
        m.ringMat.color.set(conflict ? C.conflict : C.signal);
        m.ringMat.opacity = isSel ? 1 : 0.75;
      } else if (conflict) {
        m.ring.visible = true;
        m.ringMat.color.set(C.conflict);
        m.ringMat.opacity = 0.7;
      } else m.ring.visible = false;
      m.hullMat.emissive.set(isSel || isHov ? C.signal : "#000000");
      m.hullMat.emissiveIntensity = isSel ? 0.35 : isHov ? 0.22 : 0;
    }

    const cargoFrame = this.cargoSuspended ? null : this.cargo?.update(t);
    // cranes
    if (this.cranes.length) {
      const N = this.cranes.length;
      const outN = cr ? clamp(N - cranePoolAt(this.task, Math.floor(t)), 0, N) : 0;
      const targets = craneTargets(moored, N - outN, { ...lay, railX1: lay.railX1 - outN * 31 });
      if (this.windCore) for (const tg of targets) tg.working = false;
      for (let i = 0; i < outN; i++) targets.push({ x: lay.railX1 - 16 - (outN - 1 - i) * 30, working: false, out: true });
      this.outN = outN;
      // a jump in plan time (scrubbing, plan switch, slow frames) puts the cranes straight in place
      const snap = this._snapCranes || this._lastT == null || Math.abs(t - this._lastT) > 0.75;
      this._snapCranes = false;
      this._lastT = t;
      this.cranes.forEach((c, i) => {
        targets[i].cargo = cargoFrame?.craneStates.get(`${targets[i].ship}:${targets[i].x.toFixed(3)}`);
        c.update(dt, this.realTime, targets[i], snap);
        c.setGrey(targets[i].out ? 1 : 0);
      });
    }

    for (const c of this.closures) {
      c.group.visible = t >= c.block.start && t < c.block.end;
      // Closures can start in the same integer hour a pre-existing ship departs.
      // Keep the administrative closure visible, but do not spawn a workboat under
      // a hull that is still making its final lateral clearance manoeuvre.
      c.workboat.visible = ![...this.actors.values()].some(a => a.pose && trafficBodies(a.pose,
        { len: a.model.length, beam: a.model.beam }).some(body => hullsOverlap(body, body, { x: c.cx, z: 42, th: 0 }, { len: c.x1 - c.x0, beam: 76 }, 6)));
    }
    const states = [...this.actors.values()].filter(a => a.pose);
    const along = states.filter(a => a.pose.phase === "alongside").length;
    const moving = states.filter(a => ["berthing", "departing"].includes(a.pose.phase));
    const waiting = states.filter(a => ["at anchor", "plan blocked"].includes(a.pose.phase));
    const blocked = waiting.filter(a => a.sched.reason).length;
    const heldAlongside = states.filter(a => a.pose.phase === "alongside" && a.sched.reason).length;
    const lag = states.filter(a => a.sched.delay > 0.01 && a.pose.phase !== "at anchor" && a.pose.phase !== "plan blocked").length;
    const detail = moving.length ? `${moving[0].name} · ${moving[0].pose.lateral ? "tug-assisted manoeuvre" : moving[0].pose.reverse ? "stern-first under tow" : `${(moving[0].pose.speed * 1.944).toFixed(1)} kn`}` : "Channel clear";
    const info = `<div class="traffic-title"><i></i> HARBOUR CONTROL</div><div class="traffic-counts"><b>${along}</b> alongside <b>${moving.length}</b> moving <b>${waiting.length}</b> waiting</div><div class="traffic-detail">${escapeHtml(detail)}</div>${blocked ? `<div class="traffic-note">${blocked} held offshore · plan / clearance conflict</div>` : ""}${heldAlongside ? `<div class="traffic-note">${heldAlongside} held alongside · awaiting safe clearance</div>` : ""}${lag ? `<div class="traffic-note">Traffic delay in 3D · chart shows planned hours</div>` : ""}`;
    const cs = cargoFrame?.summary;
    const cargoInfo = cs && this.cargo.jobs.length ? `<div class="traffic-cargo" title="Illustrative deck transfers; the berth task has no container manifest.">${cs.lifting} on hoist · ${cs.quay} on quay · ${cs.transit} to yard · ${cs.stored} stored${cs.phase ? `<br>${escapeHtml(cs.phase)}` : ''}</div>` : '';
    if (this.trafficEl.innerHTML !== info + cargoInfo) this.trafficEl.innerHTML = info + cargoInfo;
    this._render();
    this._labels();
  }

  /* ---------------- labels ---------------- */

  _shipLabel(a, rich) {
    const pose = a.pose;
    const fmt = this.fmt;
    let sub = "";
    let warn = "";
    if (rich) {
      const row = a.row;
      if (a.block) {
        sub = `already alongside ${a.block.first}–${a.block.last} · ` + (a.sched.visualDep == null ? `held for ${a.sched.reason || "safe clearance"}` : `sails ${fmt(a.sched.visualDep)}`);
      } else if (row) {
        const s = row.ship;
        if (pose.phase === "plan blocked") sub = "held offshore · " + a.sched.reason;
        else if (pose.phase === "approaching") sub = `approaching · arrives ${fmt(s.arrival)}`;
        else if (pose.phase === "at anchor") sub = row.placed ? `waiting · docks ${fmt(row.berth)}` : "waiting · not placed in this plan";
        else if (pose.phase === "berthing") sub = `docking at ${row.section}–${row.last}`;
        else if (pose.phase === "alongside") {
          const held = this.time >= a.sched.visualBerth + row.work;
          sub = a.sched.reason ? `held alongside · ${a.sched.reason}` : held ? `work complete · awaiting departure ${fmt(a.sched.visualDep)}` : `alongside ${row.section}–${row.last}${row.cranes != null ? ` · ${row.cranes} cranes` : ""} · sails ${fmt(a.sched.visualDep)}`;
          if (row.delay > 0) sub += ` · ${row.delay} h late`;
        }
        else sub = "departing";
        if (a.kind && a.kind.kind === "divert") sub = `diverted from ${a.kind.from} · ${sub}`;
        else if (a.unscheduled) sub = `unscheduled call · ${sub}`;
        if (row.conflicts.length) warn = row.conflicts.length === 1 ? row.conflicts[0] : `${row.conflicts.length} conflicts`;
        if (a.sched.delay > 0.01 && a.sched.visualBerth != null) sub += ` · visual traffic delay ${Math.round(a.sched.delay * 60)} min`;
      }
    }
    const sh = a.ship;
    let badges = "";
    if (sh && (sh.weight || 1) > 1) badges += ` <em class="bdg" title="Priority: lateness counts ×${sh.weight}">×${sh.weight}</em>`;
    if (sh && sh.berth_deadline != null) badges += ` <em class="bdg red" title="Emergency">dock by h ${sh.berth_deadline}</em>`;
    return `<b>${escapeHtml(a.name)}</b>${badges}${sub ? `<span>${escapeHtml(sub)}</span>` : ""}${warn ? `<span class="warn">${escapeHtml(warn)}</span>` : ""}`;
  }

  /** Wind windows: rough water, a banner, and (in the all-ships core) booms raised. */
  _wind(t, dt) {
    const ws = ((this.task && this.task.rules && this.task.rules.no_moves) || []).filter((w) => t >= w.start && t < w.end);
    const core = ws.find((w) => !(w.min_length > 0));
    this.windCore = !!core;
    const want = core ? 1 : ws.length ? 0.55 : 0;
    const u = this.water.material.uniforms.uRough;
    const jump = this._lastWindT == null || Math.abs(t - this._lastWindT) > 0.75;
    this._lastWindT = t;
    if (u) u.value = jump ? want : u.value + (want - u.value) * Math.min(1, dt * 2.5);
    if (!ws.length) {
      if (!this.windEl.hidden) this.windEl.hidden = true;
      return;
    }
    const w = core || ws[0];
    const end = Math.max(...ws.map((x) => x.end));
    const txt = `${w.reason ? w.reason.charAt(0).toUpperCase() + w.reason.slice(1) : "Wind"} · ${core ? "no ship may dock or leave" : `ships of ${w.min_length} m or more may not dock or leave`} · until ${this.fmt(end)}`;
    if (this.windEl._t !== txt) {
      this.windEl.textContent = txt;
      this.windEl._t = txt;
    }
    this.windEl.hidden = false;
  }

  _labels() {
    const want = [];
    for (const a of this.actors.values()) {
      if (!a.pose) continue;
      const sel = this.selected != null && this.selected === a.id;
      const hov = this.hover != null && this.hover === a.id;
      if (!(sel || hov || this.names)) continue;
      if (!(sel || hov) && a.pose.phase === "departing" && a.pose.u > 0.5) continue;
      const rich = sel || hov;
      const prio = sel ? 0 : hov ? 1 : a.pose.phase === "alongside" ? 3 : 4;
      want.push({ key: a.key, html: this._shipLabel(a, rich), pos: new THREE.Vector3(a.pose.x, WATER_Y + a.model.height + 6, a.pose.z), prio, cls: rich ? "lbl rich" : "lbl" });
    }
    if (this.outN > 0 && this.cranes.length) {
      const o = ((this.task.rules && this.task.rules.crane_outages) || []).find((x) => this.time >= x.start && this.time < x.end);
      const c = this.cranes[this.cranes.length - 1];
      want.push({ key: "outage", html: `<b>${this.outN} crane${this.outN > 1 ? "s" : ""} out of service</b>${o ? `<span>until ${escapeHtml(this.fmt(o.end))}</span>` : ""}`, pos: new THREE.Vector3(c.st.x - (this.outN - 1) * 15, 100, this.lay.landRail), prio: 2, cls: "lbl closed" });
    }
    for (const c of this.closures) {
      if (!c.group.visible) continue;
      const b = c.block;
      want.push({ key: c.key, html: `<b>Closed ${b.first}–${b.last}</b><span>${escapeHtml(b.label)} · until ${escapeHtml(this.fmt(b.end))}</span>`, pos: new THREE.Vector3(c.cx, 14, -24), prio: 2, cls: "lbl closed" });
    }
    want.sort((a, b) => a.prio - b.prio);
    const w = this.el.clientWidth;
    const h = this.el.clientHeight;
    const placed = [];
    const seen = new Set();
    for (const it of want) {
      let el = this.labels.get(it.key);
      if (!el) {
        el = document.createElement("div");
        el.dataset.key = it.key;
        this.labelLayer.appendChild(el);
        this.labels.set(it.key, el);
      }
      seen.add(it.key);
      if (el._html !== it.html || el.className !== it.cls) {
        el.innerHTML = it.html;
        el.className = it.cls;
        el._html = it.html;
        el._size = null;
      }
      const v = it.pos.project(this.camera);
      const sx = (v.x * 0.5 + 0.5) * w;
      const sy = (-v.y * 0.5 + 0.5) * h;
      if (v.z > 1 || sx < -50 || sx > w + 50 || sy < -20 || sy > h + 40) {
        el.style.visibility = "hidden";
        continue;
      }
      if (!el._size) el._size = { w: el.offsetWidth, h: el.offsetHeight };
      const rect = { x0: sx - el._size.w / 2 - 2, x1: sx + el._size.w / 2 + 2, y0: sy - el._size.h - 8, y1: sy };
      const clash = placed.some((p) => rect.x0 < p.x1 && rect.x1 > p.x0 && rect.y0 < p.y1 && rect.y1 > p.y0);
      if (clash && it.prio > 1) {
        el.style.visibility = "hidden";
        continue;
      }
      placed.push(rect);
      el.style.visibility = "visible";
      const lx = Math.max(4, Math.min(w - el._size.w - 4, sx - el._size.w / 2));
      el.style.setProperty("--tick", `${Math.round(Math.max(6, Math.min(el._size.w - 6, sx - lx)))}px`);
      el.style.transform = `translate(${Math.round(lx)}px, ${Math.round(sy - el._size.h - 8)}px)`;
      el.style.zIndex = String(10 - it.prio);
    }
    for (const [k, el] of this.labels) {
      if (!seen.has(k)) {
        el.remove();
        this.labels.delete(k);
      }
    }
  }

  snapshot() {
    this._render();
    return this.canvas.toDataURL("image/png");
  }
}

let shared = null;
export function getScene() {
  if (!shared) shared = new PortScene();
  return shared;
}
export const isLite = LITE;

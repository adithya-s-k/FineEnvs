// Ship-to-shore gantry cranes as the terminals have them:
//   BEST (36A): super post-Panamax A-frame cranes, white legs and portal, carmine boom, girders and bracing,
//               white machinery house; 64 m outreach, ~45 m lift.
//   APM (24B):  yellow A-frame cranes with grey-blue machinery houses, and the three 2025 ZPMC "Triple-E" cranes
//               in light APM blue with a white house and red apex; 66 m outreach, 56 m lift.
// The trolley/spreader cycle and the crane allocation along the quay are deterministic functions of time.
import * as THREE from "three";
import { C, Parts } from "./scene-kit.js";
import { BOX_H, BOX_L, BOX_W, GAUGE } from "./scene-world.js";

const RAISED = -1.38; // boom-up angle (about 79 degrees)

const STYLES = {
  best: { leg: C.bestWhite, portal: C.bestWhite, brace: C.bestCarmine, boom: C.bestCarmine, house: C.bestWhite, apex: C.bestCarmine, tip: C.bestCarmine, lift: 46, outreach: 64, back: 21, apex_h: 84 },
  apm: { leg: C.apmYellow, portal: C.apmYellow, brace: C.apmYellow, boom: C.apmYellow, house: C.apmHouse, apex: C.apmYellow, tip: C.apmYellow, lift: 44, outreach: 60, back: 18, apex_h: 80 },
  triplee: { leg: C.apmBlue, portal: C.apmBlue, brace: C.apmBlue, boom: C.apmBlue, house: C.white, apex: C.red, tip: C.red, lift: 56, outreach: 66, back: 22, apex_h: 96 },
};

/** Crane style for crane i of N at a quay. */
export function craneStyle(quay, i, N) {
  if (quay === "24B") return i >= N - 3 ? "triplee" : "apm";
  return "best";
}

export function buildCrane(index, styleName, lay) {
  const S = STYLES[styleName] || STYLES.best;
  const SEA = lay.seaRail;
  const LAND = lay.landRail;
  const LEG_X = 9.2;
  const Hg = S.lift + 5; // trolley girder level
  // APM straddle carriers reach 17.1 m including the cab beacon. Keep the
  // complete lower portal and its bracing above their driving envelope.
  const SILL_Y = lay.quay === "24B" ? 20 : 15;
  const BOOM_Y = Hg;
  const OUT = S.outreach + 3;
  const BACK_Z = LAND - S.back;
  const APEX_Z = (SEA + LAND) / 2 - 2;
  const root = new THREE.Group();

  const greyU = { value: 0 };
  const cmat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.5, metalness: 0.3 });
  cmat.onBeforeCompile = (sh) => {
    sh.uniforms.uGrey = greyU;
    sh.fragmentShader = `uniform float uGrey;\n${sh.fragmentShader}`.replace(
      "#include <color_fragment>",
      "#include <color_fragment>\n  float lum = dot(diffuseColor.rgb, vec3(0.299, 0.587, 0.114));\n  diffuseColor.rgb = mix(diffuseColor.rgb, vec3(lum * 0.6 + 0.08), uGrey);",
    );
  };
  cmat.customProgramCacheKey = () => "crane-grey";

  const b = new Parts();
  const dark = C.steelDark;
  // gantry: four legs on bogies, sill beams, portal beams, side bracing
  for (const x of [-LEG_X, LEG_X]) {
    for (const z of [SEA, LAND]) {
      b.box(x, Hg / 2 + 1.5, z, 2.4, Hg - 3, 2.4, S.leg);
      b.box(x, 1.6, z, 11, 1.4, 2.2, dark); // equaliser beam
      for (const dx of [-4.2, -1.4, 1.4, 4.2]) b.cylZ(x + dx, 0.55, z, 0.55, 1.4, "#1C1E21", 12);
      b.box(x, 3.0, z, 3.4, 1.6, 2.8, S.leg);
    }
    b.box(x, SILL_Y, (SEA + LAND) / 2, 2.0, 2.2, SEA - LAND, S.portal); // sill beam
    b.box(x, Hg - 1.2, (SEA + LAND) / 2, 2.6, 3.0, SEA - LAND + 2.6, S.portal); // portal top
    b.beam([x, SILL_Y + 1, SEA], [x, Hg - 3, (SEA + LAND) / 2], 1.1, S.brace);
    b.beam([x, SILL_Y + 1, LAND], [x, Hg - 3, (SEA + LAND) / 2], 1.1, S.brace);
  }
  for (const z of [SEA, LAND]) {
    b.box(0, Hg - 1.2, z, LEG_X * 2 + 2.6, 3.0, 2.6, S.portal);
    b.box(0, SILL_Y, z, LEG_X * 2, 1.6, 1.6, S.portal);
  }
  // trolley girders over the backreach, machinery house and electrical room at the back
  for (const x of [-3.1, 3.1]) b.box(x, BOOM_Y, (SEA + BACK_Z) / 2, 1.8, 3.6, SEA - BACK_Z, S.boom);
  for (let z = BACK_Z + 4; z < SEA; z += 8) b.box(0, BOOM_Y - 1.4, z, 6.4, 0.6, 0.6, S.boom);
  b.box(0, BOOM_Y + 4.6, BACK_Z + 8, 13, 6.8, 15, S.house);
  b.box(0, BOOM_Y + 8.2, BACK_Z + 8, 13.4, 0.4, 15.4, "#C9C9C5");
  b.box(-4.5, BOOM_Y - 4.5, LAND - 6, 6, 5, 9, S.house);
  // catwalks + handrails along both girders
  for (const x of [-4.6, 4.6]) {
    b.box(x, BOOM_Y - 1.6, (SEA + BACK_Z) / 2, 1.2, 0.12, SEA - BACK_Z, "#7D8287");
    b.box(x + Math.sign(x) * 0.6, BOOM_Y - 0.6, (SEA + BACK_Z) / 2, 0.06, 1.0, SEA - BACK_Z, "#9AA0A6");
  }
  // A-frame: front and back legs on each side meeting at the apex, with a cross head
  for (const x of [-3.6, 3.6]) {
    b.beam([x * 1.6, Hg, SEA + 1], [x, S.apex_h, APEX_Z], 1.5, S.apex === C.red ? S.boom : S.apex);
    b.beam([x * 1.6, Hg, LAND - 1], [x, S.apex_h, APEX_Z], 1.5, S.apex === C.red ? S.boom : S.apex);
    // backstays to the end of the backreach
    b.beam([x, S.apex_h, APEX_Z], [x, BOOM_Y + 1.6, BACK_Z + 1], 0.7, S.boom);
  }
  b.box(0, S.apex_h + 0.8, APEX_Z, 9.4, 2.4, 3.0, S.apex);
  b.box(0, (Hg + S.apex_h) / 2, APEX_Z + 6, 8.6, 1.0, 1.0, S.brace);
  b.cyl(0, S.apex_h + 2.6, APEX_Z, 0.5, 1.2, "#D9352B", 8);
  // stair tower on one land leg, cable reel on the other
  b.box(LEG_X + 2.6, Hg / 2, LAND, 2.2, Hg - 4, 2.2, "#C8CCD0");
  b.cylX(-LEG_X - 2.2, 5.5, LAND - 1.5, 2.6, 1.8, dark, 16);
  const bodyMesh = b.mesh(cmat);
  root.add(bodyMesh);

  // forestays (only while the boom is down)
  const f = new Parts();
  for (const x of [-3.1, 3.1]) {
    f.beam([x, S.apex_h, APEX_Z], [x, BOOM_Y + 2.0, SEA + OUT * 0.42], 0.75, S.boom);
    f.beam([x, S.apex_h, APEX_Z], [x, BOOM_Y + 2.0, SEA + OUT * 0.86], 0.7, S.boom);
  }
  const foreMesh = f.mesh(cmat);
  root.add(foreMesh);

  // outreach boom, hinged just beyond the sea rail
  const outreach = new THREE.Group();
  outreach.position.set(0, BOOM_Y, SEA + 1);
  const ob = new Parts();
  for (const x of [-3.1, 3.1]) {
    ob.box(x, 0, OUT / 2, 1.8, 3.6, OUT, S.boom);
    ob.box(x + Math.sign(x) * 1.5, -1.6, OUT / 2, 1.2, 0.12, OUT, "#7D8287");
  }
  for (let z = 6; z < OUT; z += 8) ob.box(0, -1.4, z, 6.4, 0.6, 0.6, S.boom);
  ob.box(0, 0.2, OUT - 0.5, 7.4, 3.0, 1.4, S.tip);
  if (S.tip === C.red) ob.box(0, 0.2, OUT - 2.5, 7.5, 3.1, 1.2, C.white);
  outreach.add(ob.mesh(cmat));
  root.add(outreach);

  // trolley + operator cab, ropes, headblock + spreader, carried box
  const trolley = new THREE.Group();
  const tp = new Parts();
  tp.box(0, BOOM_Y + 0.6, 0, 7.6, 1.6, 6.2, "#3B4148");
  tp.box(-2.4, BOOM_Y - 3.6, 0.6, 3.0, 2.8, 3.4, C.white);
  tp.box(-2.4, BOOM_Y - 3.9, 2.35, 2.7, 1.6, 0.1, C.window);
  trolley.add(tp.mesh());
  // Four separate hoist ropes and an open twistlock spreader.
  const ropes = new THREE.Group();
  for (const x of [-4.8, 4.8]) for (const z of [-0.95, 0.95]) {
    const rope = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 1, 6), new THREE.MeshStandardMaterial({ color: '#333B40', roughness: 0.55, metalness: 0.55 }));
    rope.position.set(x, 0, z);
    ropes.add(rope);
  }
  trolley.add(ropes);
  const spreader = new THREE.Group();
  const sp = new Parts();
  for (const z of [-BOX_W / 2, BOX_W / 2]) sp.box(0, 0, z, BOX_L + 0.25, 0.45, 0.25, '#E6B42E');
  for (const x of [-BOX_L / 2, BOX_L / 2]) {
    sp.box(x, 0, 0, 0.25, 0.45, BOX_W + 0.2, '#E6B42E');
    for (const z of [-BOX_W / 2, BOX_W / 2]) sp.box(x, -0.23, z, 0.28, 0.32, 0.28, '#3E464B');
  }
  sp.box(0, 0.2, 0, 3.2, 0.8, 1.5, '#333C42');
  for (const x of [-4.8, 4.8]) sp.box(x, 0.4, 0, 0.3, 0.6, 2.1, '#E6B42E');
  spreader.add(sp.mesh());
  trolley.add(spreader);
  root.add(trolley);

  const st = { x: 0, angle: RAISED };
  function update(dt, time, target, snap = false) {
    root.visible = !target.hidden;
    const cargo = target.cargo;
    // A scheduled lift and its source share an exact coordinate, including seeks.
    const k = snap || cargo?.active ? 1 : 1 - Math.exp(-dt * 2.2);
    st.x += (target.x - st.x) * k;
    root.position.x = st.x;
    const atShip = target.working && Math.abs(target.x - st.x) < 0.1;
    const want = atShip ? 0 : RAISED;
    st.angle += (want - st.angle) * (snap || cargo?.active ? 1 : 1 - Math.exp(-dt * 2.6));
    outreach.rotation.x = st.angle;
    foreMesh.visible = st.angle > -0.04;
    const down = st.angle > -0.03 && atShip;
    trolley.position.z = down && cargo ? cargo.z : LAND + 4;
    spreader.position.y = down && cargo ? cargo.y + BOX_H / 2 + 0.4 : BOOM_Y - 12;
    const bottom = spreader.position.y + 0.7;
    const len = Math.max(0.5, BOOM_Y - 0.2 - bottom);
    for (const rope of ropes.children) { rope.scale.y = len; rope.position.y = bottom + len / 2; }
  }

  root.userData.st = st;
  const setGrey = (v) => {
    greyU.value += (v - greyU.value) * 0.2;
    if (Math.abs(v - greyU.value) < 0.01) greyU.value = v;
  };
  return { root, update, st, setGrey };
}

export { craneTargets } from "./crane-allocation.js";

export { GAUGE };

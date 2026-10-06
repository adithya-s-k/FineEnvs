// Sky, image-based light, haze and the harbour water. Scene frame: x along the quay (north-east is +x at both
// Barcelona quays), z from the quay wall out over the water (south-east), y up; one unit is one metre.
import * as THREE from "three";
import { Water } from "three/addons/objects/Water.js";

export const WATER_Y = -3;

// Presentation lighting is independent of the RL clock. Both presets use the same
// geographic sun in the sky, reflection, environment map and directional shadows.
export const ATMOSPHERES = {
  afternoon: {
    label: "Mediterranean", az: 165, el: 36,
    sky: { zenith: "#2369AC", mid: "#77B1D0", horizon: "#C5D8DC", below: "#8AA8B8" },
    haze: "#B5CDD5", sun: "#FFF0D5", sunIntensity: 3.1, hemi: 0.28, environment: 0.48,
    exposure: 0.96, fog: 0.000046, water: "#125467",
  },
  golden: {
    label: "Golden hour", az: 184, el: 18,
    sky: { zenith: "#365E92", mid: "#96B2C6", horizon: "#ECD3AE", below: "#899DAF" },
    haze: "#C3C2BD", sun: "#FFD8A8", sunIntensity: 3.2, hemi: 0.32, environment: 0.58,
    exposure: 1.02, fog: 0.000055, water: "#174B5D",
  },
};
export const SKY = ATMOSPHERES.afternoon.sky;
export const HAZE = ATMOSPHERES.afternoon.haze;
export const SUN_COLOR = ATMOSPHERES.afternoon.sun;
export const SUN_DIR = new THREE.Vector3(-0.55, 0.64, 0.54).normalize();

/** Sun direction in a quay's frame: x along the quay, z towards the water. */
export function sunFor(frame, atmosphere = ATMOSPHERES.afternoon) {
  const az = atmosphere.az * Math.PI / 180;
  const el = atmosphere.el * Math.PI / 180;
  const e = Math.sin(az) * Math.cos(el);
  const n = Math.cos(az) * Math.cos(el);
  return new THREE.Vector3(e * frame.d[0] + n * frame.d[1], Math.sin(el), e * frame.nv[0] + n * frame.nv[1]).normalize();
}

export function setSkyAtmosphere(sky, sun, atmosphere) {
  const u = sky.material.uniforms;
  for (const [key, value] of Object.entries(atmosphere.sky)) u[`u${key[0].toUpperCase()}${key.slice(1)}`].value.set(value);
  u.uSun.value.copy(sun);
  u.uSunCol.value.set(atmosphere.sun);
}

// Objects on this layer are left out of the water's mirror pass (flat or far-from-the-water things).
export const NO_REFLECT = 1;

export function noReflect(obj) {
  obj.traverse((o) => o.layers.set(NO_REFLECT));
  return obj;
}

export function buildSky(radius = 18000, sun = SUN_DIR, atmosphere = ATMOSPHERES.afternoon) {
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    fog: false,
    toneMapped: false,
    uniforms: {
      uZenith: { value: new THREE.Color(atmosphere.sky.zenith) },
      uMid: { value: new THREE.Color(atmosphere.sky.mid) },
      uHorizon: { value: new THREE.Color(atmosphere.sky.horizon) },
      uBelow: { value: new THREE.Color(atmosphere.sky.below) },
      uSun: { value: sun.clone() },
      uSunCol: { value: new THREE.Color(atmosphere.sun) },
    },
    vertexShader: /* glsl */ `
      varying vec3 vDir;
      void main() {
        vDir = normalize(position);
        vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        gl_Position = p.xyww;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uZenith, uMid, uHorizon, uBelow, uSun, uSunCol;
      varying vec3 vDir;
      float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
      float noise(vec2 p) {
        vec2 i = floor(p), f = fract(p);
        vec2 u = f * f * (3.0 - 2.0 * f);
        return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
      }
      float fbm(vec2 p) {
        float v = 0.0, a = 0.5;
        for (int k = 0; k < 5; k++) { v += a * noise(p); p = p * 2.03 + vec2(17.0, 9.0); a *= 0.5; }
        return v;
      }
      void main() {
        vec3 d = normalize(vDir);
        float h = d.y;
        vec3 col;
        if (h >= 0.0) {
          col = mix(uHorizon, uMid, smoothstep(0.0, 0.16, h));
          col = mix(col, uZenith, smoothstep(0.12, 0.85, h));
        } else {
          col = mix(uHorizon, uBelow, smoothstep(0.0, 0.08, -h));
        }
        float s = max(dot(d, normalize(uSun)), 0.0);
        col += uSunCol * (pow(s, 1400.0) * 6.0 + pow(s, 60.0) * 0.18 + pow(s, 6.0) * 0.08);
        // scattered fair-weather cumulus, thinning into the haze at the horizon
        if (h > 0.0) {
          vec2 cp = d.xz / (h + 0.12) * 1.6;
          float c = fbm(cp + vec2(3.0, 1.0));
          float cover = smoothstep(0.57, 0.76, c) * smoothstep(0.015, 0.12, h);
          vec3 cloud = mix(vec3(0.80, 0.83, 0.88), vec3(1.0, 0.99, 0.97), smoothstep(0.55, 0.9, fbm(cp * 1.7 + 5.0)));
          col = mix(col, cloud, cover * 0.72);
        }
        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
      }`,
  });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 48, 24), mat);
  mesh.frustumCulled = false;
  mesh.renderOrder = -10;
  return mesh;
}

/** Sky lighting render target. The caller owns and disposes it when changing presets. */
export function skyEnvironment(renderer, sun = SUN_DIR, atmosphere = ATMOSPHERES.afternoon) {
  const pm = new THREE.PMREMGenerator(renderer);
  const sc = new THREE.Scene();
  const sky = buildSky(100, sun, atmosphere);
  sc.add(sky);
  // a warm ground bounce under the horizon, like sunlit concrete
  const ground = new THREE.Mesh(new THREE.CircleGeometry(90, 24), new THREE.MeshBasicMaterial({ color: "#B7B2A6", side: THREE.DoubleSide }));
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -6;
  sc.add(ground);
  const target = pm.fromScene(sc, 0.02);
  sky.geometry.dispose();
  sky.material.dispose();
  ground.geometry.dispose();
  ground.material.dispose();
  pm.dispose();
  return target;
}

/** Tileable ripple normal map built in code (sum of periodic waves + periodic value noise). */
function waterNormals(size = 256) {
  const h = new Float32Array(size * size);
  const waves = [];
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  // an isotropic chop spectrum: many short waves in every direction (integer wave numbers keep the tile seamless)
  for (let i = 0; i < 90; i++) {
    const kmag = 3 + Math.pow(rnd(), 1.6) * 34;
    const ang = rnd() * Math.PI * 2;
    const kx = Math.round(Math.cos(ang) * kmag);
    const ky = Math.round(Math.sin(ang) * kmag);
    if (!kx && !ky) continue;
    waves.push({ kx, ky, a: 1 / Math.pow(Math.hypot(kx, ky), 1.15), p: rnd() * Math.PI * 2 });
  }
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let v = 0;
      for (const w of waves) v += w.a * Math.sin((2 * Math.PI * (w.kx * x + w.ky * y)) / size + w.p);
      h[y * size + x] = v;
    }
  }
  const data = new Uint8Array(size * size * 4);
  const S = 2.4;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const hx = h[y * size + ((x + 1) % size)] - h[y * size + ((x - 1 + size) % size)];
      const hy = h[((y + 1) % size) * size + x] - h[((y - 1 + size) % size) * size + x];
      const n = new THREE.Vector3(-hx * S, -hy * S, 1).normalize();
      const i = (y * size + x) * 4;
      data[i] = Math.round((n.x * 0.5 + 0.5) * 255);
      data[i + 1] = Math.round((n.y * 0.5 + 0.5) * 255);
      data[i + 2] = Math.round((n.z * 0.5 + 0.5) * 255);
      data[i + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Harbour water: three.js Water (planar mirror, sun glitter) tinted to the deep green-blue of the port basin, with
 * foam against the quay wall and wind chop/whitecaps driven by uRough (0 calm .. 1 gale).
 */
export function buildWater({ lite = false, tint = null } = {}) {
  const water = new Water(new THREE.PlaneGeometry(60000, 60000), {
    textureWidth: lite ? 512 : 1536,
    textureHeight: lite ? 512 : 1536,
    waterNormals: waterNormals(),
    sunDirection: SUN_DIR.clone(),
    sunColor: new THREE.Color(SUN_COLOR),
    waterColor: new THREE.Color(ATMOSPHERES.afternoon.water),
    distortionScale: 3.2,
    fog: true,
  });
  water.rotation.x = -Math.PI / 2;
  water.position.y = WATER_Y;
  water.renderOrder = -1;
  const m = water.material;
  m.uniforms.size.value = 3.8;
  m.uniforms.uRough = { value: 0 };
  m.uniforms.uQuayEnds = { value: new THREE.Vector2(0, 0) };
  // body colour of the water from the Sentinel-2 scene (scene x,z -> image uv)
  m.uniforms.uTint = { value: tint };
  m.uniforms.uTintM = { value: new THREE.Matrix3() };
  m.uniforms.uTintOn = { value: tint ? 1 : 0 };
  m.fragmentShader = m.fragmentShader
    // water reflects ~2-4 % at normal incidence (not 30 %), and harbour chop breaks the mirror up
    .replace("float rf0 = 0.3;", "float rf0 = 0.025;")
    .replace("noise.xzy * vec3( 1.5, 1.0, 1.5 )", "noise.xzy * vec3( 0.72 + uRough * 0.65, 1.0, 0.72 + uRough * 0.65 )")
    // ripples flatten out with distance so the far basin reads as calm water, not aliasing streaks
    .replace(
      "vec3 diffuseLight = vec3(0.0);",
      "float dEye = length(eye - worldPosition.xyz);\n surfaceNormal = normalize(mix(surfaceNormal, vec3(0.0, 1.0, 0.0), smoothstep(350.0, 3200.0, dEye) * 0.45));\n vec3 diffuseLight = vec3(0.0);",
    )
    // The stock Water shader adds neutral diffuse light and a grey reflection floor.
    // Absorbing harbour water needs a coloured body and a separate sun highlight.
    .replace(
      "vec3 albedo = mix( ( sunColor * diffuseLight * 0.3 + scatter ) * getShadowMask(), ( vec3( 0.1 ) + reflectionSample * 0.9 + reflectionSample * specularLight ), reflectance);",
      "vec3 albedo = mix(wcol * (0.55 + diffuseLight * 0.65) * (0.55 + 0.45 * getShadowMask()), reflectionSample * 0.92, reflectance) + specularLight * reflectance * 0.7;",
    )
    .replace("uniform vec3 waterColor;", "uniform vec3 waterColor;\nuniform float uRough;\nuniform vec2 uQuayEnds;\nuniform sampler2D uTint;\nuniform mat3 uTintM;\nuniform float uTintOn;")
    .replace(
      "vec3 scatter = max( 0.0, dot( surfaceNormal, eyeDirection ) ) * waterColor;",
      `vec2 tuv = (uTintM * vec3(worldPosition.x, worldPosition.z, 1.0)).xy;
      vec3 wcol = mix(waterColor, texture2D(uTint, clamp(tuv, 0.001, 0.999)).rgb * 1.65, uTintOn);
      vec3 scatter = max( 0.0, dot( surfaceNormal, eyeDirection ) ) * wcol;`,
    )
    .replace(
      "gl_FragColor = vec4( outgoingLight, alpha );",
      /* glsl */ `
      // Broad, low-amplitude variations reveal currents without moving the berth waterline.
      float current = sin(worldPosition.x * 0.006 + sin(worldPosition.z * 0.011) * 2.0 + time * 0.025);
      outgoingLight *= 0.97 + current * 0.03;
      // wind: greyer, rougher water and breaking crests
      outgoingLight = mix(outgoingLight, outgoingLight * vec3(0.92, 0.97, 1.0) + vec3(0.05, 0.06, 0.065), uRough * 0.55);
      vec2 wp = worldPosition.xz;
      float c1 = sin(wp.x * 0.11 + time * 1.3 + sin(wp.y * 0.06 + time * 0.7) * 3.0) * sin(wp.y * 0.19 - time * 1.1 + wp.x * 0.023);
      float c2 = sin(wp.x * 0.23 - time * 2.0 + sin(wp.y * 0.11 - time) * 2.0) * sin(wp.y * 0.31 + time * 1.6);
      float caps = smoothstep(0.975 - uRough * 0.05, 0.998, c1) + smoothstep(0.965, 0.996, c2) * 0.7;
      outgoingLight = mix(outgoingLight, vec3(0.93, 0.95, 0.96), clamp(caps, 0.0, 1.0) * uRough * 0.85);
      // a thin band of churned water along the quay wall (z = 0)
      float q = 1.0 - smoothstep(0.0, 3.5 + uRough * 5.0, wp.y);
      q *= smoothstep(uQuayEnds.x, uQuayEnds.x + 8.0, wp.x) * (1.0 - smoothstep(uQuayEnds.y - 8.0, uQuayEnds.y, wp.x));
      q *= step(-0.5, wp.y) * (0.55 + 0.45 * sin(wp.x * 0.37 + time * 1.7));
      outgoingLight = mix(outgoingLight, vec3(0.80, 0.86, 0.86), clamp(q, 0.0, 1.0) * 0.45);
      gl_FragColor = vec4( outgoingLight, alpha );`,
    );
  m.needsUpdate = true;
  /** Point the tint lookup at a quay frame: scene (x, z) -> twin (X, Y) -> image uv. */
  water.userData.setFrame = (frame, extent) => {
    m.uniforms.uQuayEnds.value.set(frame.quayX0, frame.quayX1);
    const W = extent[2] - extent[0];
    const H = extent[3] - extent[1];
    const { d, nv, O } = frame;
    m.uniforms.uTintM.value.set(d[0] / W, nv[0] / W, (O[0] - extent[0]) / W, d[1] / H, nv[1] / H, (O[1] - extent[1]) / H, 0, 0, 1);
  };
  water.userData.tick = (t) => {
    const r = m.uniforms.uRough.value;
    m.uniforms.time.value = t * (0.25 + r * 0.65);
    m.uniforms.distortionScale.value = 3.2 + r * 3.0;
  };
  return water;
}

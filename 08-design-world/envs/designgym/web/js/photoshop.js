// Photoshop's Color Picker, as Photoshop (and Photopea) draw it: a dialog with the colour field
// and a vertical hue strip, new / current preview, OK / Cancel / Add to Swatches, and the
// H S B, R G B, L a b, C M Y K and # fields, every one of them editable. Enter is OK.

import { clamp, h } from "./util.js";
import { closePopover, hexToRgb, hsvToRgb, popover, rgbToHex, rgbToHsv } from "./widgets.js";

// sRGB <-> CIE Lab (D65), as the Lab fields show it.
const lin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
const gam = (c) => 255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);
function rgbToLab(r, g, b) {
  const [R, G, B] = [lin(r), lin(g), lin(b)];
  const f = (t) => (t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116);
  const x = f((0.4124 * R + 0.3576 * G + 0.1805 * B) / 0.95047), y = f(0.2126 * R + 0.7152 * G + 0.0722 * B),
    z = f((0.0193 * R + 0.1192 * G + 0.9505 * B) / 1.08883);
  return [116 * y - 16, 500 * (x - y), 200 * (y - z)];
}
function labToRgb(L, a, b) {
  const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
  const inv = (t) => (t ** 3 > 216 / 24389 ? t ** 3 : (116 * t - 16) / (24389 / 27));
  const X = inv(fx) * 0.95047, Y = inv(fy), Z = inv(fz) * 1.08883;
  return [3.2406 * X - 1.5372 * Y - 0.4986 * Z, -0.9689 * X + 1.8758 * Y + 0.0415 * Z, 0.0557 * X - 0.2040 * Y + 1.0570 * Z].map((c) => clamp(gam(c), 0, 255));
}
const rgbToCmyk = (r, g, b) => {
  const k = 1 - Math.max(r, g, b) / 255;
  if (k >= 1) return [0, 0, 0, 100];
  return [(1 - r / 255 - k) / (1 - k), (1 - g / 255 - k) / (1 - k), (1 - b / 255 - k) / (1 - k), k].map((v) => v * 100);
};
const cmykToRgb = (c, m, y, k) => [c, m, y].map((v) => 255 * (1 - v / 100) * (1 - k / 100));

/** Open the Color Picker. `set(hex)` runs on OK; `addSwatch(hex)` on Add to Swatches. */
export function openColorDialog(anchor, { title = "Color Picker", get, set, addSwatch = null }) {
  const initial = (get() || "#000000").slice(0, 7);
  let [hue, sat, val] = rgbToHsv(...hexToRgb(initial));
  const current = () => rgbToHex(...hsvToRgb(hue, sat, val));
  const field = h("div", { class: "pcp-field" }, h("div", { class: "pcp-knob" }));
  const strip = h("div", { class: "pcp-strip" }, h("div", { class: "pcp-arrow" }));
  const now = h("div", { class: "pcp-new", title: "new" });
  const was = h("div", { class: "pcp-cur", title: "current", style: { background: initial } });
  const inputs = {};
  const num = (key, label, unit = "") => {
    const inp = h("input", { class: "num-input pcp-in", spellcheck: "false", "aria-label": label, dataset: { key } });
    inputs[key] = inp;
    return h("label", { class: "pcp-f" }, h("span", { class: "pcp-l" }, `${label}:`), inp, h("span", { class: "pcp-u" }, unit));
  };
  const hex = h("input", { class: "cp-hex num-input pcp-hex", spellcheck: "false", "aria-label": "Hex colour" });
  const ok = h("button", { class: "btn btn-primary pcp-ok", type: "button" }, "OK");
  const cancel = h("button", { class: "btn pcp-cancel", type: "button" }, "Cancel");
  const add = addSwatch ? h("button", { class: "btn pcp-add", type: "button" }, "Add to Swatches") : null;

  const paint = (skip = null) => {
    const c = current();
    const [r, g, b] = hexToRgb(c);
    field.style.background = `linear-gradient(to top, #000, transparent), linear-gradient(to right, #fff, hsl(${hue}, 100%, 50%))`;
    field.firstChild.style.left = `${sat * 100}%`;
    field.firstChild.style.top = `${(1 - val) * 100}%`;
    strip.firstChild.style.top = `${(1 - hue / 360) * 100}%`;
    now.style.background = c;
    const [L, A, B] = rgbToLab(r, g, b), [C, M, Y, K] = rgbToCmyk(r, g, b);
    const vals = { H: hue, S: sat * 100, Bv: val * 100, R: r, G: g, Bb: b, L, a: A, bb: B, C, M, Y, K };
    for (const [k, inp] of Object.entries(inputs)) if (k !== skip && document.activeElement !== inp) inp.value = String(Math.round(vals[k]));
    if (skip !== "hex" && document.activeElement !== hex) hex.value = c.slice(1);
  };
  const fromRgb = (r, g, b) => { [hue, sat, val] = rgbToHsv(clamp(r, 0, 255), clamp(g, 0, 255), clamp(b, 0, 255)); };
  const readField = (key) => {
    const v = (k) => parseFloat(inputs[k].value) || 0;
    if (["H", "S", "Bv"].includes(key)) { hue = clamp(v("H"), 0, 360) % 360; sat = clamp(v("S"), 0, 100) / 100; val = clamp(v("Bv"), 0, 100) / 100; }
    else if (["R", "G", "Bb"].includes(key)) fromRgb(v("R"), v("G"), v("Bb"));
    else if (["L", "a", "bb"].includes(key)) fromRgb(...labToRgb(clamp(v("L"), 0, 100), clamp(v("a"), -128, 127), clamp(v("bb"), -128, 127)));
    else fromRgb(...cmykToRgb(...["C", "M", "Y", "K"].map((k) => clamp(v(k), 0, 100))));
    paint(key);
  };
  const readHex = () => {
    let t = hex.value.trim().replace(/^#/, "");
    if (/^[0-9a-f]{3}$/i.test(t)) t = [...t].map((c) => c + c).join("");
    if (/^[0-9a-f]{6}$/i.test(t)) { [hue, sat, val] = rgbToHsv(...hexToRgb(`#${t}`)); paint("hex"); return true; }
    return false;
  };
  const done = (apply) => { closePopover(); if (apply) set(current()); };
  const onKey = (read) => (e) => {
    if (e.key !== "Escape") e.stopPropagation();
    if (e.key === "Enter") { e.preventDefault(); read(); done(true); }  // Enter is OK, as in Photoshop
    else if (e.key === "Tab") read();
  };
  for (const inp of Object.values(inputs)) {
    inp.addEventListener("keydown", onKey(() => readField(inp.dataset.key)));
    inp.addEventListener("change", () => readField(inp.dataset.key));
    inp.addEventListener("focus", () => inp.select());
  }
  hex.addEventListener("keydown", onKey(readHex));
  hex.addEventListener("change", readHex);
  hex.addEventListener("focus", () => hex.select());
  ok.addEventListener("click", () => done(true));
  cancel.addEventListener("click", () => done(false));
  add?.addEventListener("click", () => addSwatch(current()));
  was.addEventListener("click", () => { [hue, sat, val] = rgbToHsv(...hexToRgb(initial)); paint(); });
  const drag = (el, fn) => el.addEventListener("pointerdown", (e) => {
    el.setPointerCapture(e.pointerId);
    const at = (ev) => {
      const r = el.getBoundingClientRect();
      fn(clamp((ev.clientX - r.left) / r.width, 0, 1), clamp((ev.clientY - r.top) / r.height, 0, 1));
      paint();
    };
    at(e);
    const move = (ev) => at(ev);
    const up = () => { el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up); };
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
  });
  drag(field, (x, y) => { sat = x; val = 1 - y; });
  drag(strip, (_x, y) => { hue = (1 - y) * 359.9; });

  const body = h("div", { class: "pcp" },
    h("div", { class: "pcp-head float-drag" }, h("span", {}, title),
      h("button", { class: "pcp-x", type: "button", "aria-label": "Close", onclick: () => done(false) }, "×")),
    h("div", { class: "pcp-main" }, field, strip,
      h("div", { class: "pcp-side" },
        h("div", { class: "pcp-top" }, h("div", { class: "pcp-swatch" }, h("div", { class: "pcp-cap" }, "new"), now, was, h("div", { class: "pcp-cap" }, "current")),
          h("div", { class: "pcp-btns" }, ok, cancel, add)),
        h("div", { class: "pcp-grid" },
          h("div", { class: "pcp-col" }, num("H", "H", "°"), num("S", "S", "%"), num("Bv", "B", "%"), h("div", { class: "pcp-gap" }),
            num("R", "R"), num("G", "G"), num("Bb", "B")),
          h("div", { class: "pcp-col" }, num("L", "L"), num("a", "a"), num("bb", "b"), h("div", { class: "pcp-gap" }),
            num("C", "C", "%"), num("M", "M", "%"), num("Y", "Y", "%"), num("K", "K", "%"))),
        h("label", { class: "pcp-f pcp-hexrow" }, h("span", { class: "pcp-l" }, "#"), hex))));
  paint();
  popover(anchor, body, { cls: "ps-colordialog" });
  // Photoshop opens its picker in the middle of the window.
  const pop = body.parentElement;
  const r = pop.getBoundingClientRect();
  Object.assign(pop.style, { left: `${Math.max(8, (innerWidth - r.width) / 2)}px`, top: `${Math.max(8, (innerHeight - r.height) / 2 - 40)}px` });
  hex.focus();
  hex.select();
}

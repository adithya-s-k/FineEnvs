// One capability presentation for dataset readers and live environment transports.
import { esc } from "./util.js";
import { icon } from "./icons.js";

const STATES = {
  available: ["Available here", "check"],
  external: ["Use native framework", "external"],
  unavailable: ["Unavailable here", "info"],
  unknown: ["Check availability", "info"],
};

export function supportPanel(s) {
  if (!s?.capabilities?.length) return "";
  const f = s.framework || {};
  const docs = /^https:\/\//.test(f.docs || "") ? f.docs : null;
  return `<section class="support-panel" aria-label="Environment capabilities">
    <h2>What you can do here</h2>
    <div class="support-grid">${s.capabilities.map((c) => {
      const [status, glyph] = STATES[c.state] || STATES.unknown;
      return `<div class="support-item" data-capability="${esc(c.id)}" data-state="${esc(c.state)}">
        <strong>${esc(c.label)}</strong><span class="support-state">${icon(glyph, 13)}${esc(status)}</span><p>${esc(c.detail)}</p></div>`;
    }).join("")}</div>
    <details class="support-more"><summary>How ${esc(f.label || "this environment")} works</summary>
      <p>${esc(f.lifecycle)}</p><p>${esc(f.reward)}</p>
      ${s.transport ? `<p>Connection: <code>${esc(s.transport)}</code></p>` : ""}
      ${docs ? `<a class="u" href="${esc(docs)}" target="_blank" rel="noopener noreferrer">${esc(f.label)} documentation ${icon("external", 12)}</a>` : ""}
    </details></section>`;
}

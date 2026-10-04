import { esc, fmt } from "./util.js";
import { value } from "./render.js";

// Keep the task's actual inputs visible; hashes and bookkeeping belong in an
// expandable record. All content still passes through the shared safe renderer.
export function taskContent(task) {
  const primary = /^(prompt|instruction|instructions|question|description|messages|input|inputs|content|text|audio|audio_url|image|image_url|video_url|context|tools)$/i;
  const entries = Object.entries(task);
  const inputs = entries.filter(([key, v]) => primary.test(key) && v != null && v !== "");
  const metadata = Object.fromEntries(entries.filter(([key]) => !primary.test(key)));
  return `${task.media_ready === false ? `<div class="note-box"><span><b>Media loads when the episode starts.</b> Start this task to ask the Space to prepare its asset. The player appears with the observation when the server provides it.</span></div>` : ""}
    <div class="tk-inputs">${inputs.map(([key, v]) => `<section><h3>${esc(key.replaceAll("_", " "))}</h3>${value(v, key, 0, task)}</section>`).join("") || `<p class="muted sm">This task exposes structured fields. Open its record below to inspect them.</p>`}</div>
    <details class="tk-record" ${inputs.length ? "" : "open"}><summary>Task record · ${fmt.format(Object.keys(metadata).length)} fields</summary>
      ${value(metadata)}</details>`;
}

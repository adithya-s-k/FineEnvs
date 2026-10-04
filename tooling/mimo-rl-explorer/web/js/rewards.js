// Reward design: how each domain turns a rollout into a reward, and how General's rubrics are built across the
// whole dataset. A task's own Grading section has the rest: its checks, the judge prompt and the scoring bands.
import { $, esc, fmt, getJSONgz, sk, emptyState } from "./util.js";
import { icon, DOMAIN_ICON } from "./icons.js";
import * as explore from "./explore.js";

let alive = false;
export function unmount() { alive = false; }

const VERL = "https://github.com/XiaomiMiMo/verl/tree/a2ad9f6";
const MIMOAGENT = "https://github.com/XiaomiMiMo/mimoagent/tree/467f0a1";

// One row per kind of task. `pick` finds that kind in the index, for its count and an example.
const KINDS = [
  { d: "code", name: "Code", pick: () => true, reward: "1 if the hidden tests pass, otherwise 0", range: "0 or 1",
    by: "The task's hidden tests, applied after the agent finishes and run in its own image", judge: "–",
    ns: "The tests can't be applied, or the files they touch can't be reset", train: "Same" },
  { d: "cyber", name: "Cyber", pick: () => true, reward: "1 if the last proof of concept crashes in the expected function, otherwise 0", range: "0 or 1",
    by: "A root-owned server runs the real fuzz target on it, under the sanitizer", judge: "–", ns: "–", train: "Same" },
  { d: "general", name: "General, workplace", pick: (e) => e.id.startsWith("s3k_"), reward: "Σ weight × score ÷ Σ weight, over the task's rubric checks", range: "0 to 1",
    by: "Code checks on the systems' databases and the workspace, and a text judge: one 0-or-1 call per check", judge: "text",
    ns: "The judge can't be reached, or one of the task's systems stops responding", train: "The same verify.py; it first looks for Xiaomi's internal grader, not released" },
  { d: "general", name: "General, terminal", pick: (e) => !e.id.startsWith("s3k_"), reward: "1 if every test passes, otherwise 0", range: "0 or 1",
    by: "The task's pytest suite, copied in after the agent finishes", judge: "–",
    ns: "The tests crash before grading, or write no reward", train: "No reference harness in the release; follows the Terminal-Bench convention" },
  { d: "webdev", name: "Webdev", pick: () => true, reward: "mean(visual, brief fulfilment, asset quality); visual is the mean of five criteria", range: "0 to 1",
    by: "A vision judge scores a full-page render against a five-band scale per criterion", judge: "vision",
    ns: "The page can't be rendered, or the judge can't be reached", train: "Different: a relative pick among up to 8 sibling rollouts, minus a deduction for missing the brief" },
  { d: "music", name: "Music", pick: () => true, reward: "(0.85 × weighted feature groups + 0.15 × histogram similarity) ÷ 100", range: "0 to 1",
    by: "abc2midi, then 18 features compared with the range of human music", judge: "–",
    ns: "Never; a piece that can't be read or measured scores 0", train: "Same" },
];

const share = (n, of) => (of ? `${Math.round((n / of) * 100)}%` : "–");
const n = (x) => fmt.format(x || 0);

export async function mount(el) {
  alive = true;
  el.innerHTML = `<div class="wrap page">
    <div class="page-h"><div><h1>Reward design</h1><p>How each task turns a rollout into a reward. A task's Grading section has its own checks, judge prompt and scoring bands.</p></div></div>
    <div id="rw-body">${sk.lines(90, 80, 85, 70)}</div></div>`;
  let idx, stats;
  try { [idx, stats] = await Promise.all([explore.load(), getJSONgz("/data/rewards.json.gz")]); }
  catch (e) { if (alive) $("#rw-body", el).innerHTML = emptyState("alert", "Couldn't load the reward data", esc(e.message)); return; }
  if (!alive) return;
  const g = stats.general;
  const rows = KINDS.map((k) => {
    const envs = idx.envs.filter((e) => e.d === k.d && k.pick(e));
    const ex = envs[0];
    return `<tr><td><span class="rw-dom">${icon(DOMAIN_ICON[k.d], 13)}${esc(k.name)}</span>${ex ? `<br><a class="xs" href="#/task/${encodeURIComponent(ex.id)}">an example</a>` : ""}</td>
      <td class="num nw">${n(envs.length)}</td><td>${esc(k.reward)}</td><td class="nw">${esc(k.range)}</td><td>${esc(k.by)}</td><td class="nw">${esc(k.judge)}</td>
      <td>${esc(k.ns)}</td><td>${esc(k.train)}</td></tr>`;
  }).join("");
  const m = g.method || {}, t = g.tier || {}, mix = g.mix || {}, inp = g.input || {}, act = g.act || {};
  const judged = m.llm || 0;
  const facts = [
    ["Checks", `${n(g.checks)} across ${n(g.tasks)} tasks, ${g.per_task.min} to ${g.per_task.max} per task (median ${g.per_task.median})`],
    ["Judged by a model", `${n(judged)} (${share(judged, g.checks)}), each in its own call, answered 0 or 1`],
    ["Checked by code", `${n(m.rule)} (${share(m.rule, g.checks)}), scored 0 to 1 against the databases and the workspace`],
    ["Tasks", `${n(mix.judge_only)} graded by the judge alone, ${n(mix.both)} by code and the judge, ${n(mix.code_only)} by code alone`],
    ["Judge's share of a task's reward", `median ${share(g.judge_share_median, 1)}`],
    ["Tiers", `critical ${n(t.critical)} (${share(t.critical, g.checks)}), important ${n(t.important)} (${share(t.important, g.checks)}), sanity ${n(t.sanity)} (${share(t.sanity, g.checks)}). A label only: tiers don't enter the formula`],
    ["Weights", `set in ${n(g.weighted_tasks)} tasks; the rest weight every check equally (a check without a weight counts 1)`],
    ...(g.mixed_weights ? [["Mixed weights", `${n(g.mixed_weights)} tasks give some checks weights that add up to about 1 and leave others without one, which count 1 each.
      In ${n(g.mixed_heavy)} of them the unweighted checks decide 40% or more of the reward`]] : []),
    ["Facts come from", `the systems' databases in ${n(inp.db)} tasks, the workspace files in ${n(inp.workspace)}, both in ${n(inp.both)}`],
    ["The agent must", `change nothing in ${n(act.none)} tasks (its answer is graded), update databases in ${n(act.mutate_db)}, edit workspace files in ${n(act.edit_workspace)}, both in ${n(act.both)}`],
    ["What the judge reads", (g.judged_files || []).map(([f, c]) => `${esc(f)} for ${n(c)} checks`).join(", ") + ", as extracted text, up to 20,000 characters per file"],
    ["Gate and tamper checks", `supported by verify.py, used by ${n(g.uses_gates)} tasks`],
  ];
  $("#rw-body", el).innerHTML = `
    <section class="panel"><div class="panel-h"><h2>${icon("scale", 15)}By kind of task</h2><span class="aside">tasks in XiaomiMiMo/MiMo-V2.6-RL-oss</span></div>
      <div class="panel-b"><div class="tbl rw-tbl"><table><thead><tr><th>Task</th><th class="num nw">Tasks</th><th>Reward</th><th class="nw">Range</th><th>Graded by</th><th class="nw">Judge</th>
        <th>Not scored when</th><th>In training</th></tr></thead><tbody>${rows}</tbody></table></div></div></section>
    <section class="panel" style="margin-top:16px"><div class="panel-h"><h2>${icon("check", 15)}General rubrics, across the dataset</h2><span class="aside">counted from every task's verifier_meta.json</span></div>
      <div class="panel-b"><dl class="kv">${facts.map(([k, x]) => `<dt>${esc(k)}</dt><dd>${x}</dd>`).join("")}</dl></div></section>
    <section class="panel" style="margin-top:16px"><div class="panel-h"><h2>${icon("info", 15)}Everywhere</h2></div>
      <div class="panel-b"><ul class="rules" style="margin:0">
        <li>A failure that isn't the model's (the testbed, the render, the judge) is not scored: the rollout has no reward, rather than 0. Xiaomi's harness masks these the same way.</li>
        <li>Nothing that grades a task is in the sandbox while the agent works. Hidden tests, rubrics and the expected crash are uploaded after it finishes.</li>
        <li>The judge prompts are sent in Chinese, unchanged, so scores stay comparable with Xiaomi's. Each task's Grading section has its prompt with an English translation.</li>
        <li>Expected answers are never shown: not the rubric answers, not the code that checks them, not the Cyber proof of concept.</li>
        <li>Graders follow <a href="${VERL}" target="_blank" rel="noopener">XiaomiMiMo/verl a2ad9f6</a> and <a href="${MIMOAGENT}" target="_blank" rel="noopener">mimoagent 467f0a1</a>; the vendored files are unmodified.</li>
      </ul></div></section>`;
}

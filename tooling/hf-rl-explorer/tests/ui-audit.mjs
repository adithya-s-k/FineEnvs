// UI audit of the HF RL Explorer: every check clicks or types like a person would and asserts the result, then
// every page is checked at four widths in both themes for errors, sideways scroll, broken images and unlabelled
// buttons, with a screenshot of each.
//   npm i -D playwright && npx playwright install chromium     (or PW_CHROME=/path/to/chrome-headless-shell)
//   node tests/ui-audit.mjs http://localhost:8060 /tmp/rlx-shots
// Uses live Hub data and live Spaces (FineEnvs/geoguesser-env, FineEnvs/latex-ocr-env, ...): it needs network.
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const BASE = process.argv[2] || "http://localhost:8060";
const SHOTS = process.argv[3] || "/tmp/rlx-shots";
mkdirSync(SHOTS, { recursive: true });
const HARBOR = "harborframework/terminal-bench-2.1", HTASK = "tasks/feal-differential-cryptanalysis";
const MIMO = "XiaomiMiMo/MiMo-V2.6-RL-oss", MIMO_GENERAL = "s3k_0052_accounting_audit_tax_en_t2_rl_008";
const results = [];
const ok = (name, cond, detail = "") => { results.push({ name, pass: !!cond, detail }); if (!cond) console.log("  FAIL", name, detail); };

const browser = await chromium.launch(process.env.PW_CHROME ? { executablePath: process.env.PW_CHROME } : {});

async function page(width, theme = "light", height = 900) {
  const ctx = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 1, hasTouch: width < 768, isMobile: width < 768 });
  await ctx.addInitScript((t) => { try { localStorage.setItem("theme", t); localStorage.removeItem("fe-banner-closed"); } catch {} }, theme);
  const p = await ctx.newPage();
  p.errors = [];
  p.on("pageerror", (e) => p.errors.push("pageerror: " + e.message));
  p.on("console", (m) => { if (m.type() === "error" && !/favicon|Failed to load resource|net::ERR|cdn-thumbnails|hf\.space/.test(m.text())) p.errors.push("console: " + m.text()); });
  p.on("dialog", (d) => { p.errors.push("native dialog: " + d.message()); d.dismiss(); });
  p.on("response", (r) => { const u = r.url(); if (u.startsWith(BASE) && (r.status() === 403 || r.status() >= 500)) p.errors.push(`${r.status()} ${r.request().method()} ${u.slice(BASE.length, BASE.length + 120)}`); });
  return p;
}
// a live Space's interface fills in once its server answers the probe: wait for that, not for the section's skeleton
const liveIn = async (p, ms = 60000) => { try { await p.waitForFunction(() => /live/.test(document.querySelector("#sec-interface")?.textContent || "") && document.querySelector(".pg-opts, #pg-route, #pg-reset"), null, { timeout: ms }); return true; } catch { return false; } };
const here = (p) => p.evaluate(() => location.pathname + location.search + location.hash);   // pages are paths now; old #/ links redirect
const go = async (p, to, wait = 1200) => { await p.goto(BASE + "/" + to.replace(/^\/+/, ""), { waitUntil: "domcontentloaded" }); await p.waitForTimeout(wait); };   // "#/old", "path" or "/path"
const until = async (p, sel, ms = 20000) => { try { await p.waitForSelector(sel, { timeout: ms }); return true; } catch { return false; } };

async function health(p, label) {
  const r = await p.evaluate(() => {
    const over = document.documentElement.scrollWidth > document.documentElement.clientWidth + 1;
    const wide = over ? [...document.querySelectorAll("body *")].filter((e) => e.getBoundingClientRect().right > document.documentElement.clientWidth + 2
      && getComputedStyle(e).position !== "fixed" && e.offsetParent).slice(0, 4).map((e) => `${e.tagName.toLowerCase()}.${[...e.classList].join(".")}`) : [];
    const broken = [...document.images].filter((i) => i.complete && i.naturalWidth === 0 && i.src && !i.src.includes("cdn-thumbnails") && i.offsetParent).map((i) => i.src.slice(0, 80));
    const unnamed = [...document.querySelectorAll("button, a[href]")].filter((b) => b.offsetParent && !(b.textContent.trim() || b.getAttribute("aria-label") || b.getAttribute("title"))).map((b) => b.outerHTML.slice(0, 90));
    const cut = [];
    for (const e of document.querySelectorAll("body *")) {
      if (!e.offsetParent || ![...e.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim())) continue;
      if (getComputedStyle(e).textOverflow === "ellipsis") continue;
      const r = e.getBoundingClientRect();
      for (let a = e.parentElement; a && a !== document.body; a = a.parentElement) {
        const ox = getComputedStyle(a).overflowX;
        if (ox === "auto" || ox === "scroll") break;
        if (ox !== "visible") {
          if (getComputedStyle(a).textOverflow === "ellipsis") break;   // cut on purpose, with an ellipsis
          const ar = a.getBoundingClientRect();
          if (r.right > ar.right + 1 && !a.matches(".sp-facts")) cut.push(`${e.tagName.toLowerCase()}.${[...e.classList].join(".")} "${e.textContent.trim().slice(0, 30)}"`);
          break;
        }
      }
    }
    // content spilling out of its panel, card or column over its neighbour (nothing clips it, so "cut" misses it)
    const spill = [];
    const inScroller = (e, box) => { for (let a = e.parentElement; a && a !== box; a = a.parentElement) { const s = getComputedStyle(a); if (s.overflowX !== "visible" || s.textOverflow === "ellipsis") return true; } return false; };
    for (const box of document.querySelectorAll(".panel, .card, .pg-ctl, #pg-log, .tp-side, .sp-side")) {
      if (!box.offsetParent) continue;
      const br = box.getBoundingClientRect();
      for (const e of box.querySelectorAll("*")) {
        if (!e.offsetParent || getComputedStyle(e).position === "absolute") continue;
        if (e.getBoundingClientRect().right > br.right + 2 && !inScroller(e, box)) { spill.push(`${e.tagName.toLowerCase()}.${[...e.classList].join(".")} out of ${box.tagName.toLowerCase()}.${[...box.classList].join(".") || box.id}`); break; }
      }
    }
    return { over, wide, broken, unnamed, cut: [...new Set(cut)], spill: [...new Set(spill)] };
  });
  ok(`${label}: no sideways scroll`, !r.over, r.wide.join(" "));
  ok(`${label}: no broken images`, !r.broken.length, r.broken.join(" "));
  ok(`${label}: every button and link is named`, !r.unnamed.length, r.unnamed.slice(0, 3).join(" | "));
  ok(`${label}: no text cut off by its card or panel`, !r.cut.length, r.cut.slice(0, 3).join(" | "));
  ok(`${label}: nothing spills out of its panel or column`, !r.spill.length, r.spill.slice(0, 3).join(" | "));
  ok(`${label}: no errors`, !p.errors.length, p.errors.slice(0, 4).join(" | "));
  p.errors = [];
}

// ── desktop: interactions ────────────────────────────────────────────────────
{
  const p = await page(1440, "dark");
  await go(p, "#/", 600);
  ok("home: the app replaces the boot loader", await until(p, "#list .card"));
  ok("home: no collection tiles", !(await p.$("#domains, .domains")));
  ok("home: one search box (the header's hides on Explore)", await p.evaluate(() => getComputedStyle(document.querySelector(".top-search")).visibility === "hidden"));
  ok("home: the FineEnvs banner is one link", await p.evaluate(() => { const a = document.querySelector("#fe-banner a.fe-banner-link"); return a && !document.querySelector("#fe-banner").hidden && a.href.includes("huggingface.co/FineEnvs"); }));
  ok("home: heading says what it explores", /Explore RL environments on Hugging Face/.test(await p.textContent("h1")));
  const kinds = await p.$$eval("[data-kind]", (bs) => bs.map((b) => b.textContent));
  ok("home: Kind filter covers the frameworks", ["Harbor", "Spaces", "Verifiers", "NeMo Gym"].every((k) => kinds.some((x) => x.includes(k))), kinds.join(","));
  ok("home: ORS isn't a Kind of its own (its one Space is with the other Spaces)", !kinds.some((x) => /ORS/.test(x)), kinds.join(","));
  await p.click('[data-kind="openenv"]');
  await p.waitForTimeout(1500);   // the server answers (/api/search), after a short debounce
  const firstOE = await p.$$eval("#list .card", (cs) => cs.slice(0, 8).map((c) => c.getAttribute("href")));
  ok("home: OpenEnv Spaces open on FineEnvs' curated servers", firstOE.length === 8 && firstOE.every((h) => h.startsWith("/s/FineEnvs/")), firstOE.join(" "));
  await p.click('[data-kind="verifiers"]');
  await p.waitForTimeout(1500);
  ok("home: filtering to Verifiers shows only Verifiers datasets", await p.$$eval("#list .card .kind-tag", (ts) => ts.length > 0 && ts.every((t) => t.textContent.includes("Verifiers"))));
  await p.click('[data-kind="all"]');
  await p.waitForTimeout(1200);
  await p.fill("#q", "terminal-bench");
  // the server answers the search (/api/search): wait for its answer, not a fixed time (a Space is a round trip away)
  const narrowed = await p.waitForFunction(() => { const cs = [...document.querySelectorAll("#list .card")]; return cs.length > 0 && cs.every((c) => /terminal/i.test(c.textContent)); }, null, { timeout: 15000 }).then(() => true, () => false);
  ok("home: search narrows the list", narrowed);
  await p.fill("#q", "");
  const page1 = await p.textContent("#tr-page");
  await p.click("#tr-next");
  await p.waitForTimeout(200);
  ok("home: trending pages forward", (await p.textContent("#tr-page")) !== page1);
  ok("home: Space cards show the whole picture", await p.$$eval("#list .card.sp .sp-thumb", (is) => is.slice(0, 3).every((i) => Math.abs(i.getBoundingClientRect().width / i.getBoundingClientRect().height - 1200 / 630) < 0.05)));
  await health(p, "home 1440 dark");

  // a Harbor task's file viewer
  await go(p, `#/t/${HARBOR}/${HTASK}`, 600);
  ok("task: file viewer renders", await until(p, ".fv .fv-row.fv-file"));
  await p.click('.fv-row.fv-file[data-path="tests/test.sh"]');
  await p.waitForTimeout(500);
  ok("task: a shell file is highlighted with line numbers", await p.$$eval(".fv-body .tk-k", (x) => x.length > 2) && await p.$(".fv-body .ln .n"));
  ok("task: the open file is in the URL", (await here(p)).includes("f=tests%2Ftest.sh"));
  await p.click(".fv-body .ln:nth-child(3) .n");
  ok("task: a line can be linked", (await here(p)).includes("L=3"));
  await p.click('[data-act="full"]');
  ok("task: full view covers the page, above the header", await p.evaluate(() => {
    const fv = document.querySelector(".fv.full"); if (!fv) return false;
    const r = fv.getBoundingClientRect(); const hit = document.elementFromPoint(r.left + 40, r.top + 20);
    return fv.contains(hit);
  }));
  await p.keyboard.press("Escape");
  ok("task: Esc closes full view", !(await p.$(".fv.full")));
  const solution = await p.$('.fv-row.fv-file.withheld');
  if (solution) { await solution.click(); await p.waitForTimeout(200); ok("task: withheld files say so", /Not shown/.test(await p.textContent(".fv-body"))); }
  await health(p, "task 1440 dark");

  // a big Harbor dataset: walked, opens immediately
  await go(p, "#/d/microsoft/ProgramDistill", 2500);
  ok("big dataset: its tasks list without waiting for every file", await until(p, "#list .card, .ix-partial, .tk-card, [data-task]", 30000) || (await p.textContent("#body")).length > 200);
  await health(p, "ProgramDistill 1440");

  // rows: a generic dataset, paged, its facets filtering on the server
  await go(p, "#/d/FineEnvs/SmolDataEnvs", 600);
  ok("rows: list renders", await until(p, "#list .card", 30000));
  ok("rows: says how it's read", /How it's read/.test(await p.textContent(".rw-how")) && /answer/i.test(await p.textContent(".rw-roles")));
  const firstTitles = await p.$$eval("#list .card .t", (ts) => ts.slice(0, 6).map((t) => t.textContent));
  ok("rows: titles differ from row to row", new Set(firstTitles).size === firstTitles.length, firstTitles.join(" / "));
  const opt = await p.$("#facets .opt");
  if (opt) {
    await opt.click();
    await p.waitForFunction(() => /matching/.test(document.querySelector("#count").textContent) || /available/.test(document.querySelector("#note").textContent), null, { timeout: 60000 }).catch(() => {});
    ok("rows: a facet filters (or says it can't yet)", /matching/.test(await p.textContent("#count")) || /isn't available/.test(await p.textContent("#note")), await p.textContent("#count"));
  }
  await health(p, "rows SmolDataEnvs 1440");
  await go(p, "#/r/FineEnvs/SmolDataEnvs?c=default&s=train&i=3", 600);
  ok("row: an old row link opens its task page", await until(p, ".tp-sec", 30000) && (await here(p)).startsWith("/t/FineEnvs/SmolDataEnvs/train/3"));
  ok("row: the answer is withheld", /Withheld/.test(await p.textContent(".tp-side")) && !(await p.textContent(".tp-body")).includes('"answer"'));
  await p.click('.tp-nav a[title^="Next"]');
  await p.waitForTimeout(2500);
  ok("row: next task navigates", (await here(p)).includes("train/4"));
  await p.keyboard.press("ArrowLeft");
  await p.waitForTimeout(2500);
  ok("row: ← goes back a task", (await here(p)).includes("train/3"));
  await health(p, "row 1440");

  // a packed Harbor task (TaskTrove)
  await go(p, "#/t/open-thoughts/TaskTrove/train/100", 600);
  ok("packed: unpacked into a file tree", await until(p, ".fv .fv-row.fv-file", 40000));
  ok("packed: its solution is withheld", await p.$('.fv-row.fv-file.withheld[data-path="solution/solve.sh"]'));
  ok("packed: runs on Harbor", /Harbor/.test((await (await p.$("#run-card"))?.textContent()) || ""));
  await health(p, "TaskTrove task 1440");

  // MiMo: the release, as the MiMo explorer shows it
  await go(p, `#/d/${MIMO}`, 600);
  ok("MiMo: every environment listed", await until(p, "#list .card", 40000) && /7,780/.test(await p.textContent("#count")));
  ok("MiMo: a tile per domain, with its grader", (await p.$$("#tiles .dom")).length === 6 && /Graded by hidden unit tests/.test(await p.textContent("#tiles")));
  await p.click('#tiles .dom[data-tile="Cyber"]');
  await p.waitForTimeout(500);
  ok("MiMo: a domain's tile scopes the facets and the map", /Sanitizer/i.test(await p.textContent("#facets")) && /Cyber, by crash type/.test(await p.textContent("#map-title")));
  await p.fill("#q", "heap overflow png");
  await p.waitForTimeout(600);
  ok("MiMo: search ranks and highlights", /best match/.test(await p.textContent("#order-note")) && (await p.$("#list .card mark")));
  await p.evaluate(() => { document.querySelector("#ov-rewards").open = true; });
  ok("MiMo: the reward design unfolds", /General rubrics, across the dataset/.test(await p.textContent("#ov-rewards")));
  await health(p, "MiMo env 1440");
  await go(p, `#/t/${MIMO}/${MIMO_GENERAL}`, 600);
  ok("MiMo task: its systems and workspace", await until(p, ".tbl-btn", 90000) && (await p.$(".files-grid .file")));
  await p.click(".tbl-btn");
  ok("MiMo task: a table opens with the rows the agent starts with", await until(p, "#modal:not([hidden]) table", 30000));
  await p.click("#modal-close");
  await p.click('.files-grid .file[data-kind="document"]');
  ok("MiMo task: a Word file previews", await until(p, "#modal:not([hidden]) .pv-doc", 40000));
  await p.click("#modal-close");
  ok("MiMo task: the rubric's checks, answers withheld", /checks, weighted/.test(await p.textContent("#sec-grading")) && /pass_anchor/.test(await p.textContent(".tp-side")));
  ok("MiMo task: both runners offered, the MiMo harness first here", /MiMo harness/.test(await p.textContent("#run-runner")));
  ok("MiMo task: links its Harbor twin", await p.$('.tp-links a[href^="/t/FineEnvs/MiMo-V2.6-RL-harbor-general/"]'));
  await health(p, "MiMo task 1440");
  await go(p, `#/r/${MIMO}?c=code&s=train&i=6`, 600);
  ok("MiMo: an old row link opens the task by its id", await until(p, ".tp-sec", 60000) && /^\/t\/XiaomiMiMo\/MiMo-V2\.6-RL-oss\/format-code-task-/.test(await here(p)));
  const twin = await p.$eval('.tp-links a[href^="/t/FineEnvs/MiMo-V2.6-RL-harbor-code/"]', (a) => a.getAttribute("href")).catch(() => null);
  ok("MiMo: links its Harbor twin", !!twin);
  if (twin) {
    await go(p, twin, 600);
    ok("MiMo twin: links back to the release", await until(p, `.tp-links a[href^="/t/${MIMO}/"]`, 40000));
  }
  await go(p, "#/task/music-gK-0768", 600);
  ok("MiMo explorer's addresses still open", await until(p, ".tp-sec", 40000) && (await here(p)) === `/t/${MIMO}/music-gK-0768`);
  await go(p, "#/community?view=tasks", 600);
  ok("community: the tasks nobody has tried", await until(p, "#ct-rows .runrow", 60000) && /not tried yet/.test(await p.textContent("#cm-top")));
  await health(p, "community tasks 1440");

  // a live Space: playground, tasks, connect
  await go(p, "#/s/FineEnvs/geoguesser-env", 600);
  ok("space: what it offers fills in", await liveIn(p));
  ok("space: status says running", /Running/.test(await p.textContent("#sp-status")));
  await p.evaluate(() => { const o = document.querySelector(".pg-opts"); if (o) o.open = true; });
  await p.selectOption("#pg-split", "eval");
  await p.fill("#pg-index", "5");
  await p.click("#pg-reset");
  ok("space: a reset starts an episode with an image", await until(p, "#pg-log .rv-img", 40000));
  await p.selectOption("#pg-tool", "look");
  await p.fill("#pgt-heading_deg", "90");
  await p.click("#pg-call");
  await p.waitForTimeout(5000);
  ok("space: a tool call answers in the episode", /Facing 90/.test(await p.textContent("#pg-log")));
  ok("space: tasks list from its Task API", await until(p, "#tk-table [data-task]", 30000));
  await p.click("#tk-table [data-task='1']");
  ok("space: a task opens in full", await until(p, "#tk-detail:not([hidden]) .tk-detail-b"));
  await p.click('[data-agent="cursor"]');
  ok("space: connect shows each agent's config", /mcpServers/.test(await p.textContent("#sp-agent")) && (await p.textContent("#sp-agent")).includes("/mcp/FineEnvs/geoguesser-env"));
  await health(p, "space geoguesser 1440");

  await go(p, "s/FineEnvs/nayana-ocr-env", 600);   // ids and paths hundreds of characters long, no spaces
  if (await liveIn(p)) {
    await p.click("#pg-reset");
    ok("space (long ids): an episode starts", await until(p, "#pg-log .pg-entry", 40000));
    await p.waitForTimeout(800);
    await health(p, "space nayana episode 1440");
  }
  await go(p, "#/s/FineEnvs/latex-ocr-env", 600);
  ok("space (reset/step): fills in", await liveIn(p));
  await p.click("#pg-reset");
  ok("space (reset/step): reset shows the image", await until(p, "#pg-log .rv-img", 40000));
  await p.fill("#pga-latex", "x^2");
  await p.click("#pg-step");
  ok("space (reset/step): a step earns a reward", await until(p, "#pg-log .pg-entry:last-of-type .rv-reward", 40000));
  await health(p, "space latex 1440");

  // a NeMo Gym server: its own routes, with the session's cookie carrying the game
  await go(p, "#/s/FineEnvs/wordle-nemo-gym", 600);
  if (await until(p, "#pg-route", 40000)) {
    ok("nemo gym: labelled as such", /NeMo Gym/.test(await p.textContent("main")));
    await p.selectOption("#pg-route", { label: "POST /seed_session" });
    await p.click("#pg-send");
    await p.waitForTimeout(2500);
    await p.selectOption("#pg-route", { label: "POST /guess" });
    ok("nemo gym: a short string is a one-line input", await p.$("input#pgr-word"));
    await p.fill("#pgr-word", "crane");
    await p.click("#pg-send");
    await p.waitForTimeout(2500);
    await p.selectOption("#pg-route", { label: "POST /get_history" });
    await p.click("#pg-send");
    await p.waitForTimeout(2500);
    ok("nemo gym: the session keeps the game between calls", /Guess 1: crane/.test(await p.textContent("#pg-log")));
  } else ok("nemo gym: routes playground renders", false, "no #pg-route (asleep?)");
  await health(p, "nemo gym space 1440");

  await go(p, "#/s/AdithyaSK/openenv-harbor-ui-test", 600);
  if (await until(p, "#sec-interface .panel-b", 40000) && (await p.$("#sec-harbor"))) {
    ok("harbor space: maps its dataset back to the explorer", await p.$('#sec-harbor a[href="/d/FineEnvs/MiMo-V2.6-RL-harbor-terminal"]'));
  }
  await health(p, "harbor space 1440");
  await p.context().close();
}

// ── the frameworks' rows, and a model-graded task ────────────────────────────
{
  const p = await page(1440, "dark");
  await go(p, "t/nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1/train/0", 600);
  ok("NeMo Gym: the row reads as a conversation, not a table", await until(p, ".tp-sec", 40000)
    && await p.evaluate(() => { const t = document.querySelector(".tp-sec"); const turns = t ? [...t.querySelectorAll(".rl-turn")] : [];
      return turns.length > 1 && turns.every((x) => !x.closest("table")) && turns.some((x) => /user/i.test(x.querySelector(".rl-who")?.textContent || "")); }));
  // (a tool's output may be shown as a table: data, inside its turn; the turns themselves never are)
  ok("NeMo Gym: its title is the request, not a row number", !/^(Row )?\d+$/.test((await p.textContent("h1")).trim()));
  ok("NeMo Gym: the expert's next action is withheld", /expected_action/.test(await p.textContent(".tp-side")) && !(await p.textContent(".tp-body")).includes('"expected_action"'));
  ok("NeMo Gym: says how to run it with NeMo Gym", /Run it with NeMo Gym/.test(await p.textContent("main")));
  await health(p, "NeMo Gym row 1440");
  await go(p, "t/sungyub/deepscaler-preview-verl/train/0", 600);
  ok("verl: the ground truth is withheld", await until(p, ".tp-sec", 40000) && /ground_truth/.test(await p.textContent(".tp-side")));
  ok("verl: its math is typeset (KaTeX), not left as TeX", await until(p, "h1 .math .katex", 20000) && !(await p.textContent("h1")).includes("\\frac"));
  await health(p, "verl row 1440");
  await go(p, "t/FineEnvs/MiMo-V2.6-RL-harbor-general/tasks/s3k_0052_accounting_audit_tax_en_t2_rl_008", 600);
  // signed in, the run panel offers the judge; signed out (a Space without a session), the page still says how it's graded
  ok("judge: a model-graded task offers a judge, through the relay", await until(p, "#run-card", 30000) && (
    (/Judge model/.test(await p.textContent("#run-card")) && /relay/.test(await p.textContent("#run-card")))
    || (/Sign in/i.test(await p.textContent("#run-card")) && /Graded by a model/.test(await p.textContent("main")))));
  await health(p, "model-graded task 1440");
  await p.context().close();
}

// ── every page, every width, both themes ─────────────────────────────────────
const PAGES = [["home", "#/"], ["dataset", `#/d/${HARBOR}`], ["task", `#/t/${HARBOR}/${HTASK}`], ["rows", "#/d/FineEnvs/SmolDataEnvs"],
  ["row", "#/t/FineEnvs/SmolDataEnvs/train/2"], ["packed", "#/t/open-thoughts/TaskTrove/train/100"],
  ["space", "#/s/FineEnvs/geoguesser-env"], ["traces", "#/t/open-thoughts/AgentTrove/train/0"], ["mimo", `#/d/${MIMO}`],
  ["mimo-task", `#/t/${MIMO}/${MIMO_GENERAL}`], ["mimo-code", `#/t/${MIMO}/format-code-task-000720`], ["community", "#/community"]];
for (const [w, theme] of [[375, "light"], [768, "dark"], [1280, "light"], [1680, "dark"]]) {
  const p = await page(w, theme);
  for (const [name, hash] of PAGES) {
    await go(p, hash, 3500);
    await p.waitForTimeout(name === "space" ? 6000 : 1500);
    await p.screenshot({ path: `${SHOTS}/${name}-${w}-${theme}.png`, fullPage: false });
    await health(p, `${name} ${w} ${theme}`);
  }
  await p.context().close();
}

await browser.close();
const failed = results.filter((r) => !r.pass);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
for (const f of failed) console.log(`FAIL  ${f.name}${f.detail ? `  (${f.detail})` : ""}`);
process.exit(failed.length ? 1 : 0);

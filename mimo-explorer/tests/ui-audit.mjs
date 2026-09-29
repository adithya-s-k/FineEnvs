// Interactive UI/UX audit of the MiMo RL Environment Explorer. Every check clicks or types like a person would
// and asserts the result, then every page is checked for sideways scroll and errors at 4 widths in both themes.
//   npm i -D playwright && npx playwright install chromium
//   node tests/ui-audit.mjs http://localhost:8000 /tmp/shots
// Needs a local server (signed in as your machine's token) with a finished public Webdev rollout: set OWN_RUN,
// and OTHER_RUN to a copy of it whose run.json has "user" changed, to test the anonymous view.
import { chromium } from 'playwright';

const BASE = process.argv[2] || 'http://localhost:8742';
const SHOTS = process.argv[3] || '/tmp';
const OWN_RUN = process.env.OWN_RUN || '20260926-180342-0d8314';      // a finished public webdev rollout of the local user
const OTHER_RUN = process.env.OTHER_RUN || '20260926-180342-0d83ff';  // the same, owned by someone else
// a task with at least three finished rollouts of the local user, for the comparison
const CMP_TASK = process.env.CMP_TASK || 'format-code-task-001590';
const CMP_RUNS = process.env.CMP_RUNS || '20260926-133727-48e5cf,20260926-161405-d52e73,20260926-162235-4f47e5';
const results = [];
const ok = (name, cond, detail = '') => { results.push({ name, pass: !!cond, detail }); };

const b = await chromium.launch();

async function page(width, theme = 'light') {
  const ctx = await b.newContext({ viewport: { width, height: 900 }, deviceScaleFactor: 1 });
  await ctx.addInitScript((t) => { localStorage.setItem('theme', t); localStorage.setItem('source', '"hf"'); localStorage.removeItem('adv'); }, theme);
  const p = await ctx.newPage();
  p.errors = [];
  p.on('pageerror', (e) => p.errors.push(e.message));
  p.on('console', (m) => { if (m.type() === 'error' && !/favicon|404|net::ERR|Failed to load resource/.test(m.text())) p.errors.push(m.text()); });
  p.on('dialog', (d) => { p.errors.push('native dialog: ' + d.message()); d.dismiss(); });
  p.on('response', (r) => { if (r.status() === 403 || r.status() >= 500) p.errors.push(`${r.status()} ${r.request().method()} ${r.url().slice(0, 120)}`); });
  return p;
}
const go = async (p, hash, wait = 900) => { await p.goto(BASE + '/' + hash, { waitUntil: 'networkidle' }); await p.waitForTimeout(wait); };
const noOverflow = (p) => p.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);

// ── desktop, light: every interaction ────────────────────────────────────────
{
  const p = await page(1440);
  await go(p, '#/');
  ok('boot loader replaced by the app', !(await p.$('.boot')) && (await p.$('.domains')));
  ok('footer shows the version', /v\d+\.\d+\.\d+/.test(await p.textContent('#foot-version')));
  ok('footer report link opens discussions', (await p.getAttribute('.foot .report', 'href')).endsWith('/discussions'));

  // nav
  for (const [nav, sel] of [['community', '#cm-view'], ['runs', '.rs-table, .empty-state'], ['explore', '.domains']]) {
    await p.click(`[data-nav="${nav}"]`); await p.waitForTimeout(900);
    ok(`nav "${nav}" opens its view and is marked active`, (await p.$(sel)) && (await p.getAttribute(`[data-nav="${nav}"]`, 'class') || '').includes('on'));
  }
  // theme
  const before = await p.evaluate(() => document.documentElement.dataset.theme);
  await p.click('#theme'); await p.waitForTimeout(200);
  ok('theme toggle flips the theme', before !== (await p.evaluate(() => document.documentElement.dataset.theme)));
  await p.click('#theme');
  // account menu
  await p.click('.acct-btn'); ok('account menu opens', await p.isVisible('.menu'));
  ok('account menu has a prefilled-free report link', (await p.getAttribute('.menu a[href*="discussions"]', 'href')).endsWith('/discussions'));
  await p.click('.ex-head h1'); await p.waitForTimeout(150); ok('account menu closes on outside click', !(await p.isVisible('.menu')));

  // explore: domain, search, facet, pill, load more, url state
  const total = await p.textContent('#count');
  await p.click('.dom[data-dom="cyber"]'); await p.waitForTimeout(700);
  ok('domain card filters the list', /1,000/.test(await p.textContent('#count')), await p.textContent('#count'));
  await p.fill('#q', 'heap overflow'); await p.waitForTimeout(700);
  const n1 = await p.textContent('#count');
  ok('search narrows results', n1 !== total && /\d/.test(n1), n1);
  ok('search terms are highlighted', (await p.$$('.card mark')).length > 0);
  await p.click('.facet .opt:not(.zero)'); await p.waitForTimeout(500);
  ok('facet checkbox adds a filter pill', (await p.$$('.active .pill')).length === 1);
  await p.click('.active .pill'); await p.waitForTimeout(500);
  ok('clicking the pill removes the filter', (await p.$$('.active .pill')).length === 0);
  await p.reload({ waitUntil: 'networkidle' }); await p.waitForTimeout(900);
  ok('search and domain survive a reload (URL state)', (await p.inputValue('#q')) === 'heap overflow' && (await p.getAttribute('.dom[data-dom="cyber"]', 'aria-selected')) === 'true');
  await p.click('#clear'); await p.waitForTimeout(600);
  ok('clear all resets', (await p.textContent('#count')) === total);
  for (let i = 0; i < 8; i++) { await p.evaluate(() => scrollTo(0, document.body.scrollHeight)); await p.waitForTimeout(350); }
  ok('the first page plus three auto-load, then a Load more button', (await p.$$('#list > li')).length === 120 && await p.isVisible('#more'));
  await p.click('#more'); await p.waitForTimeout(400);
  ok('Load more adds 30', (await p.$$('#list > li')).length === 150);
  await p.evaluate(() => scrollTo(0, document.body.scrollHeight)); await p.waitForTimeout(300);
  ok('the footer is reachable', await p.evaluate(() => document.querySelector('.foot').getBoundingClientRect().top < innerHeight));
  await p.evaluate(() => scrollTo(0, 0));
  // treemap drill-down
  await p.click('.dom[data-dom=""]'); await p.waitForTimeout(700);
  const cell = await p.$('g.leaf rect');
  await cell.dispatchEvent('click'); await p.waitForTimeout(900);
  ok('treemap block click drills into its domain', /d=/.test(await p.evaluate(() => location.hash)));
  // random
  await p.click('#surprise'); await p.waitForTimeout(1500);
  ok('Random opens a task', /#\/task\//.test(await p.evaluate(() => location.hash)));

  // task page: general
  await go(p, '#/task/s3k_0000_accounting_audit_tax_en_t1_rl_008', 2500);
  ok('task page: every section rendered', (await p.$$('.tp-sec')).length >= 6);
  await p.click('.toc a[data-jump="grading"]'); await p.waitForTimeout(900);
  ok('outline jump scrolls to the section', await p.evaluate(() => Math.abs(document.querySelector('#sec-grading').getBoundingClientRect().top) < 200));
  for (const kind of ['spreadsheet', 'document', 'slides', 'pdf']) {
    await p.click(`.file[data-kind="${kind}"]`); await p.waitForTimeout(2500);
    const body = await p.textContent('#modal-body');
    ok(`${kind} preview opens with content`, await p.isVisible('#modal') && (body.trim().length > 50 || (kind === 'pdf' && await p.$('#modal-body iframe'))), body.slice(0, 60));
    ok(`${kind} preview has an "open original" link`, await p.isVisible('#modal-open'));
    await p.keyboard.press('Escape'); await p.waitForTimeout(200);
    ok(`Escape closes the ${kind} preview`, !(await p.isVisible('#modal')));
  }
  await p.click('.tbl-btn'); await p.waitForTimeout(1500);
  ok('database table opens with rows and a row count', (await p.$$('#modal-body tbody tr')).length > 0 && /rows?/.test(await p.textContent('#modal-body .pv-foot')));
  await p.click('#modal-close'); await p.waitForTimeout(200);
  ok('close button closes the modal', !(await p.isVisible('#modal')));
  ok('report link on the task is prefilled', /discussions\/new\?title=Task/.test(await p.getAttribute('.facts a[href*="discussions"]', 'href')));
  // picker by keyboard
  await p.click('#rb-model .pk-btn'); await p.waitForTimeout(200);
  await p.fill('#rb-model .pk-q', 'kimi'); await p.waitForTimeout(250);
  const first = await p.getAttribute('#rb-model .pk-row.act', 'data-id');
  await p.keyboard.press('Enter'); await p.waitForTimeout(250);
  ok('picker: search + Enter selects the highlighted model', first && (await p.textContent('#rb-model .pk-name')) === first.split('/')[1], first);
  await p.click('#rb-model .pk-btn'); await p.waitForTimeout(150); await p.keyboard.press('Escape'); await p.waitForTimeout(150);
  ok('picker closes on Escape', !(await p.isVisible('#rb-model .pk-pop')));
  ok('the judge picker is there for a judged task', await p.isVisible('#rb-judge .pk-btn'));
  const est1 = await p.textContent('#rb-est');
  await p.click('#rb-adv summary'); await p.selectOption('#p-think', 'none'); await p.fill('#p-steps', '120'); await p.waitForTimeout(200);
  ok('advanced settings: the summary reflects changes', /thinking off/.test(await p.textContent('#adv-sum')) && /120 steps/.test(await p.textContent('#adv-sum')));
  await p.click('[data-vis="private"]'); await p.waitForTimeout(150);
  ok('visibility: private shows the community note', /Keeping rollouts public helps/.test(await p.textContent('#vis-note')));
  await p.click('[data-vis="public"]'); await p.waitForTimeout(150);
  ok('visibility: back to public shows confetti', (await p.$$('canvas')).length > 0);
  ok('visibility: public explains anonymity and the dataset', /without your name/.test(await p.textContent('#vis-note')));
  // bring your own endpoint
  await p.click('[data-src="byo"]'); await p.waitForTimeout(200);
  ok('own endpoint: Run waits for a connection test', await p.isDisabled('#rb-go'));
  await p.fill('#ep-url', 'http://localhost:1234/v1'); await p.fill('#ep-model', 'x'); await p.click('#ep-test'); await p.waitForTimeout(1500);
  ok('own endpoint: an http/private URL is refused with a reason', /https/.test(await p.textContent('#ep-status')), await p.textContent('#ep-status'));
  await p.click('[data-src="hf"]'); await p.waitForTimeout(200);
  // run button, request intercepted
  let posted = null;
  await p.route('**/api/runs', async (route) => {
    if (route.request().method() === 'POST') { posted = route.request().postDataJSON(); await route.fulfill({ json: { id: OWN_RUN } }); }
    else route.continue();
  });
  await p.click('#rb-go'); await p.waitForTimeout(1500);
  ok('Run rollout posts the chosen settings and opens the rollout', posted && posted.visibility === 'public' && posted.params.thinking === 'none' && posted.params.steps === 120 && posted.judge
     && (await p.evaluate(() => location.hash)).includes(OWN_RUN), JSON.stringify(posted || {}).slice(0, 160));
  await p.unroute('**/api/runs');

  // task page: code repository
  await go(p, '#/task/format-code-task-001406', 2500);
  ok('code task: repository tree rendered', (await p.$$('.rt-dir')).length > 3);
  await p.click('.rt-dir'); await p.waitForTimeout(200);
  ok('repository: a folder expands', (await p.getAttribute('.rt-dir', 'aria-expanded')) === 'true');
  await p.fill('#repo-q', 'makefile'); await p.waitForTimeout(400);
  await p.click('.rt-file'); await p.waitForTimeout(1200);
  ok('repository: go-to-file and the viewer with line numbers', (await p.$$('.rt-code .ln')).length > 5);

  // compare: pick rollouts on the task page, then work the comparison
  await go(p, '#/task/' + CMP_TASK, 2500);
  const picks = await p.$$('#task-runs [data-cmp]');
  ok('compare: every rollout on the task page has a checkbox', picks.length >= 3, String(picks.length));
  await picks[0].click(); await p.waitForTimeout(150);
  ok('compare: one pick asks for another', await p.isVisible('#cmp-bar') && await p.isDisabled('#cmp-bar button.primary'));
  await picks[1].click(); await picks[2].click(); await p.waitForTimeout(150);
  await p.click('#cmp-bar a.primary'); await p.waitForTimeout(2500);
  const letters = () => p.$$eval('.cmp-col .cmp-tag', (t) => t.map((x) => x.textContent.trim()).join(''));
  ok('compare: opens with A, B and C', /#\/compare\//.test(await p.evaluate(() => location.hash)) && (await letters()) === 'ABC');
  ok('compare: summary, grade, output, how they worked and traces', (await p.$$eval('.cmp-sec', (s) => s.map((x) => x.id).join(' '))) === 'cmp-summary cmp-grade cmp-outcome cmp-work cmp-traces');
  ok('compare: the chart is drawn', await p.isVisible('#cmp-chart svg'));
  const shown = () => p.$$eval('#cmp-summary tbody tr:not(.grp)', (r) => r.filter((x) => getComputedStyle(x).display !== 'none').length);
  const all = await shown();
  await p.click('#cmp-same'); await p.waitForTimeout(150);
  const differ = await shown();
  ok('compare: "Hide rows that match" leaves only the differences', differ > 0 && differ < all, `${differ} of ${all}`);
  await p.click('[data-metric="cost"]'); await p.waitForTimeout(200);
  ok('compare: the chart switches to cost', /Model cost/.test(await p.getAttribute('#cmp-chart svg', 'aria-label')));
  await p.click('[data-remove="1"]'); await p.waitForTimeout(400);
  ok('compare: removing B keeps A and C, and the link keeps the gap', (await letters()) === 'AC' && /r=[^,]+,,/.test(await p.evaluate(() => location.hash)));
  await p.click('.cmp-add summary'); await p.waitForTimeout(200);
  await p.click('.cmp-add [data-add]:not([disabled])'); await p.waitForTimeout(1500);
  ok('compare: adding from the list fills the empty B', (await letters()) === 'ABC');
  await p.click('.cmp-add summary'); await p.waitForTimeout(150); await p.keyboard.press('Escape'); await p.waitForTimeout(150);
  ok('compare: Escape closes the list', !(await p.$('.cmp-add[open]')));
  await p.click('.cmp-toc [data-jump="traces"]'); await p.waitForTimeout(900);
  ok('compare: the slim bar follows once the cards scroll away', await p.isVisible('#cmp-strip.on') && (await p.$$('#cmp-strip .cmp-chip')).length === 3);
  ok('compare: traces side by side', (await p.$$eval('.cmp-tcol', (c) => c.filter((x) => getComputedStyle(x).display !== 'none').length)) === 3);

  // own rollout page
  await go(p, '#/run/' + OWN_RUN, 2000);
  ok('run page: Compare opens the comparison with this rollout', (await p.getAttribute('.rp-actions a[href*="#/compare/"]', 'href')).endsWith('?r=' + OWN_RUN));
  ok('run page: stepper, timeline, grade, cost and details', (await p.$$('.step')).length === 5 && (await p.$$('#rp-tl .ev')).length > 2 && await p.isVisible('.big') && await p.isVisible('.big-cost') && /Harness/.test(await p.textContent('#rp-meta')));
  ok('run page: screenshot in the grade card', await p.isVisible('.grade .shot img'));
  await p.click('#rp-expand'); await p.waitForTimeout(200);
  ok('run page: Expand all opens every step', await p.evaluate(() => [...document.querySelectorAll('#rp-tl details')].every((d) => d.open)));
  ok('run page: report link prefilled with the rollout', /discussions\/new\?title=Rollout/.test(await p.getAttribute('.rp-actions a[href*="discussions"]', 'href')));
  await p.click('#rp-vis'); await p.waitForTimeout(300);
  ok('make private asks first, with the community note', await p.isVisible('.dialog') && /helps the broader community/.test(await p.textContent('.dialog')));
  await p.click('#vis-go'); await p.waitForTimeout(800);
  ok('after confirming, the rollout is private', /Private/.test(await p.textContent('#rp-head .kick')));
  await p.click('#rp-vis'); await p.waitForTimeout(600);
  ok('making it public again: public, with confetti', /Public/.test(await p.textContent('#rp-head .kick')) && (await p.$$('canvas')).length > 0);

  // someone else's public rollout
  await go(p, '#/run/' + OTHER_RUN, 2000);
  ok("someone else's rollout: read-only, anonymous", !(await p.$('#rp-vis')) && !(await p.$('#rp-cancel')) && /shared anonymously/.test(await p.textContent('#rp-head')) && /Run it yourself/.test(await p.textContent('.rp-actions')));
  ok("someone else's rollout: nav and breadcrumb point to Community", (await p.getAttribute('[data-nav="community"]', 'class')).includes('on') && (await p.textContent('.crumbs a')) === 'Community');

  // reward design: the agent prompt, each domain's grading detail, the dataset page
  await go(p, '#/task/s3k_0000_accounting_audit_tax_en_t1_rl_008', 2000);
  ok('task: agent prompt with the harness and its limits', await p.isVisible('#sec-prompt') && /OpenCode/.test(await p.textContent('#sec-prompt')) && /500 model calls/.test(await p.textContent('#sec-prompt')));
  const folded = (await p.innerText('#sec-prompt pre')).length;   // innerText: the folded task is hidden, not absent
  await p.click('#sec-prompt [data-pr-expand]'); await p.waitForTimeout(200);
  ok('task: the folded task opens inside the prompt', !(await p.$('#sec-prompt [data-pr-expand]')) && (await p.innerText('#sec-prompt pre')).length > folded + 500);
  ok('task: General grading has the formula, the rules and the judge prompt', /Σ weight × score/.test(await p.textContent('#sec-grading')) && (await p.$$('#sec-grading .rules li')).length >= 4 && /judge prompt, in English/.test(await p.textContent('#sec-grading')));
  ok('task: a check without a weight is called out with its share', /decides 50% of the reward/.test(await p.textContent('#sec-grading')));
  ok('task: no expected answer anywhere on the page', await p.evaluate(() => !/Ground truth|pass_anchor|gold_answer|check_code/.test(document.body.innerHTML)));
  await go(p, '#/task/dasyn_260630_00886', 1500);
  ok('task: Webdev grading has the judge instructions, 35 bands and training', (await p.$$('#sec-grading .bands ol li')).length === 35 && /In training/.test(await p.textContent('#sec-grading')));
  await go(p, '#/task/music-gK-0216', 1500);
  ok('task: Music grading has the group weights and curves', /rhythm · 22%/i.test(await p.textContent('#sec-grading')) && /Histogram similarity/.test(await p.textContent('#sec-grading')));
  await go(p, '#/rewards', 1500);
  ok('reward design: nav tab, six kinds of task, the rubric counts', (await p.getAttribute('[data-nav="rewards"]', 'class')).includes('on') && (await p.$$('.rw-tbl tbody tr')).length === 6 && /5,125 across 925 tasks/.test(await p.textContent('#rw-body')));
  await p.click('.rw-tbl tbody tr a'); await p.waitForTimeout(1500);
  ok('reward design: an example opens its task', /#\/task\//.test(await p.evaluate(() => location.hash)) && await p.isVisible('#sec-grading'));

  // community
  await go(p, '#/community', 1500);
  ok('community: figures, leaderboard and rollouts', (await p.$$('.cm-figs > div')).length === 5 && (await p.$$('.cm-lb tbody tr')).length > 0 && (await p.$$('#cm-rows .runrow')).length > 0);
  ok('community: no failed or unscored rollouts shown', !(await p.textContent('#cm-rows')).includes('not scored'));
  await p.selectOption('[data-f="reward"]', 'partial'); await p.waitForTimeout(800);
  ok('community: reward filter narrows and lands in the URL', /reward=partial/.test(await p.evaluate(() => location.hash)));
  ok('community: active filters show as removable chips', await p.isVisible('[data-unset="reward"]'));
  await p.click('#cm-clear'); await p.waitForTimeout(600);
  const hcell = await p.$('[data-cell]');
  if (hcell) { await hcell.click(); await p.waitForTimeout(700); ok('community: a leaderboard score filters by model and domain', /model=.*domain=|domain=.*model=/.test(await p.evaluate(() => location.hash))); await p.click('#cm-clear'); await p.waitForTimeout(600); }
  await p.click('[data-view="tasks"]'); await p.waitForTimeout(2500);
  ok('community tasks: untried tasks listed by default', (await p.$$('#ct-rows .runrow')).length === 40 && /not tried yet/.test(await p.textContent('#cm-top')));
  await p.click('[data-has="some"]'); await p.waitForTimeout(500);
  ok('community tasks: "Has rollouts" filter', (await p.$$('#ct-rows .runrow')).length >= 1 && !(await p.textContent('#ct-rows')).includes('no rollouts yet'));

  // my rollouts
  await go(p, '#/runs', 1500);
  ok('my rollouts: table with visibility badges', (await p.$$('.rs-row:not(.head)')).length > 0 && (await p.$$('.rs-row .vis-public, .rs-row .vis-private')).length > 0);
  await p.click('.seg [data-f="done"]'); await p.waitForTimeout(300);
  ok('my rollouts: status filter', (await p.getAttribute('.seg [data-f="done"]', 'aria-pressed')) === 'true');

  // sign-in dialog
  await p.click('.acct-btn'); await p.click('.menu [data-signin]'); await p.waitForTimeout(300);
  ok('sign-in dialog opens from the menu', await p.isVisible('.dialog #tok'));
  await p.fill('#tok', 'not-a-token'); await p.click('#tok-go'); await p.waitForTimeout(800);
  ok('a malformed token gets a clear error', /start with hf_/.test(await p.textContent('#tok-err')));
  await p.keyboard.press('Escape'); await p.waitForTimeout(200);
  ok('Escape closes the dialog', !(await p.$('.dialog')));
  ok('no JavaScript errors on desktop', p.errors.length === 0, p.errors.slice(0, 3).join(' | '));
  await p.context().close();
}

// ── every width and theme: layout ────────────────────────────────────────────
const PAGES = ['#/', '#/task/s3k_0000_accounting_audit_tax_en_t1_rl_008', '#/task/format-code-task-001406', '#/task/dasyn_260630_00886',
  '#/task/music-gK-0216', '#/task/arvo_42480818', '#/run/' + OWN_RUN, '#/run/' + OTHER_RUN, '#/runs', '#/community', '#/community?view=tasks', '#/rewards',
  `#/compare/${CMP_TASK}?r=${CMP_RUNS}`];
for (const theme of ['light', 'dark']) for (const w of [390, 768, 1280, 1920]) {
  const p = await page(w, theme);
  for (const h of PAGES) {
    await go(p, h, 1400);
    await p.evaluate(() => document.querySelectorAll('details').forEach((d) => (d.open = true)));
    await p.waitForTimeout(150);
    ok(`${theme} ${w}px ${h}: no sideways scroll`, await noOverflow(p));
    if (theme === 'light' && (w === 390 || w === 768) && /#\/$|s3k_0000|001406|0d8314$|community$/.test(h))
      await p.screenshot({ path: `${SHOTS}/ui-${w}${h.replace(/[#/?=]+/g, '_').slice(0, 26)}.png` });
  }
  if (w === 390 && theme === 'light') {   // phone-only interactions, from a fresh load (the layout pass force-opened every section)
    await go(p, '#/', 300); await p.reload({ waitUntil: 'networkidle' }); await p.waitForTimeout(1200);
    ok('phone: the overview map starts collapsed', !(await p.getAttribute('#map-card', 'open')));
    await p.click('#open-filters'); await p.waitForTimeout(400);
    ok('phone: filters open as a bottom sheet', await p.evaluate(() => document.querySelector('.filters').classList.contains('open')));
    ok('phone: the sheet has a backdrop and locks the page', await p.isVisible('#sheet-scrim') && await p.evaluate(() => document.body.classList.contains('noscroll')));
    ok('phone: the sheet reaches the bottom of the screen', await p.evaluate(() => Math.abs(document.querySelector('.filters').getBoundingClientRect().bottom - innerHeight) < 2));
    ok('phone: facet sections start folded', (await p.$$('.filters .facet[open]')).length === 0);
    await p.click('.filters .facet summary'); await p.click('.filters .facet[open] .opt:not(.zero)'); await p.waitForTimeout(400);
    ok('phone: the sheet button shows the result count', /Show [\d,]+ environments/.test(await p.textContent('#show-results')));
    await p.click('#show-results'); await p.waitForTimeout(400);
    ok('phone: Show results closes the sheet and unlocks the page', await p.evaluate(() => !document.querySelector('.filters').classList.contains('open') && !document.body.classList.contains('noscroll')));
    ok('phone: the list starts on the first screen', await p.evaluate(() => document.querySelector('#list').getBoundingClientRect().top + scrollY < 400));
    ok('phone: brand text hidden, logo kept', !(await p.isVisible('.brand .name')) && await p.isVisible('.logo'));
    await go(p, `#/compare/${CMP_TASK}?r=${CMP_RUNS}`, 2500);
    const cols = () => p.$$eval('.cmp-tcol', (c) => c.filter((x) => getComputedStyle(x).display !== 'none').map((x) => x.dataset.col).join(''));
    ok('phone compare: one trace at a time, behind tabs', await p.isVisible('.cmp-tabs') && (await cols()) === '0');
    await p.click('.cmp-tabs [data-tab="1"]'); await p.waitForTimeout(200);
    ok('phone compare: a tab switches the trace', (await cols()) === '1');
    ok('phone compare: rollout cards two to a row', await p.evaluate(() => { const c = [...document.querySelectorAll('.cmp-col')]; return c.length > 1 && c[0].getBoundingClientRect().top === c[1].getBoundingClientRect().top; }));
  }
  ok(`${theme} ${w}px: no JavaScript errors`, p.errors.length === 0, p.errors.slice(0, 3).join(' | '));
  await p.context().close();
}
// ── signed out, the narrowest phones: the header carries a Sign in button instead of the avatar ─────────
for (const w of [320, 360, 390]) {
  const ctx = await b.newContext({ viewport: { width: w, height: 800 } });
  await ctx.route('**/api/me', (r) => r.fulfill({ json: { user: null, local: false, oauth: true, version: '1.0.0', source: 'test', missing_scopes: [] } }));
  const p = await ctx.newPage();
  for (const h of ['#/', '#/community', '#/rewards', '#/task/s3k_0000_accounting_audit_tax_en_t1_rl_008', '#/runs', '#/run/' + OWN_RUN, `#/compare/${CMP_TASK}?r=${CMP_RUNS}`]) {
    await p.goto(BASE + '/' + h, { waitUntil: 'networkidle' }); await p.waitForTimeout(900);
    ok(`signed out ${w}px ${h}: no sideways scroll`, await noOverflow(p));
  }
  ok(`signed out ${w}px: the Sign in button is reachable`, await p.isVisible('#account [data-signin]'));
  await ctx.close();
}
await b.close();

const failed = results.filter((r) => !r.pass);
for (const r of results) if (!r.pass || !/no sideways scroll/.test(r.name)) console.log(`${r.pass ? 'PASS' : 'FAIL'}  ${r.name}${r.detail && !r.pass ? '  — ' + r.detail : ''}`);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
process.exit(failed.length ? 1 : 0);

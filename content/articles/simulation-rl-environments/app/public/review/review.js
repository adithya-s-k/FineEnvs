// Review layer for the article, modelled on Google Docs comments. Loaded only when the page is opened
// with ?review (see index.astro); it talks to review/server.py at /api/review/ and never runs on the
// published page.
//
// Comment cards float in the right margin, level with the text they point at, on a layer above the
// page, so the article itself never moves. Clicking a highlight or a card makes it active: it expands,
// lines up with its text and pushes the others aside. On narrow screens the same cards live in the
// comments panel instead.
(() => {
  if (window.__mhrlReview) return;
  window.__mhrlReview = true;

  const API = '/api/review';
  const REACTIONS = ['👍', '❤️', '😄', '🎉', '👀', '➕', '🔥'];
  const main = document.querySelector('main');
  if (!main) return;

  const esc = (t) => String(t ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const el = (html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; };
  const ago = (iso) => {
    const s = (Date.now() - Date.parse(iso)) / 1000;
    if (s < 60) return 'just now';
    if (s < 3600) return `${Math.floor(s / 60)} min ago`;
    if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
    return new Date(iso).toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
  };
  const call = async (method, path, body) => {
    const r = await fetch(API + path, {
      method, credentials: 'same-origin',
      headers: body ? { 'content-type': 'application/json', 'x-review': '1' } : { 'x-review': '1' },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!r.ok) {
      let msg = r.statusText;
      try { msg = (await r.json()).detail || msg; } catch (e) {}
      throw new Error(msg);
    }
    return r.json();
  };

  // ------------------------------------------------------------ state
  let me = null, threads = [], lastError = '';
  let active = null;          // id of the expanded thread, or 'new' for the composer
  let composing = null;       // anchor of the comment being written
  let editing = null;         // { tid, mid } of the message being edited
  let historyOpen = false, filter = 'open';
  let anchors = {};           // thread id -> element it points at (first <mark>, or the figure)

  const isOwner = () => me && me.user && me.owner && me.user.username.toLowerCase() === me.owner.toLowerCase();
  const mine = (m) => me && me.user && m.author.username === me.user.username;

  // ------------------------------------------------------------ text index
  // The article's prose as one string, whitespace collapsed, with a map back to text nodes. Figures
  // are left out: they get their own comment button instead of text selection.
  const skip = (n) => n && n.closest && n.closest('figure.html-embed, script, style, .rv-ui');
  const buildIndex = () => {
    const walker = document.createTreeWalker(main, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => (n.nodeValue && !skip(n.parentElement) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT),
    });
    let text = '', map = [], prevSpace = true;
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const v = n.nodeValue;
      for (let i = 0; i < v.length; i++) {
        const sp = /\s/.test(v[i]);
        if (sp && prevSpace) continue;
        text += sp ? ' ' : v[i];
        map.push([n, i]);
        prevSpace = sp;
      }
    }
    return { text, map };
  };
  const posOf = (idx, node, offset) => {
    for (let i = 0; i < idx.map.length; i++) {
      const [n, o] = idx.map[i];
      if (n === node && o >= offset) return i;
      if (n !== node && node.compareDocumentPosition(n) & Node.DOCUMENT_POSITION_FOLLOWING) return i;
    }
    return idx.map.length;
  };
  const sectionOf = (node) => {
    let chapter = '', section = '';
    for (const h of main.querySelectorAll('h2, h3, h4')) {
      if (!(h.compareDocumentPosition(node) & Node.DOCUMENT_POSITION_FOLLOWING)) break;
      if (h.tagName === 'H2') { chapter = h.textContent.trim(); section = ''; } else section = h.textContent.trim();
    }
    return { chapter, section };
  };

  // ------------------------------------------------------------ anchoring
  // Figures are keyed by their title, since one embed file used several times repeats the same id.
  const figKey = (f) => f.querySelector('.html-embed__title')?.textContent.trim() || f.id;
  const figByKey = () => { const m = {}; main.querySelectorAll('figure.html-embed').forEach((f) => { const k = figKey(f); if (k && !m[k]) m[k] = f; }); return m; };
  const unmark = () => {
    main.querySelectorAll('mark.rv-hl').forEach((m) => { const p = m.parentNode; while (m.firstChild) p.insertBefore(m.firstChild, m); p.removeChild(m); p.normalize(); });
    main.querySelectorAll('.rv-fig-count, ins.rv-ins').forEach((b) => b.remove());
    main.querySelectorAll('figure.rv-fig-active').forEach((f) => f.classList.remove('rv-fig-active'));
  };
  const score = (text, at, a, q) => {
    let s = 0;
    const pre = a.prefix || '', suf = a.suffix || '';
    for (let i = 1; i <= pre.length && text[at - i] === pre[pre.length - i]; i++) s++;
    const end = at + q.length;
    for (let i = 0; i < suf.length && text[end + i] === suf[i]; i++) s++;
    return s;
  };
  const findQuote = (idx, a) => {
    const q = (a.quote || '').replace(/\s+/g, ' ').trim();
    if (!q) return null;
    let best = null, bestScore = -1;
    for (let at = idx.text.indexOf(q); at !== -1; at = idx.text.indexOf(q, at + 1)) {
      const s = score(idx.text, at, a, q);
      if (s > bestScore) { best = [at, at + q.length]; bestScore = s; }
    }
    return best;
  };
  const wrap = (idx, start, end, tid) => {
    const pieces = new Map();
    for (let i = start; i < end; i++) {
      const [n, o] = idx.map[i];
      const p = pieces.get(n) || [o, o];
      p[0] = Math.min(p[0], o); p[1] = Math.max(p[1], o + 1);
      pieces.set(n, p);
    }
    let first = null, last = null;
    for (const [n, [s, e]] of pieces) {
      if (!n.parentNode || e > n.nodeValue.length) continue;
      const mid = n.splitText(s);
      mid.splitText(e - s);
      const m = document.createElement('mark');
      m.className = 'rv-hl';
      m.dataset.thread = tid;
      mid.parentNode.insertBefore(m, mid);
      m.appendChild(mid);
      first = first || m;
      last = m;
    }
    return { first, last };
  };
  // Open threads are highlighted; a resolved one only while it is open from the comments panel.
  const shown = () => threads.filter((t) => t.status !== 'resolved' || t.id === active);
  const anchorAll = () => {
    unmark();
    const found = {};
    const idx = buildIndex(), figs = figByKey(), spots = [];
    for (const t of shown()) {
      if (t.anchor.type === 'text') {
        const r = findQuote(idx, t.anchor);
        if (r) spots.push([r, t]);
      } else if (t.anchor.type === 'figure' && figs[t.anchor.figure]) {
        found[t.id] = figs[t.anchor.figure];
      }
    }
    // Highlights can overlap (a word inside a commented sentence). Each wrap splits text nodes, so the
    // index is rebuilt before every one; the text never changes, so the offsets stay valid.
    for (const [[s, e], t] of spots) {
      const w = wrap(buildIndex(), s, e, t.id);
      if (!w.first) continue;
      found[t.id] = w.first;
      if (t.kind === 'suggestion') {
        // Suggestions read like Docs' suggesting mode: the old text struck through, the new text after it.
        main.querySelectorAll(`mark.rv-hl[data-thread="${t.id}"]`).forEach((m) => m.classList.add('rv-sug'));
        if (t.suggestion.action === 'replace') {
          const ins = document.createElement('ins');
          ins.className = 'rv-ins rv-ui'; ins.dataset.thread = t.id; ins.textContent = t.suggestion.text;
          w.last.after(ins);
        }
      }
    }
    const perFig = {};
    shown().filter((t) => t.anchor.type === 'figure' && found[t.id]).forEach((t) => { perFig[t.anchor.figure] = (perFig[t.anchor.figure] || 0) + 1; });
    Object.entries(perFig).forEach(([k, c]) => {
      const btn = figs[k]?.querySelector('.rv-fig-btn');
      if (btn) btn.insertAdjacentHTML('beforeend', `<span class="rv-fig-count">${c}</span>`);
    });
    return found;
  };
  const markActive = () => {
    main.querySelectorAll('mark.rv-hl.is-active').forEach((m) => m.classList.remove('is-active'));
    main.querySelectorAll('figure.rv-fig-active').forEach((f) => f.classList.remove('rv-fig-active'));
    if (!active || active === 'new') return;
    main.querySelectorAll(`mark.rv-hl[data-thread="${active}"]`).forEach((m) => m.classList.add('is-active'));
    const t = threads.find((x) => x.id === active);
    if (t && t.anchor.type === 'figure' && anchors[active]) anchors[active].classList.add('rv-fig-active');
  };

  // ------------------------------------------------------------ UI shell
  const ui = el(`<div class="rv-ui">
    <div class="rv-bar">
      <span class="rv-pill">Review mode</span>
      <span class="rv-user"></span>
      <button type="button" class="rv-btn" data-act="history"></button>
      <button type="button" class="rv-btn rv-quiet" data-act="exit" title="Back to the published view">Exit</button>
    </div>
    <div class="rv-margin"></div>
    <aside class="rv-panel" aria-label="Comments" hidden>
      <div class="rv-panel-head"><b>Comments</b>
        <div class="rv-tabs" role="tablist"></div>
        <button type="button" class="rv-icon" data-act="close-history" title="Close">✕</button></div>
      <div class="rv-panel-list"></div>
    </aside>
    <div class="rv-add" hidden>
      <button type="button" data-kind="comment" title="Add comment (⌘/Ctrl + Alt + M)">＋ Comment</button>
      <button type="button" data-kind="delete" title="Suggest deleting this text">Suggest delete</button>
      <button type="button" data-kind="replace" title="Suggest new wording for this text">Suggest edit</button>
    </div>
    <div class="rv-menu" hidden></div>
  </div>`);
  document.body.appendChild(ui);
  const bar = ui.querySelector('.rv-bar'), margin = ui.querySelector('.rv-margin'), panel = ui.querySelector('.rv-panel');
  const panelList = ui.querySelector('.rv-panel-list'), tabs = ui.querySelector('.rv-tabs'), addBtn = ui.querySelector('.rv-add'), menu = ui.querySelector('.rv-menu');

  const inIframe = (() => { try { return window.self !== window.top; } catch (e) { return true; } })();
  const here = () => location.pathname + location.search;

  // Margin mode needs room to the right of the text; otherwise the cards live in the panel.
  let geo = { margin: false, left: 0, width: 300 };
  const measure = () => {
    // The text column, not <main>, which is wider than the prose it holds.
    const edges = [...main.querySelectorAll('p')].filter((p) => !p.closest('figure, .rv-ui')).slice(0, 40)
      .map((p) => p.getBoundingClientRect()).filter((r) => r.width > 0).map((r) => r.right);
    const right = (edges.length ? Math.max(...edges) : main.getBoundingClientRect().right) + window.scrollX;
    const room = document.documentElement.clientWidth - (right - window.scrollX) - 24;
    geo = { margin: room >= 270, left: right + 28, width: Math.min(320, room - 12) };
  };

  // ------------------------------------------------------------ rendering
  const reactionsHtml = (t, m) => {
    const r = m.reactions || {};
    const chips = Object.entries(r).filter(([, who]) => who.length).map(([e, who]) =>
      `<button type="button" class="rv-react${me.user && who.includes(me.user.username) ? ' is-mine' : ''}" data-act="react" data-mid="${m.id}" data-emoji="${e}" title="${esc(who.join(', '))}">${e} ${who.length}</button>`).join('');
    return `<div class="rv-reacts">${chips}<button type="button" class="rv-icon rv-react-add" data-act="react-pick" data-mid="${m.id}" title="Add reaction">☺</button></div>`;
  };
  const msgHtml = (t, m, first) => {
    const body = editing && editing.tid === t.id && editing.mid === m.id
      ? `<textarea class="rv-edit" rows="3">${esc(m.body)}</textarea>
         <div class="rv-row"><button type="button" class="rv-btn rv-primary" data-act="save-edit" data-mid="${m.id}">Save</button><button type="button" class="rv-link" data-act="cancel-edit">Cancel</button></div>`
      : (m.body ? `<div class="rv-body">${esc(m.body).replace(/\n/g, '<br>')}</div>` : '');
    return `<div class="rv-msg" data-mid="${m.id}">
      <div class="rv-meta"><img src="${esc(m.author.avatar)}" alt=""><span class="rv-who"><b>${esc(m.author.name)}</b><span>${ago(m.created_at)}${m.edited_at ? ' · edited' : ''}</span></span>
        ${first && t.status !== 'resolved' ? `<button type="button" class="rv-icon rv-resolve" data-act="resolve" title="Resolve">✓</button>` : ''}
        <button type="button" class="rv-icon" data-act="menu" data-mid="${m.id}" title="More">⋮</button></div>
      ${body}
      ${reactionsHtml(t, m)}
    </div>`;
  };
  const cardHtml = (t, expanded, where) => {
    const a = t.anchor;
    const target = a.type === 'figure' ? `On figure: ${esc(a.title || a.figure)}` : `“${esc(a.quote.length > 160 ? a.quote.slice(0, 160) + '…' : a.quote)}”`;
    const msgs = expanded ? t.messages : t.messages.slice(0, 1);
    const more = !expanded && t.messages.length > 1 ? `<div class="rv-more">${t.messages.length - 1} ${t.messages.length === 2 ? 'reply' : 'replies'}</div>` : '';
    const tag = t.status === 'resolved' ? '<span class="rv-tag">Resolved</span>' : (where === 'panel' && !anchors[t.id] ? '<span class="rv-tag rv-warn">Detached</span>' : '');
    const sug = t.kind === 'suggestion' ? t.suggestion : null;
    const decided = t.decision ? `<span class="rv-tag${t.decision === 'accepted' ? ' rv-ok' : ''}">${t.decision}</span>` : '';
    const sugHtml = sug ? `<div class="rv-sugline">${decided}<span class="rv-sugkind">${sug.action === 'delete' ? 'Delete' : 'Replace'}</span><del>${esc(a.quote)}</del>${sug.action === 'replace' ? ` <span class="rv-sugkind">with</span><ins>${esc(sug.text)}</ins>` : ''}</div>
      ${t.status !== 'resolved' && isOwner() ? `<div class="rv-row rv-decide"><button type="button" class="rv-btn rv-primary" data-act="accept">✓ Accept</button><button type="button" class="rv-btn" data-act="reject">✕ Reject</button></div>` : ''}` : '';
    return `<div class="rv-card${expanded ? ' is-active' : ''}${t.status === 'resolved' ? ' is-resolved' : ''}${sug ? ' is-suggestion' : ''}" data-thread="${t.id}">
      ${!sug && (where === 'panel' || a.type === 'figure' || !anchors[t.id]) ? `<div class="rv-target">${tag}${target}</div>` : ''}
      ${sug && where === 'panel' && !anchors[t.id] && t.status !== 'resolved' ? '<span class="rv-tag rv-warn">Detached</span>' : ''}
      ${sugHtml}
      ${msgs.map((m, i) => msgHtml(t, m, i === 0 && !sug)).join('')}
      ${more}
      ${expanded ? (t.status === 'resolved'
        ? `<div class="rv-row"><button type="button" class="rv-btn" data-act="reopen">Reopen</button></div>`
        : `<div class="rv-reply"><textarea rows="1" placeholder="Reply"></textarea>
           <div class="rv-row rv-reply-actions" hidden><button type="button" class="rv-btn rv-primary" data-act="send-reply">Reply</button><button type="button" class="rv-link" data-act="cancel-reply">Cancel</button></div></div>`) : ''}
    </div>`;
  };
  const composerHtml = () => {
    const a = composing;
    const quote = esc(a.quote && a.quote.length > 220 ? a.quote.slice(0, 220) + '…' : a.quote);
    let head = '', fields = `<textarea rows="3" placeholder="Add a comment" data-role="new"></textarea>`, label = 'Comment';
    if (a.suggest === 'delete') {
      head = `<div class="rv-sugline"><span class="rv-sugkind">Suggest deleting</span><del>${quote}</del></div>`;
      fields = `<textarea rows="2" placeholder="Why? (optional)" data-role="new"></textarea>`; label = 'Suggest';
    } else if (a.suggest === 'replace') {
      head = `<div class="rv-sugline"><span class="rv-sugkind">Suggest replacing</span><del>${quote}</del></div>`;
      fields = `<textarea rows="3" data-role="replace" placeholder="New text">${esc(a.quote)}</textarea>
        <textarea rows="2" placeholder="Why? (optional)" data-role="note"></textarea>`; label = 'Suggest';
    }
    return `<div class="rv-card rv-compose is-active" data-thread="new">
      <div class="rv-meta"><img src="${esc(me.user.avatar)}" alt=""><span class="rv-who"><b>${esc(me.user.name)}</b></span></div>
      ${a.type === 'figure' ? `<div class="rv-target">On figure: ${esc(a.title || a.figure)}</div>` : ''}
      ${head}${fields}
      <div class="rv-row"><button type="button" class="rv-btn rv-primary" data-act="post">${label}</button><button type="button" class="rv-link" data-act="cancel">Cancel</button></div>
    </div>`;
  };

  const renderBar = () => {
    const u = bar.querySelector('.rv-user');
    if (!me.user) {
      u.innerHTML = inIframe
        ? `<a class="rv-btn" href="${location.origin}/?review" target="_blank" rel="noopener">Open review mode in a new tab</a>`
        : `<a class="rv-btn rv-primary" href="/oauth/huggingface/login?_target_url=${encodeURIComponent(here())}">Sign in with Hugging Face</a>`;
    } else {
      u.innerHTML = `<img src="${esc(me.user.avatar)}" alt=""><span>${esc(me.user.name)}</span>`
        + `<a class="rv-link" href="/oauth/huggingface/logout?_target_url=${encodeURIComponent(here())}">sign out</a>`;
    }
    const hb = bar.querySelector('[data-act="history"]');
    hb.hidden = !me.reviewer;
    hb.textContent = `Comments (${threads.filter((t) => t.status !== 'resolved').length})`;
    hb.classList.toggle('is-on', historyOpen);
  };

  const renderPanel = () => {
    panel.hidden = !(me.reviewer && (historyOpen || (!geo.margin && active)));
    if (panel.hidden) return;
    const counts = { open: threads.filter((t) => t.status !== 'resolved').length, resolved: threads.filter((t) => t.status === 'resolved').length, all: threads.length };
    tabs.innerHTML = ['open', 'resolved', 'all'].map((f) => `<button type="button" class="rv-tab${filter === f ? ' is-on' : ''}" data-filter="${f}">${f[0].toUpperCase() + f.slice(1)} ${counts[f]}</button>`).join('');
    const list = threads.filter((t) => filter === 'all' || (filter === 'resolved' ? t.status === 'resolved' : t.status !== 'resolved'));
    // Document order for placed threads, detached ones at the end.
    const placed = list.filter((t) => anchors[t.id]).sort((a, b) => (anchors[a.id].compareDocumentPosition(anchors[b.id]) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1));
    const loose = list.filter((t) => !anchors[t.id]);
    let html = lastError ? `<p class="rv-note rv-error">${esc(lastError)}</p>` : '';
    if (!geo.margin && active === 'new' && composing) html += composerHtml();
    html += placed.concat(loose).map((t) => cardHtml(t, !geo.margin && t.id === active, 'panel')).join('');
    if (!list.length && active !== 'new') html += `<p class="rv-note">${filter === 'resolved' ? 'Nothing resolved yet.' : 'No comments yet. Select text in the article, or use the Comment button on a figure.'}</p>`;
    panelList.innerHTML = html;
  };

  const renderMargin = () => {
    margin.innerHTML = '';
    if (!me.reviewer || !geo.margin) return;
    if (lastError) margin.insertAdjacentHTML('beforeend', `<p class="rv-note rv-error rv-float-note">${esc(lastError)}</p>`);
    if (active === 'new' && composing) margin.insertAdjacentHTML('beforeend', composerHtml());
    shown().filter((t) => anchors[t.id]).forEach((t) => margin.insertAdjacentHTML('beforeend', cardHtml(t, t.id === active, 'margin')));
    layout();
  };

  // Docs-style stacking: every card wants to sit level with its anchor. The active card gets exactly
  // that spot; cards below it flow downwards and cards above it are pushed up to make room.
  const layout = () => {
    if (!geo.margin) return;
    const cards = [...margin.querySelectorAll('.rv-card')];
    const want = (c) => {
      const id = c.dataset.thread;
      const target = id === 'new' ? composing?.el : anchors[id];
      if (!target || !target.isConnected) return null;
      return target.getBoundingClientRect().top + window.scrollY - 6;
    };
    const rows = cards.map((c) => ({ c, y: want(c), h: c.offsetHeight })).filter((r) => r.y != null).sort((a, b) => a.y - b.y);
    const gap = 10;
    let pivot = rows.findIndex((r) => r.c.classList.contains('is-active'));
    if (pivot < 0) pivot = 0;
    const top = new Array(rows.length);
    if (rows.length) top[pivot] = rows[pivot].y;
    for (let i = pivot + 1; i < rows.length; i++) top[i] = Math.max(rows[i].y, top[i - 1] + rows[i - 1].h + gap);
    for (let i = pivot - 1; i >= 0; i--) top[i] = Math.min(rows[i].y, top[i + 1] - rows[i].h - gap);
    rows.forEach((r, i) => {
      r.c.style.top = `${top[i]}px`;
      r.c.style.left = `${geo.left - (r.c.classList.contains('is-active') ? 18 : 0)}px`;
      r.c.style.width = `${geo.width}px`;
    });
    const note = margin.querySelector('.rv-float-note');
    if (note) { note.style.top = `${window.scrollY + 64}px`; note.style.left = `${geo.left}px`; note.style.width = `${geo.width}px`; }
  };

  const render = () => {
    measure();
    anchors = anchorAll();
    markActive();
    renderBar();
    renderMargin();
    renderPanel();
    const ta = ui.querySelector('textarea[data-role="replace"]') || ui.querySelector('textarea[data-role="new"]') || ui.querySelector('textarea.rv-edit');
    if (ta) { ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length); }
  };

  const activate = (tid, { scroll = false } = {}) => {
    active = tid; editing = null; closeMenu();
    if (tid && tid !== 'new') composing = null;
    render();
    if (!tid) { if (history.replaceState && location.hash.startsWith('#rv-')) history.replaceState(null, '', here()); return; }
    const card = ui.querySelector(`.rv-card[data-thread="${tid}"]`);
    if (scroll && anchors[tid]) anchors[tid].scrollIntoView({ block: 'center', behavior: 'smooth' });
    else if (card && !geo.margin) card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    if (tid !== 'new' && history.replaceState) history.replaceState(null, '', `${here()}#rv-${tid}`);
  };

  // ------------------------------------------------------------ menus
  const closeMenu = () => { menu.hidden = true; menu.innerHTML = ''; };
  const openMenu = (btn, items) => {
    menu.innerHTML = items.map(([act, label]) => `<button type="button" data-act="${act}">${label}</button>`).join('');
    menu.classList.toggle('rv-menu-emoji', items.every(([a]) => a.startsWith('react:')));
    const r = btn.getBoundingClientRect();
    menu.style.top = `${r.bottom + window.scrollY + 4}px`;
    menu.style.left = `${Math.max(window.scrollX + 8, Math.min(r.right + window.scrollX - 170, window.scrollX + document.documentElement.clientWidth - 186))}px`;
    menu.hidden = false;
    menu.dataset.tid = btn.closest('.rv-card')?.dataset.thread || '';
    menu.dataset.mid = btn.dataset.mid || '';
  };

  // ------------------------------------------------------------ figures
  const addFigureButtons = () => {
    main.querySelectorAll('figure.html-embed').forEach((f) => {
      if (!figKey(f) || f.querySelector('.rv-fig-btn')) return;
      f.classList.add('rv-fig');
      const b = el(`<button type="button" class="rv-fig-btn rv-ui" title="Comment on this figure">＋ Comment</button>`);
      b.addEventListener('click', (e) => {
        e.stopPropagation();
        composing = { type: 'figure', figure: figKey(f), title: f.querySelector('.html-embed__title')?.textContent.trim() || '', ...sectionOf(f), el: f };
        activate('new');
      });
      f.appendChild(b);
    });
  };

  // ------------------------------------------------------------ selection
  const clearProbe = () => main.querySelectorAll('span.rv-probe').forEach((p) => { const parent = p.parentNode; p.remove(); if (parent) parent.normalize(); });
  const selectionAnchor = () => {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !sel.rangeCount) return null;
    const r = sel.getRangeAt(0);
    if (!main.contains(r.commonAncestorContainer)) return null;
    const startEl = r.startContainer.nodeType === 1 ? r.startContainer : r.startContainer.parentElement;
    if (skip(startEl)) return null;
    const idx = buildIndex();
    const node = (c, o) => (c.nodeType === 3 ? [c, o] : [c.childNodes[o] || c, 0]);
    const s = posOf(idx, ...node(r.startContainer, r.startOffset));
    const e = posOf(idx, ...node(r.endContainer, r.endOffset));
    const quote = idx.text.slice(s, e).trim();
    if (quote.length < 2) return null;
    const rect = r.getBoundingClientRect();
    const a = { type: 'text', quote, prefix: idx.text.slice(Math.max(0, s - 40), s), suffix: idx.text.slice(e, e + 40), ...sectionOf(r.startContainer), rect };
    // A zero-size marker where the selection starts, so the composer can sit level with it. It is
    // placed only once the comment is started, so selecting text never changes the page.
    a.place = () => { const probe = document.createElement('span'); probe.className = 'rv-probe'; const rr = r.cloneRange(); rr.collapse(true); rr.insertNode(probe); return probe; };
    return a;
  };
  let pending = null;
  const showAdd = () => {
    if (!me || !me.reviewer) return;
    pending = selectionAnchor();
    if (!pending) { addBtn.hidden = true; return; }
    addBtn.hidden = false;
    if (geo.margin) { addBtn.style.top = `${pending.rect.top + window.scrollY}px`; addBtn.style.left = `${geo.left}px`; }
    else { addBtn.style.top = `${pending.rect.bottom + window.scrollY + 8}px`; addBtn.style.left = `${Math.min(pending.rect.right + window.scrollX - 60, window.scrollX + document.documentElement.clientWidth - 130)}px`; }
  };
  const startComment = (kind = 'comment') => {
    if (!pending) return;
    const { rect, place, ...a } = pending;
    if (kind !== 'comment') a.suggest = kind;
    clearProbe();
    a.el = place();
    composing = a; pending = null; addBtn.hidden = true;
    window.getSelection()?.removeAllRanges();
    activate('new');
  };
  document.addEventListener('mouseup', (ev) => { if (!ev.composedPath().includes(ui)) setTimeout(showAdd, 0); });
  document.addEventListener('keyup', (ev) => { if (ev.shiftKey && !ev.composedPath().includes(ui)) setTimeout(showAdd, 0); });
  addBtn.addEventListener('mousedown', (e) => e.preventDefault());
  addBtn.addEventListener('click', (e) => { const b = e.target.closest('[data-kind]'); if (b) startComment(b.dataset.kind); });

  // ------------------------------------------------------------ events
  document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.altKey && e.code === 'KeyM') { e.preventDefault(); showAdd(); startComment(); }
    if (e.key === 'Escape') {
      closeMenu();
      if (active) { if (active === 'new') { composing = null; clearProbe(); } activate(null); }
    }
  });
  // Clicking the page: a highlight opens its thread, anywhere else closes the active one, like Docs.
  // The path is read from the event, not the DOM: a click inside the layer re-renders it, which can
  // detach the clicked element before this handler runs.
  const fromUi = (e) => e.composedPath().includes(ui);
  document.addEventListener('click', (e) => {
    if (e.composedPath().includes(menu)) return;
    closeMenu();
    if (fromUi(e) || !me || !me.reviewer) return;
    const m = e.target.closest('mark.rv-hl, ins.rv-ins');
    if (m) { activate(m.dataset.thread); return; }
    if (active && active !== 'new' && !window.getSelection()?.toString()) activate(null);
  });

  bar.addEventListener('click', (e) => {
    const act = e.target.closest('[data-act]')?.dataset.act;
    if (act === 'history') { historyOpen = !historyOpen; render(); }
    if (act === 'exit') {
      try { sessionStorage.removeItem('mhrl-review'); } catch (err) {}
      const u = new URL(location.href); u.searchParams.delete('review'); u.hash = ''; location.href = u.toString();
    }
  });
  panel.addEventListener('click', (e) => {
    const f = e.target.closest('[data-filter]');
    if (f) { filter = f.dataset.filter; renderPanel(); return; }
    if (e.target.closest('[data-act="close-history"]')) { historyOpen = false; if (!geo.margin) active = null; render(); }
  });

  const replace = (t) => { threads = threads.map((x) => (x.id === t.id ? t : x)); };
  const onCard = async (e) => {
    const btn = e.target.closest('[data-act]');
    const card = e.target.closest('.rv-card');
    if (!card) return;
    const tid = card.dataset.thread;
    if (!btn) {
      // A click on a card makes it active; from the panel it also jumps to the text.
      e.stopPropagation();
      if (tid !== 'new' && tid !== active && !e.target.closest('textarea')) activate(tid, { scroll: e.composedPath().includes(panel) });
      return;
    }
    const act = btn.dataset.act;
    e.stopPropagation();
    try {
      lastError = '';
      if (act === 'cancel') { composing = null; clearProbe(); activate(null); }
      if (act === 'post') {
        const { el: _el, suggest, ...anchor } = composing;
        const body = (card.querySelector('textarea[data-role="new"]') || card.querySelector('textarea[data-role="note"]'))?.value.trim() || '';
        let suggestion = null;
        if (suggest === 'delete') suggestion = { action: 'delete' };
        if (suggest === 'replace') {
          const text = card.querySelector('textarea[data-role="replace"]').value.trim();
          if (!text || text === anchor.quote) return;
          suggestion = { action: 'replace', text };
        }
        if (!body && !suggestion) return;
        btn.disabled = true;
        const t = await call('POST', '/threads', { anchor, body, suggestion });
        threads.push(t); composing = null; clearProbe(); activate(t.id);
      }
      if (act === 'resolve') { replace(await call('PATCH', `/threads/${tid}`, { status: 'resolved' })); activate(null); }
      if (act === 'accept' || act === 'reject') { replace(await call('PATCH', `/threads/${tid}`, { decision: act === 'accept' ? 'accepted' : 'rejected' })); activate(null); }
      if (act === 'reopen') { replace(await call('PATCH', `/threads/${tid}`, { status: 'open' })); activate(tid); }
      if (act === 'send-reply') {
        const ta = card.querySelector('.rv-reply textarea'); const body = ta.value.trim(); if (!body) return;
        btn.disabled = true;
        replace(await call('POST', `/threads/${tid}/replies`, { body })); activate(tid);
      }
      if (act === 'cancel-reply') { const ta = card.querySelector('.rv-reply textarea'); ta.value = ''; ta.rows = 1; ta.blur(); card.querySelector('.rv-reply-actions').hidden = true; layout(); }
      if (act === 'menu') {
        const t = threads.find((x) => x.id === tid), m = t.messages.find((x) => x.id === btn.dataset.mid);
        const items = [];
        if (mine(m)) items.push(['edit', 'Edit']);
        if (mine(m) || isOwner()) items.push(['delete', 'Delete']);
        items.push(['link', 'Link to this comment']);
        if (t.status === 'resolved') items.push(['reopen-menu', 'Reopen']);
        openMenu(btn, items);
      }
      if (act === 'save-edit') {
        const body = card.querySelector('textarea.rv-edit').value.trim(); if (!body) return;
        replace(await call('PATCH', `/threads/${tid}/messages/${btn.dataset.mid}`, { body })); editing = null; activate(tid);
      }
      if (act === 'cancel-edit') { editing = null; render(); }
      if (act === 'react-pick') openMenu(btn, REACTIONS.map((r) => [`react:${r}`, r]));
      if (act === 'react') { replace(await call('POST', `/threads/${tid}/messages/${btn.dataset.mid}/reactions`, { emoji: btn.dataset.emoji })); render(); }
    } catch (err) { lastError = err.message; render(); }
  };
  margin.addEventListener('click', onCard);
  panelList.addEventListener('click', onCard);
  // The reply box grows and shows its buttons once it has focus, as in Docs.
  ui.addEventListener('focusin', (e) => {
    if (e.target.matches('.rv-reply textarea')) { e.target.rows = 3; e.target.closest('.rv-reply').querySelector('.rv-reply-actions').hidden = false; layout(); }
  });
  ui.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      const card = e.target.closest('.rv-card'); if (!card) return;
      card.querySelector('[data-act="post"], [data-act="send-reply"], [data-act="save-edit"]')?.click();
    }
  });
  menu.addEventListener('click', async (e) => {
    const act = e.target.closest('[data-act]')?.dataset.act; if (!act) return;
    const tid = menu.dataset.tid, mid = menu.dataset.mid;
    closeMenu();
    try {
      lastError = '';
      if (act === 'edit') { active = tid; editing = { tid, mid }; render(); }
      if (act === 'delete') {
        if (!confirm('Delete this comment?')) return;
        const r = await call('DELETE', `/threads/${tid}/messages/${mid}`);
        if (r.deleted) { threads = threads.filter((x) => x.id !== r.deleted); activate(null); } else { replace(r); render(); }
      }
      if (act === 'link') {
        const url = `${location.origin}${location.pathname}?review#rv-${tid}`;
        try { await navigator.clipboard.writeText(url); } catch (err) { prompt('Link to this comment', url); }
      }
      if (act === 'reopen-menu') { replace(await call('PATCH', `/threads/${tid}`, { status: 'open' })); activate(tid); }
      if (act.startsWith('react:')) { replace(await call('POST', `/threads/${tid}/messages/${mid}/reactions`, { emoji: act.slice(6) })); render(); }
    } catch (err) { lastError = err.message; render(); }
  });

  // Figures render after the page loads and change height, which moves every anchor below them.
  let raf = 0;
  const relayout = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => { measure(); renderMargin(); }); };
  if ('ResizeObserver' in window) new ResizeObserver(relayout).observe(main);
  window.addEventListener('resize', relayout);

  // Read-only view of the state, for debugging from the console.
  window.__rvState = () => ({ active, historyOpen, filter, geo, threads: threads.length, open: threads.filter((t) => t.status !== 'resolved').length });

  // ------------------------------------------------------------ start
  const refresh = async () => {
    if (!me || !me.reviewer) return;
    try { threads = await call('GET', '/threads'); lastError = ''; } catch (err) { lastError = err.message; }
    // Don't redraw under someone who is typing.
    const typing = document.activeElement && document.activeElement.tagName === 'TEXTAREA' && ui.contains(document.activeElement);
    if (!typing) render();
  };
  (async () => {
    try { me = await call('GET', '/status'); } catch (err) { me = null; }
    if (!me || !me.enabled) { ui.remove(); return; }  // published: leave the page untouched
    if (me.reviewer) addFigureButtons();
    await refresh();
    render();
    const m = location.hash.match(/^#rv-(.+)$/);
    if (m && threads.some((t) => t.id === m[1])) {
      const t = threads.find((x) => x.id === m[1]);
      if (t.status === 'resolved') { historyOpen = true; filter = 'resolved'; }
      activate(m[1], { scroll: true });
    }
    setInterval(() => { if (!document.hidden) refresh(); }, 30000);
  })();
})();

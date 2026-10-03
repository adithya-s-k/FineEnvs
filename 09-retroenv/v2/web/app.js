(() => {
  const data = window.RETRO_V2_DATA;
  const state = { task: 0, model: data.models[0].id, route: "1", runtime: 0, selected: null, viewBox: null, dragging: false };
  const $ = (id) => document.getElementById(id);
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

  function taskInfo() { return data.tasks[state.task]; }
  function episode() { return data.episodes.find(row => row.task_id === taskInfo().task_id && row.model_id === state.model); }
  function short(value, n=56) { value = String(value ?? ""); return value.length > n ? value.slice(0,n-1)+"…" : value; }
  function fact(value, label) { return `<div class="fact"><strong>${esc(value)}</strong><small>${esc(label)}</small></div>`; }

  function renderTabs() {
    $("task-tabs").innerHTML = data.tasks.map((task,i) => { const ep=data.episodes.find(row=>row.task_id===task.task_id&&row.model_id===state.model); return `<button class="task-tab ${i===state.task?'active':''}" data-index="${i}"><span class="task-letter">${String.fromCharCode(65+i)}</span><span><strong>${esc(task.label.replace(/^. · /,''))}</strong><small>${task.max_steps} steps · ${ep.graph_available?'graph':'no graph'} · ${Math.round(ep.score.reward*100)} reward</small></span></button>`; }).join("");
    document.querySelectorAll(".task-tab").forEach(button => button.onclick = () => { state.task=Number(button.dataset.index); state.route="1"; state.runtime=0; state.selected=null; render(); });
  }

  function renderModelTabs() {
    $("model-tabs").innerHTML = data.models.map(model => { const ep=data.episodes.find(row=>row.task_id===taskInfo().task_id&&row.model_id===model.id); return `<button type="button" class="model-tab ${model.id===state.model?'active':''}" data-model="${model.id}"><strong>${esc(model.label)}</strong><small>${esc(ep.harness_version)} · ${ep.graph_available?'graph':'no graph'} · ${Math.round(ep.score.reward*100)} reward</small></button>`; }).join("");
    $("comparison-note").textContent=data.comparison_note;
    document.querySelectorAll('[data-model]').forEach(button=>button.onclick=()=>{state.model=button.dataset.model;state.route="1";state.runtime=0;state.selected=null;render();});
  }

  function renderHeader() {
    const ep=episode(), valid=ep.score.valid;
    $("status-pill").className=`status-pill ${valid?'pass':'fail'}`;
    $("status-pill").textContent=valid?"STRICT PASS":"DIAGNOSTIC FAIL";
    $("task-id").textContent=ep.task_id;
    $("task-title").textContent=ep.label;
    $("target-smiles").textContent=ep.task.target_smiles;
    $("reward").textContent=ep.score.reward.toFixed(3);
    $("strict-result").textContent=valid?"all hard gates passed":"one or more hard gates failed";
    $("model-chip").innerHTML=`<span></span>${esc(ep.model)} · ${esc(ep.harness_label)}`;
    $("graph-note").textContent=ep.graph_note;
    $("contract-facts").innerHTML = fact(ep.task.max_steps,"max reactions / route") + fact("exactly 2","route views") + fact("≤ 20","stock hits / query") + fact("hidden","reference routes");
    $("prompt").textContent=`SYSTEM\n${ep.system_prompt}\n\nUSER\n${ep.user_prompt}`;
    $("worked").innerHTML=(ep.diagnosis.worked.length?ep.diagnosis.worked:["None"] ).map(x=>`<li>${esc(x)}</li>`).join("");
    $("failed").innerHTML=(ep.diagnosis.failed.length?ep.diagnosis.failed:["None"] ).map(x=>`<li>${esc(x)}</li>`).join("");
  }

  function renderMetrics() {
    const ep=episode();
    $("metrics").innerHTML=Object.entries(ep.score.components).map(([key,value])=>`<div class="metric-row ${value<.999?'low':''}"><span>${esc(ep.component_labels[key]||key)} <small>· ${Math.round((ep.score_weights[key]||0)*100)}%</small></span><div class="bar"><i style="width:${Math.round(value*100)}%"></i></div><strong>${Math.round(value*100)}%</strong></div>`).join("");
    const shape=ep.dataset.reference_shape.map((r,i)=>`R${i+1}: ${r.steps} steps · arity ${r.reactants_per_step.join('→')}`).join(" | ");
    $("dataset-facts").innerHTML=fact(ep.dataset.benchmark,"source")+fact(ep.dataset.stock_size.toLocaleString(),"stock molecules")+fact(ep.dataset.private_reference_routes,"private routes")+fact(shape,"reference shape");
  }

  function renderTrajectory() {
    const ep=episode(), cost=Number(ep.usage.reported_cost_usd||0); $("tool-count").textContent=`${ep.evidence.length} tool responses · $${cost.toFixed(3)}`;
    $("trajectory-title").textContent=ep.trajectory_title;
    $("trajectory-copy").textContent=ep.trajectory_copy;
    $("trajectory").innerHTML=ep.evidence.map(item=>{
      const args=Object.entries(item.arguments).slice(0,2).map(([k,v])=>`${k}: ${Array.isArray(v)?v.join(' + '):v}`).join(" · ");
      const verdict=item.summary.valid===false||item.summary.supported===false||item.summary.error?" · rejected":"";
      return `<article class="event" data-evidence="${esc(item.evidence_id)}"><div class="event-top"><span class="event-tool">${String(item.index).padStart(2,'0')} · ${esc(item.tool)}${verdict}</span><span class="event-id">${esc(item.evidence_id)}</span></div><p>${esc(short(args,110))}</p></article>`;
    }).join("");
    document.querySelectorAll(".event").forEach(el=>el.onclick=()=>highlightEvidence(el.dataset.evidence));
  }

  function runtimeFacts(ep) {
    const counts = {}, accepted = {}, rejected = {};
    ep.evidence.forEach(item => {
      counts[item.tool] = (counts[item.tool] || 0) + 1;
      const failed = item.summary.valid === false || item.summary.supported === false || Boolean(item.summary.error);
      (failed ? rejected : accepted)[item.tool] = ((failed ? rejected : accepted)[item.tool] || 0) + 1;
    });
    return { counts, accepted, rejected };
  }

  function runtimeStages(ep, facts) {
    const c = name => facts.counts[name] || 0;
    const passedCuts = facts.accepted.validate_disconnection || 0;
    const failedCuts = facts.rejected.validate_disconnection || 0;
    const exactHits = ep.evidence.filter(item => item.tool === "stock_retrieve" && item.arguments.mode === "exact" && (item.summary.hits || []).length).length;
    const validReactions = ep.harness_version === "v2" ? (ep.score.reaction_details || []).filter(item => item.valid).length : (ep.score.step_results || []).filter(item => item.valid).length;
    const graphNodes = ep.graph.nodes || [], graphEdges = ep.graph.edges || [], graphRoutes = ep.graph.routes || [];
    const reactionCount = graphNodes.filter(item => item.type === "reaction").length;
    const terminalTool = ep.harness_version === "v2" ? "emit_graph" : "emit_routes";
    return [
      {
        name: "Receive target", short: "Task envelope", tools: ["inspect_molecule", "pubchem_lookup"],
        sourceTitle: "Public task envelope",
        sourceCopy: `<p>Canonical target SMILES, a ${ep.task.max_steps}-reaction cap, and a stock snapshot ID.</p><div class="runtime-mini"><strong>${esc(short(ep.task.target_smiles, 66))}</strong><small>target only · no stock table · no references</small></div>`,
        gatewayTitle: "Prompt boundary",
        gatewayCopy: `<p>The model receives the target and rules. It may inspect structure, but private routes and the inventory remain outside context.</p>`,
        stateTitle: "Initial planning state",
        stateCopy: `<p>Identify likely reaction centers, decide what to query, and reserve room for exactly two route submissions.</p>`
      },
      {
        name: "Survey chemistry", short: "Precedents", tools: ["reaction_precedent_search", "search_literature"],
        sourceTitle: "Training-visible reaction memory",
        sourceCopy: `<p>Deduplicated product/reactant records with product Morgan fingerprints, reported conditions, and provenance. The active evaluation task is excluded.</p>`,
        gatewayTitle: "reaction_precedent_search",
        gatewayCopy: `<p>Ranks nearby products or filters by reaction class. It returns at most 20 analogues—not the hidden answer.</p><div class="runtime-mini"><strong>${c("reaction_precedent_search")} calls</strong><small>bounded analogue retrieval</small></div>`,
        stateTitle: "Candidate disconnections",
        stateCopy: `<p>The agent turns close precedents into candidate first cuts and alternative branches to test.</p>`
      },
      {
        name: "Pull inventory", short: "Stock retrieval", tools: ["stock_retrieve"],
        sourceTitle: `Frozen stock · ${ep.dataset.stock_size.toLocaleString()} molecules`,
        sourceCopy: `<p><strong>This pilot is not a million-scale inventory.</strong> It canonicalizes one immutable snapshot into exact SMILES/InChIKey maps, RDKit molecules, and Morgan fingerprints.</p><div class="scale-compare"><div><b>Pilot</b><span>in-memory exact maps; SMARTS scan; fingerprint ranking</span></div><div><b>Million-scale design</b><span>partitioned vendor snapshot; substructure index; ANN fingerprint service</span></div></div>`,
        gatewayTitle: "stock_retrieve · the only stock access",
        gatewayCopy: `<p>Modes: exact, InChIKey, class, substructure, or similarity. Every response is capped at 20 molecules.</p><div class="runtime-mini"><strong>${c("stock_retrieve")} calls · ${exactHits} exact hits</strong><small>full catalog never enters the prompt</small></div>`,
        stateTitle: "Small candidate sets",
        stateCopy: `<p>Broad queries suggest purchasable classes. Only an exact lookup may certify a terminal node as <code>in_stock</code>.</p>`
      },
      {
        name: "Ground each cut", short: "Validate + enrich", tools: ["validate_disconnection", "reaction_class_lookup", "reaction_conditions_search"],
        sourceTitle: "Private reaction evidence",
        sourceCopy: `<p>Hidden task records and trusted templates remain server-side. They are queried one proposed product/reactant cut at a time.</p>`,
        gatewayTitle: "Three deterministic chemistry tools",
        gatewayCopy: `<div class="runtime-stack"><span><b>validate</b> structure, atom inventory, dataset/template support</span><span><b>class</b> name a supported cut</span><span><b>conditions</b> retrieve frozen reported reagents</span></div>`,
        stateTitle: "Accept or backtrack",
        stateCopy: `<p><strong>${passedCuts} supported</strong> and <strong>${failedCuts} rejected</strong> validation responses in this run. ${ep.evidence_contract ? "Every response gets an evidence ID for later graph citation." : "The legacy harness records the call, but did not require graph citations."}</p>`
      },
      {
        name: "Grow the route", short: "DAG workspace", tools: [],
        sourceTitle: "Accepted cuts + exact stock hits",
        sourceCopy: `<p>The agent reuses molecule IDs, connects atom-contributing reactants, and keeps alternatives separated by first disconnection.</p>`,
        gatewayTitle: "Agent graph workspace",
        gatewayCopy: `<p>Rejected cuts trigger another proposal. Open intermediates are expanded until every route leaf is confirmed stock or the budget is exhausted.</p>`,
        stateTitle: ep.graph_available ? (ep.evidence_contract ? "Evidence-bearing DAG" : "Legacy tree → display DAG") : "No route graph emitted",
        stateCopy: `<p>${graphNodes.length} nodes, ${graphEdges.length} typed edges, and ${graphRoutes.length} route views. ${validReactions}/${reactionCount} submitted reactions passed chemistry verification.</p>`
      },
      {
        name: "Emit + score", short: "Hard gates", tools: [terminalTool],
        sourceTitle: "One structured graph submission",
        sourceCopy: `<p>${ep.evidence_contract ? "Nodes, typed edges, route membership, rationales, confidence, and cited evidence IDs are serialized once." : "The legacy model submits nested molecule/reaction trees with reaction explanations and precursor roles."}</p>`,
        gatewayTitle: `${terminalTool} → deterministic verifier`,
        gatewayCopy: `<p>The verifier independently rechecks parsing, graph integrity, chemistry, stock closure, diversity, and hidden-reference coverage${ep.evidence_contract ? ", including citation grounding" : ""}.</p>`,
        stateTitle: `${ep.score.valid ? "Strict pass" : "Diagnostic failure"} · ${ep.score.reward.toFixed(3)}`,
        stateCopy: `<p>The scalar reward stays informative, but a strict pass still requires every hard gate. The model cannot award itself points.</p>`
      }
    ];
  }

  function renderRuntime() {
    const ep = episode(), facts = runtimeFacts(ep), stages = runtimeStages(ep, facts), selected = stages[state.runtime] || stages[0];
    $("runtime-progress").textContent = `stage ${state.runtime + 1} / ${stages.length}`;
    $("runtime-stages").innerHTML = stages.map((stage, index) => `<button type="button" role="tab" aria-selected="${index === state.runtime}" class="runtime-stage ${index === state.runtime ? 'active' : ''}" data-runtime-stage="${index}"><span>${String(index + 1).padStart(2,'0')}</span><strong>${esc(stage.name)}</strong><small>${esc(stage.short)}</small></button>`).join("");
    $("runtime-source-title").textContent = selected.sourceTitle;
    $("runtime-source-copy").innerHTML = selected.sourceCopy;
    $("runtime-gateway-title").textContent = selected.gatewayTitle;
    $("runtime-gateway-copy").innerHTML = selected.gatewayCopy;
    $("runtime-state-title").textContent = selected.stateTitle;
    $("runtime-state-copy").innerHTML = selected.stateCopy;
    $("tool-belt").innerHTML = ep.tools.map(tool => {
      const count = tool === "emit_graph" || tool === "emit_routes" ? ep.terminal_calls : (facts.counts[tool] || 0);
      return `<button type="button" class="tool-chip ${selected.tools.includes(tool) ? 'active' : ''}" data-runtime-tool="${tool}"><span>${esc(tool)}</span><b>${count}</b></button>`;
    }).join("");
    document.querySelectorAll("[data-runtime-stage]").forEach(button => button.onclick = () => { state.runtime = Number(button.dataset.runtimeStage); renderRuntime(); });
    document.querySelectorAll("[data-runtime-tool]").forEach(button => button.onclick = () => {
      const tool = button.dataset.runtimeTool;
      const index = stages.findIndex(stage => stage.tools.includes(tool));
      if (index >= 0) state.runtime = index;
      renderRuntime();
      document.querySelectorAll(".event").forEach(event => event.classList.toggle("runtime-highlight", ep.evidence.some(item => item.evidence_id === event.dataset.evidence && item.tool === tool)));
    });
    const activeTools = new Set(selected.tools);
    document.querySelectorAll(".event").forEach(event => {
      const item = ep.evidence.find(row => row.evidence_id === event.dataset.evidence);
      event.classList.toggle("runtime-highlight", Boolean(item && activeTools.has(item.tool)));
    });
  }

  function routeMembership(graph) {
    const member={};
    graph.routes.forEach((route,ri)=>{
      route.reaction_node_ids.forEach(id=>member[id]=(member[id]||0)|(1<<ri));
      graph.edges.forEach(edge=>{ if(route.reaction_node_ids.includes(edge.source)||route.reaction_node_ids.includes(edge.target)){ member[edge.id]=(member[edge.id]||0)|(1<<ri); member[edge.source]=(member[edge.source]||0)|(1<<ri); member[edge.target]=(member[edge.target]||0)|(1<<ri); } });
    }); return member;
  }

  function visibleGraph() {
    const graph=episode().graph, membership=routeMembership(graph), mask=Number(state.route);
    const nodes=graph.nodes.filter(n=>(membership[n.id]||0)&mask), ids=new Set(nodes.map(n=>n.id));
    return {graph,nodes,edges:graph.edges.filter(e=>ids.has(e.source)&&ids.has(e.target)&&((membership[e.id]||0)&mask)),membership};
  }

  function layoutGraph(nodes,edges,targetId,membership) {
    const out={}, incoming={}; nodes.forEach(n=>{out[n.id]=[];incoming[n.id]=[];}); edges.forEach(e=>{out[e.source].push(e.target);incoming[e.target].push(e.source);});
    const memo={}; function distance(id,seen=new Set()){ if(id===targetId)return 0;if(memo[id]!=null)return memo[id];if(seen.has(id))return 0;const next=new Set(seen);next.add(id);memo[id]=out[id].length?1+Math.max(...out[id].map(x=>distance(x,next))):0;return memo[id]; }
    nodes.forEach(n=>distance(n.id)); const max=Math.max(0,...Object.values(memo)); const layers={}; nodes.forEach(n=>{const rank=max-(memo[n.id]||0);(layers[rank]??=[]).push(n);});
    const positions={}, xGap=255, top=80;
    Object.entries(layers).forEach(([rank,list])=>{
      list.sort((a,b)=>{const ma=membership[a.id]||0,mb=membership[b.id]||0; const order=m=>m===1?0:m===3?1:2;return order(ma)-order(mb)||a.id.localeCompare(b.id);});
      const gap=Math.max(175,700/Math.max(list.length,1)); const total=(list.length-1)*gap;
      list.forEach((n,i)=>positions[n.id]={x:70+Number(rank)*xGap,y:top+350-total/2+i*gap});
    });
    return {positions,width:140+max*xGap+220,height:860};
  }

  function renderGraph() {
    const ep=episode(), svg=$("graph");
    if(!ep.graph_available){
      state.viewBox={x:0,y:0,w:900,h:560};
      svg.innerHTML=`<text class="empty-graph-title" x="450" y="245">No route graph emitted</text><text class="empty-graph-copy" x="450" y="278">Inspect the saved tool trajectory to see where this rollout stopped.</text>`;
      setViewBox();
      $("route-filters").innerHTML="";
      return;
    }
    const {graph,nodes,edges,membership}=visibleGraph();
    const {positions,width,height}=layoutGraph(nodes,edges,graph.target_node_id,membership); state.viewBox={x:0,y:0,w:width,h:height};
    const defs=`<defs><marker id="arrow-reactant" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#3f8585"/></marker><marker id="arrow-product" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#223c4a"/></marker><marker id="arrow-other" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#b26921"/></marker></defs>`;
    const edgeHtml=edges.map(edge=>{
      const a=positions[edge.source],b=positions[edge.target], src=nodes.find(n=>n.id===edge.source),dst=nodes.find(n=>n.id===edge.target); if(!a||!b)return '';
      const ax=a.x+(src.type==='molecule'?186:27), ay=a.y+(src.type==='molecule'?71:27), bx=b.x, by=b.y+(dst.type==='molecule'?71:27), dx=Math.max(45,(bx-ax)*.52), d=`M${ax},${ay} C${ax+dx},${ay} ${bx-dx},${by} ${bx},${by}`;
      const kind=edge.type==='reactant'?'reactant':edge.type==='product'?'product':'other';
      return `<g class="graph-edge" data-kind="edge" data-id="${esc(edge.id)}"><path class="edge-path ${kind}" d="${d}" marker-end="url(#arrow-${kind})"/><path class="edge-hit" d="${d}"/><text class="edge-label" x="${(ax+bx)/2}" y="${(ay+by)/2-7}">${esc(edge.metadata.role)}</text></g>`;
    }).join('');
    const nodeHtml=nodes.map(node=>{const p=positions[node.id];if(node.type==='reaction'){const name=short(node.metadata.reaction_class,10);return `<g class="rxn-node ${state.selected?.id===node.id?'selected':''}" data-kind="node" data-id="${esc(node.id)}" transform="translate(${p.x},${p.y})"><circle cx="27" cy="27" r="25"/><text x="27" y="27">${esc(name)}</text></g>`;}
      const img=ep.drawings[node.id]?`<img src="${ep.drawings[node.id]}" alt="2D structure">`:''; const stock=node.metadata.stock_status==='confirmed';
      return `<foreignObject x="${p.x}" y="${p.y}" width="186" height="142"><div xmlns="http://www.w3.org/1999/xhtml" class="mol-card ${state.selected?.id===node.id?'selected':''}" data-kind="node" data-id="${esc(node.id)}"><div class="mol-top"><span>${esc(node.metadata.role)}</span><span class="stock ${stock?'':'no'}">${stock?'● stock':'○'}</span></div>${img}<div class="smiles">${esc(node.smiles)}</div></div></foreignObject>`;
    }).join('');
    svg.innerHTML=defs+edgeHtml+nodeHtml; setViewBox();
    svg.querySelectorAll('[data-kind]').forEach(el=>el.addEventListener('click',event=>{event.stopPropagation();selectGraphItem(el.dataset.kind,el.dataset.id);}));
    renderRouteFilters();
  }

  function renderRouteFilters(){const graph=episode().graph;$("route-filters").innerHTML=graph.routes.map((r,i)=>[String(1<<i),`Route ${i+1}`]).map(([v,l])=>`<button class="${state.route===v?'active':''}" data-route="${v}">${l}</button>`).join('');document.querySelectorAll('[data-route]').forEach(b=>b.onclick=()=>{state.route=b.dataset.route;state.selected=null;renderGraph();renderInspector();});}
  function selectGraphItem(kind,id){const graph=episode().graph;const item=kind==='node'?graph.nodes.find(x=>x.id===id):graph.edges.find(x=>x.id===id);state.selected={kind,id,item};renderGraph();renderInspector();}
  function renderInspector(){const box=$("inspector"),sel=state.selected;if(!sel){box.innerHTML='<div class="inspector-empty"><p class="eyebrow">INSPECT</p><p>Select a molecule, reaction, or edge.</p></div>';return;}const item=sel.item,meta=item.metadata||{};let title=sel.kind==='edge'?`${item.source} → ${item.target}`:item.type==='molecule'?short(item.smiles,42):item.id;let body=`<span class="kind">${esc(sel.kind==='edge'?item.type:item.type)}</span><h4>${esc(title)}</h4><dl>`;if(item.smiles)body+=`<dt>SMILES</dt><dd class="mono">${esc(item.smiles)}</dd>`;for(const [key,value] of Object.entries(meta)){if(key==='evidence_ids')continue;body+=`<dt>${esc(key.replaceAll('_',' '))}</dt><dd>${esc(Array.isArray(value)?value.join(' · '):value)}</dd>`;}body+=`<dt>Evidence</dt><dd>${(meta.evidence_ids||[]).map(x=>`<button class="evidence-chip" data-cite="${esc(x)}">${esc(x)}</button>`).join('')||'None cited'}</dd></dl>`;box.innerHTML=body;box.querySelectorAll('[data-cite]').forEach(b=>b.onclick=()=>highlightEvidence(b.dataset.cite));}
  function highlightEvidence(id){document.querySelectorAll('.event').forEach(el=>el.classList.toggle('highlight',el.dataset.evidence===id));const hit=document.querySelector(`.event[data-evidence="${CSS.escape(id)}"]`);if(hit)hit.scrollIntoView({behavior:'smooth',block:'center'});}
  function setViewBox(){const v=state.viewBox;if(v)$("graph").setAttribute('viewBox',`${v.x} ${v.y} ${v.w} ${v.h}`);}
  function zoom(factor){const v=state.viewBox,cx=v.x+v.w/2,cy=v.y+v.h/2;v.w*=factor;v.h*=factor;v.x=cx-v.w/2;v.y=cy-v.h/2;setViewBox();}
  function setupGraphControls(){ $("zoom-in").onclick=()=>zoom(.82);$("zoom-out").onclick=()=>zoom(1.22);$("fit").onclick=()=>renderGraph();const wrap=$("graph-wrap");let start=null;wrap.onpointerdown=e=>{if(e.target.closest('[data-kind]'))return;state.dragging=true;start={x:e.clientX,y:e.clientY,v:{...state.viewBox}};wrap.classList.add('dragging');wrap.setPointerCapture(e.pointerId);};wrap.onpointermove=e=>{if(!state.dragging)return;const rect=wrap.getBoundingClientRect();state.viewBox.x=start.v.x-(e.clientX-start.x)*start.v.w/rect.width;state.viewBox.y=start.v.y-(e.clientY-start.y)*start.v.h/rect.height;setViewBox();};wrap.onpointerup=()=>{state.dragging=false;wrap.classList.remove('dragging');};wrap.onwheel=e=>{e.preventDefault();zoom(e.deltaY>0?1.12:.89);};}

  function render(){renderTabs();renderModelTabs();renderHeader();renderMetrics();renderTrajectory();renderRuntime();renderGraph();renderInspector();}
  $("prompt-toggle").onclick=()=>{const p=$("prompt"),hidden=p.classList.toggle('hidden');$("prompt-toggle").textContent=hidden?'Show prompt':'Hide prompt';};
  setupGraphControls(); render();
})();

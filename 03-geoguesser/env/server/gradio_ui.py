# SPDX-License-Identifier: BSD-3-Clause

"""The human-play page: a GeoGuessr-style game in the browser.

Two viewers, both driven by the environment's own data:

- Pannellum shows the equirectangular panorama, so a person drags to look
  around exactly where the agent calls `look()`.
- MapLibre shows the guess map over OpenFreeMap tiles. No API key, no request
  limits, commercial use permitted, and self-hostable if the public instance
  ever goes away.

The page is served as its own document at `/geoguesser/play` and embedded in
the Gradio tab through an iframe, because `gr.HTML` inserts markup without
executing `<script>` tags — styles apply, but neither viewer initialises, which
looks like a blank panel and reports no error anywhere.

A human sees live tiles; the agent's map stays the offline Natural Earth
render. They agree on geometry, which is what matters, and the agent keeps a
determinism the browser does not need.
"""

from __future__ import annotations

import json
import random
import urllib.parse
from typing import Any, Dict, List, Optional

import gradio as gr


MAPLIBRE_JS = "https://cdnjs.cloudflare.com/ajax/libs/maplibre-gl/5.24.0/maplibre-gl.js"
MAPLIBRE_CSS = (
    "https://cdnjs.cloudflare.com/ajax/libs/maplibre-gl/5.24.0/maplibre-gl.css"
)
PANNELLUM_JS = "https://cdnjs.cloudflare.com/ajax/libs/pannellum/2.5.6/pannellum.js"
PANNELLUM_CSS = "https://cdnjs.cloudflare.com/ajax/libs/pannellum/2.5.6/pannellum.css"
OPENFREEMAP_STYLE = "https://tiles.openfreemap.org/styles/positron"

# The real game scores a round out of 5000 on this curve. Showing points
# rather than the RL reward makes a score comparable to GeoGuessr intuition;
# the reward is shown beside it so the two are never confused. There is no
# multi-round game here, because an episode is exactly one guess.
MAX_POINTS_PER_ROUND = 5000

_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>geoguesser_env - play</title>
<link rel="stylesheet" href="__MAPLIBRE_CSS__">
<link rel="stylesheet" href="__PANNELLUM_CSS__">
<style>
  :root {
    --panel: rgba(18, 23, 28, .92);
    --edge: #2c353d;
    --ink: #e8ecef;
    --ink-soft: #9aa5ad;
    --ink-faint: #6e7a83;
    --pin: #c4332a;
    --good: #5aa06e;
    --wire: #7fb3cc;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; height: 100%; overflow: hidden; background: #10151a;
    color: var(--ink);
    font-family: ui-monospace, "SF Mono", "IBM Plex Mono", Menlo, monospace;
  }
  #stage { position: absolute; inset: 0; }
  #pano { position: absolute; inset: 0; }
  .pnlm-zoom-controls, .pnlm-orientation-button, .pnlm-panorama-info,
  .pnlm-compass { display: none !important; }
  .pnlm-load-box { background: #10151a !important; }

  .hud {
    position: absolute; z-index: 5; background: var(--panel);
    border: 1px solid var(--edge); border-radius: 4px;
    font-size: 12px; padding: 7px 11px; backdrop-filter: blur(8px);
    line-height: 1.5;
  }
  .hud b { color: #fff; font-weight: 500; }
  .hud span { color: var(--ink-soft); }
  #top { top: 12px; left: 12px; }
  #top .ep { color: var(--wire); }
  #compass { top: 12px; left: 50%; transform: translateX(-50%); letter-spacing: .1em; }
  #score { top: 12px; right: 12px; text-align: right; }
  #score .pts { font-size: 15px; color: #fff; }
  #credit {
    top: 74px; right: 12px; font-size: 10.5px; color: var(--ink-soft);
    max-width: 34vw; text-align: right; z-index: 14;
  }
  #credit a { color: #8fb8cc; text-decoration: none; }
  #actions {
    bottom: 12px; left: 50%; transform: translateX(-50%); display: flex;
    gap: 6px; align-items: center; transition: opacity .3s ease;
  }
  #actions {
    gap: 0; padding: 0; display: flex; align-items: stretch; overflow: hidden;
    bottom: 12px; left: 12px; transform: none;
  }
  /* One group per kind of environment action, each labelled, each showing what
     it costs through its tooltip rather than shouting a number. A group whose
     tool is not registered is removed rather than greyed: a control you cannot
     use is noise. */
  .pad {
    display: flex; flex-direction: column; gap: 4px; padding: 8px 14px;
    justify-content: center;
  }
  .pad + .pad { border-left: 1px solid var(--edge); }
  .pad.gone { display: none; }
  .padlabel {
    font-size: 9.5px; letter-spacing: .14em; text-transform: uppercase;
    color: var(--ink-faint);
  }
  .padlabel b { color: var(--ink); font-weight: 500; letter-spacing: 0; }
  .btns { display: flex; align-items: center; gap: 5px; }
  #actions button {
    font-family: inherit; cursor: pointer; color: var(--ink);
    background: #232c33; border: 1px solid var(--edge); border-radius: 4px;
    display: flex; flex-direction: column; align-items: center; gap: 1px;
    min-width: 46px; padding: 5px 7px; line-height: 1;
  }
  #actions button .glyph { font-size: 12px; }
  #actions button .tag {
    font-size: 8.5px; letter-spacing: .06em; color: var(--ink-soft);
  }
  #actions button:hover {
    border-color: #6d8493; background: #2b353d; color: #fff;
  }
  #actions button:hover .tag { color: var(--ink); }
  #actions button:active { background: #1c242a; }
  #actions button:focus-visible { outline: 2px solid var(--wire); outline-offset: 1px; }
  #actions button.gone { display: none; }
  #actions.working { opacity: .55; }
  #actions.working button { cursor: progress; }
  #actions.working #padBudget b { color: var(--wire); }
  #actions button.preset {
    font-size: 10px; letter-spacing: .04em; min-width: 44px; padding: 7px 8px;
  }
  #actions button.preset.on { border-color: var(--wire); color: #fff; }
  #fovRange { width: 104px; accent-color: var(--wire); margin-left: 4px; }
  .budget { gap: 4px; font-size: 10px; color: var(--ink-faint); white-space: nowrap; }
  .budget .sep { color: var(--edge); margin: 0 3px; }
  .budget b {
    font-size: 14px; color: var(--ink); font-weight: 500;
    font-variant-numeric: tabular-nums;
  }

  /* ---- rollout trace: the observation stream the agent would receive ---- */
  /* The trace lives on the left and the guess map on the right, so an
     expanded map can never cover the observation stream. */
  #trace {
    position: absolute; top: 58px; left: 12px; width: 310px; z-index: 7;
    max-height: calc(100% - 130px); display: flex; flex-direction: column;
    background: var(--panel); border: 1px solid var(--edge); border-radius: 4px;
    backdrop-filter: blur(8px); overflow: hidden;
  }
  #trace.hidden { display: none; }
  #trace h4 {
    margin: 0; padding: 8px 11px; font-size: 10.5px; font-weight: 500;
    letter-spacing: .12em; text-transform: uppercase; color: var(--wire);
    border-bottom: 1px solid var(--edge); display: flex;
    justify-content: space-between;
  }
  #trace h4 em { color: var(--ink-faint); font-style: normal; letter-spacing: 0; }
  #trace h4 > span:last-child { display: flex; align-items: center; gap: 8px; }
  #collapse {
    font-family: inherit; font-size: 13px; line-height: 1; cursor: pointer;
    background: #232c33; border: 1px solid var(--edge); border-radius: 3px;
    color: var(--ink); width: 22px; height: 19px; padding: 0;
  }
  #expand {
    font-family: inherit; font-size: 11px; line-height: 1; cursor: pointer;
    background: #232c33; border: 1px solid var(--edge); border-radius: 3px;
    color: var(--ink); padding: 5px 9px;
  }
  #collapse:hover, #expand:hover { border-color: #5b6b78; color: #fff; }
  /* With the trace hidden, a small control stays where it was. */
  #expandWrap {
    position: absolute; top: 58px; left: 12px; z-index: 7; display: none;
    padding: 4px 5px;
  }
  #expandWrap.show { display: block; }
  #steps { overflow-y: auto; padding: 4px 0; font-size: 11px; }
  .step { padding: 6px 11px; border-bottom: 1px solid #1e262c; line-height: 1.5; }
  .step:last-child { border-bottom: none; }
  .step .op { color: var(--wire); }
  .step .rw { float: right; color: var(--ink-faint); }
  .step .fb { color: var(--ink-soft); display: block; margin-top: 2px; }
  #agentview { border-top: 1px solid var(--edge); padding: 8px 11px; }
  #agentview .cap {
    font-size: 9.5px; letter-spacing: .1em; text-transform: uppercase;
    color: var(--ink-faint); margin-bottom: 5px;
  }
  #agentview img { width: 100%; display: block; border-radius: 3px; }
  /* The agent's view is where the pin's rendered map arrives, so that is where
     the wait belongs. */
  #agentview.loading img { opacity: .35; filter: saturate(.4); }
  #agentview.loading .cap { color: var(--pin); }
  #agentview .bar {
    display: none; height: 2px; margin-top: 6px; border-radius: 2px;
    background: linear-gradient(90deg, transparent, var(--pin), transparent);
    background-size: 40% 100%; background-repeat: no-repeat;
    animation: sweep 1.1s linear infinite;
  }
  #agentview.loading .bar { display: block; }
  @keyframes sweep {
    0% { background-position: -40% 0; } 100% { background-position: 140% 0; }
  }

  /* ---- guess map ------------------------------------------------------- */
  #mapwrap {
    position: absolute; right: 12px; bottom: 12px; z-index: 6;
    width: 300px; height: 210px; border: 1px solid var(--edge);
    border-radius: 5px; overflow: hidden; opacity: .9; background: #191f24;
    transition: width .24s ease, height .24s ease, opacity .24s ease,
                right .24s ease, bottom .24s ease;
  }
  #mapwrap:hover, #mapwrap.big {
    width: min(560px, 48vw); height: min(400px, 58vh); opacity: 1;
  }
  #mapwrap.reveal {
    right: 50%; bottom: 50%; transform: translate(50%, 50%);
    width: min(1000px, 86vw); height: min(600px, 72vh); opacity: 1;
  }
  /* The reveal is the whole point of the round, so it may cover the trace. */
  #mapwrap.reveal { z-index: 13; }
  #map { position: absolute; inset: 0; }
  #submit {
    position: absolute; left: 0; right: 0; bottom: 0; z-index: 3; width: 100%;
    font-family: inherit; font-size: 12px; letter-spacing: .07em;
    text-transform: uppercase; padding: 10px; border: none;
    border-top: 1px solid var(--edge); background: #2b3540; color: #7e8b96;
    cursor: not-allowed;
  }
  #submit.ready { background: var(--pin); color: #fff; cursor: pointer; }
  #submit.ready:hover { background: #d64236; }
  /* A pin is a charged step whose reply carries a freshly rendered map, and at
     a tight zoom that render takes a moment. The button used to switch to
     "submit guess" the instant you clicked the map, so it looked ready while
     the step was still in flight -- and `step()` drops calls while busy, so the
     click did nothing at all, silently. It now says what it is waiting for. */
  #submit.waiting {
    background: #2b3540; color: var(--ink-faint); cursor: progress;
  }
  #submit.waiting::after {
    content: ""; display: inline-block; width: 6px; height: 6px;
    margin-left: 7px; border-radius: 50%; background: currentColor;
    animation: pulse 1s ease-in-out infinite; vertical-align: middle;
  }
  @keyframes pulse { 0%, 100% { opacity: .25; } 50% { opacity: 1; } }
  #mapwrap.reveal #submit { display: none; }
  /* The map drops clicks while a step is in flight, so it should not look
     clickable. Without this a second pin feels like it was ignored. */
  #mapwrap.busy { cursor: progress; }
  #mapwrap.busy #map { pointer-events: none; opacity: .75; }

  /* ---- result bar ------------------------------------------------------ */
  #result {
    position: absolute; left: 0; right: 0; bottom: 0; z-index: 12; display: none;
    background: rgba(14, 18, 22, .96); border-top: 1px solid var(--edge);
    padding: 13px 20px; backdrop-filter: blur(8px);
  }
  #result.show { display: block; }
  #result .inner {
    max-width: 1180px; margin: 0 auto; display: flex; align-items: center;
    gap: 22px; flex-wrap: wrap;
  }
  #verdict { font-size: 17px; color: #fff; min-width: 140px; }
  .stat { font-size: 10.5px; color: var(--ink-soft); }
  .stat b {
    display: block; font-size: 14px; color: var(--ink); font-weight: 500;
    font-variant-numeric: tabular-nums; margin-top: 2px;
  }
  .stat.env b { color: var(--wire); }
  #bar { flex: 1 1 140px; min-width: 100px; height: 6px; background: #232b32;
         border-radius: 4px; overflow: hidden; }
  #bar i { display: block; height: 100%; background: var(--good); width: 0; }
  #advance {
    font-family: inherit; font-size: 12px; text-transform: uppercase;
    letter-spacing: .06em; padding: 10px 18px; background: #232b32;
    color: var(--ink); border: 1px solid var(--edge); border-radius: 3px;
    cursor: pointer;
  }
  #advance:hover { border-color: #465360; color: #fff; }

  .dot {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    margin-right: 7px; background: var(--ink-faint);
    vertical-align: 1px;
  }
  .dot.live { background: var(--good); }
  .dot.bad { background: var(--pin); }
</style>
</head>
<body>
<div id="stage">
  <div id="pano"></div>

  <div class="hud" id="top">
    <i id="conn" class="dot" title="environment session"></i><b>geoguesser_env</b>
    <span>episode</span> <b class="ep" id="task">-</b>
    <span>· frame</span> <b id="frame">-</b>
    <span>· steps</span> <b id="stepsLeft">-</b>
  </div>
  <div class="hud" id="compass"><span>facing</span> <b id="heading">-</b>
    <span>· fov</span> <b id="fov">-</b>
    <span>· map</span> <b id="mapspan">-</b></div>
  <div class="hud" id="score">
    <div class="pts"><b id="total">0</b><span> pts</span></div>
    <span id="played">0 episodes</span> · <span id="captured"></span>
  </div>
  <div class="hud" id="credit"></div>
  <div class="hud" id="actions">
    <div class="pad" id="padTurn">
      <span class="padlabel">look</span>
      <div class="btns">
        <button id="turn-left" title="Turn 45° left and look (costs 0.01) — key: A or ←">
          <span class="glyph">&#9664;</span><span class="tag">left</span></button>
        <button id="look-now" title="Look again where you are facing (costs 0.01) — key: L">
          <span class="glyph">&#9678;</span><span class="tag">look</span></button>
        <button id="turn-right" title="Turn 45° right and look (costs 0.01) — key: D or →">
          <span class="glyph">&#9654;</span><span class="tag">right</span></button>
      </div>
    </div>

    <div class="pad" id="padMove">
      <span class="padlabel">walk</span>
      <div class="btns">
        <button id="move-fwd" title="Walk 15 m forward (costs 0.05) — key: W or ↑">
          <span class="glyph">&#9650;</span><span class="tag">forward</span></button>
        <button id="move-back" title="Walk 15 m back (costs 0.05) — key: S or ↓">
          <span class="glyph">&#9660;</span><span class="tag">back</span></button>
      </div>
    </div>

    <div class="pad" id="padZoom">
      <span class="padlabel">zoom &middot; <b id="fovValue">90&deg;</b></span>
      <div class="btns">
        <button class="preset" data-fov="90" title="Wide view, 90° (costs 0.01)">wide</button>
        <button class="preset" data-fov="50" title="Street level, 50° (costs 0.01)">street</button>
        <button class="preset" data-fov="30" title="Read a sign, 30° (costs 0.01)">sign</button>
        <input id="fovRange" type="range" min="20" max="110" step="5" value="90"
          aria-label="field of view in degrees">
      </div>
    </div>

    <div class="pad" id="padBudget">
      <span class="padlabel">budget</span>
      <div class="btns budget">
        <b id="stepsBudget">12</b><span>left</span>
        <span class="sep">&middot;</span>
        <b id="costBudget">0.00</b><span>spent</span>
      </div>
    </div>
  </div>

  <div class="hud" id="expandWrap">
    <button id="expand" title="show the rollout trace (T)">+ trace</button>
  </div>

  <div id="trace">
    <h4>
      <span>rollout trace</span>
      <span><em id="cost">cost 0.00</em>
        <button id="collapse" title="hide the trace (T)">&#8211;</button></span>
    </h4>
    <div id="steps"></div>
    <div id="agentview" style="display:none">
      <div class="cap" id="agentcap">what the agent sees</div>
      <img id="agentimg" alt="the environment's own rendered observation">
      <div class="bar"></div>
    </div>
  </div>

  <div id="mapwrap">
    <div id="map"></div>
    <button id="submit">click the map to place a pin</button>
  </div>

  <div id="result">
    <div class="inner">
      <div id="verdict">-</div>
      <div class="stat">distance<b id="dist">-</b></div>
      <div class="stat">points<b id="points">-</b></div>
      <div class="stat env">env reward<b id="reward">-</b></div>
      <div class="stat">score - cost<b id="breakdown2">-</b></div>
      <div class="stat">true location<b id="truth">-</b></div>
      <div id="bar"><i></i></div>
      <button id="advance">load another episode</button>
    </div>
  </div>

</div>

<script src="__MAPLIBRE_JS__"></script>
<script src="__PANNELLUM_JS__"></script>
<script>
(function () {
  "use strict";
  const N_TASKS = __N_TASKS__;
  const SPLIT = __SPLIT__;
  const MAX_POINTS = __MAX_POINTS__;
  // Every task-scoped request has to name its split, or index 12 of eval and
  // index 12 of train are indistinguishable and the page reveals the wrong
  // ground truth.
  const q = (extra) => "?split=" + encodeURIComponent(SPLIT) + (extra || "");
  const $ = (id) => document.getElementById(id);

  let socket = null, ready = false, pending = null;
  let viewer = null, map = null, guessMarker = null, truthMarker = null;
  let lineAdded = false, guess = null, taskIndex = 0, compass = 0;
  let taskMeta = null, frameIndex = 0;
  let total = 0, played = 0, busy = false, cost = 0;

  const DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
  const fmt = (la, lo) => la.toFixed(4) + ", " + lo.toFixed(4);
  const yaw = () => (viewer ? ((viewer.getYaw() % 360) + 360) % 360 : 0);
  const hfov = () => (viewer ? viewer.getHfov() : 90);

  // ---- the environment, over the same WebSocket session API a client uses --
  // Plain REST /step builds a fresh environment per request, so a stateful
  // episode has to run over /ws. This page therefore plays exactly the
  // rollout an agent would: one reset, a few charged steps, one terminal guess.
  function connect() {
    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(scheme + "//" + location.host + "/ws");
    socket.onopen = function () {
      ready = true;
      $("conn").className = "dot live";
      $("conn").title = "environment session: connected";
      startRound(requestedTask());
    };
    socket.onclose = function () {
      ready = false;
      $("conn").className = "dot bad";
      $("conn").title = "environment session: disconnected";
    };
    socket.onerror = function () {
      $("conn").className = "dot bad";
      $("conn").title = "environment session: error";
    };
    socket.onmessage = function (event) {
      const message = JSON.parse(event.data);
      if (message.type === "error") {
        addStep("error", message.data ? JSON.stringify(message.data) : "", null);
        setBusy(false);
        return;
      }
      if (message.type !== "observation") return;
      // The wire format nests the observation and carries reward and done as
      // siblings: {observation: {...}, reward, done, metadata}. Flatten it so
      // callers read one object.
      const payload = message.data || {};
      const observation = Object.assign(
        {}, payload.observation || payload,
        { reward: payload.reward, done: payload.done }
      );
      const handler = pending;
      pending = null;
      if (handler) handler(observation);
    };
  }

  function send(type, data, handler) {
    if (!ready) return;
    pending = handler || null;
    socket.send(JSON.stringify({ type: type, data: data || {} }));
  }

  // ---- trace ------------------------------------------------------------
  function addStep(op, feedback, observation) {
    const row = document.createElement("div");
    row.className = "step";
    let right = "";
    if (observation && observation.steps_remaining !== undefined) {
      right = "<span class='rw'>" + observation.steps_remaining + " left</span>";
    }
    row.innerHTML = "<span class='op'>" + op + "</span>" + right +
                    "<span class='fb'>" + (feedback || "") + "</span>";
    $("steps").appendChild(row);
    $("steps").scrollTop = $("steps").scrollHeight;
    if (observation && observation.image_base64) {
      const mime = observation.image_kind === "map" ? "png" : "jpeg";
      $("agentimg").src = "data:image/" + mime + ";base64," + observation.image_base64;
      $("agentview").style.display = "block";
    }
  }

  function applyObservation(observation) {
    if (!observation) return;
    if (observation.steps_remaining !== undefined) {
      $("stepsLeft").textContent = observation.steps_remaining;
      $("stepsBudget").textContent = observation.steps_remaining;
    }
    if (observation.action_cost !== null && observation.action_cost !== undefined) {
      cost = observation.action_cost;
      $("cost").textContent = "cost " + cost.toFixed(2);
      $("costBudget").textContent = cost.toFixed(2);
    }
    setControls(observation);
  }

  // ---- rounds -----------------------------------------------------------
  function loadPano(index) {
    fetch("/geoguesser/task/" + index + q())
      .then((response) => response.json())
      .then((meta) => {
        taskMeta = meta;
        frameIndex = meta.start_frame || 0;
        const who = (meta.attribution || {}).creator_username;
        $("credit").innerHTML =
          "imagery &copy; " + (who ? who : "Mapillary contributor") +
          " via <a href='https://www.mapillary.com' target='_blank' rel='noopener'>Mapillary</a>" +
          ", <a href='https://creativecommons.org/licenses/by-sa/4.0/' target='_blank' rel='noopener'>CC BY-SA 4.0</a>";
        showFrame(frameIndex, 0, 90);
      });
  }

  /**
   * Point the main viewer at one frame of the sequence.
   *
   * Called on reset and again after every move(), so walking forward actually
   * changes what you are looking at rather than only what the trace shows.
   * Heading and zoom carry over, because losing your orientation on every step
   * would make navigation useless.
   */
  function showFrame(index, keepYaw, keepHfov) {
    frameIndex = index;
    const frames = (taskMeta && taskMeta.frames) || [];
    const frame = frames[index] || {};
    compass = frame.compass_angle || 0;
    if (frame.captured_at) {
      $("captured").textContent = "captured " + frame.captured_at;
    }
    $("frame").textContent = index + "/" + Math.max(0, frames.length - 1);
    if (viewer) { viewer.destroy(); viewer = null; }
    viewer = pannellum.viewer("pano", {
      type: "equirectangular",
      panorama: "/geoguesser/pano/" + taskIndex + "/" + index + q(),
      autoLoad: true, showControls: false, northOffset: compass,
      yaw: keepYaw, hfov: keepHfov,
      minHfov: 20, maxHfov: 110, compass: false, friction: 0.15,
    });
    viewer.on("mouseup", updateHud);
    viewer.on("touchend", updateHud);
    viewer.on("zoomchange", updateHud);
    viewer.on("load", updateHud);
    setTimeout(updateHud, 400);
  }

  function setControls(observation) {
    if (!observation) return;
    const tools = observation.available_tools || [];
    const canLook = tools.indexOf("look") !== -1;
    const canMove = tools.indexOf("move") !== -1;
    $("padTurn").classList.toggle("gone", !canLook);
    $("padZoom").classList.toggle("gone", !canLook);
    $("look-now").classList.toggle("gone", !canLook);
    $("padMove").classList.toggle(
      "gone",
      !canMove ||
        (!observation.can_move_forward && !observation.can_move_backward)
    );
    $("move-fwd").classList.toggle("gone", !observation.can_move_forward);
    $("move-back").classList.toggle("gone", !observation.can_move_backward);
    if (observation.fov_deg) {
      const fov = Math.round(observation.fov_deg);
      $("fovRange").value = String(fov);
      $("fovValue").textContent = fov + "\u00b0";
      markPreset(fov);
    }
  }

  function updateHud() {
    if (!viewer) return;
    const y = yaw();
    $("heading").textContent = DIRS[Math.round(y / 45) % 8] + " " + y.toFixed(0) + "°";
    $("fov").textContent = hfov().toFixed(0) + "°";
  }

  // ---- charged actions, executed by the environment ---------------------
  function setBusy(value, waitingFor) {
    busy = value;
    $("actions").classList.toggle("working", value);
    // The wait is shown where the result will appear: the agent's view is what
    // the pin re-renders, and at a tight zoom that render is slow enough to
    // read as the UI having ignored you.
    $("agentview").classList.toggle("loading", value);
    $("mapwrap").classList.toggle("busy", value);
    if (value) {
      $("stepsBudget").textContent = "\u2026";
      $("agentcap").textContent = waitingFor || "rendering\u2026";
    } else {
      $("agentcap").textContent = "what the agent sees";
    }
  }

  function step(op, data, label, waitingFor) {
    if (busy || !ready) return;
    setBusy(true, waitingFor);
    send("step", Object.assign({ op: op }, data), function (observation) {
      setBusy(false);
      applyObservation(observation);
      addStep(label, observation.feedback, observation);
      const meta = observation.metadata || {};
      if (op === "move" && meta.frame_index !== undefined &&
          meta.frame_index !== frameIndex) {
        // Keep the player facing the same way through the step.
        showFrame(meta.frame_index, yaw(), hfov());
      }
      if (op === "look" && data && data.fov_deg && viewer) {
        // The main view and the agent's view should never disagree.
        viewer.setHfov(data.fov_deg);
        if (data.heading_deg !== undefined) viewer.setYaw(data.heading_deg);
      }
      // Only now is the guess actually submittable: until the pin's step has
      // come back the environment has not registered it, and a click would be
      // dropped by the `busy` guard above without any visible effect.
      if (observation.done) {
        reveal(observation);
      } else if (op === "pin" && guess) {
        $("submit").className = "ready";
        $("submit").textContent = "submit guess";
      }
    });
  }

  // The pad is the agent's action set, not a viewer control: every button is a
  // charged environment step, and the panorama follows the result. Dragging the
  // scene stays free, for orientation only.
  function lookAt(heading, fov) {
    const wrapped = ((Math.round(heading) % 360) + 360) % 360;
    step(
      "look",
      { heading_deg: wrapped, pitch_deg: 0, fov_deg: Math.round(fov) },
      "look(heading=" + wrapped + ", fov=" + Math.round(fov) + ")"
    );
  }

  $("turn-left").onclick = function () { lookAt(yaw() - 45, hfov()); };
  $("turn-right").onclick = function () { lookAt(yaw() + 45, hfov()); };
  $("look-now").onclick = function () { lookAt(yaw(), hfov()); };

  Array.prototype.forEach.call(
    document.querySelectorAll("#padZoom .preset"),
    function (button) {
      button.onclick = function () {
        const fov = parseInt(button.dataset.fov, 10);
        $("fovRange").value = String(fov);
        $("fovValue").textContent = fov + "\u00b0";
        markPreset(fov);
        lookAt(yaw(), fov);
      };
    }
  );

  function markPreset(fov) {
    Array.prototype.forEach.call(
      document.querySelectorAll("#padZoom .preset"),
      function (button) {
        button.classList.toggle("on", parseInt(button.dataset.fov, 10) === fov);
      }
    );
  }
  $("move-fwd").onclick = function () {
    step("move", { direction: "forward", meters: 15 }, "move(forward, 15m)");
  };
  $("move-back").onclick = function () {
    step("move", { direction: "backward", meters: 15 }, "move(backward, 15m)");
  };

  // The slider reads out live but only spends a step on release, so dragging it
  // does not burn the budget.
  $("fovRange").addEventListener("input", function () {
    const fov = parseInt($("fovRange").value, 10);
    $("fovValue").textContent = fov + "\u00b0";
    markPreset(fov);
  });
  $("fovRange").addEventListener("change", function () {
    lookAt(yaw(), parseInt($("fovRange").value, 10));
  });
  function setTrace(visible) {
    $("trace").classList.toggle("hidden", !visible);
    $("expandWrap").classList.toggle("show", !visible);
  }
  $("collapse").onclick = function () { setTrace(false); };
  $("expand").onclick = function () { setTrace(true); };

  // The picker is the human equivalent of reset(task_index=k): the same call an
  // eval harness makes, so a person can replay exactly the episode an agent saw.
  /** Task index requested in the page URL, when the host supplied one. */
  function requestedTask() {
    const value = new URLSearchParams(location.search).get("task");
    if (value === null || value === "" || value === "random") return undefined;
    const parsed = parseInt(value, 10);
    return Number.isFinite(parsed) ? parsed : undefined;
  }

  function startRound(index) {
    cost = 0;
    $("cost").textContent = "cost 0.00";
    $("steps").innerHTML = "";
    $("agentview").style.display = "none";
    clearRound();
    const wanted = index === undefined
      ? Math.floor(Math.random() * N_TASKS)
      : ((index % N_TASKS) + N_TASKS) % N_TASKS;
    taskIndex = wanted;
    send("reset", { split: SPLIT, index: wanted }, function (observation) {
      const meta = observation.metadata || {};
      taskIndex = meta.task_index !== undefined ? meta.task_index : wanted;
      $("task").textContent = taskIndex;
      $("captured").textContent = "captured " + (observation.captured_at || "unknown");
      applyObservation(observation);
      addStep(
        "reset(split='" + SPLIT + "', index=" + taskIndex + ")",
        "episode started · " + (observation.available_tools || []).length +
        " tools registered",
        observation
      );
      loadPano(taskIndex);
    });
  }

  // ---- map --------------------------------------------------------------
  const WORLD = [[-179, -58], [179, 76]];
  map = new maplibregl.Map({
    container: "map", style: "__OPENFREEMAP_STYLE__",
    center: [0, 12], zoom: 0, minZoom: -2,
    attributionControl: { compact: true }, dragRotate: false,
    // Without this the world repeats horizontally, which reads as a rendering
    // bug at the zoom levels a small guess map uses.
    renderWorldCopies: false,
  });
  map.on("load", () => map.fitBounds(WORLD, { padding: 6, duration: 0 }));
  // Exposed so the page can be driven from a test harness or the console.
  window.__ggMap = map;
  window.__ggState = function () {
    return {
      busy: busy,
      ready: ready,
      pendingHandler: !!pending,
      socket: socket ? socket.readyState : null,
      steps: document.querySelectorAll('#steps .step').length,
    };
  };

  /**
   * Half-width of the visible map, in degrees.
   *
   * The environment renders its own map from this, so a pin dropped while
   * zoomed into a city comes back as a street-level map rather than a
   * continental one. Without it the agent's view and the player's would
   * disagree about how precisely the pin could be aimed.
   */
  function currentSpanDeg() {
    const bounds = map.getBounds();
    const span = Math.abs(bounds.getEast() - bounds.getWest()) / 2;
    return Math.min(180, Math.max(0.03, span));
  }

  function updateMapSpan() {
    const span = currentSpanDeg();
    $("mapspan").textContent =
      span >= 1 ? span.toFixed(0) + "\u00b0" : (span * 111).toFixed(0) + " km";
  }
  map.on("zoomend", updateMapSpan);
  map.on("moveend", updateMapSpan);
  map.on("load", updateMapSpan);

  map.on("click", function (event) {
    if (busy || $("result").classList.contains("show")) return;
    // Once you have committed to a pin the map stays open; letting it collapse
    // on mouse-out makes it easy to lose the guess you were adjusting.
    $("mapwrap").classList.add("big");
    map.resize();
    guess = event.lngLat;
    if (guessMarker) guessMarker.remove();
    guessMarker = new maplibregl.Marker({ color: "#c4332a" })
      .setLngLat(guess).addTo(map);
    // Deliberately not "ready" yet -- see the note on `#submit.waiting`.
    $("submit").className = "waiting";
    $("submit").textContent = "placing pin";
    // A pin is a real, charged environment step, so the map the agent would see
    // comes back in the trace panel — framed at the zoom you are looking at, so
    // the two views agree about how precisely the pin was aimed.
    const span = currentSpanDeg();
    step("pin", { lat: guess.lat, lon: guess.lng, span_deg: span },
         "place_pin(" + guess.lat.toFixed(2) + ", " + guess.lng.toFixed(2) +
         ", span=" + span.toFixed(2) + ")",
         "rendering the map at this zoom\u2026");
  });

  $("submit").onclick = function () {
    if (!guess) return;
    // A click while the pin is still rendering is not an error, but it must not
    // look like the button is broken either.
    if (busy) {
      $("submit").textContent = "still placing the pin";
      return;
    }
    step("guess", { lat: guess.lat, lon: guess.lng },
         "submit_guess(" + guess.lat.toFixed(2) + ", " + guess.lng.toFixed(2) + ")",
         "scoring the guess\u2026");
  };

  function reveal(observation) {
    const hasDistance = observation.distance_km !== null &&
      observation.distance_km !== undefined;
    const km = hasDistance ? observation.distance_km : null;
    const reward = observation.reward === null ? 0 : observation.reward;
    const score = observation.score === null ? 0 : observation.score;
    const points = Math.round(score * MAX_POINTS);
    const truthLat = observation.true_lat, truthLon = observation.true_lon;
    total += points;
    played += 1;
    $("played").textContent = played + (played === 1 ? " episode" : " episodes");

    $("verdict").textContent = hasDistance
      ? (km < 0.025 ? "Perfect." : km < 25 ? "Pinpoint." : km < 200 ? "Close." :
         km < 1500 ? "Right region." : "Wrong continent.")
      : "Out of actions.";
    $("dist").textContent = hasDistance
      ? (km < 10 ? (km * 1000).toFixed(0) + " m" : km.toFixed(0) + " km")
      : "no guess";
    $("points").textContent = points + " / " + MAX_POINTS;
    $("reward").textContent = reward.toFixed(3);
    $("breakdown2").textContent =
      score.toFixed(3) + " - " + (observation.action_cost || 0).toFixed(2);
    $("truth").textContent = fmt(truthLat, truthLon);
    $("bar").firstElementChild.style.width = (reward * 100).toFixed(1) + "%";
    $("total").textContent = total;
    $("result").classList.add("show");
    document.body.classList.add("revealing");
    $("actions").style.opacity = "0";

    if (truthLat !== null && truthLat !== undefined &&
        truthLon !== null && truthLon !== undefined) {
      truthMarker = new maplibregl.Marker({ color: "#5aa06e" })
        .setLngLat([truthLon, truthLat]).addTo(map);
    }
    if (guess && truthLat !== null && truthLat !== undefined &&
        truthLon !== null && truthLon !== undefined) {
      const line = {
        type: "Feature",
        geometry: {
          type: "LineString",
          coordinates: [[guess.lng, guess.lat], [truthLon, truthLat]],
        },
      };
      if (lineAdded) {
        map.getSource("shot").setData(line);
      } else {
        map.addSource("shot", { type: "geojson", data: line });
        map.addLayer({
          id: "shot", type: "line", source: "shot",
          paint: { "line-color": "#c4332a", "line-width": 2.5, "line-dasharray": [2, 1.6] },
        });
        lineAdded = true;
      }
    }
    $("mapwrap").classList.add("reveal");
    map.resize();
    setTimeout(function () {
      if (guess && truthLat !== null && truthLat !== undefined &&
          truthLon !== null && truthLon !== undefined) {
        map.fitBounds(
          [[Math.min(guess.lng, truthLon), Math.min(guess.lat, truthLat)],
           [Math.max(guess.lng, truthLon), Math.max(guess.lat, truthLat)]],
          { padding: 80, maxZoom: 7, duration: 900 }
        );
      } else if (truthLat !== null && truthLat !== undefined &&
                 truthLon !== null && truthLon !== undefined) {
        map.flyTo({ center: [truthLon, truthLat], zoom: 5, duration: 900 });
      }
    }, 260);
  }

  function clearRound() {
    guess = null;
    setBusy(false);
    if (guessMarker) { guessMarker.remove(); guessMarker = null; }
    if (truthMarker) { truthMarker.remove(); truthMarker = null; }
    if (lineAdded) {
      map.getSource("shot").setData({
        type: "Feature", geometry: { type: "LineString", coordinates: [] },
      });
    }
    $("mapwrap").classList.remove("reveal", "big");
    map.resize();
    map.fitBounds(WORLD, { padding: 6, duration: 700 });
    $("submit").className = "";
    $("submit").textContent = "click the map to place a pin";
    $("result").classList.remove("show");
    document.body.classList.remove("revealing");
    $("actions").style.opacity = "1";
  }

  // An episode is one guess, so there is no round to advance: the terminal
  // control simply starts another episode.
  $("advance").onclick = function () {
    clearRound();
    startRound();
  };

  document.addEventListener("keydown", function (event) {
    if (event.key === "m" || event.key === "M") {
      $("mapwrap").classList.toggle("big");
      map.resize();
      if (!guess && !busy) map.fitBounds(WORLD, { padding: 6, duration: 250 });
    } else if (event.key === "t" || event.key === "T") {
      setTrace($("trace").classList.contains("hidden"));
    } else if (event.key === "ArrowUp" || event.key === "w") {
      if (!$("move-fwd").classList.contains("gone")) $("move-fwd").click();
    } else if (event.key === "ArrowDown" || event.key === "s") {
      if (!$("move-back").classList.contains("gone")) $("move-back").click();
    } else if (event.key === "l" || event.key === "L") {
      if (!$("padTurn").classList.contains("gone")) $("look-now").click();
    } else if (event.key === "ArrowLeft" || event.key === "a") {
      if (!$("padTurn").classList.contains("gone")) $("turn-left").click();
    } else if (event.key === "ArrowRight" || event.key === "d") {
      if (!$("padTurn").classList.contains("gone")) $("turn-right").click();
    } else if (event.key === "Enter") {
      if ($("result").classList.contains("show")) { $("advance").click(); }
      else { $("submit").click(); }
    }
  });

  connect();
})();
</script>
</body>
</html>
"""


def play_page_html(splits: list[dict] | int, split: str | None = None) -> str:
    """
    Return the standalone play page.

    Served at `/geoguesser/play` and embedded in the Gradio tab through an
    iframe. It is a full document rather than a fragment because `gr.HTML`
    inserts markup without running `<script>` tags.

    Args:
        splits (`list[dict]` or `int`):
            Split descriptors from [`~GeoGuesserEnvironment.list_splits`]. A
            bare integer is accepted as a task count for callers predating
            splits.
        split (`str`, *optional*):
            Which split the page should play. Defaults to the split marked
            `default`, else the first one.

    Returns:
        `str`: A complete HTML document.
    """
    if isinstance(splits, int):
        descriptors = [
            {"name": "train", "num_tasks": splits, "default": True, "type": "train"}
        ]
    else:
        descriptors = list(splits) or [
            {"name": "train", "num_tasks": 1, "default": True, "type": "train"}
        ]
    chosen = next(
        (d for d in descriptors if d["name"] == split),
        next((d for d in descriptors if d.get("default")), descriptors[0]),
    )
    replacements = {
        "__SPLIT__": json.dumps(chosen["name"]),
        "__N_TASKS__": str(max(1, int(chosen.get("num_tasks", 1)))),
        "__MAX_POINTS__": str(MAX_POINTS_PER_ROUND),
        "__MAPLIBRE_JS__": MAPLIBRE_JS,
        "__MAPLIBRE_CSS__": MAPLIBRE_CSS,
        "__PANNELLUM_JS__": PANNELLUM_JS,
        "__PANNELLUM_CSS__": PANNELLUM_CSS,
        "__OPENFREEMAP_STYLE__": OPENFREEMAP_STYLE,
    }
    page = _TEMPLATE
    for token, value in replacements.items():
        page = page.replace(token, value)
    return page


def _iframe(task: str | int = "random", split: str = "") -> str:
    """Markup for the play iframe, pointed at one task of one split.

    Args:
        task (`str` or `int`, *optional*, defaults to `"random"`):
            Task index to open, or `"random"`.
        split (`str`, *optional*):
            Split to play. Empty means the server's default split.

    Returns:
        `str`: An iframe element. Re-rendering it with a different task is what
        makes the Gradio controls reload the round, since the page reads its
        task from the URL.
    """
    query = f"?task={task}"
    if split:
        query += f"&split={urllib.parse.quote(split)}"
    return (
        f'<iframe src="/geoguesser/play{query}" '
        'style="width:100%;height:720px;border:1px solid #2c353d;'
        'border-radius:6px" allow="fullscreen"></iframe>'
    )


def build_geoguesser_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[Any],
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> gr.Blocks:
    """
    Build the human-play tab.

    The episode controls live here, on the Gradio side, rather than inside the
    page: picking a task is orchestration, the same `reset(task_index=k)` an
    eval harness calls, so it belongs with the host controls and not among the
    in-game HUD.

    Args:
        web_manager (`Any`):
            The playground's environment manager, unused here.
        action_fields (`list[dict]`):
            Action schema fields, unused here.
        metadata (`Any`, *optional*):
            Environment metadata, unused here.
        is_chat_env (`bool`):
            Whether the env is chat-shaped, unused here.
        title (`str`):
            Playground title.
        quick_start_md (`str`):
            Quick-start markdown, unused here.

    Returns:
        `gradio.Blocks`: The play tab, hosting `/geoguesser/play` in an iframe.
    """
    # Ask the server which splits it actually serves, rather than re-deriving
    # them here from environment variables and drifting out of step with it.
    descriptors: list[dict] = []
    try:
        from .app import ACTIVE_DEFAULT_SPLIT, create_geoguesser_environment

        descriptors = create_geoguesser_environment().list_splits()
        default_split = ACTIVE_DEFAULT_SPLIT
    except Exception:  # pragma: no cover - the page still works without counts
        default_split = "train"
    if not descriptors:
        descriptors = [
            {"name": default_split, "num_tasks": 1, "default": True, "type": "train"}
        ]

    counts = {d["name"]: max(1, int(d.get("num_tasks", 1))) for d in descriptors}
    names = list(counts)
    if default_split not in counts:
        default_split = names[0]

    def _label(split: str) -> str:
        return f"reset(index=)  ·  0 to {counts[split] - 1}"

    with gr.Blocks(title="Geoguesser Environment") as blocks:
        with gr.Row():
            split_box = gr.Dropdown(
                choices=names,
                value=default_split,
                label="reset(split=)",
                scale=1,
                interactive=len(names) > 1,
            )
            task_box = gr.Number(
                value=0,
                minimum=0,
                maximum=counts[default_split] - 1,
                step=1,
                precision=0,
                label=_label(default_split),
                scale=2,
            )
            load_button = gr.Button("load episode", variant="primary", scale=1)
            random_button = gr.Button("random episode", scale=1)
        frame = gr.HTML(value=_iframe("random", default_split), show_label=False)

        def _on_split(split: str):
            """Re-range the index box so it cannot address a missing task."""
            split = split or default_split
            return gr.update(maximum=counts[split] - 1, value=0, label=_label(split))

        split_box.change(fn=_on_split, inputs=split_box, outputs=task_box)
        load_button.click(
            fn=lambda index, split: _iframe(int(index or 0), split or default_split),
            inputs=[task_box, split_box],
            outputs=frame,
        )
        random_button.click(
            fn=lambda split: _iframe(
                random.randrange(counts[split or default_split]),
                split or default_split,
            ),
            inputs=split_box,
            outputs=frame,
        )
    return blocks


__all__ = ["build_geoguesser_gradio_app", "play_page_html"]

"""pass@k on the DataAgent eval suite, through the same OpenEnv x Harbor server.

Mirrors `experiments/rollout_control/harbor_trl/eval_harbor.py` so the numbers are comparable to the
sync run's, with three differences that matter:

  * **Same harness as training.** That script used `agent="bash"` because the sync run TRAINED on the
    bash env. We train on mini-swe-agent, so evaluating with bash would measure a different agent than
    the one being optimised.
  * **No second server.** The engine arrives per request, so this job stands up its own vLLM and names
    it; the dataset server is shared with the training run. The engine is deliberately FLAGLESS (no
    `--return-tokens-as-token-ids`, no `--logprobs-mode`), so the server probes it as `eval` tier and
    captures no tokens — there is nothing to train on here and pretending otherwise costs memory.
  * **Infra failures are excluded, not scored 0.** `data_agent_reward` returns `None` when the verifier
    never ran, which is precisely the `infra_failed` exclusion the sync script hand-rolled: a dead
    sandbox is missing data, and counting it as a wrong answer silently deflates pass@k.

`k > 1` requires temperature > 0. At temperature 0 every sample of a task is identical and pass@k
collapses to pass@1 while looking like a real k-sample number.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE.parents[1] / "OpenEnv" / "src"))
sys.path.insert(0, str(HERE.parents[1] / "OpenEnv" / "envs"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval")

EVAL_SPLIT = "AdithyaSK/data_agent_rl_environment_eval"


def _supported_harnesses() -> list[str]:
    """The harnesses this CLI will run, read from tools/harnesses_supported.txt.

    A list, not a hardcoded tuple, so a support decision is made once and every stage inherits it.
    Excluded harnesses are absent rather than silently accepted: `goose` cannot be budgeted (6 turns
    on one rollout, 347 on another of the same task), `openclaw`'s CLI exits 1 without making a model
    call, and `kimi-cli` produced no reward at all. Offering them in the CLI would invite a sweep
    whose numbers are not comparable to the others.
    """
    path = Path(__file__).resolve().parent / "harnesses_supported.txt"
    names = []
    for line in path.read_text().splitlines():
        name = line.split("#", 1)[0].strip()
        if name:
            names.append(name)
    if not names:
        raise SystemExit(f"{path} lists no harnesses")
    return names


SUPPORTED_HARNESSES = _supported_harnesses()


# ── per-harness circuit breaker ─────────────────────────────────────────────────────────────────────
# A harness that cannot produce a graded rollout at all should stop consuming sandboxes. The threshold
# counts CONSECUTIVE UNSCORABLE results only — never low rewards. A harness scoring 0.0 on hard tasks is
# working correctly and telling us something; one whose verifier never runs is broken, and the two must
# not be confused or the sweep would silently drop the weakest models instead of the broken harnesses.
_state_lock = threading.Lock()
_streak: dict[str, int] = defaultdict(int)
_graded: dict[str, int] = defaultdict(int)
DISABLED: dict[str, str] = {}

# A GLOBAL brake, separate from the per-harness one. The per-harness breaker cannot see a systemic
# failure: when the SERVER stops accepting connections every harness fails together, each has already
# graded something, so none looks broken and the sweep burns its whole budget on refusals. That
# happened — 96 concurrent clients against a single-process server drove it to 3132 open fds until it
# stopped responding, and ~58% of 2800 rollouts came back `[Errno 111] Connection refused` while the
# per-harness breaker stayed correctly silent.
_conn_fail_streak = 0
SERVER_DOWN: str = ""
# "timed out" belongs here: a server can be LISTENING but wedged, accepting connections and never
# answering. That produces httpx.ReadTimeout, not a refusal, and without this marker the brake stayed
# silent through exactly that phase while the sweep burned rollouts. Measured on one run: 157 refusals,
# 88 timeouts and 24 resets, and only the first group was being counted.
_CONN_MARKERS = (
    "No rollout result returned by Harbor transport",
    "Connection refused",
    "Errno 111",
    "Connect call failed",
    "ConnectionResetError",
    "Errno 104",
    "timed out",
    "ReadTimeout",
    "ConnectTimeout",
)


def _note_connectivity(err: str | None, threshold: int) -> None:
    """Track consecutive connection-level failures across ALL harnesses."""
    global _conn_fail_streak, SERVER_DOWN
    with _state_lock:
        if err and any(m in err for m in _CONN_MARKERS):
            _conn_fail_streak += 1
            if _conn_fail_streak >= threshold and not SERVER_DOWN:
                SERVER_DOWN = (
                    f"{_conn_fail_streak} consecutive connection failures — the server is refusing "
                    f"connections; stopping so the remaining rollouts are not wasted"
                )
                logger.error("HALTING SWEEP: %s", SERVER_DOWN)
        else:
            _conn_fail_streak = 0


def _note_result(harness: str, graded: bool, threshold: int) -> None:
    with _state_lock:
        if graded:
            _streak[harness] = 0
            _graded[harness] += 1
            return
        _streak[harness] += 1
        # Never disable a harness that has already proved it can be graded: a later run of unscorable
        # results is a flaky sandbox, not a broken integration.
        if _graded[harness] == 0 and _streak[harness] >= threshold and harness not in DISABLED:
            DISABLED[harness] = (
                f"{_streak[harness]} consecutive unscorable rollouts and never once graded"
            )
            logger.warning(
                "PAUSING %s — %s; its remaining rollouts are skipped", harness, DISABLED[harness]
            )


def _as_plain(m):
    """A message as plain JSON. Pydantic models and dicts both arrive here depending on the client."""
    if hasattr(m, "model_dump"):
        return m.model_dump()
    if isinstance(m, dict):
        return m
    return {"role": "unknown", "content": str(m)}


def one_sample(args, harness: str, index: int, rep: int) -> dict:
    """One rollout of one task. `reward=None` marks it as infra, not as wrong."""
    from harbor_env.harness import HarborSessionFactory

    with _state_lock:
        halted = SERVER_DOWN
    if halted:
        return {"harness": harness, "index": index, "rep": rep, "reward": None, "n_turns": 0,
                "ok": False, "skipped": True, "skip_reason": halted, "messages": []}

    with _state_lock:
        paused = DISABLED.get(harness)
    if paused:
        # Skipped, not failed: recorded distinctly so the summary can say a harness was paused rather
        # than implying it was measured and scored nothing.
        return {"harness": harness, "index": index, "rep": rep, "reward": None,
                "n_turns": 0, "ok": False, "skipped": True, "skip_reason": paused, "messages": []}

    f = HarborSessionFactory(
        args.server,
        split=args.split,
        harness=harness,
        sandbox=args.sandbox,
        llm_url=args.vllm_url,
        model=args.model,
        api_key=args.api_key,
        auth_header=args.auth_header,
        agent_timeout_sec=args.agent_timeout,
        agent_step_limit=args.agent_step_limit,
        indices=[index],
    )
    session = None
    try:
        session = f.create(f.prompt_rows()[0]["prompt"])
        session.wait_for_completion()
        result = session.result
        reward = session.verify([]).env_reward

        # THE CONVERSATION, from `result.conversations`, not `fetch_proxy_trace()`. The proxy trace is
        # deliberately empty for an eval-tier rollout (no token fields to train on), so reading it here
        # is why every cell said "no trace recorded". `conversations[].messages` is the readable form —
        # system prompt, every assistant turn, every tool result — and it is present at either tier.
        #
        # Read n_turns first: the trace checks below compare against it.
        n_turns = getattr(result, "n_turns", 0) or 0

        # Agent-role conversations only: a rollout is a tree, and auxiliary calls (title generation,
        # summarisers) are not what the task was solved with. Including them would pad the trace with
        # work the grader never saw.
        #
        # TAKING THE FIRST ONE LOSES MOST OF THE TRACE. Several harnesses report one agent
        # conversation PER STEP, each carrying the accumulated history, so the conversations are
        # strictly nested prefixes and the first is the SHORTEST. Measured on this suite:
        #
        #   terminus-2   n_turns=5  ->  convs of 2, 4, 6, 8, 10 messages   (every pair prefix=True)
        #   claude-code  n_turns=5  ->  convs of 3, 5, 7, 9, 11 messages   (every pair prefix=True)
        #
        # so the stored trace was 1 assistant turn out of 5. Concatenating them is equally wrong --
        # that yields 30 messages of duplicated history for a 10-message rollout.
        #
        # Correct reduction: drop any conversation that is a PREFIX of another, then use what remains.
        # For nested harnesses exactly one survives (the longest, which contains all the others). For a
        # harness with genuinely disjoint sub-conversations, several survive and are concatenated in
        # order, which is right for that shape and would have been wrong for this one. n_agent_convs
        # and trace_reduction are recorded so a trace can always be audited back to what produced it.
        _agent_raw = [
            conv
            for conv in (getattr(result, "conversations", None) or [])
            if getattr(conv, "role", "agent") == "agent"
        ]
        agent_convs = [
            [_as_plain(m) for m in (getattr(conv, "messages", None) or [])] for conv in _agent_raw
        ]
        # The AGENT conversations' own turn count, which is not the rollout's. A rollout counts every
        # model call it made, including calls that belong to no agent conversation -- measured on
        # opencode: rollout n_turns=7 while its single agent conversation reported n_turns=6 and
        # carried exactly 6 assistant messages. Comparing a trace against the rollout figure made 15
        # of 15 complete traces look short by one. This is the number a trace can actually be
        # checked against.
        agent_turns = sum(getattr(c, "n_turns", 0) or 0 for c in _agent_raw)
        agent_convs = [c for c in agent_convs if c]

        def _sig(conv):
            """Comparable shape of a conversation, for the prefix test.

            The SYSTEM message is compared by role only, never by content. claude-code re-renders its
            system prompt per step with something varying inside it (identical for the first 150
            chars, divergent later), so comparing system content made five strictly-nested
            conversations look disjoint and concatenated them: 13 turns became 101 assistant
            messages. Everything else is compared exactly.
            """
            out = []
            for m in conv:
                role = m.get("role")
                if role == "system":
                    out.append(("system", ""))
                    continue
                c = m.get("content")
                out.append((role, c if isinstance(c, str) else json.dumps(c, sort_keys=True)))
            return out

        sigs = [_sig(c) for c in agent_convs]
        keep = [
            i
            for i in range(len(agent_convs))
            if not any(
                j != i and len(sigs[j]) >= len(sigs[i]) and sigs[j][: len(sigs[i])] == sigs[i]
                for j in range(len(agent_convs))
            )
        ]
        messages = [m for i in keep for m in agent_convs[i]]
        conv_role = "agent" if messages else ""
        trace_reduction = (
            "" if len(agent_convs) <= 1
            else ("longest-of-nested" if len(keep) == 1 else f"concatenated-{len(keep)}-disjoint")
        )

        # INFLATION GUARD. If the prefix test misjudges nested conversations as disjoint, the trace
        # silently multiplies -- measured once at 101 assistant messages for a 13-turn rollout. There
        # is no honest reason for a trace to hold many more assistant messages than the rollout had
        # turns, so say so loudly rather than writing it into the dataset unremarked.
        _asst = sum(1 for m in messages if m.get("role") == "assistant")
        trace_warning = ""
        # Require ACTUAL DUPLICATION, not just a high count ratio. codex emits a prose message and a
        # separate tool-call message every step, so ~2 assistant messages per turn is its normal
        # shape; and agent_turns undercounts for it (4 turns -> 2). Ratio alone produced 13 false
        # CRITICALs on traces with zero duplicated messages. Compared against the LARGER of the two
        # turn counts so an undercounting field cannot trigger this by itself.
        _ref_turns = max(agent_turns or 0, n_turns or 0)
        _sigs = [
            (m.get("role"), m.get("content") if isinstance(m.get("content"), str)
             else json.dumps(m.get("content"), sort_keys=True))
            for m in messages
        ]
        _dups = sum(n - 1 for s, n in Counter(_sigs).items() if n > 1 and (s[1] or "").strip())
        if _ref_turns and _asst > 1.5 * _ref_turns + 2 and _dups:
            trace_warning = (
                f"trace holds {_asst} assistant messages for {_ref_turns} turns with {_dups} duplicate(s) "
                f"({len(agent_convs)} agent convs, {trace_reduction or 'single'}) — likely duplicated"
            )
            logger.warning("%s task %d rep %d: %s", harness, index, rep, trace_warning)


        # A ZERO-TURN ROLLOUT IS NOT A MEASUREMENT. Both test.sh variants write a 0.0 when
        # `answer.txt` is missing -- the scalar one via `echo "0.0" > reward.txt`, the json one via its
        # early-exit branch -- so a harness whose CLI crashed before making a single model call comes
        # back with a clean, countable 0.0 that is indistinguishable from a model that tried and was
        # wrong. Measured: openclaw exited 1 under nvm and scored 0.0/0.0 at 0 turns.
        #
        # Excluding it is the honest reading: with no model call there is nothing about the policy to
        # score. This is deliberately narrow -- turns > 0 with reward 0.0 stays a real zero, because a
        # model that answered badly SHOULD score zero.
        infra_zero = ""
        if reward is not None and n_turns == 0:
            infra_zero = (
                "0 turns: the agent made no model call, so the graded 0.0 is the verifier's "
                "missing-answer default rather than a measurement"
            )
            logger.warning("%s task %d rep %d excluded — %s", harness, index, rep, infra_zero)
            reward = None

        return {
            "harness": harness,
            "index": index,
            "rep": rep,
            "reward": None if reward is None else float(reward),
            "infra_zero": infra_zero,
            "n_turns": n_turns,
            "ok": bool(getattr(result, "ok", False)),
            "task_id": getattr(result, "task_id", "") or "",
            "trial_name": getattr(result, "trial_name", "") or "",
            "wall_s": getattr(result, "wall_s", None),
            "rewards": dict(getattr(result, "rewards", None) or {}),
            "rollout_type": getattr(result, "rollout_type", ""),
            "n_trainable_tokens": getattr(result, "n_trainable_tokens", 0) or 0,
            "conversation_role": conv_role,
            "n_agent_convs": len(agent_convs),
            "agent_turns": agent_turns,
            "trace_reduction": trace_reduction,
            "trace_warning": trace_warning,
            "messages": messages,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s task %d rep %d failed: %s", harness, index, rep, str(exc)[:160])
        return {"harness": harness, "index": index, "rep": rep, "reward": None, "n_turns": 0,
                "ok": False, "error": str(exc)[:300], "messages": []}
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:  # noqa: BLE001
                pass


def score(rows: list[dict], k: int, n_tasks: int) -> dict:
    """pass@k and pass@1 for one harness.

    pass@k is over TASKS that produced at least one graded sample; pass@1 is over SAMPLES. They can
    therefore invert when tasks contribute unequal numbers of graded samples — an earlier run reported
    pass@1 0.375 above pass@4 0.333 for exactly that reason. Both definitions match the sync run's
    eval_harbor.py, so the numbers stay comparable; the asymmetry is just worth knowing.
    """
    by_task: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["index"]].append(r)

    solved = measured = all_infra = 0
    per_sample: list[float] = []
    turns: list[int] = []
    for rs in by_task.values():
        graded = [r for r in rs if r["reward"] is not None]
        if not graded:
            all_infra += 1
            continue
        measured += 1
        if any(r["reward"] > 0 for r in graded):
            solved += 1
        per_sample.extend(r["reward"] for r in graded)
        turns.extend(r["n_turns"] for r in graded)

    return {
        "n_tasks": n_tasks,
        "n_measured": measured,
        "n_all_infra": all_infra,
        f"pass@{k}": round(solved / measured, 4) if measured else None,
        "pass@1": round(statistics.mean(per_sample), 4) if per_sample else None,
        "mean_turns": round(statistics.mean(turns), 2) if turns else None,
    }


def load_prior(paths: list[str]) -> tuple[dict[str, list[dict]], set[tuple[str, int]]]:
    """Rows already collected, and the (harness, task) pairs that need no more work.

    A pair is DONE when it has at least one GRADED attempt. Not "has rows" — a task whose every attempt
    came back unscorable is exactly the work a resume exists to redo, and treating it as finished would
    bake an infrastructure failure into the result permanently.

    Prior rows are carried forward so the resumed run's summary covers everything, not just the tail.
    """
    rows: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        f = Path(path)
        if not f.exists():
            logger.warning("resume source missing, skipping: %s", path)
            continue
        blob = json.loads(f.read_text())
        for harness, rs in (blob.get("rows") or {}).items():
            rows[harness].extend(rs)

    graded: dict[tuple[str, int], int] = defaultdict(int)
    for harness, rs in rows.items():
        for r in rs:
            if r.get("reward") is not None:
                graded[(harness, r["index"])] += 1
    done = set(graded)
    logger.info(
        "resume: loaded %d prior rows across %d harness(es); %d (harness, task) pairs already graded",
        sum(len(v) for v in rows.values()), len(rows), len(done),
    )
    return rows, done


def _dump(args, rows_by_harness, started, partial: bool) -> dict:
    """Write the summary as it stands. Used for the periodic partial write and the final one."""
    per_harness = {
        h: score(rows, args.eval_k, args.n_tasks) for h, rows in sorted(rows_by_harness.items())
    }
    with _state_lock:
        paused = dict(DISABLED)
    with _state_lock:
        halted = SERVER_DOWN
    # Recorded in the output, because a sweep that halted is not a sweep that measured low: the
    # difference has to survive into whatever reads this.
    summary = {
        "dataset": args.split, "model": args.model, "step": args.step, "k": args.eval_k,
        "elapsed_s": round(time.monotonic() - started),
        "partial": partial,
        "paused_harnesses": paused,
        "halted": halted or None,
        "harnesses": per_harness,
    }
    out = Path(args.out) if args.out else HERE / f"logs/eval_step{args.step}_k{args.eval_k}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"summary": summary, "rows": rows_by_harness}, indent=2, default=str))
    tmp.replace(out)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--vllm-url", required=True)
    ap.add_argument("--model", required=True, help="checkpoint dir or hub id, as served by the engine")
    ap.add_argument("--split", default=EVAL_SPLIT)
    ap.add_argument(
        "--harness",
        default="mini-swe-agent",
        choices=SUPPORTED_HARNESSES,
        metavar="NAME",
    )
    # One engine serves every harness, because `harness` is a per-request parameter on run_rollout.
    # Four harnesses in one job costs one GPU, not four, and holds the model fixed so the comparison is
    # between AGENTS rather than between engines.
    ap.add_argument(
        "--harnesses",
        default="",
        metavar="A,B,C",
        help=(
            "comma list for a multi-harness sweep; overrides --harness. Supported: "
            + ", ".join(SUPPORTED_HARNESSES)
        ),
    )
    ap.add_argument(
        "--allow-unsupported-harness",
        action="store_true",
        help="run a harness excluded in tools/harnesses_supported.txt anyway (for re-testing it)",
    )
    # Hosted providers (Anthropic, OpenAI, the HF router) need a credential on every request the
    # sandbox makes. It is read from the environment by default so a key never has to appear in a
    # command line -- and so it never lands in a log, a shell history, or a saved invocation.
    ap.add_argument(
        "--api-key-env",
        default="",
        help="name of the env var holding the upstream credential, e.g. OPENAI_API_KEY",
    )
    ap.add_argument(
        "--auth-header",
        default="",
        help="header to send the credential under; defaults to Authorization: Bearer",
    )
    ap.add_argument("--sandbox", default="e2b")
    ap.add_argument("-k", "--eval-k", type=int, default=4, help="samples per task; needs temperature>0")
    ap.add_argument("--n-tasks", type=int, default=366)
    # An explicit index list beats --n-tasks: the eval suite's difficulty_level is very unevenly
    # distributed (269 tasks at level 0, 12 at level 3), so the first N tasks are not a difficulty
    # sample of anything.
    ap.add_argument("--indices", default="", help="comma list or @file of task indices")
    # The server's sandbox ceiling is set far above this (300), so this number really is the throttle
    # rather than a request that gets queued behind a training run.
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--agent-timeout", type=float, default=300.0)
    ap.add_argument("--agent-step-limit", type=int, default=12)
    ap.add_argument("--resume-from", nargs="*", default=[],
                    help="prior sweep JSONs; (harness, task) pairs already GRADED are skipped and "
                         "their rows carried into this run's summary")
    ap.add_argument("--halt-after-conn-fails", type=int, default=15,
                    help="stop the whole sweep after this many CONSECUTIVE connection-level failures; "
                         "a server that stops accepting fails every harness at once, which the "
                         "per-harness breaker cannot detect")
    ap.add_argument("--pause-after", type=int, default=8,
                    help="pause a harness after this many CONSECUTIVE unscorable rollouts with none "
                         "ever graded; low rewards never count")
    ap.add_argument("--partial-every", type=int, default=20,
                    help="write the summary every N rollouts so the run is watchable live; 0 disables")
    ap.add_argument("--step", type=int, default=-1, help="training step this checkpoint came from")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # Resolve the credential from the environment, never from the command line. Refuse rather than
    # run unauthenticated: a hosted provider answering 401 on every call would come back as 250
    # unscorable rollouts, which reads as a broken harness rather than a missing key.
    args.api_key = ""
    if args.api_key_env:
        args.api_key = os.environ.get(args.api_key_env, "")
        if not args.api_key:
            raise SystemExit(f"--api-key-env {args.api_key_env} is set but that variable is empty")

    harnesses = [h.strip() for h in args.harnesses.split(",") if h.strip()] or [args.harness]

    # Reject up front, not per-rollout. An unsupported harness discovered 200 rollouts in has already
    # spent the sandboxes, and its results are not comparable with the rest of the sweep.
    unsupported = [h for h in harnesses if h not in SUPPORTED_HARNESSES]
    if unsupported and not args.allow_unsupported_harness:
        raise SystemExit(
            f"unsupported harness: {', '.join(unsupported)}\n"
            f"supported: {', '.join(SUPPORTED_HARNESSES)}\n"
            f"see tools/harnesses_supported.txt for why each exclusion was made, or pass "
            f"--allow-unsupported-harness to re-test one."
        )
    if unsupported:
        logger.warning(
            "running UNSUPPORTED harness(es) %s — results are not comparable with a supported sweep",
            ", ".join(unsupported),
        )
    if args.indices:
        spec = args.indices
        if spec.startswith("@"):
            spec = (HERE / spec[1:]).read_text() if not Path(spec[1:]).is_absolute() else Path(spec[1:]).read_text()
        # Dedupe but PRESERVE the caller's order. This used to sort, which silently destroyed the whole
        # point of passing a shuffled list: the suite is ordered easy->hard, so a pass that dies before
        # finishing loses a contiguous hard tail. Sorting made every partial pass an easy-prefix
        # measurement -- openhands-sdk never scored a task above index 149 across four passes, and its
        # 0.864 was really 0.850-on-the-easy-133 rather than a 250-task score.
        _seen: set[int] = set()
        indices = []
        for x in spec.replace("\n", ",").split(","):
            if not x.strip():
                continue
            i = int(x)
            if i not in _seen:
                _seen.add(i)
                indices.append(i)
    else:
        indices = list(range(args.n_tasks))
    args.n_tasks = len(indices)
    # ROUND-ROBIN by harness, not harness-major. `pool.map` consumes the list in order, so grouping by
    # harness means the first workers are all one harness and the sweep completes them roughly in
    # series: a run that dies at the halfway mark leaves half the harnesses with full data and half with
    # none. Interleaving means every harness has partial coverage at any moment, which is both what a
    # live view should show and the more useful thing to salvage from an interrupted sweep.
    prior_rows, already_done = ({}, set())
    if args.resume_from:
        prior_rows, already_done = load_prior(args.resume_from)

    # Skip whole (harness, task) pairs, not individual attempts. pass@k is a property of the k attempts
    # of one task, so a task with 2 of 4 attempts done cannot be topped up without mixing samples taken
    # under different server conditions — it is redone.
    per_harness_jobs = [
        [(h, i, r) for i in indices if (h, i) not in already_done for r in range(args.eval_k)]
        for h in harnesses
    ]
    # zip_longest, not zip: with a resume each harness has a DIFFERENT number of remaining tasks, and
    # plain zip truncates to the shortest — which would silently drop the work of whichever harness has
    # the most left to do, the exact opposite of what a resume is for.
    from itertools import zip_longest

    jobs = [j for wave in zip_longest(*per_harness_jobs) for j in wave if j is not None]
    assert len(jobs) == sum(len(x) for x in per_harness_jobs), "lost jobs while interleaving"
    logger.info(
        "%d harness(es) x %d tasks x k=%d = %d rollouts, %d at a time",
        len(harnesses), args.n_tasks, args.eval_k, len(jobs), args.concurrency,
    )
    started = time.monotonic()

    rows_by_harness: dict[str, list[dict]] = defaultdict(list)
    for h, rs in prior_rows.items():
        rows_by_harness[h].extend(rs)
    done = 0
    # Interleaved across harnesses on purpose: a slow harness then overlaps a fast one instead of
    # serialising behind it, and a mid-run failure does not leave one harness entirely unmeasured.
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for row in pool.map(lambda j: one_sample(args, j[0], j[1], j[2]), jobs):
            rows_by_harness[row["harness"]].append(row)
            _note_result(row["harness"], row.get("reward") is not None, args.pause_after)
            _note_connectivity(row.get("error"), args.halt_after_conn_fails)
            done += 1
            if done % 10 == 0:
                with _state_lock:
                    paused = dict(DISABLED)
                logger.info(
                    "%d/%d rollouts, %.0fs%s",
                    done, len(jobs), time.monotonic() - started,
                    f", paused: {sorted(paused)}" if paused else "",
                )
            # Partial write, so the run is watchable while it runs instead of only at the end. Written
            # to a temp file and moved, because a reader hitting a half-written JSON would look like a
            # corrupt result rather than an in-progress one.
            if args.partial_every and done % args.partial_every == 0:
                _dump(args, rows_by_harness, started, partial=True)

    per_harness = {
        h: score(rows, args.eval_k, args.n_tasks) for h, rows in sorted(rows_by_harness.items())
    }
    summary = {
        "dataset": args.split,
        "model": args.model,
        "step": args.step,
        "k": args.eval_k,
        "elapsed_s": round(time.monotonic() - started),
        "harnesses": per_harness,
    }
    logger.info("--- %s", json.dumps(summary, indent=2))
    key = f"pass@{args.eval_k}"
    logger.info("%-18s %8s %8s %11s %10s", "harness", key, "pass@1", "mean_turns", "measured")
    for h, m in sorted(per_harness.items(), key=lambda kv: -(kv[1][key] or 0)):
        logger.info(
            "%-18s %8s %8s %11s %10s",
            h, m[key], m["pass@1"], m["mean_turns"], f"{m['n_measured']}/{m['n_tasks']}",
        )

    summary = _dump(args, rows_by_harness, started, partial=False)
    per_harness = summary["harnesses"]
    logger.info("--- %s", json.dumps(summary, indent=2))
    key = f"pass@{args.eval_k}"
    logger.info("%-18s %8s %8s %11s %10s", "harness", key, "pass@1", "mean_turns", "measured")
    for h, m in sorted(per_harness.items(), key=lambda kv: -(kv[1][key] or 0)):
        logger.info("%-18s %8s %8s %11s %10s", h, m[key], m["pass@1"], m["mean_turns"],
                    f"{m['n_measured']}/{m['n_tasks']}")
    if summary["paused_harnesses"]:
        for h, why in summary["paused_harnesses"].items():
            logger.warning("PAUSED %s: %s", h, why)
    logger.info("wrote %s", args.out or "logs/")
    return 0 if any(m["n_measured"] for m in per_harness.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

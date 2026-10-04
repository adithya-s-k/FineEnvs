# MiMo-V2.6-RL in Harbor format

A Harbor adapter for [XiaomiMiMo/MiMo-V2.6-RL-oss](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss):
all 7,780 environments as [Harbor](https://github.com/harbor-framework/harbor) tasks, set up and graded the way
Xiaomi's harness does it (XiaomiMiMo/verl `a2ad9f6` + XiaomiMiMo/mimoagent `467f0a1`).

The adapter mirrors the MiMo RL Environment Explorer's runner (`app/runner/domains.py`), which is the "original"
side of the parity runs below. Both use the same vendored graders, byte for byte.

| Dataset | Tasks | Agent works in | Graded by | Step limit |
|---|---:|---|---|---:|
| `code` | 2,698 | the repo at its base commit | the hidden test patch and the task's test command | 500 |
| `cyber` | 1,000 | `/home/agent`, as the unprivileged `agent` user | the image's PoC verify server (crash in the expected function) | 300 |
| `general` | 925 | a simulated workplace behind 5 to 15 MCP systems, as `agent` | the task's `run_verify.py` (rules plus an LLM judge) | 500 |
| `terminal` | 64 | the Terminal-Bench task's workspace | its own `test.sh` (pytest plus an anti-hack guard) | 500 |
| `webdev` | 2,093 | `/workspace`, delivering `dist/` | a full-page render judged by a vision model | 64 |
| `music` | 1,000 | `/app`, writing one reply | Xiaomi's music scorer (abc2midi, 18 features, validity gate) | 4 |

## Generate

```bash
uv run python -m mimo_harbor.main --output-dir ../mimo-v2.6-rl-harbor                  # all 7,780
uv run python -m mimo_harbor.main --output-dir OUT --split parity                       # 6 per dataset
uv run python -m mimo_harbor.main --output-dir OUT --task-ids arvo_42480818 --overwrite
python -m mimo_harbor.check ../mimo-v2.6-rl-harbor                                       # static checks (needs harbor)
```

Conversion is deterministic. It reads the source at one pinned revision (`source.py`) and the images from
`images.lock.json` (tag to digest, refreshed only by `python -m mimo_harbor.images`). The output has no timestamps
and fixed archive metadata, so two runs give byte-identical tasks. Each dataset's `MANIFEST.json` records the
sha256 of every task directory.

## Run

```bash
cd mimo-v2.6-rl-harbor
PYTHONPATH=agents HF_TOKEN=hf_... harbor run -y -c jobs/code.yaml
```

The `jobs/*.yaml` configs carry what a `task.toml` can't hold: the agent and its settings. Each one runs
`agents/mimo_opencode.py` on HF Sandbox with the explorer's settings:
- OpenCode 1.18.32, with the reference step limit for that dataset;
- replies of up to 65,536 tokens (100,000 for Music);
- thinking set to low;
- web fetch and search denied.

Change `model_name` and the provider block to run another model. Any other Harbor agent works on these tasks too,
but only this one applies the two harness steps below.

## How a task is built

```
task.toml         image pinned by digest; setup runs once from [environment.healthcheck]; judge settings in [verifier.env]
instruction.md    the exact prompt the explorer gives OpenCode
environment/      Dockerfile (FROM the same digest) and setup/, the readable copy of what the healthcheck runs
tests/            test.sh and everything grading needs; Harbor uploads it only after the agent finishes
```

- **Setup before the agent.** HF Sandbox and other prebuilt-image backends can't build a Dockerfile, so any
  per-task setup travels inside `task.toml`: the healthcheck unpacks a small archive into `/var/lib/mimo` and runs
  `setup.sh` once.
  - **Code:** records the base commit, runs mimoagent's anti-hack cleanup verbatim, and hides `.git` if the
    image's history reaches past the base.
  - **Cyber:** starts the PoC verify server.
  - **General:** downloads the workplace files from the dataset at the pinned revision, checking every file's
    hash, starts the MCP systems, and makes their data root-only.
- **Nothing that grades a task is reachable while the agent works.** That covers hidden tests, rubrics and verifier
  scripts. They all live in `tests/`, which Harbor uploads after the agent finishes.
- **"Not scored" is not 0.** When the testbed fails, the grader writes no reward and exits non-zero, so Harbor
  records the trial as a verifier error rather than a score. Testbed failures include:
  - a test patch that won't apply;
  - a system that died;
  - a judge that never answered;
  - a render that failed;
  - Terminal-Bench's `-1` crash sentinel.

### The agent: `agents/mimo_opencode.py`

This is Harbor's OpenCode agent plus three steps from Xiaomi's harness that a task can't express:
1. **The answer-leak blocklist.** It adds 32 hosts to `/etc/hosts` right after install, since the install itself
   needs github.com. If that fails, the agent stops, as mimoagent's does.
2. **The unprivileged user.** Cyber and General set `[agent].user = "agent"`. HF Sandbox can't switch users, so on
   it this agent runs every command through `runuser -u agent`.
3. **The output ceiling.** It sets `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX` itself instead of in the job's env,
   because Harbor treats any env var named `*TOKEN*` as a secret and blanks its value in every output file.

## Deviations from the reference, and why

| What | Reference | Here | Why |
|---|---|---|---|
| Music | one chat completion, no tools | a reply written to `/app/answer.md`, at most 4 steps with only the `write` tool; the final message counts if no file was written | Harbor tasks are agent sessions. The scorer reads the reply exactly as it read the completion. |
| `/logs` in Code's residue scrub | `rm -rf /logs` | the contents are emptied, and Harbor's `/logs/agent`, `/logs/verifier` and `/logs/artifacts` are kept | `/logs` belongs to Harbor |
| General's judge | whatever the training infra set | `thinkingmachines/Inkling` via the HF router, overridable with `MIMO_TEXT_JUDGE` / `MIMO_JUDGE_URL` | gpt-oss-120b echoed placeholder ids on Chinese tasks (explorer finding); Inkling gave 14/14 usable verdicts |
| Webdev's judge | whatever the training infra set | `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`, overridable with `MIMO_VISION_JUDGE` | the same as the explorer |
| Terminal-Bench `allow_internet = false` | not applicable (no reference) | network on | the agent calls its model from inside the container; the original flag is kept in `[metadata]` |
| Cyber reward | the server's last result | the grader re-runs the last submitted PoC with the server's own code | a root agent could overwrite `/root/last_result.json`; the verdict is the same otherwise |
| MCP sidecar venv | `pip install 'mcp<2'` | `pip install 'mcp==1.26.0'` | pinned, so reruns get the same library |

## Validation

Everything below ran on HF Sandbox against the generated tasks. Results are in [the parity notes](#parity).

- **Static checks** (`check.py`), on every task:
  - Harbor loads it;
  - the image is pinned by digest, and the Dockerfile uses the same one;
  - the setup archive equals `environment/setup/`;
  - no rubric is reachable before grading;
  - Cyber and General run as `agent`;
  - the directory matches its manifest hash.
- **Determinism:** two conversions give byte-identical manifests.
- **No-op agent** (Harbor's `nop`): every dataset scores 0, and every verifier runs to completion. There are no
  free rewards: untouched Code repos fail their hidden tests, and nothing delivered or submitted scores 0.
- **Real agent, both sides:** the same tasks with GLM-5.3 (deepinfra, thinking low) in the explorer and in Harbor.

## Parity

See `parity.md`; it is filled in from the runs.

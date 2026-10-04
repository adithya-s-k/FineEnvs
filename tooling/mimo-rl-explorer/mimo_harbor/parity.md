# Parity: explorer (original) vs Harbor

**Original side:** the MiMo RL Environment Explorer's runner (`app/runner/domains.py`), which follows Xiaomi's
harness.

**Harbor side:** the tasks this adapter generates, run with `agents/mimo_opencode.py` and `jobs/*.yaml`.

Both sides use the same settings:
- Agent: OpenCode 1.18.32.
- Model: `zai-org/GLM-5.3` via deepinfra, thinking low, replies up to 65,536 tokens.
- Step limit: the reference limit for each dataset.
- Sandbox: HF Sandbox, same flavors.
- Judges: Inkling (text) and Llama-4-Maverick (vision).
- Graders: the same vendored code.

## Sanity check: 1 task per dataset, 1 run per side (2026-09-26)

The first parity task of each dataset, chosen by `--split parity`.

| Dataset | Task | Explorer | Harbor | Notes |
|---|---|---:|---:|---|
| code | format-code-task-000192 | 0.0 | 0.0 | hidden tests fail on both sides |
| cyber | arvo_13428 | 1.0 | 1.0 | Harbor's earlier valid run scored 0: the model patched the bug instead of submitting a PoC |
| general | s3k_0057_accounting_audit_tax_en_t2_rl_008 | 0.625 | 0.625 | the same rubric score |
| terminal | candidate-0109-science-robotics | 0.0 | 1.0 | the explorer run passed 6 of 7 tests (one edge case failed); the task scores all or nothing |
| webdev | dasyn_260630_00561 | 0.76 | 0.747 | the same judge on two different sites |
| music | music-gK-0229 | 0.0 | 0.0 | both rejected by the validity gate for bars of the wrong length |

These runs cost about $2.40 on the explorer side and about the same on Harbor.

This is one run per side. It shows that both sides set up, run and grade each dataset the same way. It doesn't yet
measure parity, which Harbor defines as overlapping score ranges over at least 2 runs per side.

### What the sanity runs found and fixed

- **Webdev's grader couldn't import its rubric.** The vendored files use package-relative imports, so they now ship
  as the `webdev` package.
- **General's verifier printed the first characters of the judge key.** Harbor scrubs whole secrets only, so the
  grader now redacts key-shaped text in its output and in every verifier file.
- **Harbor blanked every `65536` in the results.** It treats env vars named `*TOKEN*` as secrets, and
  `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX` qualifies. The agent class now sets that variable itself.
- **A trial with no model access still records reward 0.** Harbor keeps the trial and scores it 0, with no verifier
  error. When comparing, look at `agent/opencode.txt`: 0 steps with an auth error means an infrastructure failure,
  not a model result.
- **The explorer's Music runner had been broken since the security audit** (`httpx.stream` with `extensions=`).
  It is fixed in `app/runner/domains.py`.

## Full parity: 36 tasks, 3 runs per side (2026-09-27)

The parity subset (`--split parity`): 6 tasks per dataset, fixed by the sha256 of the task id. Run 1 was done on both
sides before runs 2 and 3, as Harbor's parity guide asks. Each run's score is the mean reward over the tasks that side
scored. Scores are mean ± sample SEM over the three runs.

| Dataset | Tasks | Original (mean ± SEM) | Harbor (mean ± SEM) | Original runs | Harbor runs | Ranges overlap |
|---|---:|---:|---:|---|---|:---:|
| Code | 6 | 0.556 ± 0.056 | 0.556 ± 0.056 | 0.667, 0.500, 0.500 | 0.667, 0.500, 0.500 | yes |
| Cyber | 6 | 0.822 ± 0.011 | 0.833 ± 0.096 | 0.800, 0.833, 0.833 | 1.000, 0.667, 0.833 | yes |
| General | 6 | 0.844 ± 0.008 | 0.844 ± 0.024 | 0.842, 0.832, 0.859 | 0.890, 0.811, 0.832 | yes |
| Music | 6 | 0.111 ± 0.063 | 0.317 ± 0.093 | 0.000, 0.114, 0.219 | 0.138, 0.449, 0.363 | yes |
| Terminal | 6 | 0.278 ± 0.056 | 0.111 ± 0.056 | 0.333, 0.333, 0.167 | 0.167, 0.167, 0.000 | yes |
| Webdev | 6 | 0.749 ± 0.049 | 0.738 ± 0.054 | 0.650, 0.803, 0.793 | 0.799, 0.631, 0.785 | yes |

**All six datasets meet Harbor's criterion: the run-score ranges overlap.** Code, Cyber, General and Webdev agree
closely.

- **Terminal (0.278 vs 0.111):** each task is all or nothing, and outcomes flip in both directions on both sides
  (for example `candidate-0109` scored 0/1/1 in the explorer and 0/1/0 in Harbor). With six tasks, that is noise.
- **Music (0.111 vs 0.317):** the validity gate makes each piece close to pass or fail, and it flips run to run on
  both sides. Harbor passes the gate more often (7 of 18 pieces vs 4 of 18). That fits Music's one intended
  difference: in Harbor the agent writes its reply to a file with up to 4 steps, where the original is a single
  completion.

**Infrastructure failures were rerun, not counted.**
- Some first attempts timed out talking to the sandbox API (about 26 sandboxes ran at once): 9 original-side and 6
  Harbor-side tasks in run 1, fewer in later runs. Those were rerun at lower concurrency.
- `arvo_49407` run 1 failed on both sides the same way: the agent started 16 parallel fuzzers on a 2-vCPU
  `cpu-basic` sandbox and starved the sandbox server. It is excluded from run 1 on both sides.

The explorer side of these runs cost about $77; the Harbor side about the same. `parity_experiment.json` has the raw
numbers in Harbor's format.

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

## Next

Harbor's order:
1. the full parity subset (36 tasks) once on each side;
2. then 3 runs on each side;
3. report mean ± sample SEM per dataset, with the overlap criterion.

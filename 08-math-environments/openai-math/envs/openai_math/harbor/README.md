# Generator and verifier: technical notes

This directory produces the Harbor tasks in `../tasks/` from a checkout of
[openai/math](https://github.com/openai/math), builds the sandbox image they run in, and contains
the verifier. The dataset card (`../README.md`) covers what the environment is and how to run it;
this file covers how the tasks are generated and graded, the threat model, and the checks behind
the validation.

## Layout

```
openai_math_harbor/
  challenges.py      reads the challenges, maps them to families, applies the hole policy
  import_graph.py    import graph of the openai/math Lean library (reference-proof closures)
  oracle.py          closure and packaging of the reference proofs
  reference.py       places a reference proof as a submission (copied into each task's solution/)
  adapter.py         writes tasks/, registry.json, manifest.parquet, the card and NOTICE
  snapshot.py        builds the shared Daytona snapshot from image/
  report.py          turns a Harbor oracle job into per-task verdicts
  attacks.py         adversarial submissions used to test the verifier
  templates/         instruction, verifier (tests/grade.py, test.sh), oracle (solve.sh), card
image/               Dockerfile and setup scripts for the shared image (= every task's environment/)
tests/               unit tests for the verifier's submission handling
```

## Regenerating the tasks

```bash
# A sparse checkout of openai/math at the pinned commit:
git init openai-math && cd openai-math
git remote add origin https://github.com/openai/math.git
git sparse-checkout set --no-cone /lean/ComparatorChallenges/ /lean/docs/ /overview.tex /LICENSE
git fetch --depth 1 --filter=blob:none origin adc7f1241b42e322a6451854ab7e4b4c146bf78a && git checkout FETCH_HEAD
cd ..

# Optional: the import graph, for reference-proof sizes and the oracle's file lists. Needs the full
# lean/ directory (about 1.8 GB checked out).
python -m openai_math_harbor.import_graph openai-math/lean graph.json

python -m openai_math_harbor.adapter --openai-math openai-math --graph graph.json --out out/
# Stamp oracle verdicts into the tasks after an oracle run:
python -m openai_math_harbor.report <harbor job dir> oracle_results.json
python -m openai_math_harbor.adapter ... --oracle-results oracle_results.json
```

Every challenge goes through the same code; no task is written by hand. Three things are decided
per challenge, all from the data:

| What | Count | Handling |
|---|---|---|
| Definition holes | 9 configs | Comparator checks only the name and type of a `definition_names` entry, never its body, so a solution could redefine `ProperColoring := False` and prove the theorem trivially. A hole whose body is written out is removed from the config, so it must match exactly (8 configs). A hole whose body is `sorry` stays open only if its name appears nowhere else in the file (`ElementaryPositivity`, whose type is the claim). `DefocusingNLS` has a `sorry` inside a definition the theorems use and is excluded. |
| Reference proofs needing other Lean libraries | 23 | PrimeNumberTheoremAnd, StrongPNT, RellichKondrachov and others. Every statement imports only Mathlib, so agents are unaffected; only those official proofs cannot be built in this image. |
| Grading time | all | 3 h plus 15 min per MB of the reference proof, at most 6 h. |

The 137 result families without Lean statements have no machine-checkable target and are not part
of the dataset.

## How a trial runs

```
agent sandbox (snapshot)                    verifier sandbox (same snapshot, fresh)
  /workspace  Lake project, Mathlib           never runs agent code
  agent writes Submission.lean                 /tests: Challenge.lean, task.json, grade.py
  (+ Submission/**.lean)                       /logs/artifacts/submission.tar  <- only input
  no network while the agent runs                1. accept regular Submission*.lean files only
        |                                        2. fresh Lake project, prebuilt read-only Mathlib
  collect hook: pack_submission.py               3. iptables: grader uid gets no sockets
  -> /logs/artifacts/submission.tar ---------->  4. Comparator as `grader`: landrun builds, statement
     (one plain file crosses)                       + axiom check, kernel replay
                                                 5. kill grader processes, delete the build,
                                                    root writes reward.txt
```

Harbor's `[verifier] environment_mode = "separate"` (0.24+) provides the second sandbox. Both
sandboxes start from one Daytona snapshot built from `image/Dockerfile`; every task carries the
identical `environment/`, so the image is built once. Harbor's own Daytona auto-snapshot gives up
after 10 minutes, which is less than the image build takes, hence `snapshot.py`.

## Threat model

| Attempt | What stops it | Test |
|---|---|---|
| `sorry`, `admit`, a tactic macro that expands to `sorry` | `sorryAx` is not a permitted axiom | `sorry`, `sorry-macro` |
| Declare `axiom cheat : False` | only `propext`, `Quot.sound`, `Classical.choice` allowed | `axiom` |
| `implemented_by` + `native_decide` to prove `False` | `native_decide` introduces an axiom of its own | `native-decide-false` |
| Same theorem name, weaker statement | the statement must equal the challenge's exactly | `weaken-statement` |
| Redefine a definition the statement uses | every constant in the statement must match; openai/math's loose hole configs are tightened | `redefine-definition` |
| Keep a structure's name and type but drop a proof obligation from its fields | Comparator also compares the structure's constructor, whose type lists the fields | `tamper-structure` |
| Re-export the trusted Challenge module | its theorems are `sorry` | `import-challenge` |
| Plant `reward.txt` or a fake `comparator` in the agent sandbox | grading runs in a separate fresh sandbox | `agent-side-reward` |
| Lean code run while the verifier builds the proof (`run_cmd`, `#eval`, macros) writes `reward.txt`, runs a shell or opens a connection | landrun (read-only file system except `.lake`, execution limited to Lean and git, TCP denied), unprivileged `grader` user, iptables owner rule; root writes the reward only after killing every grader process | `compile-time-reward` |
| Exhaust the verifier's disk or memory so grading cannot finish | Lake by-products are emptied as the build runs; the build is deleted before the verdict is written, so a full disk still yields reward 0; memory and time are bounded (8 GB, `grade_timeout_sec`) and a killed build scores 0 | `fill-disk` |
| Symlinks, path traversal, device files, hard links, huge archives | the collect hook packs only regular files; `grade.py` accepts only regular `Submission.lean` and `Submission/**.lean` whose path components are Lean identifiers | `symlink`, `tests/test_grade_unpack.py` |
| Read OpenAI's proof | not in the image; the agent phase has no network | network probe (below) |
| A background process in the agent sandbox interfering with grading | Harbor stops the agent sandbox before the verifier starts and empties the verifier's log directory first; on Daytona the two sandboxes share no files | by construction |
| A Lean kernel soundness bug | not covered (Comparator's second kernel, nanoda, is off) | none |
| Memorized public proofs | not covered | none |

Grading fails closed: a rejected or non-compiling submission scores 0. If the trusted side breaks
(the Challenge fails to build, or times out before the submission is reached, or the network block
cannot be installed), no reward is written and Harbor reports the trial as an error rather than 0.

**What each isolation layer was shown to do** on Daytona's kernel (Linux 6.17): under landrun, the
grader's build cannot write outside the build directory and gets `EACCES` on TCP connects. Without
landrun, the grader user *can* reach Daytona's in-sandbox control service on port 2280, so
`grade.py` also installs an iptables rule for that user before Comparator runs and refuses to grade
if the rule does not take effect.

**Trusted, not re-checked here:** the Lean kernel, Comparator's export parser and `lean4export` (a
crafted `.olean` written at build time is only accepted if the kernel replay accepts it), landrun,
and Daytona's sandbox boundary.

## Checks behind the validation

All through `harbor run` on Daytona with the separate verifier; none involves a language model.

| Check | Expected | Result |
|---|---|---|
| Oracle over every task (official proofs) | 1 where the proof fits the image | see the dataset card and each task's `oracle_result` |
| All 405 statements built in one sandbox (`lake build`, 3 threads) | compile | 405 of 405; median 5 s, 90th percentile 7 s, slowest 220 s |
| No-op agent | 0 | 0, "no submission" |
| 12 adversarial submissions (`attacks.py`) | 0 | 0 on all twelve, each rejected for the intended reason (`fill-disk` on Oct 9 with the final grader: "ran out of disk space" after 85 s); the build-time probe printed `write denied`, and neither a shell nor git's HTTP helper could be executed |
| Agent-phase view (oracle agent printing its environment) | no network | DNS fails, HTTPS to github.com times out, a direct-IP connection is refused; `LEAN_NUM_THREADS=3` is set; `/tests` is absent |
| Oracle with `--allow-agent-host github.com` | 1 | 1 (positive control for the network check) |
| landrun and iptables in a Daytona sandbox | both enforce | as described above |

## The image

`image/lean_setup.sh` installs Lean v4.34.1 and Mathlib at the commits openai/math pins, builds
Comparator (`leanprover/comparator@d03acab`) and lean4export with the same toolchain, builds landrun,
copies every challenge statement to `/opt/openai-math/challenges/`, and prepares `/workspace`. Three
details matter:

- **Size.** Lean plus Mathlib is about 11 GB as installed, more than a 10 GB Daytona sandbox. The
  script drops what importing Mathlib never reads: static libraries, LLVM and gcc (needed only to link
  the executables built earlier in the script), and Mathlib's generated C and `.ilean` files,
  truncated with their `.hash` files kept so Lake still treats Mathlib as built. A smoke test fails
  the build if a second `lake build` still compiles any dependency.
- **Threads.** Lean sees every host CPU (64 on Daytona), not the sandbox's 4, so by default Lake
  starts one Mathlib-loaded `lean` per module and runs out of memory. Agents get
  `LEAN_NUM_THREADS=3` (`task.toml`); the verifier uses 2, which reaches Comparator's landrun
  sandbox through a one-line `sed` patch to Comparator's `envPass` lists.
- **Permissions.** Mathlib's cache tool unpacks files as mode 0600. The script makes the toolchain
  and packages world-readable (still root-owned), and a second smoke test builds as the `grader` user.

## Limits

- **Daytona's per-sandbox caps: 4 CPUs, 8 GB memory, 10 GB disk.** The 10 GB is working space on
  top of the image. Lake compiles every module to C as well as to `.olean`; grading never reads the
  C, but for the largest proofs it reached 9.4 GB and filled the disk. `grade.py` therefore empties
  the generated `.c` files every 15 seconds while the build runs (as the `grader` user; Lake keeps
  the recorded hash and does not rebuild), and deletes the build before writing its verdict, so a
  build that still fills the disk scores 0 instead of leaving Harbor waiting for a verifier that
  can no longer report. Some proofs still need more than 8 GB of memory or 6 hours; the oracle
  results record which. Higher per-sandbox limits remove the constraint.
- **The agent sandbox contains Comparator,** since agent and verifier share one image. Nothing in it
  is secret: the statement and the rules are public, and `lake build` plus `#print axioms` already
  tell an agent whether a proof is complete.
- **Difficulty.** These are open research problems; expect a reward of 0 on almost every task.

## Licensing

The statements are redistributed unchanged from openai/math (OpenAI, Apache-2.0); the dataset root
carries `LICENSE-openai-math` and a `NOTICE` describing what was copied, what was adapted (the
Comparator configs) and what is not included (the proofs). The image bundles Lean, Mathlib,
Comparator and lean4export (Apache-2.0) and landrun (MIT). This code is under the Apache License 2.0.

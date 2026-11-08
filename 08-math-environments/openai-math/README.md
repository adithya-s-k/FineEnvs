<div align="center">

<h1>openai/math as Harbor tasks</h1>

<h3>Prove a research-level result in Lean 4, graded by Comparator in a separate sandbox</h3>

<p>369 statements from OpenAI's openai/math release whose official proof the grader accepts end to end.</p>

<a href="https://huggingface.co/datasets/FineEnvs/openai-math"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-369%20Harbor%20tasks-4F46E5?style=for-the-badge&labelColor=1a1a1a" alt="Dataset" height="32"></a>
<a href="https://github.com/harbor-framework/harbor"><img src="https://img.shields.io/badge/framework-Harbor-3B82F6?style=for-the-badge&labelColor=1a1a1a" alt="Harbor" height="32"></a>
<a href="https://github.com/adithya-s-k/FineEnvs/tree/main/08-math-environments/openai-math"><img src="https://img.shields.io/badge/GitHub-08--math--environments%2Fopenai--math-181717?style=for-the-badge&logo=github&labelColor=1a1a1a" alt="Code on GitHub" height="32"></a>

</div>

---

> **Experimental.** Built on a research release that is itself new and not yet independently reviewed. Read any
> proof that scores 1 before relying on it.

On 6 October 2026 OpenAI released [openai/math](https://github.com/openai/math): 722 manuscripts written by an
internal model, grouped into 372 result families, with Lean 4 formalizations for 235 families. For 405 results the
repository ships a *Comparator challenge*: the statement in Lean with `sorry` for the proof, and a config for
[Comparator](https://github.com/leanprover/comparator), the Lean FRO's proof checker.

This project turns those statements into [Harbor](https://github.com/harbor-framework/harbor) tasks that any coding
agent can attempt. The agent gets a sandbox with Lean 4 and a prebuilt Mathlib, no internet, and the statement with
`sorry` where the proof should be. A separate sandbox then runs Comparator: the reward is **1** if the proof is
accepted (exactly the original statement, standard axioms only, checked by the Lean kernel) and **0** otherwise.

- Tasks on the Hub: [FineEnvs/openai-math](https://huggingface.co/datasets/FineEnvs/openai-math) (369 tasks, the
  dataset card lists the rest with the reason each was left out)
- Generator, grader and image: [`envs/openai_math/harbor/`](envs/openai_math/harbor/)
- How the tasks were curated, and why: [DESIGN.md](DESIGN.md)
- Grading design, threat model and limits: [`envs/openai_math/harbor/README.md`](envs/openai_math/harbor/README.md)

<!-- BEGIN:matrix -->
| Env | Tools | Backend | `harbor` |
|---|---|---|---|
| **openai_math** | shell (Lean, Lake, Mathlib) | `daytona` | ✅ |
<!-- END:matrix -->

## Quick start

You need Harbor 0.24+ with the Daytona backend (`DAYTONA_API_KEY`, `DAYTONA_API_URL`) and credentials for the agent
you run.

```bash
pip install "harbor[daytona]>=0.24" daytona
hf download FineEnvs/openai-math --repo-type dataset --local-dir openai-math
cd openai-math

# Build the shared sandbox image once (Lean 4, Mathlib, Comparator; ~9 GB, about an hour).
(cd generator && python -m openai_math_harbor.snapshot)
SNAP=$(cd generator && python -m openai_math_harbor.snapshot --print-name)

# Run any agent Harbor supports on a few tasks (task names are the folders under tasks/):
harbor run -p tasks -e daytona --ek snapshot_template_name=$SNAP \
  -a claude-code -m anthropic/claude-opus-5-5 -i plane-coloring -i hilbert-crouzeix -n 2

# Check that the grader accepts OpenAI's own proof of a task (the oracle fetches it from GitHub):
harbor run -p tasks -e daytona --ek snapshot_template_name=$SNAP \
  -a oracle --allow-agent-host github.com -i plane-coloring
```

Each task asks for 4 CPUs, 8 GB of memory and 10 GB of disk, gives the agent 4 hours, and allows up to 6 hours for
grading. An accepted official proof took 8 minutes to grade at the median.

## What a task looks like

`instruction.md` gives the agent the result family, the full Lean statement and the rules. `/workspace` is a Lake
project with Mathlib prebuilt; `lake build Submission` checks a proof in about 15 seconds. The agent writes
`/workspace/Submission.lean` (optionally splitting long proofs into `Submission/*.lean` modules), starting from a copy
of the statement and replacing every `sorry`.

The grader accepts the submission only if

- every listed theorem has **exactly** the challenge's statement (names, binders, types, and every definition and
  structure the statement depends on),
- the proofs use no axioms beyond `propext`, `Quot.sound` and `Classical.choice` (so `sorry`, `admit`, new `axiom`s
  and `native_decide` all fail), and
- the Lean kernel re-checks the proof, independently of the elaborator.

## How grading is isolated

```
agent sandbox                                   verifier sandbox (fresh, same image)
  /workspace, Mathlib, no network                 never ran agent code
  agent writes Submission.lean                     only input: submission.tar
        │                                          1. accept regular Submission*.lean files only
  after the agent: pack only the .lean files ───►  2. fresh Lake project, read-only Mathlib
  into one archive                                 3. grader user: no sockets (iptables), landrun
                                                   4. Comparator: statement match, axiom check,
                                                      kernel replay
                                                   5. delete the build, then root writes reward.txt
```

Harbor's `[verifier] environment_mode = "separate"` provides the second sandbox, so nothing the agent does in its
own sandbox (planting reward files, replacing Comparator, background processes) reaches the verifier. The threat
model, with one probe per attack, is in the [technical README](envs/openai_math/harbor/README.md#threat-model).

## Validation

Harbor's oracle agent submits OpenAI's own proof of each task and has it graded by the same verifier an agent faces.
Only tasks whose official proof was accepted are published. All runs on Daytona, 7 to 9 October 2026. 365 of the
369 were re-checked with the final grader; the other four (`sk-ratio`, `triangular-energy`, `triangular-gaussian`,
`uniform-commutator`) were accepted by the previous version, and their re-check was cut short by a network drop.

| Oracle result | Statements | Meaning |
|---|---:|---|
| accepted | 369 | published |
| missing-package | 21 | the official proof needs a Lean library beyond Mathlib (PrimeNumberTheoremAnd, RellichKondrachov, …); agents are unaffected, since every statement imports only Mathlib |
| memory | 10 | the official proof needs more than the sandbox's 8 GB to build |
| definition-mismatch | 2 | the official proof restates a definition that elaborates differently in its own library, so the exact-match check fails |
| timeout | 1 | `erdos-reciprocal`: grading exceeded 6 hours |
| disk | 1 | `mub-six`: 205 MB of Lean whose compiled output exceeds the 10 GB disk |
| left out by design | 1 | `DefocusingNLS`: a `sorry` inside a definition its theorems depend on |

| Other check | Expected | Result |
|---|---|---|
| 12 reward-hacking probes (`sorry` in several disguises, fake axiom, `native_decide`, weakened statement, redefined definition, tampered structure, re-exported challenge, planted reward files, build-time code, symlinks, a build that fills the disk) | 0 | 0 on all twelve with the final grader, each rejected for the intended reason |
| An agent that does nothing | 0 | 0 |
| Agent-phase network | none | DNS, HTTPS and direct IP all fail; the positive control (`--allow-agent-host github.com`) scores 1 |
| All 405 challenge statements built in one sandbox | compile | 405 of 405, median 5 s, slowest 220 s |

No model has been evaluated at scale yet. These are open research problems, so a reward of 0 on nearly every task
is the expected result; the value is in the rare 1 and in the trajectories.

## Limits

- **Daytona sandbox caps (4 CPUs, 8 GB, 10 GB disk).** They decide which official proofs grade (see the table).
  Lake writes a setup file and a C file for every module it builds; the grader empties them once each module is done
  and deletes the build before writing its verdict, so a build that still fills the disk scores 0 instead of
  leaving Harbor waiting for a verifier that can no longer report.
- **Lean parallelism.** Lean sees every host CPU, not the sandbox's 4; tasks set `LEAN_NUM_THREADS=3` for agents and
  the grader uses 2, retrying once single-threaded if a build runs out of memory.
- **A strict reward.** Only a complete proof of the statement exactly as formalized scores. A correct proof of an
  equivalent statement, or partial progress, scores 0. [FineProofEnvs](../fineproof/) is the other end of that trade-off.
- **Contamination.** The statements *and proofs* have been public since 6 October 2026. Report the model's training
  cutoff with every result.
- **One kernel.** Comparator's optional second kernel (nanoda) is off. The trust base is the Lean kernel,
  Comparator's export parser and `lean4export`, landrun, and Daytona's sandbox boundary.

## A note on openai/math's Comparator configs

Comparator lets a challenge leave *definition holes*: `definition_names` entries whose body the solver supplies. By
design it checks only a hole's name and type, never its body, and Comparator's README says such solutions need an
additional check.

Nine of the 405 configs use `definition_names`, and in eight of them the listed definitions are written out in the
challenge rather than left open. Run as shipped, a solution could redefine one of them (for instance
`ProperColoring := False` in the five-colouring challenge) and prove the theorem trivially. This looks like a
config-generation oversight, not a flaw in the mathematics or in Comparator.

The generator handles it from the files alone: a hole whose body is written out is removed from `definition_names`,
so the solution's definition must match exactly; a hole whose body is `sorry` and whose name appears nowhere else in
the file stays open (one case, `ElementaryPositivity`, where the hole's type is the claim); and the one challenge
with a `sorry` inside a definition its theorems use (`DefocusingNLS`) is left out. The redefinition and a tampered
structure are among the probes above, and both are rejected.

## Licensing and attribution

- **Statements** (`tests/Challenge.lean` in each task, quoted in `instruction.md`) are copied unchanged from
  [openai/math](https://github.com/openai/math) at commit `adc7f12`, © OpenAI, under the
  [Apache License 2.0](https://github.com/openai/math/blob/main/LICENSE). The dataset carries that license
  (`LICENSE-openai-math`) and a `NOTICE`. Configs were adapted as described above; each task records the change in
  `task.toml` (`fixed_definitions`).
- **Proofs** are OpenAI's and are not redistributed. The oracle fetches them from GitHub at run time.
- **Tooling** in the image: [Lean 4](https://github.com/leanprover/lean4) and
  [Mathlib](https://github.com/leanprover-community/mathlib4) (Apache-2.0),
  [Comparator](https://github.com/leanprover/comparator) and [lean4export](https://github.com/leanprover/lean4export)
  (Apache-2.0, with a one-line patch to pass `LEAN_NUM_THREADS` into its sandbox),
  [landrun](https://github.com/Zouuup/landrun) (MIT).
- **This project's code** is under the repository's Apache License 2.0.
- If you build on the results, cite the openai/math manuscripts as that repository asks.

## Reproducing the dataset

```bash
cd envs/openai_math/harbor
# A sparse checkout of openai/math with lean/ComparatorChallenges, lean/docs, lean/OAI and overview.tex.
python -m openai_math_harbor.import_graph /path/to/openai-math/lean graph.json
python -m openai_math_harbor.adapter --openai-math /path/to/openai-math --graph graph.json --out out/openai-math
# Oracle over every task, then keep only the validated ones and stage the Hub folder:
harbor run -p out/openai-math/tasks -e daytona --ek snapshot_template_name=$SNAP -a oracle --allow-agent-host github.com -o jobs --job-name oracle
python -m openai_math_harbor.report jobs/oracle oracle_results.json
python -m openai_math_harbor.adapter ... --oracle-results oracle_results.json --only-validated --out out/release
python -m openai_math_harbor.package out/release out/hub
# The reward-hacking probes, and the grader's unit tests:
python -m openai_math_harbor.attacks --dataset out/openai-math/tasks --out out/attacks
python -m unittest discover -s tests
```

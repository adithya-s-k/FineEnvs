---
license: apache-2.0
task_categories:
- other
tags:
- harbor
- rl-environment
- reinforcement-learning
- code-agent
- lean4
- mathlib
- theorem-proving
- openai-math
pretty_name: "openai/math as Harbor environments"
---

# openai/math as Harbor environments

> **Experimental.** Built on a research release that is itself new and not yet independently
> reviewed. Expect rough edges, and read any proof that scores 1 before relying on it.

We ported the math problems from OpenAI's [openai/math](https://github.com/openai/math) release
(commit `$source_commit`) into [Harbor](https://github.com/harbor-framework/harbor) environments.
The release formalizes **$n_statements** of its results as Lean statements. This dataset contains
**$n_tasks environments**, the ones whose official proof we verified end to end in the environment,
covering $n_families result families across $n_subjects fields. The other $n_not_included are listed
under [Validation](#validation) with the reason each could not be validated.

## How a task works

The agent gets a sandbox with Lean 4 and a prebuilt Mathlib, no internet, and one of OpenAI's
theorems written in Lean with `sorry` where the proof should be.

**What `sorry` means.** In Lean, `sorry` is a placeholder for a missing proof: it lets the file
compile as if the step were proven, but nothing has actually been shown, and Lean flags it. The
statement is given exactly as OpenAI formalized it, and the agent's job is to replace every `sorry`
with a real proof.

When the agent finishes, a separate, fresh sandbox checks its proof with
[Comparator](https://github.com/leanprover/comparator), the Lean FRO's proof checker:

- **reward 1** if the theorem has exactly the original statement, the proof uses only Lean's
  standard axioms (`propext`, `Quot.sound`, `Classical.choice`, so no leftover `sorry` and no new
  axioms), and the Lean kernel accepts it;
- **reward 0** otherwise.

OpenAI's own proofs serve as the reference solutions: Harbor's oracle fetches them from GitHub at
run time. They are not included here and the agent never sees them.

## A strict reward

The grader only accepts a complete proof of the statement exactly as OpenAI formalized it. That
makes it hard to game, but it is also a real limitation for problems like these:

- a correct proof of an equivalent statement, written with different definitions or a different
  formulation, scores 0;
- partial progress (useful lemmas, a proof of a special case, a correct strategy with a gap) scores 0;
- nothing rewards a creative approach unless it ends in a complete proof of this exact statement.

For open research problems, a more general reward function that recognizes any valid solution to the
underlying problem, including creative reformulations and partial progress, would be a better
training signal. We use the strict check here because it is precise and cannot be fooled; treat it
as one end of that design space.

## What's included

- `tasks/` — one Harbor task per validated problem: instruction, Lean statement, sandbox image recipe,
  verifier and oracle
- `registry.json` — Harbor registry entry (`openai-math@1.0`)
- `manifest.parquet` — one row per task: field, family, theorem names, reference-proof size, oracle
  result
- `generator/` — the code that builds the sandbox image and regenerates the tasks from openai/math
- `LICENSE`, `NOTICE`, `LICENSE-openai-math` — licensing and attribution

## Run it with the Harbor CLI

You need Harbor 0.24+ with the Daytona backend (`DAYTONA_API_KEY`, `DAYTONA_API_URL`) and
credentials for the agent you run.

```bash
pip install "harbor[daytona]>=0.24" daytona
hf download FineEnvs/openai-math --repo-type dataset --local-dir openai-math
cd openai-math

# Build the shared sandbox image once (Lean 4, Mathlib, Comparator; ~9 GB, about an hour).
(cd generator && python -m openai_math_harbor.snapshot)
SNAP=$$(cd generator && python -m openai_math_harbor.snapshot --print-name)

# Run any agent Harbor supports (`harbor agent list`) on a few tasks:
harbor run -p tasks -e daytona --ek snapshot_template_name=$$SNAP \
  -a <agent> -m <model> -i plane-coloring -i hilbert-crouzeix -n 2

# Check that the grader accepts OpenAI's own proof of a task:
harbor run -p tasks -e daytona --ek snapshot_template_name=$$SNAP \
  -a oracle --allow-agent-host github.com -i plane-coloring
```

Each task asks for 4 CPUs, 8 GB of memory and 10 GB of disk, and gives the agent 4 hours. Task
names are the folder names under `tasks/`.

## Validation

We generated a task for every statement except one left out by design (`DefocusingNLS`, whose
statement has a `sorry` inside a definition its theorems depend on), and ran Harbor's oracle on each:
it submits OpenAI's own proof and has it graded by the same verifier an agent faces. Only tasks whose
official proof was accepted are included.

$validation

$not_validated

The verifier was also tested against submissions that try to cheat (`sorry` in several disguises,
fake axioms, `native_decide`, weakened statements, redefined definitions, planted reward files, code
that runs during grading, a build that fills the verifier's disk). All scored 0. A no-op agent scores 0.

## Notes

- **Configs.** In eight of openai/math's Comparator configs, definitions written out in the statement
  were listed as "holes" that Comparator does not check, so a solution could redefine them. Here they
  must match exactly; affected tasks list them under `fixed_definitions` in `task.toml`.
- **Contamination.** The statements and proofs have been public since 6 October 2026.
- **Difficulty.** These are open research problems; expect reward 0 on almost every task.

Technical details (grading design, threat model, image recipe) are in `generator/README.md`.

## License

- The Lean statements are copied unchanged from [openai/math](https://github.com/openai/math)
  (© OpenAI), licensed under the
  [Apache License 2.0](https://github.com/openai/math/blob/main/LICENSE); see `LICENSE-openai-math`
  and `NOTICE`. OpenAI's proofs are not redistributed.
- The packaging, verifier and generator are Apache-2.0 (`LICENSE`).
- The sandbox image is built from [Lean 4](https://github.com/leanprover/lean4) and
  [Mathlib](https://github.com/leanprover-community/mathlib4) (Apache-2.0),
  [Comparator](https://github.com/leanprover/comparator) and
  [lean4export](https://github.com/leanprover/lean4export) (Apache-2.0), and
  [landrun](https://github.com/Zouuup/landrun) (MIT).
- If you use the results, cite the openai/math manuscripts as that repository asks.

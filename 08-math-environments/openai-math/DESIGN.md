# How the openai/math tasks were curated

The decisions behind the 369 published tasks, in the order they came up. The [README](README.md) has what the tasks
are and how to run them; the [technical README](envs/openai_math/harbor/README.md) has the grader's threat model.

## 1. Which results become tasks

openai/math has 722 manuscripts in 372 result families. Only some of them can be graded by a machine:

| Layer | Count | Gradable? |
|---|---:|---|
| manuscripts | 722 | no, prose |
| result families | 372 | — |
| families with a Lean formalization | 235 | partly |
| Comparator challenges (statement with `sorry` + config) | 405 | **yes** |

A Comparator challenge is the only object with an unambiguous success condition: the statement is fixed in Lean and
Comparator decides whether a proof proves exactly that. So a task is a challenge, one per challenge, and the 137
families without Lean are left out rather than graded by a judge. Every task comes out of one generator
(`adapter.py`); none is written by hand, so a fix to the generator fixes every task.

## 2. Why Harbor

The agent needs a real Lean toolchain: it edits files, runs `lake build`, searches Mathlib and iterates for hours.
That is a coding-agent task in a sandbox, which is what Harbor is for, and Harbor 0.24 can grade in a **separate
sandbox** that never ran agent code. An in-process environment (OpenEnv, Verifiers) would have had to rebuild both.

## 3. Assume the agent is root in its sandbox

Rather than list what an agent might try, the design assumes it controls its sandbox completely, then makes that
irrelevant to the grade:

- **One file crosses.** After the agent finishes, a collect hook packs only regular `Submission.lean` and
  `Submission/**.lean` files into one archive. The verifier re-checks every entry (no links, no paths outside
  `Submission/`, UTF-8, size limits).
- **The verifier is fresh.** It starts from the same image, never ran agent code, and gets the trusted statement
  from `tests/`, not from the agent.
- **Comparator decides.** Exactly the challenge's statement (names, binders, types, and every definition and
  structure it depends on), only `propext`, `Quot.sound` and `Classical.choice`, then a kernel replay independent of
  the elaborator.
- **Code that runs at build time is contained.** Lean can execute code while building (`#eval`, `run_cmd`, macros),
  so the build runs as an unprivileged user under landrun (writes only to `.lake`, no TCP), and an iptables rule
  blocks that user's sockets. That rule matters: without it, the user could reach the sandbox's control daemon.
- **Probe every assumption.** Each defence has a submission that tries to get past it (`attacks.py`, twelve
  probes). All score 0, each for the intended reason.

## 4. Fix the configs before trusting them

Reading the configs turned up the one real hole. Comparator's `definition_names` lets the solver supply a
definition's body, and checks only its name and type. At the pinned commit `adc7f12`, eight configs list
definitions that are written out in the statement, so a solution could redefine one (`ProperColoring := False`) and
prove the theorem trivially. openai/math has since fixed these configs upstream in `42907c9` (26 October 2026, PR
#3); the generator fix is still needed for the pinned commit. The generator decides from the files alone:

- a written-out definition loses its hole, so it must match exactly;
- a `sorry`-bodied hole stays open only if its name appears nowhere else in the file (`ElementaryPositivity`, where
  the hole's type is the claim);
- `DefocusingNLS`, with a `sorry` inside a definition its theorems use, is left out.

Each affected task records the change under `fixed_definitions` in `task.toml`. Five of the published tasks have
one (`Brenier`, `EuclideanFiveColor`, `KServer`, `Naimark`, `Rokhlin`); of the other three corrected configs,
`OccupiedOverlap` and `SpinAngle` did not validate (section 6) and `DefocusingNLS` is left out.

## 5. Make every task solvable in the image, and prove it

A task is only useful if a correct proof can pass. So the official proofs serve as the oracle: for each challenge,
the import graph of openai/math gives the closure of modules the official proof needs; Harbor's oracle agent fetches
exactly those files from GitHub at run time (they are never redistributed), places them under `Submission/`, and the
real verifier grades them. A task is published only if that proof is accepted.

The image pins what openai/math pins (Lean v4.34.1, Mathlib `d13f23b`) and builds Comparator, lean4export and
landrun with the same toolchain. Three details decided whether official proofs could pass at all:

- **Size.** Lean plus Mathlib is about 11 GB installed, more than a 10 GB sandbox, so the image drops what importing
  Mathlib never reads (static libraries, LLVM, generated C) while keeping the hashes Lake checks.
- **Threads.** Lean sees every host CPU (64), so Lake starts one Mathlib-loaded process per module and runs out of
  memory. `LEAN_NUM_THREADS` bounds it, which needed a one-line patch so Comparator passes it into its sandbox.
- **Permissions.** Mathlib's cache unpacks files readable by root only; the unprivileged grader could not read them.

## 6. What the oracle runs taught us

Running the oracle over all 404 tasks was the real test of the grader, and most of what it found was in the grader,
not in the proofs:

| Symptom | Cause | Fix |
|---|---|---|
| many proofs "out of disk" | a `chmod -R` in the grader copied 8.7 GB of the image into the sandbox's writable layer | change only what needs changing (an `lstat` walk); re-grade the affected tasks |
| big proofs "out of disk", grading hangs for hours | Lake leaves a 5.5 MB setup file and a C file per module (10 GB at 1,800 modules); with the disk full the sandbox can't report that the verifier exited | empty both once each module is built; delete the build before writing the verdict; a probe now fills the disk on purpose |
| out of memory on heavy modules | two Lean processes at once on one module | retry once single-threaded in a fresh directory |
| `Submission/…/Kappa₁.lean` rejected | file-name check allowed ASCII only | accept any Lean identifier |
| grading outlived its limit | `timeout` signalled only its direct child | kill the whole process tree |
| small proofs timed out | a few heavy tactic calls take 45+ minutes | 3 h grading floor, plus 15 min per MB of proof |

What remained after the fixes were genuine limits, each listed on the dataset card: 21 official proofs need Lean
libraries beyond Mathlib (agents are unaffected: every statement imports only Mathlib), 10 need more than 8 GB, and 2
restate a definition that elaborates differently in their own library. Two more (`erdos-reciprocal`, `mub-six`) did
not complete in the Harbor oracle run and are listed as "not completed"; a detached re-run of the same grader
outside Harbor showed that one takes more than 6 hours and the other is too big for the disk.

## 7. Publish only what was validated, and say what was not

The dataset holds the 369 tasks whose official proof the grader accepted. The other 35, and `DefocusingNLS`, are
listed on the card with the reason each was left out. The proofs stay on GitHub; the statements are copied unchanged
under openai/math's Apache-2.0 license, with its `LICENSE` and a `NOTICE` that also records the config fix.

## 8. What the strictness costs

The reward is 1 only for a complete proof of the exact statement. That makes it hard to game and easy to trust, but
a correct proof of an equivalent statement, or real partial progress, scores 0, and on open research problems
almost every attempt will score 0. A reward that recognises valid alternative formulations and partial progress
would be a better training signal; [FineProofEnvs](../fineproof/) explores that end, with a rubric judge.

import Lake
open Lake DSL

-- Mathlib at the commit openai/math pins. Agents write `Submission.lean` and `Submission/**.lean`.
package openai_math where
  leanOptions := #[⟨`autoImplicit, false⟩]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "d13f23b723b8a846827a245b89c10fc7d3f11612"

-- `Submission` and every `Submission.*` module it imports.
lean_lib Submission

# Vendored code

Unmodified copies of Xiaomi's reference graders and harness pieces, so rollouts here are set up and scored the
way MiMo-V2.6 was trained and evaluated. Pinned to XiaomiMiMo/verl `a2ad9f6` and XiaomiMiMo/mimoagent `467f0a1`.

| Path | Source | License |
|---|---|---|
| `server_arvo.py` | mimoagent `src/mimoagent/environments/datasets/resources/server_arvo.py` | MIT (`LICENSE-mimoagent.md`) |
| `music_scorer/` | verl `recipes/design/music/scorer/` | Apache-2.0 (`LICENSE-verl.txt`) |
| `webdev/eval_rubric.py` | verl `recipes/design/webdev/eval_rubric.py` | Apache-2.0 |
| `webdev/verdict.py` | verl `recipes/design/webdev/eval_mode.py` `parse_verdict` | Apache-2.0 |
| `webdev/shot.py` | verl `recipes/design/webdev/shot.py`, byte for byte (the in-pod full-page render) | Apache-2.0 |
| `webdev/agent_prompt.txt` | verl `config/agent/design/webdev-eval.yaml` `system_template`, minus its "You are Claude Code" line | Apache-2.0 |

Also ported verbatim, inline in `app/runner/domains.py`: mimoagent's `_RESIDUE_SCRUB_GLOBAL`, `_GLOBAL_CACHE_SCRUB`,
the `git clean` keep list, `get_patch_touched_files` and `build_reset_test_files_cmd` (base.py), `_parse_description`
(arvo.py, in `app/catalog.py`), and the step limits and instruction templates from verl's `config/agent/*`.

The General-domain verifiers (`verify.py`, `run_verify.py`) are not vendored: every task ships its own, and they
are uploaded into the sandbox from the dataset after the agent has finished.

# Vendored code

Unmodified copies of Xiaomi's reference graders, so rollouts here are scored the way MiMo-V2.6 was trained and evaluated.

| Path | Source | License |
|---|---|---|
| `server_arvo.py` | [XiaomiMiMo/mimoagent](https://github.com/XiaomiMiMo/mimoagent) `src/mimoagent/environments/datasets/resources/server_arvo.py` | MIT (`LICENSE-mimoagent.md`) |
| `music_scorer/` | [XiaomiMiMo/verl](https://github.com/XiaomiMiMo/verl) `recipes/design/music/scorer/` | Apache-2.0 (`LICENSE-verl.txt`) |
| `webdev/eval_rubric.py` | [XiaomiMiMo/verl](https://github.com/XiaomiMiMo/verl) `recipes/design/webdev/eval_rubric.py` | Apache-2.0 (`LICENSE-verl.txt`) |

The General-domain verifiers (`verify.py`, `run_verify.py`) are not vendored: every task ships its own, and they
are uploaded into the sandbox from the dataset after the agent has finished.
| `webdev/agent_prompt.txt` | XiaomiMiMo/verl `config/agent/design/webdev-eval.yaml` `system_template`, minus its "You are Claude Code" line | Apache-2.0 |
| `webdev/verdict.py` | XiaomiMiMo/verl `recipes/design/webdev/eval_mode.py` `parse_verdict` | Apache-2.0 |

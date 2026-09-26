---
title: MiMo RL Environments Explorer
emoji: 🧭
colorFrom: indigo
colorTo: green
sdk: static
pinned: false
license: apache-2.0
short_description: Browse and search the 7,780 MiMo-V2.6 RL environments
tags:
  - reinforcement-learning
  - rl-environments
  - agents
  - dataset-explorer
datasets:
  - XiaomiMiMo/MiMo-V2.6-RL-oss
---

# MiMo RL Environments Explorer

An unofficial explorer for [XiaomiMiMo/MiMo-V2.6-RL-oss](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss):
7,780 RL environments across five domains, each checked a different way.

| Domain | Environments | The task | Checked by | Browse by |
|---|---:|---|---|---|
| Code | 2,698 | Fix real issues in real repos | Executable tests | programming language |
| Webdev | 2,093 | Build a website from a brief | Visual grading | site type, framework, style, brief language |
| Cyber | 1,000 | Reproduce a real crash (ARVO) | Rule checks | crash type, project, sanitizer |
| Music | 1,000 | Compose in ABC notation | Rule checks | family, style, meter, voices, tempo |
| General | 989 | Knowledge work in a simulated workplace | Rubric judging | industry, tier, brief language |

Search everything, filter by category, or click a block in the map. Open an environment for its full
brief and what makes it tick: the mock enterprise systems (MCP servers) and workspace files of a
General environment, the files a Code task's tests touch, the crash a Cyber task reproduces.

## How the categories are made
Most come straight from the dataset's metadata. Where it has none, they are derived:

- **Code language** is the language of the files each task's test patch touches.
- **General industry and tier** are encoded in each environment's id (`s3k_<n>_<industry>_<lang>_<tier>_rl_<n>`).
- **Webdev site type, framework and style** are read from the brief's own wording and labelled "mentioned".
  The site type is the earliest specific kind of site the brief names.
- **Music styles** are the dataset's own tags, translated from Chinese and grouped into families.

Rubric questions are shown with their tier and weight. Their answers (`pass_anchor`, `gold_answer`) are not.

## Rebuild
```bash
uv run build_data.py      # fetches the dataset and writes data/
```

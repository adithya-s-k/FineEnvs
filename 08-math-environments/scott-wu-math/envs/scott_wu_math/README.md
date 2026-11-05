---
title: Scott Wu Math
emoji: 🃏
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Scott Wu's card game as an OpenEnv RL environment
tags:
  - openenv
  - reinforcement-learning
  - math
  - reasoning
---

# Scott Wu Math

Deal some playing cards, name a number, and make that number from the cards with + − × ÷, using every
card exactly once. People and language models play the same environment: the page at `/` drives the
same OpenEnv session over `/ws` that an agent does.

Named after the game Scott Wu plays in the clip below, as a fan tribute. It is not affiliated with or
endorsed by Scott Wu or Cognition.

## Where it comes from

In a clip filmed at a pub, Cognition's CEO Scott Wu explains the game he plays with a deck of cards.
It starts as the 24 game: four cards, make 24. "Once that got too easy", you take eight cards and make
the current year. Dealt K Q J 10 7 9 5 5, he says it is quite easy because 13 − 12 is 1, 11 − 10 is 1,
1 + 1 + 7 is 9, and 9 × 9 × 5 × 5 is exactly 2025. Then someone names a four-digit number, 6,843, and
with nine cards (6 4 Q 10 10 8 J 7 3) he goes: 6 + 4 is 10, 12 − 10 is 2, 2 × 10 × 10 is 200, plus 8 is
208, times 11 is 2,288, minus 7 is 2,281, times 3 is 6,843.

Both hands are in the `video` split, as he played them.

## Rules

- Use every card exactly once. A = 1, J = 11, Q = 12, K = 13.
- Combine with + − × ÷ and parentheses. Division is exact, so fractions along the way are allowed, as in the 24 game.
- No joining cards into longer numbers, no powers, no unary minus.
- One graded answer per deal.

## Levels

| Level | Cards | Target | Notes |
|---|---|---|---|
| `24` | 4 | 24 | The classic 24 game. About a third of random deals have no solution; those are redealt. |
| `three-digit` | 6 | 100–999 | A step between 24 and the year. |
| `year` | 8 | 1900–2100 | The play page and a bare `reset()` use the current year, as in the video. |
| `four-digit` | 9 | 1000–9999 | "Give me your favourite four-digit number". |
| custom | 4–9 | any up to 1,000,000 | `reset(target=6843)`; card count by size unless `n_cards` is given. |

Cards come from one 52-card deck, so a value appears at most four times. Every generated deal is checked
with the solver and redealt until it has a solution, so a task is never impossible.

## Episode

```
reset(...)  ->  observation: cards, target, prompt          (done = false)
step(CardAction(expression="..."))  ->  reward, result, a reference solution   (done = true)
```

`reset` arguments, all optional:

| Argument | Meaning |
|---|---|
| `split`, `index` | A fixed task from `train` (4,000), `test` (200) or `video` (2). Without `index`, a random one. |
| `level` | A fresh deal at that level. |
| `target`, `n_cards` | A fresh deal for your own number. |
| `cards` + `target` | Your own hand, as values 1–13. It may have no solution. |
| `seed` | Makes a fresh deal reproducible. |
| `reward_mode` | `shaped` (default) or `binary`. |

The observation carries `prompt` and `system_prompt`, the exact text a model reads. The answer goes in
`CardAction.expression`; an empty expression gives up. After the step, the observation adds `result`
(the grader's report), `solution` and `solution_steps` (one way to make the target, one combination
per line), and `elapsed_s`. The reference solution is never shown before the episode ends.

## Reward

| Answer | Reward |
|---|---|
| Uses every card once and equals the target | 1.0 |
| Uses every card once, within 10% of the target (shaped mode) | `0.3 × c²`, `c = 1 − |value − target| / (0.1 × target)` |
| Anything else: no answer, parse error, division by zero, a missing or extra card | 0 |

The grader parses the expression with Python's `ast`, allows only non-negative integers, + − × ÷ and
parentheses, compares the multiset of numbers to the cards, and evaluates with exact fractions. × ÷ −
and the letters A J Q K are accepted; anything after `=` is ignored. Partial credit keeps a GRPO group of
near misses from all scoring 0, and is capped far below an exact answer, so getting close never pays
more than getting it right.

## Use it

```python
from scott_wu_math import CardAction, ScottWuMathEnv

with ScottWuMathEnv(base_url="https://fineenvs-scottwumath.hf.space").sync() as env:
    obs = env.reset(split="video", index=1).observation
    print(obs.prompt)
    result = env.step(CardAction(expression="(((12 - 10) * (6 + 4) * 10 + 8) * 11 - 7) * 3"))
    print(result.reward, result.observation.result["check"]["value"])   # 1.0 6843
```

Over plain WebSocket (what the play page does): connect to `/ws`, send
`{"type": "reset", "data": {"level": "four-digit"}}`, then `{"type": "step", "data": {"expression": "..."}}`.

Task discovery follows OpenEnv's Task API: `GET /scott_wu_math/splits`, `POST /scott_wu_math/tasks`
with `{"split": "test"}`, and so on. Task listings leave out the reference solutions.

For text-only RL (no server), `scott_wu_math.game.reward.grade(expression, card_values, target)` and
`extract_answer(completion)` are the same grader as a plain function, and `game.prompts.messages(cards, target)`
builds the same prompt.

## The solver

Every expression tree splits the cards at its root into two groups, and the smaller group has at most
half the cards. The solver enumerates every value a group of up to four cards can make, then asks the
remaining cards for the one number that completes the target (`target = v + rest` needs
`rest = target − v`, and so on for the six ways to place `v`), recursing until four cards remain, where
the answer is a table lookup. That is complete for up to nine cards, and finds a solution for an
eight- or nine-card deal in a few milliseconds. A first pass keeps every intermediate value a whole
non-negative number, the way people play; a second pass allows fractions.

## Run locally

```bash
pip install -e .
uvicorn scott_wu_math.server.app:app --port 8012     # play at http://localhost:8012
python scripts/build_tasks.py                          # rebuild data/tasks (deterministic)
pytest
```

## Evaluations

`eval/rollout.py` in the [FineEnvs repository](https://github.com/adithya-s-k/FineEnvs/tree/main/08-math-environments/scott-wu-math)
runs OpenAI, Anthropic and open models on Hugging Face Inference Providers against a running server, in
parallel, one OpenEnv session per episode.

### Results (2026-10-23)

Twelve models on the first five `test` deals of each level (20 deals, one sample each) and on the two
`video` hands (three samples each), in three modes:

- **Reasoning**: the model's own reasoning at high effort, up to 32k output tokens.
- **No reasoning**: reasoning switched off, or as low as the API allows (`*`), up to 32k tokens, so the
  model may still work it out in its visible reply.
- **At once**: no reasoning, a system line asking for the answer tag alone, and 300 output tokens: the
  mental-arithmetic version of the game.

| Model | Reasoning | No reasoning | At once | Video hands: reasoning / none / at once |
|---|---|---|---|---|
| GPT-6.1 Sol | 20/20 | 20/20* | 14/20* | 6/6 / 6/6 / 3/6 |
| Claude Opus 5.5 | 20/20 | 19/20* | 8/20* | 6/6 / 6/6 / 5/6 |
| GLM-5.3 | 17/20 | 19/20 | 5/20 | 6/6 / 5/6 / 0/6 |
| Qwen3.8 2.4T | 20/20 | 19/20 | 4/20 | 6/6 / 5/6 / 0/6 |
| Claude Sonnet 5.5 | 20/20 | 15/20 | 4/20 | 6/6 / 4/6 / 0/6 |
| Claude Haiku 5.5 | 20/20 | 11/20 | 3/20 | 6/6 / 5/6 / 0/6 |
| Kimi K2.6 | 19/20 | 20/20 | 2/20 | 6/6 / 6/6 / 0/6 |
| DeepSeek V4 Pro | 20/20 | 10/20 | 2/20 | 6/6 / 3/6 / 0/6 |
| gpt-oss-120b | 16/20 | 6/20* | 2/20* | 6/6 / 1/6 / 0/6 |
| GPT-6 Luna | 20/20 | 2/20 | 2/20 | 5/6 / 0/6 / 0/6 |
| Qwen3.8 27B | 18/20 | 20/20 | 1/20 | 6/6 / 6/6 / 0/6 |
| Gemma 4 31B | 19/20 | 18/20 | 1/20 | 5/6 / 5/6 / 0/6 |

`*` lowest effort the API allows: GPT-6.1 Sol and gpt-oss-120b at `low`, Claude Opus 5.5 at `low` (its
thinking cannot be disabled). Open models ran on Hugging Face Inference Providers (Qwen3.8 2.4T and Kimi
K2.6 on DeepInfra; DeepSeek V4 Pro, GLM-5.3 and Qwen3.8 27B on Novita; gpt-oss-120b on Cerebras; Gemma 4
31B on DeepInfra). Gemma and Kimi each lost one or two episodes to provider errors, counted as failures.

With reasoning the game is close to solved: every model makes 6,843 on the first try. Answering at once
separates them: only GPT-6.1 Sol and Claude Opus 5.5, which still reason briefly at their lowest effort,
get far; most other models start working out loud and run out of room. That gap, and the no-reasoning
column for the smaller models, is where there is something to train.

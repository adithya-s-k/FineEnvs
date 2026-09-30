# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch"]
# ///
"""CPU GRPO on a tiny Wordle policy.

The estimator in alive.py is the contribution. This file is where it has to
earn a number: a 32-d pointer over the 2,309-word answer list, trained with
the same group size GeoGuesser used (8), on a machine with no GPU.

Four arms, one seed each, same budget:

    sparse  + grpo     the default TRL setup, win/loss reward
    sparse  + alive    skip dead groups, replay cliffs, rank advantages
    process + grpo     information-gain reward, vanilla advantages
    process + alive    both

The policy never sees the remaining set. It sees the same letter constraints
a person tracks on paper. Information gain is computed behind the reward.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_TRAIN = Path(__file__).resolve().parent
if str(_TRAIN) not in sys.path:
    sys.path.insert(0, str(_TRAIN))

from envs.wordle.core.game import WordleGame, feedback_pattern  # noqa: E402
from envs.wordle.core.rewards import reward_for  # noqa: E402
from envs.wordle.core.tasks import eval_tasks, train_tasks  # noqa: E402
from envs.wordle.core.words import ANSWERS  # noqa: E402
from alive import (  # noqa: E402
    AliveStats,
    ReplayQueue,
    advantages,
    classify_group,
    drop_zero_advantage_group,
    should_skip,
)

STATE_DIM = 26 * 4 + 5 * 27 + 7  # letter status, greens, remaining-guesses
HIDDEN = 128
EMBED = 32


def letter_status(guesses: list[str], patterns: list[str]) -> tuple[np.ndarray, list[str]]:
    """Per-letter {0 unknown, 1 grey, 2 yellow, 3 green} and the 5 green slots."""
    status = np.zeros(26, dtype=np.int64)
    greens = [" "] * 5
    for guess, pattern in zip(guesses, patterns):
        for i, (ch, colour) in enumerate(zip(guess, pattern)):
            idx = ord(ch) - 97
            if colour == "🟩":
                status[idx] = 3
                greens[i] = ch
            elif colour == "🟨":
                status[idx] = max(int(status[idx]), 2)
            elif colour == "⬛" and status[idx] == 0:
                status[idx] = 1
    return status, greens


def state_vector(guesses: list[str], patterns: list[str], remaining_guesses: int) -> torch.Tensor:
    status, greens = letter_status(guesses, patterns)
    letter_oh = np.eye(4, dtype=np.float32)[status].reshape(-1)  # 104
    green_idx = np.array([0 if g == " " else (ord(g) - 96) for g in greens], dtype=np.int64)
    green_oh = np.eye(27, dtype=np.float32)[green_idx].reshape(-1)  # 135
    rg = min(max(remaining_guesses, 0), 6)
    rg_oh = np.eye(7, dtype=np.float32)[rg]
    vec = np.concatenate([letter_oh, green_oh, rg_oh], axis=0)
    return torch.from_numpy(vec)


def observation_mask(guesses: list[str], patterns: list[str], word_codes: torch.Tensor) -> torch.Tensor:
    """Ban words that contradict the colouring a player can see.

    Grey letters are out. Green positions are locked. This is the paper
    tracker, not the remaining-set oracle — yellows are *not* used to
    filter, because doing that well is the policy's job.
    """
    n = word_codes.size(0)
    allowed = torch.ones(n, dtype=torch.bool)
    if not guesses:
        return allowed
    status, greens = letter_status(guesses, patterns)
    grey = torch.tensor(status == 1)
    if grey.any():
        uses_grey = grey[word_codes.long()]
        allowed &= ~uses_grey.any(dim=1)
    for i, ch in enumerate(greens):
        if ch == " ":
            continue
        allowed &= word_codes[:, i] == (ord(ch) - 97)
    if not bool(allowed.any()):
        return torch.ones(n, dtype=torch.bool)
    return allowed


class WordPolicy(nn.Module):
    """Pointer over the answer list, conditioned on letter constraints."""

    def __init__(self, n_words: int, word_of: list[str]):
        super().__init__()
        self.state = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, EMBED),
        )
        self.emb = nn.Embedding(n_words, EMBED)
        nn.init.normal_(self.emb.weight, std=0.02)
        codes = [[ord(c) - 97 for c in w] for w in word_of]
        self.register_buffer("word_codes", torch.tensor(codes, dtype=torch.long))

    def dist(self, guesses: list[str], patterns: list[str], remaining_guesses: int) -> Categorical:
        h = self.state(state_vector(guesses, patterns, remaining_guesses))
        logits = self.emb.weight @ h
        allowed = observation_mask(guesses, patterns, self.word_codes)
        logits = logits.masked_fill(~allowed, -1e9)
        return Categorical(logits=logits)


def play_episode(
    policy: WordPolicy,
    answer: str,
    word_of: list[str],
    *,
    max_guesses: int = 6,
    greedy: bool = False,
) -> dict:
    game = WordleGame(answer=answer, max_guesses=max_guesses)
    logprob = torch.zeros(())
    entropy = 0.0
    chosen: list[int] = []
    with torch.set_grad_enabled(not greedy):
        for _ in range(max_guesses):
            dist = policy.dist(game.guesses, game.patterns, game.max_guesses - len(game.guesses))
            entropy += float(dist.entropy().detach())
            if greedy:
                idx = int(torch.argmax(dist.logits))
            else:
                idx = int(dist.sample())
                logprob = logprob + dist.log_prob(torch.tensor(idx))
            chosen.append(idx)
            game.guess(word_of[idx])
            if game.done:
                break
    return {
        "won": game.won,
        "n_guesses": len(game.guesses),
        "guesses": list(game.guesses),
        "patterns": list(game.patterns),
        "invalid_count": game.invalid_count,
        "logprob": logprob,
        "entropy": entropy / max(len(game.guesses), 1),
        "fingerprint": tuple(chosen),
    }


def episode_reward(episode: dict, shape: str) -> float:
    return reward_for(
        shape,
        won=episode["won"],
        n_guesses=episode["n_guesses"],
        guesses=episode["guesses"],
        patterns=episode["patterns"],
        invalid_count=episode["invalid_count"],
    )


@dataclass
class Arm:
    name: str
    reward: str
    advantage: str
    skip_dead: bool
    replay: bool
    resample_cliff: int = 0


ARMS = [
    Arm("sparse-grpo", "sparse", "grpo", False, False),
    Arm("sparse-alive", "sparse", "rank", True, True, resample_cliff=2),
    Arm("process-grpo", "process", "grpo", False, False),
    Arm("process-alive", "process", "rank", True, True, resample_cliff=2),
]


@dataclass
class TrainConfig:
    steps: int = 150
    group_size: int = 8
    tasks_per_step: int = 4
    lr: float = 3e-3
    seed: int = 0
    eval_every: int = 25
    eval_limit: int = 200
    max_guesses: int = 6


def evaluate(policy: WordPolicy, word_of: list[str], answers: tuple[str, ...], cfg: TrainConfig) -> dict:
    policy.eval()
    wins = 0
    guesses = []
    rewards_sparse = []
    rewards_process = []
    with torch.no_grad():
        for answer in answers[: cfg.eval_limit]:
            ep = play_episode(policy, answer, word_of, max_guesses=cfg.max_guesses, greedy=True)
            wins += int(ep["won"])
            guesses.append(ep["n_guesses"] if ep["won"] else cfg.max_guesses)
            rewards_sparse.append(episode_reward(ep, "sparse"))
            rewards_process.append(episode_reward(ep, "process"))
    policy.train()
    n = max(len(guesses), 1)
    return {
        "solve_rate": wins / n,
        "mean_guesses": float(sum(guesses) / n),
        "sparse_reward": float(sum(rewards_sparse) / n),
        "process_reward": float(sum(rewards_process) / n),
    }


def train_arm(arm: Arm, cfg: TrainConfig, word_of: list[str]) -> dict:
    torch.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)
    policy = WordPolicy(len(word_of), word_of)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    train = list(train_tasks())
    held = eval_tasks()
    replay = ReplayQueue()
    stats = AliveStats()
    curve = []
    t0 = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        batch_logprobs = []
        batch_advs = []
        step_kinds = []
        for _ in range(cfg.tasks_per_step):
            task_i = replay.sample(range(len(train)), np_rng) if arm.replay else int(np_rng.integers(0, len(train)))
            answer = train[task_i]
            episodes = None
            rewards = None
            kind = "cliff"
            resampled = False
            attempts = 1 + (arm.resample_cliff if arm.skip_dead else 0)
            for attempt in range(attempts):
                episodes = [
                    play_episode(policy, answer, word_of, max_guesses=cfg.max_guesses)
                    for _ in range(cfg.group_size)
                ]
                rewards = [episode_reward(ep, arm.reward) for ep in episodes]
                fps = [ep["fingerprint"] for ep in episodes]
                kind = classify_group(rewards, fps)
                if kind == "live" or not arm.skip_dead:
                    break
                if kind == "collapse":
                    break
                resampled = True
            assert episodes is not None and rewards is not None
            stats.record(kind, skipped=should_skip(kind) if arm.skip_dead else False, resampled=resampled)
            replay.observe(task_i, kind)
            step_kinds.append(kind)
            if arm.skip_dead and should_skip(kind):
                continue
            adv = advantages(rewards, kind=arm.advantage)
            # Vanilla GRPO keeps a tied group in the batch mean. Dropping it
            # here would rescale the live groups that share the step, so the
            # *-grpo arms would not be the TRL baseline. Alive arms skip it.
            if drop_zero_advantage_group(arm.skip_dead, adv):
                stats.skipped += 1
                continue
            for ep, a in zip(episodes, adv):
                batch_logprobs.append(ep["logprob"])
                batch_advs.append(float(a))

        if batch_logprobs:
            lp = torch.stack(batch_logprobs)
            adv_t = torch.tensor(batch_advs, dtype=lp.dtype)
            loss = -(lp * adv_t).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            loss_v = float(loss.detach())
        else:
            loss_v = 0.0

        if step == 1 or step % cfg.eval_every == 0 or step == cfg.steps:
            ev = evaluate(policy, word_of, held, cfg)
            row = {
                "step": step,
                "arm": arm.name,
                "loss": loss_v,
                **ev,
                **stats.as_dict(),
            }
            curve.append(row)
            print(
                f"{arm.name:16s} step {step:4d}  solve {ev['solve_rate']:.3f}  "
                f"guesses {ev['mean_guesses']:.2f}  dead {row['frac_dead']:.2f}  "
                f"cliff {row['frac_cliff']:.2f}  collapse {row['frac_collapse']:.2f}",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    final = curve[-1]
    return {
        "arm": asdict(arm),
        "config": asdict(cfg),
        "seconds": elapsed,
        "final": final,
        "curve": curve,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arm", choices=[a.name for a in ARMS] + ["all"], default="all")
    p.add_argument("--out", type=Path, default=_ROOT / "results" / "tiny-ablation.json")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--tasks-per-step", type=int, default=4)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--smoke", action="store_true", help="8 steps, 40 eval words")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        steps=8 if args.smoke else args.steps,
        seed=args.seed,
        eval_every=4 if args.smoke else args.eval_every,
        eval_limit=40 if args.smoke else 200,
        tasks_per_step=2 if args.smoke else args.tasks_per_step,
        group_size=4 if args.smoke else args.group_size,
    )
    word_of = list(ANSWERS)
    chosen = ARMS if args.arm == "all" else [a for a in ARMS if a.name == args.arm]
    reports = []
    for arm in chosen:
        print(f"\n== {arm.name} ==", flush=True)
        reports.append(train_arm(arm, cfg, word_of))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "answers": len(word_of),
        "train": len(train_tasks()),
        "eval": len(eval_tasks()),
        "arms": reports,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    # Compact table for the README.
    print(f"{'arm':16s} {'solve':>7s} {'guesses':>8s} {'dead':>6s} {'cliff':>6s} {'collapse':>9s}")
    for report in reports:
        f = report["final"]
        print(
            f"{report['arm']['name']:16s} {f['solve_rate']:7.3f} {f['mean_guesses']:8.2f} "
            f"{f['frac_dead']:6.2f} {f['frac_cliff']:6.2f} {f['frac_collapse']:9.2f}"
        )


if __name__ == "__main__":
    main()

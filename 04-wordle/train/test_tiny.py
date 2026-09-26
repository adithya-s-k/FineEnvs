"""Smoke the tiny policy without running the full ablation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_TRAIN = Path(__file__).resolve().parent
if str(_TRAIN) not in sys.path:
    sys.path.insert(0, str(_TRAIN))

from envs.wordle.core.words import ANSWERS
from tiny_grpo import WordPolicy, episode_reward, play_episode, state_vector


def test_state_vector_has_the_advertised_width():
    vec = state_vector([], [], 6)
    assert vec.shape == (26 * 4 + 5 * 27 + 7,)


def test_episode_produces_a_gradient():
    policy = WordPolicy(len(ANSWERS), list(ANSWERS))
    ep = play_episode(policy, "crane", list(ANSWERS), max_guesses=2)
    assert ep["n_guesses"] >= 1
    assert ep["logprob"].requires_grad
    reward = episode_reward(ep, "process")
    assert 0.0 <= reward <= 2.0
    ep["logprob"].backward()
    grads = [p.grad.abs().sum().item() for p in policy.parameters() if p.grad is not None]
    assert sum(grads) > 0.0

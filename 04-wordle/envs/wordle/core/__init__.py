"""Wordle domain logic for training.

The 50-word list in `00-environments-101` is a demo. Training against it is
memorising a list. This package uses the 2,309-word original answer list, keeps
the answer out of the observation, and scores a guess on information gain as
well as on a solve.
"""

from .game import WordleGame, feedback_pattern
from .information import information_gain, remaining_words
from .rewards import process_reward, sparse_reward, terminal_reward
from .tasks import eval_tasks, train_tasks
from .words import ANSWERS, load_answers

__all__ = [
    "ANSWERS",
    "WordleGame",
    "eval_tasks",
    "feedback_pattern",
    "information_gain",
    "load_answers",
    "process_reward",
    "remaining_words",
    "sparse_reward",
    "terminal_reward",
    "train_tasks",
]

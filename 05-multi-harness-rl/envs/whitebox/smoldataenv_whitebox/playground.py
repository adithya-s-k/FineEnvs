"""One browser session, using the same tools and grader as training."""

import math
import threading
import time

from .catalog import task_by_name
from .environment import BashEnvironment


class PlaygroundSession:
    def __init__(self):
        self.environment = BashEnvironment()
        self.lock = threading.RLock()
        self.task = None
        self.history = []
        self.output = "Start a task to open your workspace."
        self.status = "Ready when you are."
        self.reward = None
        self.correctness = None
        self.active = False

    def __deepcopy__(self, memo):
        # Gradio copies the empty initial state for each browser session.
        return type(self)()

    def start(self, split, name):
        with self.lock:
            self.close()
            self.task = task_by_name(split, name)
            self.history = []
            self.reward = self.correctness = None
            self.output = ""
            self.environment.reset(self.task["folder"])
            self.active = True
            self.status = "Sandbox ready. Explore the files, then submit your answer."

    def run(self, command):
        with self.lock:
            if not self.active:
                raise ValueError("Start a task before running a command.")
            if not command.strip():
                raise ValueError("Enter a command first.")
            started = time.monotonic()
            self.output = self.environment.bash(command)
            self.history.append(
                {
                    "command": command[:4000],
                    "output": self.output[-4000:],
                    "seconds": time.monotonic() - started,
                }
            )
            self.history = self.history[-100:]
            self.status = "Command finished. Continue exploring or submit your answer."

    def grade(self, answer):
        with self.lock:
            if not self.active:
                raise ValueError("Start a task before submitting an answer.")
            if not answer.strip():
                raise ValueError("Write your final answer first.")
            try:
                self.environment.submit_solution(answer)
                value = self.environment.get_reward()
                self.reward = value if math.isfinite(value) else None
                self.correctness = (
                    self.environment._correctness if self.reward is not None else None
                )
                self.status = (
                    "Answer graded. Your sandbox has been released."
                    if self.reward is not None
                    else "Grading was unavailable. This attempt is unscored."
                )
            finally:
                self.close()

    def close(self):
        with self.lock:
            self.active = False
            self.environment._close()


def close_session(session):
    if session is not None:
        session.close()

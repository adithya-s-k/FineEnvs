"""Open the current training project first in the pinned Trackio 0.33.0 UI."""
import os

import trackio
from trackio import server


DEFAULT_PROJECT = os.environ.get(
    "TRACKIO_DEFAULT_PROJECT", "multi4-qwen35-2b-prod-20260915"
)
_get_all_projects = server.get_all_projects


def get_all_projects() -> list[str]:
    # The browser selects the first project at the bare Space URL. The project
    # argument to show() only changes its printed/browser-launch URL.
    # Keep every historical project available in the normal project picker.
    return sorted(_get_all_projects(), key=lambda name: name != DEFAULT_PROJECT)


server.get_all_projects = get_all_projects

if __name__ == "__main__":
    trackio.show()

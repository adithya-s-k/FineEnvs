# SPDX-License-Identifier: BSD-3-Clause

"""FastAPI application for the watercolour environment."""

import sys
from pathlib import Path

# Two deliberate departures from the layout in CONTRIBUTING.md, both forced by the
# same thing: this folder is called `openenv` and so is the installed package.
#
# 1. There is no `__init__.py` here. With one, `import openenv` resolves to this
#    folder whenever the interpreter's working directory is the environment root,
#    which is what the Dockerfile sets, and `openenv.core` then does not exist.
# 2. The path entry is appended rather than inserted, and points at the folder
#    holding `core/`, so an installed package always wins over a local directory.
_ENV_ROOT = str(Path(__file__).resolve().parents[1])
if not __package__ and _ENV_ROOT not in sys.path:
    sys.path.append(_ENV_ROOT)



from openenv.core.env_server import create_app  # noqa: E402

if __package__:
    from .models import WatercolourAction, WatercolourObservation
    from .watercolour_environment import WatercolourEnvironment
else:
    from models import WatercolourAction, WatercolourObservation
    from watercolour_environment import WatercolourEnvironment

# The class is passed rather than an instance so each session gets its own
# environment and its own sampled subject.
app = create_app(
    WatercolourEnvironment,
    WatercolourAction,
    WatercolourObservation,
    env_name="watercolour_env",
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

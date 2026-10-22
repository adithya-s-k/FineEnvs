"""The env server: airops-occ3 on streamable HTTP, as its image runs it."""

import os

from .occ.live3 import OccEnv3


def main() -> None:
    OccEnv3().create_app().run(transport="http", host=os.environ.get("MCP_HOST", "0.0.0.0"),
                               port=int(os.environ.get("MCP_PORT", "18765")), show_banner=False)


if __name__ == "__main__":
    main()

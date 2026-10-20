"""The env server: airops-occ on streamable HTTP, as its image runs it."""

import os

from .occ.live import OccEnv


def main() -> None:
    OccEnv().create_app().run(transport="http", host=os.environ.get("MCP_HOST", "0.0.0.0"),
                              port=int(os.environ.get("MCP_PORT", "18765")), show_banner=False)


if __name__ == "__main__":
    main()

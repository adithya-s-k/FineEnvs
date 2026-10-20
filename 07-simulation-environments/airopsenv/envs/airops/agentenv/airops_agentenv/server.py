"""The env server: airops-day on streamable HTTP, as the image runs it."""

import os

from .live import AirOpsLiveEnv


def main() -> None:
    AirOpsLiveEnv().create_app().run(transport="http", host=os.environ.get("MCP_HOST", "0.0.0.0"),
                                     port=int(os.environ.get("MCP_PORT", "18765")), show_banner=False)


if __name__ == "__main__":
    main()

"""CLI wrapper for the packaged deployment smoke."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from ui_smoke import main

if __name__ == "__main__":
    main()

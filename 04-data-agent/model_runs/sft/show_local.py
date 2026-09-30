"""View local Trackio data, preserving the append-only event files."""
import argparse
import json
from pathlib import Path
import threading
import time

from local_logging import configure, materialize

p = argparse.ArgumentParser()
p.add_argument('--run-root', type=Path, required=True)
args = p.parse_args()
configure(args.run_root)
import trackio
project = json.loads((args.run_root / 'config.json').read_text())['configuration']['project']
materialize()


def refresh():
    while True:
        time.sleep(10)
        materialize()


threading.Thread(target=refresh, daemon=True).start()
trackio.show(project=project)

"""Load the existing runtime credentials before the production SFT entry point."""
from pathlib import Path
import runpy
import sys
import os

root = Path(sys.argv[sys.argv.index('--workspace') + 1]).resolve()
sys.path.insert(0, str(root / 'HuggingEnvs/04-data-agent/hf'))
from deploy import credentials
os.environ.update(credentials(root / 'experiments/.env'))
runpy.run_path(str(Path(__file__).with_name('train.py')), run_name='__main__')

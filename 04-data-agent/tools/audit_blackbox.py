"""Use the original baseline's exact-token auditor on the Daytona captures."""
import argparse
from pathlib import Path
from baseline_checks import audit_captures

p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','resume','final'],required=True)
p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[1]/'logs/20260915/blackbox')
args=p.parse_args()
root=args.output_root
audit_captures(root,root,args.phase)

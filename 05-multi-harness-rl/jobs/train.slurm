#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=smoldataenv-%j.log
set -euo pipefail
# Submit from 05-multi-harness-rl after installation and prepare.py.
# Activate your Python 3.12 environment before sbatch; it is inherited by the job.
if [[ "$1" == "smoke" ]]; then
    shift
    exec python jobs/smoke.py "$@"
fi
python jobs/run.py "$@"

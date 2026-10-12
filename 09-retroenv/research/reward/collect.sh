#!/usr/bin/env bash
# Trajectory collection for the reward ablation: one OpenAI capability ladder over a
# fixed task list, so reward variants can be re-scored offline from runs/rewardlab/.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . ../.env; set +a
export RETROENV_BENCHMARK_DIR=$PWD/data/release/RetroEnv-RL RETROENV_MAX_TOOL_CALLS=48 MAX_CONCURRENT_ENVS=96 ENABLE_WEB_INTERFACE=false
PORT=${PORT:-8791}
OUT=runs/rewardlab
URL="http://127.0.0.1:$PORT"

uv run uvicorn retroenv_openenv.server:app --host 127.0.0.1 --port "$PORT" --log-level warning > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 200); do curl -sf -m 2 "$URL/health" > /dev/null && break; sleep 3; done
curl -sf -m 1800 "$URL/retro_route/splits" | head -c 200; echo

run() {  # run <model> <split> <max-cost> [extra args...]
  local model=$1 split=$2 cost=$3; shift 3
  local slug; slug=$(echo "$model" | tr '/.' '__')
  uv run python eval/run_eval.py --provider openrouter --model "$model" --server "$URL" \
    --split "$split" --task-ids "$OUT/$split.txt" --max-turns 20 --concurrency 5 \
    --max-cost "$cost" --output "$OUT/$split/$slug" "$@" > "$OUT/$split.$slug.log" 2>&1
}

# gpt-5* reject temperature=0 and spend most of max_tokens on reasoning.
for split in test_id test_hard; do
  ( run openai/gpt-4.1-nano "$split" 1.5 --max-tokens 4000 ) &
  ( run openai/gpt-4o-mini  "$split" 1.5 --max-tokens 4000 ) &
  ( run openai/gpt-4.1-mini "$split" 3.0 --max-tokens 4000 ) &
  ( run openai/gpt-5-nano   "$split" 2.0 --max-tokens 10000 --temperature 1 ) &
  ( run openai/gpt-5-mini   "$split" 4.0 --max-tokens 10000 --temperature 1 ) &
  wait
done

# The expensive end of the ladder only needs the primary split.
( run openai/gpt-4.1 test_id 8.0 --max-tokens 4000 ) &
( run openai/gpt-4o  test_id 8.0 --max-tokens 4000 ) &
( run openai/gpt-5   test_id 12.0 --max-tokens 10000 --temperature 1 ) &
wait
echo ALL-DONE

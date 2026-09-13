# Export HF_TOKEN for Harbor, WITHOUT printing it.
#
# Harbor's task containers read HF_TOKEN from the HOST environment -- each task.toml declares it in
# [environment.env] for the pull_bucket.py healthcheck -- and harbor refuses the rollout outright with
# "Environment variable 'HF_TOKEN' not found in host environment" when it is missing.
#
# It lives in three different places here, so try all of them: experiments/.env calls it HF_API_KEY,
# huggingface_hub keeps a cached login of its own, and HF_TOKEN may already be set.
if [ -z "${HF_TOKEN:-}" ]; then
  if [ -n "${HF_API_KEY:-}" ]; then
    export HF_TOKEN="$HF_API_KEY"
  else
    HF_TOKEN=$("$(dirname "${BASH_SOURCE[0]}")/../.venv/bin/python" -c \
      'from huggingface_hub import get_token; print(get_token() or "")' 2>/dev/null)
    export HF_TOKEN
  fi
fi
# `${VAR:+present}` ALONE. Combining it with `${VAR:-...}` prints the value when it IS set, which is
# exactly the case being tested for -- that leaked a key into a transcript once already.
echo "  HF_TOKEN: ${HF_TOKEN:+present}"

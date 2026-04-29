#!/usr/bin/env bash
set -euo pipefail

output="$1"
error_output="$2"
seed="$3"
limit="${4:-850}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY must be set in the environment" >&2
  exit 2
fi

cd "$(dirname "$0")"
uv run python -m charstreamer_span_generator.simple \
  --output "$output" \
  --error-output "$error_output" \
  --limit "$limit" \
  --max-records 200000 \
  --annotation-protocol per-label \
  --label-strategy all \
  --sample-focus-strategy round-robin \
  --min-chars 80 \
  --max-chars 700 \
  --context-chars 160 \
  --seed "$seed" \
  --shuffle-buffer-size 50000 \
  --model gpt-5.4-mini \
  --reasoning-effort low \
  --verbosity low \
  --max-attempts 3 \
  --max-output-tokens 12000 \
  --progress-every 25

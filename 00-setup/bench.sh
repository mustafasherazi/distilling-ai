#!/usr/bin/env bash
# Load-test the server from serve.sh with aiperf: 512-token prompts, 256-token answers,
# at several concurrency levels. Prints time to first token, per-user speed and total throughput.
#
#   bash 00-setup/bench.sh                                       # concurrency 1 8 16 32
#   MODEL=Qwen/Qwen2.5-14B-Instruct-AWQ CONCS="1 8 16" bash 00-setup/bench.sh
#
# One-time install:  uv venv ~/bench-env && source ~/bench-env/bin/activate && uv pip install "aiperf==0.11.0"
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
CONCS="${CONCS:-1 8 16 32}"
ISL="${ISL:-512}"   # input tokens per request
OSL="${OSL:-256}"   # output tokens per request
OUT="${OUT:-$HOME/bench-results}"
mkdir -p "$OUT" && cd "$OUT"

for C in $CONCS; do
  N=$(( C <= 2 ? 8 : (C <= 8 ? 32 : 64) ))   # enough requests for stable numbers
  echo "== concurrency $C ($N requests) =="
  aiperf profile --url http://localhost:8000 --model "$MODEL" --endpoint-type chat --streaming \
    --synthetic-input-tokens-mean "$ISL" --output-tokens-mean "$OSL" \
    --extra-inputs ignore_eos:true --concurrency "$C" --request-count "$N" \
    --tokenizer "$MODEL" > "aiperf_c$C.log" 2>&1
  CSV=$(ls -t "$OUT"/artifacts/*concurrency${C}/profile_export_aiperf.csv 2>/dev/null | head -1)
  if [ -z "$CSV" ]; then echo "  no results, see $OUT/aiperf_c$C.log"; continue; fi
  grep -E "^Time to First Token|^Output Token Throughput Per User" "$CSV" \
    | awk -F, '{printf "  %-34s p50=%s\n",$1,$10}'
  grep -E "^Output Token Throughput \(t" "$CSV" | awk -F, '{printf "  %-34s %s\n",$1,$2}'
done

#!/usr/bin/env bash
#
# Run vLLM inference for every generated prompt condition in the prompt manifest.
#
# This is the batch counterpart to scripts/run_vllm.sh. It uses the manifest
# produced by examples/generate_prompt_batches.ipynb as the source of truth,
# skips completed result files, and runs missing/incomplete conditions
# sequentially so a single GPU can work through the full prompt set.
#
# Usage:
#   ./scripts/run_vllm_all_prompt_results.sh
#   GPU=1 VLLM_TP_SIZE=1 ./scripts/run_vllm_all_prompt_results.sh
#   DRY_RUN=1 ./scripts/run_vllm_all_prompt_results.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif [[ -x "$PROJECT_ROOT/.venv311/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv311/bin/python"
  elif [[ -x "$PROJECT_ROOT/../miniconda3/envs/ai_terrarium/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/../miniconda3/envs/ai_terrarium/bin/python"
  else
    echo "Error: no Python interpreter found. Set PYTHON_BIN to a Python executable." >&2
    exit 1
  fi
fi

PROMPT_MANIFEST="${PROMPT_MANIFEST:-${PROJECT_ROOT}/prompts/new_nl/post_ICA/vote/_manifests/latest_prompt_manifest.csv}"
PROMPT_SPEC="${PROMPT_SPEC:-section_combinations}"
PROMPT_QID="${PROMPT_QID:-Q21}"
PROMPT_TARGET="${PROMPT_TARGET:-president}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_SLUG="${MODEL//\//__}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results/vote/${MODEL_SLUG}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

GPU="${GPU:-1}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.75}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-fp8}"
BACKGROUND=0

DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
RUN_TAG="${RUN_TAG:-all_prompts_$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="${MASTER_LOG:-${PROJECT_ROOT}/logging/${RUN_TAG}.log}"

mkdir -p "$(dirname "$MASTER_LOG")"

_condition_plan() {
  "$PYTHON_BIN" - "$PROMPT_MANIFEST" "$PROMPT_SPEC" "$PROMPT_QID" "$PROMPT_TARGET" "$PROJECT_ROOT" "$RESULT_ROOT" <<'PY'
import csv
import sys
from pathlib import Path

manifest, spec, qid, target, project_root, result_root = sys.argv[1:]
project_root = Path(project_root)
result_root = Path(result_root)
manifest = Path(manifest)

with manifest.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    if row.get("spec") != spec or row.get("qid") != qid or row.get("target") != target:
        continue

    hypothesis = row.get("hypothesis", "")
    prompt_path = Path(row.get("prompt_csv_abs") or row.get("prompt_csv") or "")
    truth_path = Path(row.get("ground_truth_csv_abs") or row.get("ground_truth_csv") or "")
    output_dir = row.get("output_dir", "")
    n_truth = row.get("n_truth") or row.get("n_prompts") or "0"

    if not prompt_path.is_absolute():
        prompt_path = project_root / prompt_path
    if not truth_path.is_absolute():
        truth_path = project_root / truth_path

    marker = "prompts/new_nl/post_ICA/vote/"
    if marker in output_dir:
        rel_dir = output_dir.split(marker, 1)[1]
    else:
        rel_dir = output_dir
    result_path = result_root / rel_dir / "results.csv"

    print("\t".join([
        hypothesis,
        str(prompt_path),
        str(truth_path),
        str(result_path),
        str(n_truth),
    ]))
PY
}

_csv_row_count() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    echo 0
    return
  fi
  "$PYTHON_BIN" - "$file" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    try:
        next(reader)
    except StopIteration:
        print(0)
    else:
        print(sum(1 for _ in reader))
PY
}

_log() {
  local msg="$1"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" | tee -a "$MASTER_LOG"
}

_log "Starting all-prompt vLLM batch"
_log "Manifest: $PROMPT_MANIFEST"
_log "Model: $MODEL"
_log "GPU=$GPU TP=$VLLM_TP_SIZE quant=$VLLM_QUANTIZATION max_len=$VLLM_MAX_MODEL_LEN gpu_mem=$VLLM_GPU_MEMORY_UTIL"
_log "Master log: $MASTER_LOG"

total=0
skipped=0
started=0
failed=0

while IFS=$'\t' read -r hypothesis prompt_path truth_path result_path n_truth; do
  [[ -z "${hypothesis:-}" ]] && continue
  total=$((total + 1))

  rows_done="$(_csv_row_count "$result_path")"
  if [[ "$FORCE" != "1" && "$rows_done" -ge "$n_truth" && "$n_truth" -gt 0 ]]; then
    skipped=$((skipped + 1))
    _log "SKIP complete: $hypothesis ($rows_done/$n_truth) -> $result_path"
    continue
  fi

  _log "QUEUE: $hypothesis ($rows_done/$n_truth complete) -> $result_path"
  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi

  mkdir -p "$(dirname "$result_path")"
  started=$((started + 1))

  set +e
  PROMPT_MANIFEST="$PROMPT_MANIFEST" \
  PROMPT_SPEC="$PROMPT_SPEC" \
  PROMPT_QID="$PROMPT_QID" \
  PROMPT_HYPOTHESIS="$hypothesis" \
  TASK_NAME="${PROMPT_QID}_${hypothesis}_${RUN_TAG}" \
  PROMPT_PATH="$prompt_path" \
  GROUND_TRUTH_CSV="$truth_path" \
  RESULT_PATH="$result_path" \
  PYTHON_BIN="$PYTHON_BIN" \
  MODEL="$MODEL" \
  GPU="$GPU" \
  VLLM_TP_SIZE="$VLLM_TP_SIZE" \
  VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
  VLLM_GPU_MEMORY_UTIL="$VLLM_GPU_MEMORY_UTIL" \
  VLLM_QUANTIZATION="$VLLM_QUANTIZATION" \
  PYTHON_BIN="$PYTHON_BIN" \
  BACKGROUND=0 \
  "$SCRIPT_DIR/run_vllm.sh" 2>&1 | tee -a "$MASTER_LOG"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "$status" -ne 0 ]]; then
    failed=$((failed + 1))
    _log "FAILED: $hypothesis (exit $status)"
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      _log "STOP_ON_ERROR=1; stopping batch."
      exit "$status"
    fi
  else
    rows_done="$(_csv_row_count "$result_path")"
    _log "DONE: $hypothesis ($rows_done/$n_truth) -> $result_path"
  fi
done < <(_condition_plan)

_log "Finished all-prompt vLLM batch: total=$total skipped=$skipped started=$started failed=$failed"
exit "$failed"

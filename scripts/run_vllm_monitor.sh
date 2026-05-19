#!/usr/bin/env bash
#
# Monitor vLLM inference logs.
#
# Usage:
#   ./scripts/run_vllm_monitor.sh                      # tail -f latest log
#   ./scripts/run_vllm_monitor.sh <log_file>            # tail -f specific log
#   ./scripts/run_vllm_monitor.sh --list                # list all logs with summary
#   ./scripts/run_vllm_monitor.sh --status [log]        # one-shot status check
#   ./scripts/run_vllm_monitor.sh --poll <pid> [log]    # poll every 60s until PID exits
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logging"

_ts() { date +"%Y-%m-%d %H:%M:%S"; }

_find_latest_log() {
  local f
  f=$(ls -t "${LOG_DIR}"/*.log 2>/dev/null | head -1)
  [[ -n "$f" ]] && echo "$f" || return 1
}

_parse_header_field() {
  local file="$1" field="$2"
  grep -m1 " ${field}" "$file" 2>/dev/null | sed "s/.*: //" | xargs
}

_show_status() {
  local LOG="$1"
  if [[ -z "$LOG" || ! -f "$LOG" ]]; then
    echo "No log found."
    return 1
  fi
  echo "Log file     : $LOG"
  echo "Last modified : $(stat -c %y "$LOG" 2>/dev/null || stat -f %Sm "$LOG" 2>/dev/null)"

  local task combo
  task=$(_parse_header_field "$LOG" "Task")
  combo=$(_parse_header_field "$LOG" "Prompt combo")
  [[ -n "$task" ]]  && echo "Task         : $task"
  [[ -n "$combo" ]] && echo "Prompt combo : $combo"

  if grep -Eq " FAILED |Traceback|Error|Exception" "$LOG" 2>/dev/null; then
    echo "Status       : FAILED"
    grep -E "FAILED|Traceback|Error|Exception" "$LOG" | tail -3
  elif grep -q "Finished" "$LOG" 2>/dev/null; then
    echo "Status       : COMPLETED"
    grep "Runtime" "$LOG" | tail -1
    grep "Results" "$LOG" | tail -1
  else
    echo "Status       : RUNNING (or interrupted)"
  fi
  echo ""
  echo "Last 5 lines:"
  tail -5 "$LOG"
}

case "${1:-}" in
  --list)
    if ! ls "${LOG_DIR}"/*.log &>/dev/null; then
      echo "No logs in $LOG_DIR"
      exit 0
    fi
    printf "%-12s %-20s %-20s %-10s %s\n" "DATE" "TASK" "PROMPT_COMBO" "STATUS" "FILE"
    printf "%-12s %-20s %-20s %-10s %s\n" "----" "----" "------------" "------" "----"
    for f in $(ls -t "${LOG_DIR}"/*.log 2>/dev/null); do
      local_task=$(_parse_header_field "$f" "Task")
      local_combo=$(_parse_header_field "$f" "Prompt combo")
      local_date=$(_parse_header_field "$f" "Date")
      local_date=${local_date%% *}
      if grep -Eq " FAILED |Traceback|Error|Exception" "$f" 2>/dev/null; then
        local_status="FAILED"
      elif grep -q "Finished" "$f" 2>/dev/null; then
        local_status="DONE"
      else
        local_status="RUN/FAIL"
      fi
      printf "%-12s %-20s %-20s %-10s %s\n" \
        "${local_date:-?}" "${local_task:-?}" "${local_combo:-?}" "$local_status" "$(basename "$f")"
    done
    ;;
  --poll)
    PID="${2:-}"
    LOG="${3:-$(_find_latest_log)}"
    if [[ -z "$PID" ]]; then
      echo "Usage: $0 --poll <pid> [log_file]"
      exit 1
    fi
    echo "[$(_ts)] Polling PID $PID (log: ${LOG:-<none>})"
    while kill -0 "$PID" 2>/dev/null; do
      echo "[$(_ts)] Status: running (PID: $PID)"
      sleep 60
    done
    echo "[$(_ts)] Process $PID exited."
    if [[ -n "$LOG" && -f "$LOG" ]]; then
      echo ""
      _show_status "$LOG"
    fi
    ;;
  --status)
    LOG="${2:-$(_find_latest_log)}"
    _show_status "$LOG"
    ;;
  "")
    LOG="$(_find_latest_log)" || { echo "No *.log in $LOG_DIR"; exit 1; }
    echo "Tailing: $LOG"
    exec tail -f "$LOG"
    ;;
  -*)
    echo "Usage: $0 [log_file]"
    echo "       $0 --list"
    echo "       $0 --status [log_file]"
    echo "       $0 --poll <pid> [log_file]"
    exit 1
    ;;
  *)
    LOG="$1"
    if [[ ! -f "$LOG" ]]; then
      echo "File not found: $LOG"
      exit 1
    fi
    echo "Tailing: $LOG"
    exec tail -f "$LOG"
    ;;
esac

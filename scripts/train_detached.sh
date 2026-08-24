#!/usr/bin/env bash
# Launch training in its own session, detached from the invoking shell.
#
# Isaac Sim traps SIGTERM and starts a graceful shutdown that can hang with the
# main thread spinning at 100% CPU and the GPU idle. If the shell that launched a
# background run is later cleaned up, the signal reaches the run through the shared
# process group and stalls it silently mid-training: the process stays alive, so
# nothing looks wrong until you notice the checkpoints stopped advancing.
set -uo pipefail
PROJECT_DIR="${MARINE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_NAME=""
for ((i = 1; i <= $#; i++)); do
  if [ "${!i}" = "--run-name" ]; then j=$((i + 1)); RUN_NAME="${!j}"; fi
done
if [ -z "$RUN_NAME" ]; then echo "--run-name is required" >&2; exit 2; fi
mkdir -p "$PROJECT_DIR/logs"
LOG="$PROJECT_DIR/logs/${RUN_NAME}.log"
setsid env PYTHONUNBUFFERED=1 nohup bash "$PROJECT_DIR/scripts/train.sh" "$@" \
  > "$LOG" 2>&1 < /dev/null &
echo "$!" > "$PROJECT_DIR/logs/${RUN_NAME}.sid"
echo "launched ${RUN_NAME}; log ${LOG}"

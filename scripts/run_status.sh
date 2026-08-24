#!/usr/bin/env bash
# Report whether a training run is progressing, using artefacts on disk rather
# than process liveness: a hung Isaac Sim stays alive but stops writing.
set -uo pipefail
PROJECT_DIR="${MARINE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_NAME="${1:?usage: run_status.sh RUN_NAME}"
DIR="$PROJECT_DIR/outputs/training/$RUN_NAME"
EVENTS=$(ls -t "$DIR"/events.out.tfevents.* 2>/dev/null | head -1)
if [ -z "$EVENTS" ]; then echo "$RUN_NAME: no tfevents yet"; exit 0; fi
AGE=$(( $(date +%s) - $(stat -c %Y "$EVENTS") ))
LATEST=$(ls "$DIR"/model_*.pt 2>/dev/null | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
ALIVE=$(pgrep -f "scripts/train\.py .*--run-name $RUN_NAME\$" | wc -l)
printf '%s: latest checkpoint %s, tfevents %ss old, %s process(es)\n' \
  "$RUN_NAME" "${LATEST:-none}" "$AGE" "$ALIVE"
if [ "$AGE" -gt 300 ]; then echo "  STALLED: no metric written in ${AGE}s"; fi

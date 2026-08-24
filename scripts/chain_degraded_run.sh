#!/usr/bin/env bash
# Sequence the remaining pipeline on one GPU: wait out the clean run, train the
# sensor-degraded policy, then sweep both controllers across measurement delay.
#
# Runs in its own session (see train_detached.sh) because Isaac Sim traps SIGTERM
# and can hang mid-shutdown with the process still alive, which stalls a run
# silently rather than failing it.
set -uo pipefail
PROJECT_DIR="${MARINE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

CLEAN_RUN=ikseeded_4096_seed940_10k
DEGRADED_RUN=degraded_4096_seed941_10k

wait_for_run() {
  local name="$1"
  while true; do
    sleep 60
    if grep -q "fresh_train_summary" "logs/${name}.log" 2>/dev/null; then return 0; fi
    if ! pgrep -f "scripts/train\.py .*--run-name ${name}\$" > /dev/null; then
      echo "run ${name} exited without writing a summary" >&2
      return 1
    fi
  done
}

echo "waiting for $CLEAN_RUN"
wait_for_run "$CLEAN_RUN" || exit 1
echo "$CLEAN_RUN finished"
sleep 30  # let Isaac release the GPU

# 6000 rather than 10000: the clean run's error plateaued by iteration 500, and the
# late-collapse question that motivated a long run is already answered by it.
bash scripts/train_detached.sh \
  --task Marine-UR3-Random6DoFBase-WorldLineSensorDegraded-v0 \
  --num-envs 4096 --max-iterations 6000 --seed 941 --run-name "$DEGRADED_RUN"
sleep 90
echo "waiting for $DEGRADED_RUN"
wait_for_run "$DEGRADED_RUN" || exit 1
echo "$DEGRADED_RUN finished"
sleep 30

bash scripts/sweep_delay.sh "$DEGRADED_RUN"

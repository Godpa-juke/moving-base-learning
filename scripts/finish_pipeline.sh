#!/usr/bin/env bash
# Stop the parallel training runs once they have plateaued, then sweep measurement
# delay for every condition and both analytic controller formulations.
#
# The clean run's tracking error flattened by iteration 500 and stayed flat through
# 9000, so training the remaining conditions to 6000 buys nothing; they are cut off at
# TARGET_ITER and the best checkpoint is selected from the curve. Killing them also
# hands the GPU to the evaluations, which are the only remaining work.
#
# SIGKILL rather than SIGTERM on purpose: Isaac Sim traps SIGTERM and can hang
# mid-shutdown with the process alive and one core spinning.
set -uo pipefail
PROJECT_DIR="${MARINE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

TARGET_ITER="${TARGET_ITER:-2500}"
RUNS="degraded_4096_seed941_6k nohistory_4096_seed942_6k widedelay_4096_seed943_6k seam_4096_seed944_6k"

echo "waiting for iteration $TARGET_ITER in: $RUNS"
while true; do
  sleep 60
  pending=""
  for run in $RUNS; do
    if ls "outputs/training/$run/model_$TARGET_ITER.pt" > /dev/null 2>&1; then continue; fi
    if ! pgrep -f "scripts/train\.py .*--run-name $run\$" > /dev/null; then
      echo "run $run stopped before iteration $TARGET_ITER" >&2
      continue
    fi
    pending="$pending $run"
  done
  [ -z "$pending" ] && break
done
echo "all runs reached iteration $TARGET_ITER"

for run in $RUNS; do
  for pid in $(pgrep -f "scripts/train\.py .*--run-name $run\$"); do
    kill -9 "$pid" 2>/dev/null && echo "stopped $run (pid $pid)"
  done
done
sleep 45  # let the GPU drain before the evaluations start

DEGRADED_TASK=Marine-UR3-Random6DoFBase-WorldLineSensorDegraded-Play-v0
SEAM_TASK=Marine-UR3-Random6DoFBase-WorldLineDegradedSeam-Play-v0

# Base motion measured late, seam offset still exact. The policy can close the loop on
# its own error here, so this is the control that shows base delay alone costs little.
bash scripts/sweep_delay.sh --run degraded_4096_seed941_6k --tag base_pol \
  --task "$DEGRADED_TASK" --controllers "policy"
bash scripts/sweep_delay.sh --tag base_ikpose --task "$DEGRADED_TASK" \
  --controllers "ik" --ik-mode pose
bash scripts/sweep_delay.sh --tag base_ikseam --task "$DEGRADED_TASK" \
  --controllers "ik" --ik-mode seam

# Ablation: one measured frame instead of twenty, same degradation.
bash scripts/sweep_delay.sh --run nohistory_4096_seed942_6k --tag nohist \
  --task Marine-UR3-Random6DoFBase-WorldLineDegradedNoHistory-Play-v0 \
  --controllers "policy"

# Both channels late: the condition under which predicting the disturbance is
# necessary rather than optional.
bash scripts/sweep_delay.sh --run seam_4096_seed944_6k --tag seam_pol \
  --task "$SEAM_TASK" --controllers "policy"
bash scripts/sweep_delay.sh --tag seam_ikpose --task "$SEAM_TASK" \
  --controllers "ik" --ik-mode pose
bash scripts/sweep_delay.sh --tag seam_ikseam --task "$SEAM_TASK" \
  --controllers "ik" --ik-mode seam

# Wider delays, where a reactive controller has the most to lose.
bash scripts/sweep_delay.sh --run widedelay_4096_seed943_6k --tag wide_pol \
  --task Marine-UR3-Random6DoFBase-WorldLineDegradedWideDelay-Play-v0 \
  --controllers "policy" --delays "0 0.0833 0.1667 0.2500 0.3333"
bash scripts/sweep_delay.sh --tag wide_ikseam \
  --task Marine-UR3-Random6DoFBase-WorldLineDegradedWideDelay-Play-v0 \
  --controllers "ik" --ik-mode seam --delays "0 0.0833 0.1667 0.2500 0.3333"

echo PIPELINE_COMPLETE

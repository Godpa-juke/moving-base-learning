#!/usr/bin/env bash
# Four loose ends the main sweep left, each of which would otherwise put a wrong
# number in the results.
#
# 1. The closed-loop analytic controller went unstable past 67 ms of delay. Unity gain
#    nulls the whole measured error in one step, which cannot be stable once the loop
#    carries three or four steps of lag - that is a tuning choice of ours, not a
#    property of feedback control, and must not be reported as one.
# 2. The seam-degraded policy scores 3.27 mm on its training metric and 12.10 mm in
#    evaluation. Training samples sensor noise uniformly from zero, while the sweep pins
#    every environment at the midpoint; this measures how much of the gap that explains.
# 3. base_ikseam ran before the fallback fix, so the controller was reading a delayed
#    seam offset on a task whose policy reads an exact one.
# 4. The wide-delay comparison has no open-loop analytic curve, so the regime-1 gap at
#    large delay is currently an extrapolation.
set -uo pipefail
PROJECT_DIR="${MARINE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

SEAM_TASK=Marine-UR3-Random6DoFBase-WorldLineDegradedSeam-Play-v0
BASE_TASK=Marine-UR3-Random6DoFBase-WorldLineSensorDegraded-Play-v0
WIDE_TASK=Marine-UR3-Random6DoFBase-WorldLineDegradedWideDelay-Play-v0
SEAM_CKPT=$(python3 scripts/best_checkpoint.py outputs/training/seam_4096_seed944_6k --json \
  | python3 -c "import json,sys;print(json.load(sys.stdin)[0]['checkpoint'])")

cell() {  # cell RUN_NAME extra-args...
  local name="$1"; shift
  if bash scripts/evaluate.sh --num-envs 4096 --steps 600 --seed 44 --run-name "$name" "$@" \
     > /dev/null 2>&1; then
    echo "followup done: $name"
  else
    echo "followup FAILED: $name"
  fi
}

# 1. Closed-loop gain, at the delays where unity gain blew up.
for GAIN in 1.0 0.5 0.3 0.15; do
  for DELAY in 0.1000 0.1333; do
    MS=$(python3 -c "print(f'{round($DELAY*1000):03d}')")
    G=${GAIN/./p}
    cell "gain_g${G}_d${MS}ms" --controller ik --ik-mode seam --ik-gain "$GAIN" \
      --task "$SEAM_TASK" --sensor-delay-s "$DELAY" \
      --sensor-position-noise-m 0.001 --sensor-rotation-noise-rad 0.0015
  done
done

# 2. How much of the policy's train/eval gap is the pinned noise level.
for NOISE in 0 0.0005 0.001 0.002; do
  ROT=$(python3 -c "print($NOISE * 1.5)")
  N=${NOISE/./p}
  cell "seamnoise_n${N}" --controller policy --checkpoint "$SEAM_CKPT" \
    --task "$SEAM_TASK" --sensor-delay-s 0 \
    --sensor-position-noise-m "$NOISE" --sensor-rotation-noise-rad "$ROT"
done

# 3. Closed-loop analytic controller on the base-degraded task, with the seam fallback
#    now reading the exact offset the policy also sees.
bash scripts/sweep_delay.sh --tag base_ikseam2 --task "$BASE_TASK" \
  --controllers "ik" --ik-mode seam

# 4. Open-loop analytic controller across the wide delay range.
bash scripts/sweep_delay.sh --tag wide_ikpose --task "$WIDE_TASK" \
  --controllers "ik" --ik-mode pose --delays "0 0.0833 0.1667 0.2500 0.3333"

echo FOLLOWUPS_COMPLETE

#!/usr/bin/env bash
# Evaluate a learned policy and/or the analytic controller across measurement delay,
# in the same environment, reading the same degraded base measurement.
#
# Delay is the axis that separates the two: the analytic controller inverts whatever
# it measures right now, so a late measurement costs it roughly (base speed x delay),
# while a policy that sees a window of past measurements can estimate the phase of a
# periodic disturbance and command ahead of it.
#
# Usage:
#   sweep_delay.sh --run RUN_NAME [--task TASK] [--delays "0 0.05 0.1"]
#                  [--controllers "ik policy"] [--tag PREFIX] [--ik-mode seam|pose]
set -uo pipefail
PROJECT_DIR="${MARINE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

RUN=""
TASK=Marine-UR3-Random6DoFBase-WorldLineSensorDegraded-Play-v0
DELAYS="0 0.0333 0.0667 0.1000 0.1333"
CONTROLLERS="ik policy"
TAG=sweep
IK_MODE=seam
ENVS="${ENVS:-4096}"
SEED="${SEED:-44}"
# Noise pinned to the middle of the training range so the sweep isolates delay.
POSITION_NOISE="${POSITION_NOISE:-0.001}"
ROTATION_NOISE="${ROTATION_NOISE:-0.0015}"

while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN="$2"; shift 2;;
    --task) TASK="$2"; shift 2;;
    --delays) DELAYS="$2"; shift 2;;
    --controllers) CONTROLLERS="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --ik-mode) IK_MODE="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

CKPT=""
case " $CONTROLLERS " in
  *" policy "*)
    if [ -z "$RUN" ]; then echo "--run is required to evaluate a policy" >&2; exit 2; fi
    CKPT=$(python3 scripts/best_checkpoint.py "outputs/training/$RUN" --json \
      | python3 -c "import json,sys;print(json.load(sys.stdin)[0]['checkpoint'])")
    if [ -z "$CKPT" ]; then echo "no checkpoint for $RUN" >&2; exit 1; fi
    echo "$TAG policy checkpoint: $CKPT"
    ;;
esac

for DELAY in $DELAYS; do
  MS=$(python3 -c "print(f'{round($DELAY*1000):03d}')")
  for CONTROLLER in $CONTROLLERS; do
    ARGS=(--controller "$CONTROLLER" --task "$TASK" --num-envs "$ENVS" --steps 600
          --seed "$SEED" --sensor-delay-s "$DELAY"
          --sensor-position-noise-m "$POSITION_NOISE"
          --sensor-rotation-noise-rad "$ROTATION_NOISE"
          --run-name "${TAG}_${CONTROLLER}_d${MS}ms")
    if [ "$CONTROLLER" = policy ]; then
      ARGS+=(--checkpoint "$CKPT")
    else
      ARGS+=(--ik-mode "$IK_MODE")
    fi
    if bash scripts/evaluate.sh "${ARGS[@]}" > /dev/null 2>&1; then
      echo "sweep cell done: ${TAG} ${CONTROLLER} ${MS}ms"
    else
      echo "sweep cell FAILED: ${TAG} ${CONTROLLER} ${MS}ms"
    fi
  done
done
echo "SWEEP_COMPLETE ${TAG}"

#!/usr/bin/env bash
# Reproduce Herland & Bach's conditions, then test whether a residual policy recovers
# the headroom finding 17 opened up.
#
# Finding 17 established the precondition: under *actuation* delay the analytic
# controller degrades 1.6-2.1x, where measurement delay cost it nothing. That is the
# regime the previous two rounds never entered, and the one in which a policy has
# something to win.
#
# Three of Herland & Bach's four differences from our setup are applied here - actuation
# delay, a residual architecture, and a recurrent policy. The fourth, a target that moves
# with a second vessel, is not, and stays a stated difference.
set -uo pipefail
PROJECT_DIR="${MARINE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export MARINE_PROJECT_DIR="$PROJECT_DIR"
cd "$PROJECT_DIR"

STATUS="$PROJECT_DIR/logs/residual_pipeline.status"
EVAL="$PROJECT_DIR/outputs/evaluation"
TRAIN="$PROJECT_DIR/outputs/training"
ITER="${RESIDUAL_ITERATIONS:-6000}"
ENVS="${RESIDUAL_ENVS:-4096}"
SEEDS="${RESIDUAL_SEEDS:-970 971 972}"
DELAYS="${RESIDUAL_DELAYS:-0,1,2,3}"
FREQS="${RESIDUAL_FREQS:-1.0,2.0,4.0}"

TRAIN_TASK=Marine-UR3-ResidualIkDelay-v0
PLAY_TASK=Marine-UR3-ResidualIkDelay-Play-v0
REF_TASK=Marine-UR3-Random6DoFBase-WorldLineIkSeeded-Play-v0

note() { printf '%s  %s\n' "$(date -Is)" "$*" >> "$STATUS"; }
have_eval() { [ -s "$EVAL/$1/summary.json" ]; }
have_model() { ls "$TRAIN/$1"/model_*.pt >/dev/null 2>&1; }

note "residual pipeline start: iter=$ITER envs=$ENVS seeds='$SEEDS'"

# Baseline A on the reference task, and the zero-residual control on the residual task.
# The two must agree cell for cell; if they stop agreeing, the residual plumbing has
# started contributing something of its own and no policy number above it is meaningful.
for seed in 44 45 46; do
  if ! have_eval "residual_ref_seed${seed}"; then
    note "start residual_ref_seed${seed}"
    ./scripts/run_py.sh scripts/uvms_sweep.py --run-name "residual_ref_seed${seed}" \
      --task "$REF_TASK" --controllers ik --num-envs 256 --steps 800 --seed "$seed" \
      --actuation-delays "$DELAYS" --wave-frequency-scales "$FREQS" \
      > "logs/residual_ref_seed${seed}.log" 2>&1
    note "done residual_ref_seed${seed}"
  fi
  if ! have_eval "residual_zero_seed${seed}"; then
    note "start residual_zero_seed${seed}"
    ./scripts/run_py.sh scripts/uvms_sweep.py --run-name "residual_zero_seed${seed}" \
      --task "$PLAY_TASK" --controllers zero --num-envs 256 --steps 800 --seed "$seed" \
      --actuation-delays "$DELAYS" --wave-frequency-scales "$FREQS" \
      > "logs/residual_zero_seed${seed}.log" 2>&1
    note "done residual_zero_seed${seed}"
  fi
done

# Training. Sequential: one GPU.
for seed in $SEEDS; do
  name="residual_${ENVS}env_seed${seed}_${ITER}"
  if have_model "$name"; then note "skip train $name"; continue; fi
  note "start train $name"
  ./scripts/run_py.sh scripts/train.py --task "$TRAIN_TASK" \
    --num-envs "$ENVS" --max-iterations "$ITER" --seed "$seed" \
    --run-name "$name" > "logs/$name.log" 2>&1
  if have_model "$name"; then note "done train $name"; else note "FAILED train $name"; fi
done

# Checkpoint selection - never the last one.
CKPT=""
for seed in $SEEDS; do
  name="residual_${ENVS}env_seed${seed}_${ITER}"
  have_model "$name" || continue
  best=$(cd "$PROJECT_DIR/scripts" && python3 best_checkpoint.py --json "$TRAIN/$name" 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["checkpoint"])' 2>/dev/null)
  if [ -n "${best:-}" ] && [ -f "$best" ]; then
    note "seed $seed best: $best"
    CKPT="${CKPT:+$CKPT,}seed${seed}=$best"
  else
    note "seed $seed: no checkpoint selected"
  fi
done
note "checkpoints: ${CKPT:-none}"

# The comparison, at three evaluation seeds.
if [ -n "$CKPT" ]; then
  for seed in 44 45 46; do
    have_eval "residual_full_seed${seed}" && { note "skip residual_full_seed${seed}"; continue; }
    note "start residual_full_seed${seed}"
    ./scripts/run_py.sh scripts/uvms_sweep.py --run-name "residual_full_seed${seed}" \
      --task "$PLAY_TASK" --controllers zero --checkpoints "$CKPT" \
      --num-envs 256 --steps 800 --seed "$seed" \
      --actuation-delays "$DELAYS" --wave-frequency-scales "$FREQS" \
      > "logs/residual_full_seed${seed}.log" 2>&1
    note "done residual_full_seed${seed}"
  done
fi

note "residual pipeline complete"

#!/usr/bin/env bash
# The whole free-floating round, start to finish, in the order docs/UVMS_PLAN.md sets.
#
# Written to run unattended overnight, so every phase appends a line to a status file
# rather than relying on anyone watching the log. A phase that has already produced its
# artefact is skipped, which makes the script restartable: if it dies at seed 3 of
# training, re-running it picks up there instead of redoing the twelve hours before it.
#
# Isaac Sim traps SIGTERM and can hang mid-shutdown with the main thread spinning, so
# each phase runs to completion in the foreground of *this* script and this script is
# what gets detached (see the launcher at the bottom of the file's usage note). Nothing
# here backgrounds a simulator.
set -uo pipefail

PROJECT_DIR="${MARINE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export MARINE_PROJECT_DIR="$PROJECT_DIR"
cd "$PROJECT_DIR"

STATUS="$PROJECT_DIR/logs/uvms_pipeline.status"
LOGDIR="$PROJECT_DIR/logs"
EVAL="$PROJECT_DIR/outputs/evaluation"
TRAIN="$PROJECT_DIR/outputs/training"
mkdir -p "$LOGDIR"

# Sized from a measured 1.0 s/iteration at 4096 environments: three seeds at 6000
# iterations is about five hours, which leaves room for the sweeps either side of it.
ITERATIONS="${UVMS_ITERATIONS:-6000}"
NUM_ENVS="${UVMS_NUM_ENVS:-4096}"
SEEDS="${UVMS_SEEDS:-960 961 962}"
# Training randomises drag over (0.5, 2.0); 0.25 and 4.0 sit outside it on purpose,
# because inside the range randomisation is just training data and the claim the plan
# makes is about extrapolation.
FACTORS="${UVMS_FACTORS:-0.25,0.5,0.7,1.0,1.43,2.0,4.0}"
SWEEP_ENVS="${UVMS_SWEEP_ENVS:-256}"
SWEEP_STEPS="${UVMS_SWEEP_STEPS:-800}"

UVMS_TASK=Marine-UR3-Uvms-WorldLineFreeFloating-Play-v0
UVMS_TRAIN_TASK=Marine-UR3-Uvms-WorldLineFreeFloatingDragRandom-v0
FIXED_TASK=Marine-UR3-Random6DoFBase-WorldLineIkSeeded-Play-v0

note() { printf '%s  %s\n' "$(date -Is)" "$*" >> "$STATUS"; }

# Phase guard: skip when the artefact this phase produces already exists.
have_eval() { [ -s "$EVAL/$1/summary.json" ]; }
have_model() { ls "$TRAIN/$1"/model_*.pt >/dev/null 2>&1; }

run_sweep() {
  local name="$1"; shift
  if have_eval "$name"; then note "skip sweep $name (already present)"; return 0; fi
  note "start sweep $name"
  ./scripts/run_py.sh scripts/uvms_sweep.py --run-name "$name" \
    --num-envs "$SWEEP_ENVS" --steps "$SWEEP_STEPS" "$@" \
    > "$LOGDIR/$name.log" 2>&1
  if have_eval "$name"; then note "done sweep $name"; else note "FAILED sweep $name (see logs/$name.log)"; fi
}

note "pipeline start: iterations=$ITERATIONS envs=$NUM_ENVS seeds='$SEEDS' factors=$FACTORS"

# ---------------------------------------------------------------------------------
# Stage 1e - the hydrodynamic model does what it claims, and the dynamic model the
# coupled baseline inverts matches what PhysX actually does. Both are cheap and both
# are re-run here so the pipeline's own artefacts carry their provenance.
# ---------------------------------------------------------------------------------
if ! have_eval uvms_verify; then
  note "start verify_uvms"
  ./scripts/run_py.sh scripts/verify_uvms.py --num-envs 16 --seconds 6.0 \
    --run-name uvms_verify > "$LOGDIR/uvms_verify.log" 2>&1
  note "done verify_uvms"
fi
if ! have_eval uvms_coupling_model; then
  note "start verify_coupling_model"
  ./scripts/run_py.sh scripts/verify_coupling_model.py \
    --run-name uvms_coupling_model > "$LOGDIR/uvms_coupling_model.log" 2>&1
  note "done verify_coupling_model"
fi

# ---------------------------------------------------------------------------------
# Stage 0 decisive test + Stage 2 baselines.
#
# Four conditions. With the sea state silenced, the only thing moving the hull is the
# arm's own reaction, so fixed-versus-free *is* the coupling and nothing else. With it
# on, the same pair measures the coupling on top of a realistic disturbance. Baseline B
# only exists on the free-floating task: on a fixed base there is no hull to predict.
# ---------------------------------------------------------------------------------
run_sweep stage2_fixed_nowaves --task "$FIXED_TASK" --controllers ik --drag-factors 1.0 --no-waves
run_sweep stage2_uvms_nowaves  --task "$UVMS_TASK"  --controllers ik,coupled --drag-factors 1.0 --no-waves
run_sweep stage2_fixed_waves   --task "$FIXED_TASK" --controllers ik --drag-factors 1.0
run_sweep stage2_uvms_waves    --task "$UVMS_TASK"  --controllers ik,coupled --drag-factors 1.0

# ---------------------------------------------------------------------------------
# Stage 3 - the mismatch axis, baselines only. Run before any training so the kill
# criterion is read off a curve that no policy result can have influenced.
# ---------------------------------------------------------------------------------
run_sweep stage3_baselines --task "$UVMS_TASK" --controllers ik,coupled --drag-factors "$FACTORS"

# ---------------------------------------------------------------------------------
# Stage 4 - the policy, three seeds. Sequential: one GPU.
# ---------------------------------------------------------------------------------
for seed in $SEEDS; do
  name="uvms_drag_${NUM_ENVS}env_seed${seed}_${ITERATIONS}"
  if have_model "$name"; then note "skip train $name (checkpoints present)"; continue; fi
  note "start train $name"
  ./scripts/run_py.sh scripts/train.py --task "$UVMS_TRAIN_TASK" \
    --num-envs "$NUM_ENVS" --max-iterations "$ITERATIONS" --seed "$seed" \
    --run-name "$name" > "$LOGDIR/$name.log" 2>&1
  if have_model "$name"; then note "done train $name"; else note "FAILED train $name"; fi
done

# ---------------------------------------------------------------------------------
# Checkpoint selection. Never the last one: every run in this project peaks early and
# then degrades (docs/FINDINGS.md), and reporting the final checkpoint is on the plan's
# list of things not to repeat.
# ---------------------------------------------------------------------------------
CKPT_ARGS=""
for seed in $SEEDS; do
  name="uvms_drag_${NUM_ENVS}env_seed${seed}_${ITERATIONS}"
  have_model "$name" || continue
  best=$(cd "$PROJECT_DIR/scripts" && python3 best_checkpoint.py --json "$TRAIN/$name" 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["checkpoint"])' 2>/dev/null)
  if [ -n "${best:-}" ] && [ -f "$best" ]; then
    note "seed $seed best checkpoint: $best"
    CKPT_ARGS="${CKPT_ARGS:+$CKPT_ARGS,}seed${seed}=$best"
  else
    note "seed $seed: could not select a checkpoint"
  fi
done
note "checkpoints: ${CKPT_ARGS:-none}"

# ---------------------------------------------------------------------------------
# Stage 3 again, now with the policies alongside the baselines, all on identical
# episodes at each pinned mismatch.
# ---------------------------------------------------------------------------------
if [ -n "$CKPT_ARGS" ]; then
  run_sweep stage3_full --task "$UVMS_TASK" --controllers ik,coupled \
    --checkpoints "$CKPT_ARGS" --drag-factors "$FACTORS"
  run_sweep stage4_nowaves --task "$UVMS_TASK" --controllers ik,coupled \
    --checkpoints "$CKPT_ARGS" --drag-factors 1.0 --no-waves
fi

# ---------------------------------------------------------------------------------
# Tables.
# ---------------------------------------------------------------------------------
note "start report"
python3 "$PROJECT_DIR/scripts/uvms_report.py" > "$LOGDIR/uvms_report.log" 2>&1 \
  && note "done report" || note "FAILED report (see logs/uvms_report.log)"

note "pipeline complete"

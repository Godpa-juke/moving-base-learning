#!/usr/bin/env bash
# Two follow-ups the first pass left as assertions rather than measurements.
#
# 1. **Mass sweep.** Stage 0's kill criterion failed at 100 kg: releasing the base cost
#    the controller nothing measurable. That has two possible causes with opposite
#    implications — the vehicle is too heavy to be pushed, or the task's arm motion is
#    too gentle to push anything — and only a sweep separates them. The plan asked for
#    {50, 100, 200, 500, 2000} kg and the first pass ran one of them.
#
# 2. **Term ablation.** Baseline B beat baseline A by 39% under waves. B adds two things
#    to A: a feedforward for where the hull will drift, and a Jacobian correction for
#    the hull's recoil from the arm's own command. Reporting the gap without saying
#    which half produced it is exactly the unverified-mechanism failure the plan's last
#    section forbids, so each half is run alone.
#
# Same settings as the first pass throughout, so the numbers drop into the same tables.
set -uo pipefail

PROJECT_DIR="${MARINE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export MARINE_PROJECT_DIR="$PROJECT_DIR"
cd "$PROJECT_DIR"

STATUS="$PROJECT_DIR/logs/uvms_pipeline.status"
LOGDIR="$PROJECT_DIR/logs"
EVAL="$PROJECT_DIR/outputs/evaluation"
SWEEP_ENVS="${UVMS_SWEEP_ENVS:-256}"
SWEEP_STEPS="${UVMS_SWEEP_STEPS:-800}"
FACTORS="${UVMS_FACTORS:-0.25,0.5,0.7,1.0,1.43,2.0,4.0}"

note() { printf '%s  %s\n' "$(date -Is)" "$*" >> "$STATUS"; }
have_eval() { [ -s "$EVAL/$1/summary.json" ]; }

run_sweep() {
  local name="$1"; shift
  if have_eval "$name"; then note "skip sweep $name (already present)"; return 0; fi
  note "start sweep $name"
  ./scripts/run_py.sh scripts/uvms_sweep.py --run-name "$name" \
    --num-envs "$SWEEP_ENVS" --steps "$SWEEP_STEPS" "$@" \
    > "$LOGDIR/$name.log" 2>&1
  if have_eval "$name"; then note "done sweep $name"; else note "FAILED sweep $name"; fi
}

note "followups start"

# Mass sweep. Sea state off, so the arm's own reaction is the only thing moving the hull
# and the comparison against the fixed base is the coupling alone.
for mass in 50 200 500 2000; do
  run_sweep "stage0_mass${mass}_nowaves" \
    --task "Marine-UR3-Uvms-WorldLineFreeFloating${mass}kg-Play-v0" \
    --controllers ik,coupled --drag-factors 1.0 --no-waves
done

# The same masses with the sea state on: a lighter hull is thrown further by the same
# wave force, so this is where a mass dependence should show up if there is one.
for mass in 50 200 500 2000; do
  run_sweep "stage0_mass${mass}_waves" \
    --task "Marine-UR3-Uvms-WorldLineFreeFloating${mass}kg-Play-v0" \
    --controllers ik,coupled --drag-factors 1.0
done

# Term ablation, across the full mismatch axis: which half of baseline B carries it, and
# does that answer change as the drag model goes wrong.
run_sweep stage2_ablation \
  --controllers ik,coupled,coupled:drift,coupled:reaction,coupled:neither \
  --drag-factors "$FACTORS"

note "start report"
python3 "$PROJECT_DIR/scripts/uvms_report.py" > "$LOGDIR/uvms_report.log" 2>&1 \
  && note "done report" || note "FAILED report"
note "followups complete"

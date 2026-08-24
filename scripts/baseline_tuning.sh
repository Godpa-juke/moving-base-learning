#!/usr/bin/env bash
# Is baseline A fairly tuned, or is the residual policy beating a strawman?
#
# The residual result is a near-constant absolute gain present even at zero delay, which
# says the analytic controller is suboptimal generally rather than delay-specifically.
# Baseline A runs a fixed proportional gain of 1.0 at every condition. Under a delay
# inside the loop, the textbook classical response is to lower the gain until the loop
# is stable again - so a fixed gain is exactly the handicap that would manufacture our
# headline.
#
# Sweeps the gain at every delay and frequency. The fair baseline is then the best gain
# per cell, not the gain we happened to pick. This applies finding 15's rule - name the
# control and run it - to our own headline result.
set -uo pipefail
PROJECT_DIR="${MARINE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export MARINE_PROJECT_DIR="$PROJECT_DIR"
cd "$PROJECT_DIR"
STATUS="$PROJECT_DIR/logs/baseline_tuning.status"
note() { printf '%s  %s\n' "$(date -Is)" "$*" >> "$STATUS"; }
TASK=Marine-UR3-Random6DoFBase-WorldLineSensorDegraded-Play-v0

note "baseline tuning start"
for gain in 0.2 0.4 0.6 0.8; do
  name="tune_gain${gain/./p}"
  [ -s "$PROJECT_DIR/outputs/evaluation/$name/summary.json" ] && { note "skip $name"; continue; }
  note "start $name"
  ./scripts/run_py.sh scripts/uvms_sweep.py --run-name "$name" \
    --task "$TASK" --controllers ik --num-envs 256 --steps 800 --seed 44 \
    --ik-gain "$gain" --actuation-delays 0,1,2,3 --wave-frequency-scales 1.0,2.0,4.0 \
    --sensor-delays-s 0.0 > "logs/$name.log" 2>&1
  note "done $name"
done
note "baseline tuning complete"

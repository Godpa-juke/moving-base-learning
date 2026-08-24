#!/usr/bin/env bash
# Reproduce the conditions under which Herland & Bach (2023) report RL beating an IK
# controller, starting with the baseline alone.
#
# They report their IK baseline degrading roughly fourfold on 40 ms of *actuation* delay.
# The previous round swept *measurement* delay to 333 ms and found the analytic
# controller flat (docs/FINDINGS.md 10). Four differences separate the two studies; this
# script tests the two that cost nothing to test, before any policy is trained:
#
#   1. delay type   - actuation (inside the loop) against measurement (outside it)
#   2. disturbance  - this study's 0.06-0.3 Hz against their effective 0.32-0.53 Hz
#
# Baseline only, on purpose. If the analytic controller does not degrade under these
# conditions, no policy result on top of them would mean anything.
set -uo pipefail
PROJECT_DIR="${MARINE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export MARINE_PROJECT_DIR="$PROJECT_DIR"
cd "$PROJECT_DIR"
STATUS="$PROJECT_DIR/logs/delay_repro.status"
note() { printf '%s  %s\n' "$(date -Is)" "$*" >> "$STATUS"; }

# The sensor-degraded task is the one that defines the measurement-delay event, and it
# inherits the IK-seeded task otherwise, so both delay types can be swept on one task.
TASK=Marine-UR3-Random6DoFBase-WorldLineSensorDegraded-Play-v0

note "start delay_repro"

# 1. Actuation delay against wave frequency, no measurement delay.
./scripts/run_py.sh scripts/uvms_sweep.py --run-name repro_actuation \
  --task "$TASK" --controllers ik --num-envs 256 --steps 800 \
  --actuation-delays 0,1,2,3 --wave-frequency-scales 1.0,2.0,4.0 \
  --sensor-delays-s 0.0 > logs/repro_actuation.log 2>&1
note "done repro_actuation"

# 2. Measurement delay over the same span, actuation delay held at zero. 33/67/100 ms
#    match the actuation steps above so the two axes are directly comparable.
./scripts/run_py.sh scripts/uvms_sweep.py --run-name repro_measurement \
  --task "$TASK" --controllers ik --num-envs 256 --steps 800 \
  --actuation-delays 0 --wave-frequency-scales 1.0,2.0,4.0 \
  --sensor-delays-s 0.0,0.0333,0.0667,0.1 > logs/repro_measurement.log 2>&1
note "done repro_measurement"

note "delay_repro complete"

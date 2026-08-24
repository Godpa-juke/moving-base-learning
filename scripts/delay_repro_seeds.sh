#!/usr/bin/env bash
# The same two sweeps at three evaluation seeds.
#
# The single-seed pass had one cell (freq x1, zero delay) carrying a 378 mm worst sample
# and an RMSE of 7.26 mm against ~2.9 mm elsewhere - the reach-limit tail of findings 5
# and 8. RMSE is sensitive to exactly that, and finding 13 already established that cells
# at the ends of an axis need a median across seeds before anything is read off them.
set -uo pipefail
PROJECT_DIR="${MARINE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export MARINE_PROJECT_DIR="$PROJECT_DIR"
cd "$PROJECT_DIR"
STATUS="$PROJECT_DIR/logs/delay_repro.status"
note() { printf '%s  %s\n' "$(date -Is)" "$*" >> "$STATUS"; }
TASK=Marine-UR3-Random6DoFBase-WorldLineSensorDegraded-Play-v0

for seed in 45 46; do
  note "start repro_actuation_seed${seed}"
  ./scripts/run_py.sh scripts/uvms_sweep.py --run-name "repro_actuation_seed${seed}" \
    --task "$TASK" --controllers ik --num-envs 256 --steps 800 --seed "$seed" \
    --actuation-delays 0,1,2,3 --wave-frequency-scales 1.0,2.0,4.0 \
    --sensor-delays-s 0.0 > "logs/repro_actuation_seed${seed}.log" 2>&1
  note "done repro_actuation_seed${seed}"

  note "start repro_measurement_seed${seed}"
  ./scripts/run_py.sh scripts/uvms_sweep.py --run-name "repro_measurement_seed${seed}" \
    --task "$TASK" --controllers ik --num-envs 256 --steps 800 --seed "$seed" \
    --actuation-delays 0 --wave-frequency-scales 1.0,2.0,4.0 \
    --sensor-delays-s 0.0,0.0333,0.0667,0.1 > "logs/repro_measurement_seed${seed}.log" 2>&1
  note "done repro_measurement_seed${seed}"
done
note "delay_repro_seeds complete"

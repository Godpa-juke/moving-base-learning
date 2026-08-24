#!/usr/bin/env bash
# Gain sweep at two more evaluation seeds. The claim "the tuned baseline beats the policy
# at 100 ms with a slow disturbance" currently rests on two cells at -3% and -2%, from a
# single evaluation seed. Finding 13 already showed single cells swinging 3.09/1.32/1.51 mm.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MARINE_PROJECT_DIR="$PWD"
S="$PWD/logs/tune_seeds.status"
note() { printf '%s  %s\n' "$(date -Is)" "$*" >> "$S"; }
TASK=Marine-UR3-Random6DoFBase-WorldLineSensorDegraded-Play-v0
note "start"
for seed in 45 46; do
  for g in 1.2 1.5 2.0 2.5 3.0; do
    name="tune_gain${g/./p}_seed${seed}"
    [ -s "outputs/evaluation/$name/summary.json" ] && continue
    ./scripts/run_py.sh scripts/uvms_sweep.py --run-name "$name" --task "$TASK" \
      --controllers ik --num-envs 256 --steps 800 --seed "$seed" --ik-gain "$g" \
      --actuation-delays 0,1,2,3 --wave-frequency-scales 1.0,2.0,4.0 --sensor-delays-s 0.0 \
      > "logs/$name.log" 2>&1
  done
  note "seed $seed done"
done
note "complete"

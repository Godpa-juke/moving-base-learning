#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${MARINE_PROJECT_DIR:-$HOME/00_dev/08-marine-manipulator}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/00_dev/02-isaacsim/robot-poc/IsaacLab}"
TASK="Marine-UR3-ResidualIkDelay-v0"
NUM_ENVS=4096
ITERATIONS=6000
SEEDS=(970 971 972)
BATCH_DIR="$PROJECT_DIR/outputs/training/zerohead_batch_4096env_6000"
mkdir -p "$BATCH_DIR"
LOG="$BATCH_DIR/batch.log"
exec > >(tee -a "$LOG") 2>&1

export MARINE_PROJECT_DIR="$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$ISAACLAB_DIR/source/isaaclab:$ISAACLAB_DIR/source/isaaclab_assets:$ISAACLAB_DIR/source/isaaclab_tasks:$ISAACLAB_DIR/source/isaaclab_rl:$ISAACLAB_DIR/source/isaaclab_mimic:$ISAACLAB_DIR/source/isaaclab_contrib:${PYTHONPATH:-}"

printf '[zerohead] started=%s\n' "$(date --iso-8601=seconds)"
for seed in "${SEEDS[@]}"; do
  run="residual_zerohead_4096env_seed${seed}_6000"
  out="$PROJECT_DIR/outputs/training/$run"
  if python3 - "$out/summary.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.exists():
    raise SystemExit(1)
d = json.loads(p.read_text())
ok = (
    d.get("status") == "fresh_train_ok"
    and d.get("num_envs") == 4096
    and d.get("max_iterations") == 6000
    and bool(d.get("zero_init_note"))
    and d.get("init_noise_std") is None
)
raise SystemExit(0 if ok else 1)
PY
  then
    echo "[zerohead] skip verified run=$run"
    continue
  fi
  if [[ -d "$out" ]]; then
    archived="${out}.partial.$(date +%Y%m%d_%H%M%S)"
    mv "$out" "$archived"
    echo "[zerohead] archived incomplete run to $archived"
  fi
  echo "[zerohead] train seed=$seed run=$run"
  cd "$ISAACLAB_DIR"
  ./isaaclab.sh -p "$PROJECT_DIR/scripts/train.py" \
    --task "$TASK" \
    --num-envs "$NUM_ENVS" \
    --max-iterations "$ITERATIONS" \
    --seed "$seed" \
    --run-name "$run" \
    --zero-init-residual \
    --device cuda:0
  echo "[zerohead] completed seed=$seed run=$run"
done

python3 - "$PROJECT_DIR" "$BATCH_DIR/batch_summary.json" <<'PY'
import json, pathlib, sys
project = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
rows = []
for seed in (970, 971, 972):
    run = f"residual_zerohead_4096env_seed{seed}_6000"
    p = project / "outputs" / "training" / run / "summary.json"
    d = json.loads(p.read_text())
    rows.append({
        "seed": seed,
        "run": run,
        "status": d.get("status"),
        "elapsed_sec": d.get("elapsed_sec"),
        "latest_model": d.get("latest_model"),
        "zero_init_note": d.get("zero_init_note"),
        "init_noise_std": d.get("init_noise_std"),
    })
payload = {
    "task": "Marine-UR3-ResidualIkDelay-v0",
    "num_envs": 4096,
    "iterations": 6000,
    "initialization": "zero actor output layer; unchanged PPO initial std 0.5",
    "rows": rows,
    "status": "complete" if all(r["status"] == "fresh_train_ok" for r in rows) else "incomplete",
}
out.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, sort_keys=True))
PY
printf '[zerohead] finished=%s\n' "$(date --iso-8601=seconds)"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${MARINE_PROJECT_DIR:-$HOME/00_dev/08-marine-manipulator}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/00_dev/02-isaacsim/robot-poc/IsaacLab}"
TASK="Marine-UR3-ResidualIkDelay-Play-v0"
TRAIN_BATCH="$PROJECT_DIR/outputs/training/zerohead_batch_4096env_6000/batch_summary.json"
EVAL_BATCH="$PROJECT_DIR/outputs/evaluation/zerohead_matched_grid_batch"
mkdir -p "$EVAL_BATCH"
exec > >(tee -a "$EVAL_BATCH/batch.log") 2>&1

export MARINE_PROJECT_DIR="$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$ISAACLAB_DIR/source/isaaclab:$ISAACLAB_DIR/source/isaaclab_assets:$ISAACLAB_DIR/source/isaaclab_tasks:$ISAACLAB_DIR/source/isaaclab_rl:$ISAACLAB_DIR/source/isaaclab_mimic:$ISAACLAB_DIR/source/isaaclab_contrib:${PYTHONPATH:-}"

printf '[zerohead-eval] waiter_started=%s\n' "$(date --iso-8601=seconds)"
for _ in $(seq 1 1080); do
  if python3 - "$TRAIN_BATCH" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
if not p.exists(): raise SystemExit(1)
d=json.loads(p.read_text())
raise SystemExit(0 if d.get("status") == "complete" and len(d.get("rows", [])) == 3 else 1)
PY
  then break; fi
  sleep 30
done
[[ -f "$TRAIN_BATCH" ]] || { echo "training batch did not complete before timeout"; exit 1; }

runs=(
  "$PROJECT_DIR/outputs/training/residual_zerohead_4096env_seed970_6000"
  "$PROJECT_DIR/outputs/training/residual_zerohead_4096env_seed971_6000"
  "$PROJECT_DIR/outputs/training/residual_zerohead_4096env_seed972_6000"
)
python3 "$PROJECT_DIR/scripts/best_checkpoint.py" --json "${runs[@]}" > "$EVAL_BATCH/checkpoint_selection.json"
checkpoints=$(python3 - "$EVAL_BATCH/checkpoint_selection.json" <<'PY'
import json, sys
rows=json.load(open(sys.argv[1]))
labels=("z970","z971","z972")
print(",".join(f"{label}={row['checkpoint']}" for label,row in zip(labels,rows)))
PY
)
echo "[zerohead-eval] checkpoints=$checkpoints"

for seed in 44 45 46; do
  run="residual_zerohead_grid_seed${seed}"
  summary="$PROJECT_DIR/outputs/evaluation/$run/summary.json"
  if python3 - "$summary" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
if not p.exists(): raise SystemExit(1)
d=json.loads(p.read_text())
ok=(d.get("status")=="uvms_sweep_ok" and d.get("num_envs")==256 and d.get("steps")==800 and len(d.get("rows",[]))==36)
raise SystemExit(0 if ok else 1)
PY
  then echo "[zerohead-eval] skip verified run=$run"; continue; fi
  if [[ -d "$PROJECT_DIR/outputs/evaluation/$run" ]]; then
    mv "$PROJECT_DIR/outputs/evaluation/$run" "$PROJECT_DIR/outputs/evaluation/${run}.partial.$(date +%Y%m%d_%H%M%S)"
  fi
  cd "$ISAACLAB_DIR"
  ./isaaclab.sh -p "$PROJECT_DIR/scripts/uvms_sweep.py" \
    --task "$TASK" --controllers '' --checkpoints "$checkpoints" \
    --drag-factors 1.0 --actuation-delays 0,1,2,3 \
    --wave-frequency-scales 1.0,2.0,4.0 \
    --num-envs 256 --steps 800 --seed "$seed" --run-name "$run" --device cuda:0
  echo "[zerohead-eval] completed seed=$seed run=$run"
done

python3 - "$PROJECT_DIR" "$EVAL_BATCH/batch_summary.json" <<'PY'
import json, pathlib, sys
project,out=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2])
runs=[]
for seed in (44,45,46):
 p=project/"outputs"/"evaluation"/f"residual_zerohead_grid_seed{seed}"/"summary.json"
 d=json.loads(p.read_text()); runs.append({"seed":seed,"path":str(p),"rows":len(d["rows"]),"status":d["status"]})
payload={"status":"complete" if all(r["status"]=="uvms_sweep_ok" and r["rows"]==36 for r in runs) else "incomplete","runs":runs}
out.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,sort_keys=True))
PY
printf '[zerohead-eval] finished=%s\n' "$(date --iso-8601=seconds)"

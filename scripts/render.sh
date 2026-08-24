#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${MARINE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/00_dev/02-isaacsim/robot-poc/IsaacLab}"
export MARINE_PROJECT_DIR="$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$ISAACLAB_DIR/source/isaaclab:$ISAACLAB_DIR/source/isaaclab_assets:$ISAACLAB_DIR/source/isaaclab_tasks:$ISAACLAB_DIR/source/isaaclab_rl:$ISAACLAB_DIR/source/isaaclab_mimic:$ISAACLAB_DIR/source/isaaclab_contrib:${PYTHONPATH:-}"
cd "$ISAACLAB_DIR"
exec ./isaaclab.sh -p "$PROJECT_DIR/scripts/render.py" "$@"

#!/usr/bin/env bash
# Run an arbitrary project script under the Isaac Lab interpreter.
#
# scripts/train.sh hard-codes scripts/train.py; the staged UVMS work needs the
# same environment for one-off probes, so the launcher is factored out here.
set -euo pipefail
PROJECT_DIR="${MARINE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/00_dev/02-isaacsim/robot-poc/IsaacLab}"
SCRIPT="${1:?usage: run_py.sh SCRIPT [args...]}"
shift
export MARINE_PROJECT_DIR="$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$ISAACLAB_DIR/source/isaaclab:$ISAACLAB_DIR/source/isaaclab_assets:$ISAACLAB_DIR/source/isaaclab_tasks:$ISAACLAB_DIR/source/isaaclab_rl:$ISAACLAB_DIR/source/isaaclab_mimic:$ISAACLAB_DIR/source/isaaclab_contrib:${PYTHONPATH:-}"
cd "$ISAACLAB_DIR"
case "$SCRIPT" in /*) TARGET="$SCRIPT" ;; *) TARGET="$PROJECT_DIR/$SCRIPT" ;; esac
exec ./isaaclab.sh -p "$TARGET" "$@"

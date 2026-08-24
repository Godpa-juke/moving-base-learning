#!/usr/bin/env python3
"""Pick the best checkpoint of an rsl_rl run instead of the last one.

Every run in this project peaks early and then degrades, so ``model_<last>.pt``
is systematically the worst checkpoint. This selects by the smoothed tracking
metric logged in the run's tfevents file and snaps to the nearest saved model.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tfevents import read_run

DEFAULT_METRIC = "Metrics/ee_pose/position_error"


def smooth(values: list[float], window: int) -> list[float]:
    half = max(1, window // 2)
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def saved_iterations(run_dir: Path) -> list[int]:
    iterations = []
    for path in run_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)", path.stem)
        if match:
            iterations.append(int(match.group(1)))
    return sorted(iterations)


def select(run_dir: Path, metric: str, window: int, warmup: int, maximize: bool) -> dict:
    # Absolute path: evaluate.sh runs from the Isaac Lab tree, so a relative checkpoint
    # resolves against the wrong directory and every policy evaluation fails on a
    # missing file while the analytic-controller cells, which need no checkpoint, pass.
    run_dir = Path(run_dir).resolve()
    series = read_run(run_dir).get(metric)
    if not series:
        raise SystemExit(f"metric {metric!r} not found in {run_dir}")
    steps = [s for s, _ in series]
    values = [v for _, v in series]
    curve = smooth(values, window)
    start = min(warmup, len(curve) - 1)
    pick = max if maximize else min
    best_index = pick(range(start, len(curve)), key=lambda i: curve[i])
    best_step = steps[best_index]

    available = saved_iterations(run_dir)
    if not available:
        raise SystemExit(f"no model_*.pt in {run_dir}")
    nearest = min(available, key=lambda it: abs(it - best_step))
    final = available[-1]
    final_index = min(range(len(steps)), key=lambda i: abs(steps[i] - final))
    return {
        "run": str(run_dir),
        "metric": metric,
        "smooth_window": window,
        "best_iteration": best_step,
        "best_metric_smoothed": curve[best_index],
        "best_metric_raw": values[best_index],
        "checkpoint": str(run_dir / f"model_{nearest}.pt"),
        "checkpoint_iteration": nearest,
        "final_iteration": final,
        "final_metric_smoothed": curve[final_index],
        "degradation_ratio": (
            curve[final_index] / curve[best_index] if curve[best_index] else float("nan")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--window", type=int, default=101, help="smoothing window in iterations")
    parser.add_argument("--warmup", type=int, default=50, help="ignore this many leading points")
    parser.add_argument("--maximize", action="store_true", help="metric is better when larger")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = [select(d, args.metric, args.window, args.warmup, args.maximize) for d in args.run_dirs]
    if args.json:
        print(json.dumps(results, indent=2))
        return
    for r in results:
        print(f"{Path(r['run']).name}")
        print(
            f"  best   iter {r['best_iteration']:>6}  {r['best_metric_smoothed'] * 1000:8.2f} mm"
            f"   -> {Path(r['checkpoint']).name}"
        )
        print(
            f"  final  iter {r['final_iteration']:>6}  {r['final_metric_smoothed'] * 1000:8.2f} mm"
            f"   ({r['degradation_ratio']:.1f}x worse)"
        )


if __name__ == "__main__":
    main()

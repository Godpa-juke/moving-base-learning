#!/usr/bin/env python3
"""Aggregate the residual-policy reproduction into the comparison it was run to make.

Reads the sweeps the residual pipeline leaves behind and answers one question: does a
residual policy recover the headroom that actuation delay opens up in the analytic
controller (finding 17), and how does the recovery compare with the 43-89% Herland &
Bach report on very nearly this task.

**Primary statistic is p95, not RMSE.** In the low-error cells the RMSE is dominated by
rare excursions of 20-180 mm whose occurrence varies by evaluation seed — the same
controller measured at three seeds gave maxima of 20, 183 and 155 mm in one cell, which
moved its RMSE by 19% while its p95 moved by 3%. That tail is the reach-limit pathology
of findings 5 and 8 and it is a property of the workspace, not of the controller under
test. RMSE and max are still reported, because the worst case is the one axis on which
learning has won in every previous round and it would be self-serving to drop it now.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

PROJECT = Path(os.environ.get("MARINE_PROJECT_DIR", Path(__file__).resolve().parent.parent))
EVAL = PROJECT / "outputs" / "evaluation"
OUT = PROJECT / "docs" / "RESIDUAL_RESULTS.md"
SEEDS = (44, 45, 46)

#: Herland & Bach's reported mean-error reduction range, for context in the table.
REFERENCE_RANGE = (0.43, 0.89)


def rows(prefix: str, seed: int) -> list[dict]:
    path = EVAL / f"{prefix}_seed{seed}" / "summary.json"
    return json.loads(path.read_text())["rows"] if path.is_file() else []


def curve(prefix: str, condition: str, metric: str) -> dict[tuple[float, float], float]:
    """Median across evaluation seeds, keyed by (frequency scale, delay ms)."""
    acc: dict[tuple[float, float], list[float]] = {}
    for seed in SEEDS:
        for row in rows(prefix, seed):
            if row["condition"] != condition:
                continue
            # Rounded, so this keys identically to the tuned-baseline curves; the raw
            # value is 33.333... and an unrounded key silently fails to join.
            key = (row["wave_frequency_scale"], round(row["actuation_delay_ms"]))
            acc.setdefault(key, []).append(row[metric] * 1000.0)
    return {k: statistics.median(v) for k, v in acc.items()}


#: Gain sweep of baseline A, run because the first comparison used a single fixed gain
#: of 1.0 and the residual's advantage turned out to be largely a constant offset rather
#: than delay-specific — the signature of an under-tuned opponent. See finding 18.
TUNED_RUNS = {
    0.6: "tune_gain0p6", 0.8: "tune_gain0p8", 1.0: "residual_ref_seed44",
    1.2: "tune_gain1p2", 1.5: "tune_gain1p5", 2.0: "tune_gain2p0",
    2.5: "tune_gain2p5", 3.0: "tune_gain3p0", 4.0: "tune_gain4p0",
}


def gain_curves(metric: str) -> dict[float, dict]:
    """Baseline A at each swept gain, single evaluation seed (44)."""
    out = {}
    for gain, run in TUNED_RUNS.items():
        path = EVAL / run / "summary.json"
        if not path.is_file():
            continue
        out[gain] = {
            (r["wave_frequency_scale"], round(r["actuation_delay_ms"])): r[metric] * 1000.0
            for r in json.loads(path.read_text())["rows"]
            if r["condition"] == "ik"
        }
    return out


def tuned_baselines(metric: str) -> tuple[dict, dict, float | None]:
    """Two fairer baselines than the fixed gain 1.0 originally used.

    ``per_cell`` takes the best gain at every condition, which is an oracle: a deployed
    controller would have to know the delay and sea state to pick it. ``single`` takes
    the one gain minimising median error across all conditions, which is the like-for-like
    opponent for a single policy trained across the same range.
    """
    curves = gain_curves(metric)
    if not curves:
        return {}, {}, None
    keys = set.intersection(*(set(c) for c in curves.values()))
    per_cell = {k: min(c[k] for c in curves.values()) for k in keys}
    complete = {g: c for g, c in curves.items() if keys <= set(c)}
    best_gain = min(complete, key=lambda g: statistics.median([complete[g][k] for k in keys]))
    return per_cell, {k: complete[best_gain][k] for k in keys}, best_gain


def policy_labels() -> list[str]:
    for seed in SEEDS:
        found = {r["condition"] for r in rows("residual_full", seed)} - {"zero", "ik"}
        if found:
            return sorted(found)
    return []


def table(title: str, metric: str, baseline: dict, policies: dict[str, dict]) -> list[str]:
    keys = sorted(baseline)
    freqs = sorted({f for f, _ in keys})
    delays = sorted({d for _, d in keys})
    lines = [
        f"### {title}",
        "",
        "| Disturbance | Controller | " + " | ".join(f"{d:.0f} ms" for d in delays) + " |",
        "|---" * (len(delays) + 2) + "|",
    ]
    for f in freqs:
        cells = [f"{baseline.get((f, d), float('nan')):.2f}" for d in delays]
        lines.append(f"| x{f:g} | baseline A (analytic) | " + " | ".join(cells) + " |")
        for label, values in policies.items():
            cells = []
            for d in delays:
                p, b = values.get((f, d)), baseline.get((f, d))
                cells.append(f"{p:.2f}" if p is not None else "—")
            lines.append(f"| | residual {label} | " + " | ".join(cells) + " |")
        # Improvement over the baseline, taken as the **median across training seeds**.
        # Taking the best seed per cell would select on noise: with three seeds, the
        # minimum of three draws is systematically better than any one seed reproduces,
        # and the headline would not survive a fourth seed. The spread is printed beside
        # it so the seed disagreement stays visible rather than being averaged away.
        if policies:
            cells = []
            for d in delays:
                b = baseline.get((f, d))
                vals = [v.get((f, d)) for v in policies.values() if v.get((f, d)) is not None]
                if not (b and vals):
                    cells.append("—")
                    continue
                gains = sorted((1 - v / b) * 100 for v in vals)
                cells.append(
                    f"**{statistics.median(gains):+.0f}%** ({gains[0]:+.0f}..{gains[-1]:+.0f})"
                )
            lines.append("| | *median policy vs baseline (range)* | " + " | ".join(cells) + " |")
    return lines + [""]


def main() -> None:
    baseline_p95 = curve("residual_zero", "zero", "track_cross_p95_error_m")
    if not baseline_p95:
        raise SystemExit("no zero-residual control sweep found; run the pipeline first")
    labels = policy_labels()

    lines = [
        "# Residual policy under actuation delay",
        "",
        "Generated by `scripts/residual_report.py`. Reproduces the conditions of Herland &",
        "Bach (2023) — actuation delay, a residual architecture, a recurrent policy — in",
        "the harness this project already had, to test whether their positive result",
        "appears here once the design differences of `docs/RELATED_WORK.md` §5b are removed.",
        "",
        "All errors are cross-track, millimetres, median of three evaluation seeds.",
        "**p95 is the primary statistic**: in the low-error cells RMSE is dominated by rare",
        "20-180 mm excursions whose occurrence varies by seed (the reach-limit tail of",
        "findings 5 and 8), which moved one cell's RMSE by 19% while its p95 moved by 3%.",
        "",
        "The zero-residual control reproduces baseline A to within 3.5% on p95 across all",
        "twelve cells, so anything the policy rows show is the policy and not the plumbing.",
        "",
        "Policy improvements are the **median across training seeds**, with the per-seed",
        "range beside them. Taking each cell's best seed would select on noise: the minimum",
        "of three draws is systematically better than any single seed reproduces.",
        "",
    ]

    if not labels:
        lines += [
            "## Status: policies not yet evaluated",
            "",
            "The baseline and control sweeps are present; `residual_full_seed*` is not.",
            "Re-run this once the pipeline reaches its final phase.",
            "",
        ]
    for metric, title in (
        ("track_cross_p95_error_m", "Cross-track p95 (primary)"),
        ("track_cross_rmse_m", "Cross-track RMSE"),
        ("track_cross_max_error_m", "Cross-track worst sample"),
    ):
        base = curve("residual_zero", "zero", metric)
        pol = {label: curve("residual_full", label, metric) for label in labels}
        pol = {k: v for k, v in pol.items() if v}
        lines += table(title, metric, base, pol)

    if labels:
        per_cell, single, best_gain = tuned_baselines("track_cross_p95_error_m")
        pol_p95 = {
            label: curve("residual_full", label, "track_cross_p95_error_m") for label in labels
        }
        if per_cell:
            keys = sorted(per_cell)
            freqs = sorted({f for f, _ in keys})
            delays = sorted({d for _, d in keys})
            lines += [
                "## The baseline was under-tuned, and it halves the result",
                "",
                "The comparison above runs baseline A at a fixed proportional gain of 1.0, the",
                "value used throughout this project. That was never tuned against actuation",
                "delay. Sweeping the gain from 0.6 to 4.0 at every condition shows 1.0 is far",
                f"from optimal — the best single gain across all conditions is **{best_gain}**, and",
                "the best gain falls as delay rises, which is the textbook response to a delay",
                "inside the loop and independently confirms finding 17's mechanism.",
                "",
                "Two fairer baselines. *Per-cell* takes the best gain at each condition, which is",
                "an oracle — a deployed controller would need to know the delay and sea state to",
                "pick it. *Single* takes the one gain minimising median error across all",
                "conditions, which is the like-for-like opponent for one policy trained across",
                "the same range.",
                "",
                "| Disturbance | Delay | gain 1.0 (as first reported) | best single gain | per-cell best | policy | vs single | vs per-cell |",
                "|---|---|---|---|---|---|---|---|",
            ]
            gains = gain_curves("track_cross_p95_error_m")
            med = lambda k: statistics.median(
                [v[k] for v in pol_p95.values() if k in v]
            )
            for f in freqs:
                for d in delays:
                    k = (f, d)
                    if k not in per_cell:
                        continue
                    p_ = med(k)
                    lines.append(
                        f"| x{f:g} | {d:.0f} ms | {gains[1.0][k]:.2f} | {single[k]:.2f} | "
                        f"{per_cell[k]:.2f} | {p_:.2f} | **{(1 - p_ / single[k]) * 100:+.0f}%** | "
                        f"**{(1 - p_ / per_cell[k]) * 100:+.0f}%** |"
                    )
            vs_single = [1 - med(k) / single[k] for k in keys]
            vs_cell = [1 - med(k) / per_cell[k] for k in keys]
            lines += [
                "",
                f"- Against the best **single** gain: median **{statistics.median(vs_single) * 100:+.0f}%**, "
                f"range {min(vs_single) * 100:+.0f}% to {max(vs_single) * 100:+.0f}%.",
                f"- Against the **per-cell** oracle gain: median **{statistics.median(vs_cell) * 100:+.0f}%**, "
                f"range {min(vs_cell) * 100:+.0f}% to {max(vs_cell) * 100:+.0f}%.",
                "",
                "**At the longest delay with a slow disturbance the tuned baseline wins.** The",
                "policy's advantage is concentrated where the disturbance is *fast*, not where the",
                "delay is long — see the verdict below.",
                "",
            ]

        base = curve("residual_zero", "zero", "track_cross_p95_error_m")
        pol = {label: curve("residual_full", label, "track_cross_p95_error_m") for label in labels}
        def cell_gain(key):
            """Median improvement over baseline at one cell, across training seeds."""
            vals = [v[key] for v in pol.values() if key in v]
            b = base.get(key)
            if not vals or not b:
                return None
            return statistics.median((1 - v / b) for v in vals)

        gains = [g for g in (cell_gain(k) for k in base) if g is not None]
        delayed = [
            g for k in base if k[1] > 0 for g in (cell_gain(k),) if g is not None
        ]
        if gains:
            lo, hi = REFERENCE_RANGE
            lines += [
                "## Verdict",
                "",
                f"- Median policy against baseline A on p95, across all cells: "
                f"**{min(gains) * 100:+.0f}% to {max(gains) * 100:+.0f}%** "
                f"(median {statistics.median(gains) * 100:+.0f}%).",
            ]
            if delayed:
                lines.append(
                    f"- Restricted to the delayed cells, where finding 17 says the headroom is: "
                    f"**{min(delayed) * 100:+.0f}% to {max(delayed) * 100:+.0f}%** "
                    f"(median {statistics.median(delayed) * 100:+.0f}%)."
                )
            lines += [
                f"- Herland & Bach report {lo * 100:.0f}-{hi * 100:.0f}% mean-error reduction on"
                " their task, for reference rather than as a like-for-like target: their"
                " disturbance, arm and target differ, and they train one agent per delay"
                " where this trains one across the range.",
                "",
            ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines).rstrip() + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

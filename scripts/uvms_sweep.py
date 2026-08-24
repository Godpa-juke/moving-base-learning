#!/usr/bin/env python3
"""Measure every controller against every drag mismatch in one process.

The Stage 3 axis crosses seven mismatch factors with two analytic controllers and one
policy per training seed. Run through ``scripts/evaluate.py`` that is forty-odd separate
Isaac Sim start-ups, most of the wall clock spent booting rather than measuring. This
boots once and re-pins the mismatch between rollouts.

The metric code is not duplicated: both this and ``evaluate.py`` call
``marine_manipulator.evaluation.rollout``, so a threshold changed in one place cannot
leave the sweep and the single-condition numbers disagreeing.

Conditions are specified as ``--controllers`` (``ik`` is baseline A, ``coupled`` is
baseline B) and ``--checkpoints label=path`` for policies. Every condition sees the same
environment, the same seed and the same pinned mismatch, which is the fairness the plan's
rule 2 is about: whatever the drag is wrong by, it is wrong by that much for all of them.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Marine-UR3-Uvms-WorldLineFreeFloating-Play-v0")
parser.add_argument(
    "--drag-factors",
    default="1.0",
    help="comma-separated multipliers on the drag coefficients the simulation uses, "
    "while the controllers and the policy keep the nominal ones",
)
parser.add_argument(
    "--controllers",
    default="ik,coupled",
    help="comma-separated analytic controllers: 'ik' is baseline A (arm-only Jacobian, "
    "assumes the hull is an anchor), 'coupled' is baseline B (generalized Jacobian plus "
    "predicted hull drift)",
)
parser.add_argument(
    "--checkpoints",
    default="",
    help="comma-separated label=path policy checkpoints, each evaluated at every factor",
)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--seed", type=int, default=44)
parser.add_argument("--run-name", required=True)
parser.add_argument("--capture-threshold", type=float, default=0.02)
parser.add_argument("--ik-gain", type=float, default=1.0)
parser.add_argument("--ik-iterations", type=int, default=2)
parser.add_argument(
    "--no-waves",
    action="store_true",
    help="silence the sea state so the arm's own reaction is the only thing moving the hull",
)
parser.add_argument(
    "--actuation-delays",
    default="0",
    help="comma-separated actuation delays in whole control steps (33.3 ms each at "
    "30 Hz). Distinct from --sensor-delay-s: this one sits inside the feedback loop. "
    "Sweeping it tests whether the delay type explains the gap between this study's "
    "flat measurement-delay result and Herland & Bach's fourfold actuation-delay one.",
)
parser.add_argument(
    "--sensor-delays-s",
    default="",
    help="comma-separated measurement delays in seconds, swept alongside the actuation "
    "delays so the two kinds can be contrasted in one run on one task. Requires a task "
    "that defines the sensor-degradation event.",
)
parser.add_argument(
    "--wave-frequency-scales",
    default="1.0",
    help="comma-separated multipliers on the disturbance frequency. 1.0 is this "
    "study's 0.06-0.3 Hz; about 2.0 reaches the 0.32-0.53 Hz of the vessel "
    "trajectories Herland & Bach used.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from marine_manipulator import compat

    cudnn_note = compat.disable_cudnn_rnn_if_unsupported()

    import isaaclab_tasks  # noqa: F401
    import marine_manipulator.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
    from marine_manipulator import evaluation, provenance
    from marine_manipulator.tasks.random_base_line import mdp

    from marine_manipulator.controllers import AnalyticIkController, CoupledUvmsController

    device = args.device or "cuda:0"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs, use_fabric=True)
    env_cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = device

    raw = gym.make(args.task, cfg=env_cfg)
    if args.no_waves:
        mdp.set_wave_scale(raw.unwrapped, 0.0)
        mdp._state(raw.unwrapped)["amplitude"][:] = 0.0
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    base = env.unwrapped

    checkpoints = [
        entry.split("=", 1) for entry in args.checkpoints.split(",") if entry.strip()
    ]
    runner = None
    if checkpoints:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    def build(name: str):
        """Instantiate one condition's controller. Policies reload into one runner."""
        if name == "zero":
            # Emits nothing. On the residual task this leaves the analytic controller
            # running untouched, so it must reproduce baseline A cell for cell; that
            # equality is what proves the residual plumbing adds nothing of its own.
            zeros = torch.zeros((base.num_envs, base.action_manager.total_action_dim), device=base.device)
            return lambda obs: zeros
        if name == "ik":
            return AnalyticIkController(
                base, iterations=args.ik_iterations, mode="seam", gain=args.ik_gain
            )
        if name.startswith("coupled"):
            # "coupled" is baseline B as the plan specifies it; "coupled:drift" and
            # "coupled:reaction" keep one of its two extra terms and drop the other,
            # which is what attributes the A-to-B gap to a mechanism instead of
            # leaving it asserted.
            _, _, terms = name.partition(":")
            # "coupled:oracle" is the confound control, not a baseline: it reads the
            # simulation's true drag, so comparing it against nominal-B at the same
            # factor separates model error from the task simply being easier.
            oracle = terms == "oracle"
            return CoupledUvmsController(
                base,
                gain=args.ik_gain,
                terms="both" if oracle else (terms or "both"),
                oracle_drag=oracle,
            )
        runner.load(dict(checkpoints)[name])
        return runner.get_inference_policy(device=base.device)

    conditions = [name for name in args.controllers.split(",") if name.strip()]
    conditions += [label for label, _ in checkpoints]
    factors = [float(value) for value in args.drag_factors.split(",")]

    delays = [int(value) for value in args.actuation_delays.split(",")]
    sensor_delays = [float(v) for v in args.sensor_delays_s.split(",") if v.strip()] or [None]
    frequency_scales = [float(value) for value in args.wave_frequency_scales.split(",")]

    rows = []
    # One rollout per (frequency, delay, mismatch, controller). The two outer loops set
    # environment properties and the two inner ones vary what is being compared, so an
    # environment property is written once per block rather than once per rollout.
    for frequency_scale in frequency_scales:
      for sensor_delay in sensor_delays:
        for delay in delays:
            # Set before the reset below, so each episode is sampled at this frequency
            # and opens with an empty delay buffer.
            mdp.set_wave_frequency_scale(base, frequency_scale)
            mdp.set_actuation_delay(base, delay)
            # Takes precedence over the reset event that randomises these for training,
            # which fires after the pin is set and would otherwise overwrite it.
            mdp.pin_actuation_conditions(base)
            if sensor_delay is not None:
                mdp.set_sensor_degradation(base, delay_s=sensor_delay)
            for factor in factors:
                for name in conditions:
                    # Reset before pinning so the episode counters, the hull pose and
                    # the sampled line start from the same place for every condition,
                    # and the comparison is between controllers rather than between
                    # rollout phases.
                    #
                    # Under inference mode, like the rollout that produced the state
                    # being reset. Buffers written inside `torch.inference_mode()`
                    # become inference tensors, and writing to one from ordinary mode
                    # raises - which only shows up on the *second* condition, once a
                    # rollout has already run.
                    with torch.inference_mode():
                        env.reset()
                    mdp.set_drag_mismatch(base, factor)
                    if sensor_delay is not None:
                        # The reset event redraws the degradation, so the pin is
                        # re-applied here rather than once per block.
                        mdp.set_sensor_degradation(base, delay_s=sensor_delay)
                    metrics = evaluation.rollout(
                        env,
                        build(name),
                        steps=args.steps,
                        capture_threshold=args.capture_threshold,
                        tcp_offset=mdp.TCP_OFFSET,
                    )
                    row = {
                        "condition": name,
                        "drag_mismatch": factor,
                        "actuation_delay_steps": delay,
                        "actuation_delay_ms": delay * base.step_dt * 1000.0,
                        "wave_frequency_scale": frequency_scale,
                        "sensor_delay_s": sensor_delay,
                    }
                    row.update(metrics)
                    rows.append(row)
                    print(
                        f"  {name:16s} drag x{factor:<5.2f}"
                        f"  delay {delay * base.step_dt * 1000.0:5.1f} ms"
                        f"  freq x{frequency_scale:<4.1f}"
                        f"  sensor {0.0 if sensor_delay is None else sensor_delay * 1000:5.1f} ms"
                        f"  cross_rmse {row['track_cross_rmse_m'] * 1000:7.3f} mm"
                        f"  cross_max {row['track_cross_max_error_m'] * 1000:8.2f} mm",
                        flush=True,
                    )

    project = Path(os.environ["MARINE_PROJECT_DIR"])
    out_dir = project / "outputs" / "evaluation" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "probe": "uvms_sweep",
        "task": args.task,
        "provenance": provenance.record(project, out_dir),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "no_waves": args.no_waves,
        "capture_threshold_m": args.capture_threshold,
        "ik_gain": args.ik_gain,
        "reaction_factor": CoupledUvmsController.REACTION_FACTOR,
        "checkpoints": dict(checkpoints),
        "drag_factors": factors,
        "actuation_delays_steps": delays,
        "wave_frequency_scales": frequency_scales,
        "sensor_delays_s": sensor_delays,
        "step_dt": base.step_dt,
        "conditions": conditions,
        "cudnn_note": cudnn_note,
        "rows": rows,
        "status": "uvms_sweep_ok",
    }
    env.close()
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("uvms_sweep_summary", json.dumps({"run_name": args.run_name, "rows": len(rows)}), flush=True)
except BaseException:
    import traceback

    traceback.print_exc()
    os._exit(1)
else:
    app.close()

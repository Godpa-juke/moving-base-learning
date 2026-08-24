#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Marine-UR3-Random6DoFBase-WorldLine-Play-v0")
parser.add_argument("--checkpoint", default=None)
parser.add_argument(
    "--controller",
    choices=("policy", "ik", "coupled"),
    default="policy",
    help="'ik' replaces the learned policy with an analytic inverse-kinematics "
    "controller that reads the true base pose each step: the model-based baseline "
    "the learned policy has to beat. 'coupled' is the free-floating round's baseline B, "
    "which additionally predicts what its own command does to the hull it stands on; "
    "'ik' is then baseline A, the same controller with that prediction removed.",
)
parser.add_argument(
    "--no-waves",
    action="store_true",
    help="silence the sea state, leaving the arm's own reaction as the only thing that "
    "moves the hull. Running this on the fixed-base task and on the free-floating one "
    "isolates the coupling from the disturbance, which is the plan's stage 0 decisive test.",
)
parser.add_argument(
    "--drag-mismatch",
    type=float,
    default=None,
    help="multiply the drag coefficients the *simulation* uses by this factor while the "
    "controllers and the policy keep the nominal ones. This is the Stage 3 x-axis: 1.0 "
    "is a perfectly known model, 0.5 and 2.0 are the plus/minus 100%% ends.",
)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--run-name", required=True)
parser.add_argument("--target-x-offset", type=float, default=0.0)
parser.add_argument(
    "--sensor-delay-s",
    type=float,
    default=None,
    help="pin the measured base-state delay instead of randomising it per episode; "
    "sweeping this is what separates a reactive controller from a predictive one",
)
parser.add_argument("--sensor-position-noise-m", type=float, default=None)
parser.add_argument("--sensor-rotation-noise-rad", type=float, default=None)
parser.add_argument(
    "--ik-mode",
    choices=("seam", "pose"),
    default="seam",
    help="'seam' drives the analytic controller from the measured tool-to-line offset, "
    "closing the loop on the same signal the policy observes; 'pose' is the earlier "
    "open-loop form that reconstructs the target from the measured base pose alone. "
    "'seam' is the fair comparison - an open-loop baseline loses to any feedback "
    "controller for reasons that have nothing to do with learning.",
)
parser.add_argument(
    "--ik-gain",
    type=float,
    default=1.0,
    help="fraction of the measured error the seam-driven controller tries to null per step",
)
parser.add_argument(
    "--ik-iterations",
    type=int,
    default=2,
    help="damped-least-squares steps per control step for --controller ik. Few steps "
    "on purpose: this is a resolved-rate controller, warm-started from the measured "
    "configuration. Solving all the way to convergence lets the solution jump to a "
    "different IK branch, which the joint rate limiter then crawls toward, and the "
    "error tail grows monotonically with the step count (max 68 mm at 1 step, 350 mm "
    "at 100).",
)
parser.add_argument(
    "--capture-threshold",
    type=float,
    default=0.02,
    help="error (m) at which an episode is considered to have captured the line; "
    "samples before capture are transit, not tracking",
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
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
    from marine_manipulator import provenance
    from marine_manipulator.tasks.random_base_line import mdp


    from marine_manipulator.controllers import (
        AnalyticIkController,
        CoupledUvmsController,
    )

    project = Path(os.environ["MARINE_PROJECT_DIR"])
    out_dir = project / "outputs" / "evaluation" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs, use_fabric=True)
    x_low, x_high = env_cfg.commands.ee_pose.ranges.pos_x
    env_cfg.commands.ee_pose.ranges.pos_x = (x_low + args.target_x_offset, x_high + args.target_x_offset)
    env_cfg.seed = args.seed
    degradation = getattr(env_cfg.events, "sample_sensor_degradation", None)
    pins = (
        ("delay_range_s", args.sensor_delay_s),
        ("position_noise_range_m", args.sensor_position_noise_m),
        ("rotation_noise_range_rad", args.sensor_rotation_noise_rad),
    )
    for key, value in pins:
        if value is None:
            continue
        if degradation is None:
            raise SystemExit(f"{args.task} has no sensor degradation to pin ({key})")
        degradation.params[key] = (value, value)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = args.device or "cuda:0"
    raw = gym.make(args.task, cfg=env_cfg)
    if args.no_waves:
        # Before the wrapper, and therefore before the first reset draws an amplitude.
        mdp.set_wave_scale(raw.unwrapped, 0.0)
        mdp._state(raw.unwrapped)["amplitude"][:] = 0.0
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    if args.drag_mismatch is not None:
        # Pin the simulation's drag error. Set after the environment exists and before
        # the first step, so every episode of the rollout runs at the same mismatch
        # rather than redrawing one per reset.
        mdp.set_drag_mismatch(env.unwrapped, args.drag_mismatch)
    if args.controller == "ik":
        policy = AnalyticIkController(
            env.unwrapped,
            iterations=args.ik_iterations,
            mode=args.ik_mode,
            gain=args.ik_gain,
        )
    elif args.controller == "coupled":
        policy = CoupledUvmsController(env.unwrapped, gain=args.ik_gain)
    else:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required unless --controller ik")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
    from marine_manipulator import evaluation

    measured = evaluation.rollout(
        env,
        policy,
        steps=args.steps,
        capture_threshold=args.capture_threshold,
        tcp_offset=mdp.TCP_OFFSET,
    )


    result = {
        "task": args.task,
        "provenance": provenance.record(project, out_dir),
        "controller": args.controller,
        "cudnn_note": cudnn_note,
        "checkpoint": args.checkpoint,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "seed": args.seed,
        "target_x_offset_m": args.target_x_offset,
        "capture_threshold_m": args.capture_threshold,
        "ik_iterations": args.ik_iterations if args.controller == "ik" else None,
        "ik_mode": args.ik_mode if args.controller == "ik" else None,
        "ik_gain": args.ik_gain if args.controller in ("ik", "coupled") else None,
        "drag_mismatch": args.drag_mismatch,
        "no_waves": args.no_waves,
        "reaction_factor": (
            CoupledUvmsController.REACTION_FACTOR if args.controller == "coupled" else None
        ),
        "sensor_delay_s": args.sensor_delay_s,
        "sensor_position_noise_m": args.sensor_position_noise_m,
        "sensor_rotation_noise_rad": args.sensor_rotation_noise_rad,
        "ik_convergence_rate": (
            sum(policy.solved) / len(policy.solved)
            if args.controller in ("ik", "coupled") and policy.solved
            else None
        ),
        "status": "fresh_eval_ok",
    }
    # "all_" keeps the original whole-rollout numbers; "track_" is the steady-state
    # tracking performance, which is what the task is actually about.
    result.update(measured)
    # Backwards-compatible top-level aliases (whole-rollout, as before).
    result.update(
        {
            "samples": result["all_samples"],
            "rmse_m": result["all_rmse_m"],
            "mean_error_m": result["all_mean_error_m"],
            "p95_error_m": result["all_p95_error_m"],
            "max_error_m": result["all_max_error_m"],
            "success_1cm": result["all_success_1cm"],
            "success_5cm": result["all_success_5cm"],
            "success_10cm": result["all_success_10cm"],
        }
    )
    env.close()
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("fresh_eval_summary", json.dumps(result, sort_keys=True), flush=True)
except BaseException:
    import traceback

    traceback.print_exc()
    os._exit(1)
else:
    app.close()

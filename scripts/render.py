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
parser.add_argument("--controller", choices=("policy", "ik"), default="policy")
parser.add_argument("--ik-gain", type=float, default=2.0)
parser.add_argument("--ik-iterations", type=int, default=2)
parser.add_argument("--wave-frequency-scale", type=float, default=None)
parser.add_argument("--actuation-delay-steps", type=int, choices=(0, 1, 2, 3), default=None)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--seed", type=int, default=818)
parser.add_argument("--run-name", default="fresh_smoke_rollout")
parser.add_argument("--target-x-offset", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab.sim as sim_utils
    import isaaclab_tasks  # noqa: F401
    import marine_manipulator.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
    from isaaclab.utils.math import quat_apply
    from marine_manipulator import compat
    from marine_manipulator.controllers import AnalyticIkController
    from marine_manipulator.tasks.random_base_line import mdp

    cudnn_note = compat.disable_cudnn_rnn_if_unsupported()

    project = Path(os.environ["MARINE_PROJECT_DIR"])
    out_dir = project / "outputs" / "rollout_videos" / args.run_name
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=1, use_fabric=True)
    x_low, x_high = env_cfg.commands.ee_pose.ranges.pos_x
    env_cfg.commands.ee_pose.ranges.pos_x = (x_low + args.target_x_offset, x_high + args.target_x_offset)
    env_cfg.commands.ee_pose.debug_vis = False
    env_cfg.seed = args.seed
    env_cfg.viewer.eye = (-1.10, 1.10, 0.80)
    env_cfg.viewer.lookat = (-0.22, 0.0, 0.25)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = args.device or "cuda:0"

    raw = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    if args.wave_frequency_scale is not None:
        mdp.set_wave_frequency_scale(raw.unwrapped, args.wave_frequency_scale)
    if args.actuation_delay_steps is not None:
        mdp.set_actuation_delay(raw.unwrapped, args.actuation_delay_steps)
    cylinder = sim_utils.CylinderCfg(
        radius=0.025,
        height=0.12,
        axis="Z",
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.58, 0.62), metallic=0.2),
    )
    cylinder.func(
        "/World/envs/env_0/Robot/wrist_3_link/ToolCylinder",
        cylinder,
        translation=(0.0, 0.0, 0.06),
    )
    tip = sim_utils.SphereCfg(
        radius=0.018,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.25, 1.0), emissive_color=(0.0, 0.1, 0.6)),
    )
    tip.func(
        "/World/envs/env_0/Robot/wrist_3_link/TcpTip",
        tip,
        translation=mdp.TCP_OFFSET,
    )
    app.update()

    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.util.debug_draw")
    app.update()
    from isaacsim.util.debug_draw import _debug_draw

    debug = _debug_draw.acquire_debug_draw_interface()
    raw = gym.wrappers.RecordVideo(
        raw,
        video_folder=str(video_dir),
        step_trigger=lambda step: step == 0,
        video_length=args.steps,
        disable_logger=True,
        name_prefix=args.run_name,
    )
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    if args.controller == "policy":
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for --controller policy")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
    else:
        policy = AnalyticIkController(
            env.unwrapped,
            iterations=args.ik_iterations,
            mode="seam",
            gain=args.ik_gain,
        )
    obs = env.get_observations()
    robot = env.unwrapped.scene["robot"]
    wrist = robot.body_names.index("wrist_3_link")
    offset = torch.tensor(mdp.TCP_OFFSET, device=env.unwrapped.device).repeat(1, 1)
    command = env.unwrapped.command_manager._terms["ee_pose"]
    bar_center_w = env.unwrapped.scene.env_origins[0] + command.line_center_e[0]
    target_bar = sim_utils.CylinderCfg(
        radius=0.012,
        height=float(2.0 * command.line_amplitude[0]),
        axis="Y",
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.35, 1.0),
            emissive_color=(0.0, 0.15, 0.8),
        ),
    )
    target_bar.func(
        "/World/TargetLineBar",
        target_bar,
        translation=tuple(float(value) for value in bar_center_w),
    )
    app.update()
    errors = []
    cross_track_errors = []
    post_capture_cross_track_errors = []
    trajectory = []
    captured = False

    with torch.inference_mode():
        for step in range(args.steps):
            tcp = robot.data.body_pos_w[:, wrist] + quat_apply(robot.data.body_quat_w[:, wrist], offset)
            target = command.pose_command_w[:, :3]
            line_start, line_end = mdp.line_endpoints_w(env.unwrapped, "ee_pose")
            start = tuple(float(x) for x in line_start[0])
            end = tuple(float(x) for x in line_end[0])
            target_tuple = tuple(float(x) for x in target[0])
            tcp_tuple = tuple(float(x) for x in tcp[0])
            line_direction = line_end[0] - line_start[0]
            line_direction = line_direction / torch.linalg.norm(line_direction).clamp_min(1.0e-9)
            line_relative = tcp[0] - line_start[0]
            along_track = torch.dot(line_relative, line_direction)
            cross_track = torch.linalg.norm(line_relative - along_track * line_direction)
            debug.clear_lines()
            debug.draw_lines(
                [start, target_tuple],
                [end, tcp_tuple],
                [(0.0, 0.55, 1.0, 1.0), (1.0, 0.45, 0.0, 1.0)],
                [6.0, 3.0],
            )
            errors.append(float(torch.linalg.norm(tcp[0] - target[0])))
            cross_track_errors.append(float(cross_track))
            captured = captured or errors[-1] <= 0.02
            if captured:
                post_capture_cross_track_errors.append(cross_track_errors[-1])
            trajectory.append(
                {
                    "step": step,
                    "time_s": step * env.unwrapped.step_dt,
                    "tcp_w": list(tcp_tuple),
                    "target_w": list(target_tuple),
                    "line_start_w": list(start),
                    "line_end_w": list(end),
                    "error_m": errors[-1],
                    "cross_track_error_m": cross_track_errors[-1],
                    "along_track_coordinate_m": float(along_track),
                    "captured": captured,
                }
            )
            obs, _, dones, _ = env.step(policy(obs))
            if bool(dones.any()) and hasattr(policy, "reset"):
                policy.reset(dones)

    env.close()
    videos = sorted(str(p) for p in video_dir.glob("*.mp4"))
    result = {
        "task": args.task,
        "controller": args.controller,
        "cudnn_note": cudnn_note,
        "checkpoint": args.checkpoint,
        "ik_gain": args.ik_gain if args.controller == "ik" else None,
        "ik_iterations": args.ik_iterations if args.controller == "ik" else None,
        "steps": args.steps,
        "seed": args.seed,
        "wave_frequency_scale": args.wave_frequency_scale,
        "actuation_delay_steps": args.actuation_delay_steps,
        "target_x_offset_m": args.target_x_offset,
        "rmse_m": math.sqrt(sum(x * x for x in errors) / len(errors)),
        "cross_track_rmse_m": math.sqrt(sum(x * x for x in cross_track_errors) / len(cross_track_errors)),
        "post_capture_cross_track_p95_m": (
            float(torch.quantile(torch.tensor(post_capture_cross_track_errors), 0.95))
            if post_capture_cross_track_errors
            else None
        ),
        "captured": captured,
        "success_1cm": sum(x <= 0.01 for x in errors) / len(errors),
        "success_5cm": sum(x <= 0.05 for x in errors) / len(errors),
        "videos": videos,
        "status": "fresh_rollout_ok" if videos else "fresh_rollout_missing",
    }
    (out_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("fresh_rollout_summary", json.dumps(result, sort_keys=True), flush=True)
    if not videos:
        raise RuntimeError("No rollout video generated")
except BaseException:
    import traceback

    traceback.print_exc()
    os._exit(1)
else:
    app.close()

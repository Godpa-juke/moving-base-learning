#!/usr/bin/env python3
"""Check that the IK-seeded task really starts each episode on the line.

Reports, at the first step of an episode, the cylinder-tip distance to the
commanded line point and the angle between the cylinder axis and straight down.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Marine-UR3-Random6DoFBase-WorldLineIkSeeded-Play-v0")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--seed", type=int, default=44)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    import marine_manipulator.tasks  # noqa: F401
    from isaaclab.utils.math import quat_apply
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from marine_manipulator.tasks.random_base_line import mdp

    device = args.device or "cuda:0"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs, use_fabric=True)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset(seed=args.seed)

    robot = env.scene["robot"]
    wrist = robot.body_names.index("wrist_3_link")
    offset = torch.tensor(mdp.TCP_OFFSET, device=env.device).repeat(args.num_envs, 1)
    local_axis = torch.zeros((args.num_envs, 3), device=env.device)
    local_axis[:, 2] = 1.0
    down = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(args.num_envs, 3)
    command = env.command_manager._terms["ee_pose"]

    start_errors = []
    start_tilts = []
    seed_errors = []
    zero_action = torch.zeros((args.num_envs, env.action_space.shape[1]), device=env.device)

    def measure():
        tcp = robot.data.body_pos_w[:, wrist] + quat_apply(
            robot.data.body_quat_w[:, wrist], offset
        )
        target = env.scene.env_origins + command.pose_command_b[:, :3]
        axis = quat_apply(robot.data.body_quat_w[:, wrist], local_axis)
        tilt = torch.acos(((axis * down).sum(dim=1)).clamp(-1.0, 1.0))
        return torch.linalg.norm(tcp - target, dim=1), tilt

    # Pure IK accuracy: seed every environment and read the pose straight back,
    # with no physics step in between to blur it.
    all_ids = torch.arange(args.num_envs, device=env.device)
    mdp.reset_joints_to_ik_line_start(env, all_ids)
    seed_error, seed_tilt = measure()
    seed_errors = seed_error.cpu().tolist()
    seed_tilts = seed_tilt.cpu().tolist()

    with torch.inference_mode():
        for _ in range(args.steps):
            env.step(zero_action)
            # After step(), envs whose episode just began have been IK-seeded and
            # advanced by exactly one control step.
            fresh = env.episode_length_buf == 1
            if not bool(fresh.any()):
                continue
            error, tilt = measure()
            start_errors.extend(error[fresh].cpu().tolist())
            start_tilts.extend(tilt[fresh].cpu().tolist())

    errors = torch.tensor(start_errors)
    tilts = torch.rad2deg(torch.tensor(start_tilts))
    seed_error_t = torch.tensor(seed_errors)
    seed_tilt_t = torch.rad2deg(torch.tensor(seed_tilts))
    result = {
        "task": args.task,
        "ik_only_error_mean_mm": float(seed_error_t.mean() * 1000),
        "ik_only_error_max_mm": float(seed_error_t.max() * 1000),
        "ik_only_tilt_max_deg": float(seed_tilt_t.max()),
        "episode_starts_sampled": int(errors.numel()),
        "start_error_mean_mm": float(errors.mean() * 1000),
        "start_error_p95_mm": float(errors.quantile(0.95) * 1000),
        "start_error_max_mm": float(errors.max() * 1000),
        "start_tool_tilt_mean_deg": float(tilts.mean()),
        "start_tool_tilt_max_deg": float(tilts.max()),
    }
    env.close()
    print("ik_seeding_check " + json.dumps(result, sort_keys=True), flush=True)
except BaseException:
    import traceback

    traceback.print_exc()
    os._exit(1)
else:
    app.close()

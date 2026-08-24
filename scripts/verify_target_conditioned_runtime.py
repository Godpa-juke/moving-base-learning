#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import traceback
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import torch
    import isaaclab_tasks  # noqa: F401
    import marine_manipulator.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from marine_manipulator.tasks.random_base_line import mdp
    from marine_manipulator.tasks.random_base_line.env_cfg import TargetConditionedWorldLineEnvCfg

    task = "Marine-UR3-Random6DoFBase-WorldLineTargetConditioned-Play-v0"
    assert TargetConditionedWorldLineEnvCfg().scene.num_envs == 4096
    cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=16, use_fabric=True)
    env = gym.make(task, cfg=cfg)
    raw = env.unwrapped
    env.reset(seed=921)
    command = raw.command_manager._terms["ee_pose"]
    start, end = mdp.line_endpoints_w(raw, "ee_pose")
    origins = raw.scene.env_origins
    start_e = start - origins
    end_e = end - origins
    center0 = command.line_center_e.clone()
    direction0 = command.line_direction_e.clone()
    target0 = command.pose_command_w[:, :3].clone()
    endpoint_radius = torch.maximum(torch.linalg.norm(start_e, dim=1), torch.linalg.norm(end_e, dim=1))
    endpoint_xyz_radius = torch.maximum(
        torch.linalg.norm(start_e, dim=1), torch.linalg.norm(end_e, dim=1)
    )
    unique_centers = torch.unique(torch.round(center0 * 10000), dim=0).shape[0]
    unique_directions = torch.unique(torch.round(direction0 * 10000), dim=0).shape[0]
    expected_direction = torch.tensor([0.0, 1.0, 0.0], device=raw.device)
    zeros = torch.zeros((raw.num_envs, raw.action_manager.total_action_dim), device=raw.device)
    hold_targets = []
    root0 = raw.scene["robot"].data.root_link_pos_w.clone()
    root2 = None
    hold_steps = int(round(cfg.commands.ee_pose.hold_duration_s / raw.step_dt))
    probe_step = hold_steps + 31
    for step in range(1, probe_step + 1):
        env.step(zeros)
        if step <= hold_steps:
            hold_targets.append(command.pose_command_w[:, :3].clone())
        if step == 2:
            root2 = raw.scene["robot"].data.root_link_pos_w.clone()
    target61 = command.pose_command_w[:, :3].clone()
    hold = torch.stack(hold_targets)
    hold_span = float((hold.amax(dim=0) - hold.amin(dim=0)).abs().max())
    target_motion = torch.linalg.norm(target61 - target0, dim=1)
    root_motion = torch.linalg.norm(root2 - root0, dim=1)
    projection = torch.sum((target61 - origins - center0) * direction0, dim=1)
    perpendicular = target61 - origins - center0 - projection[:, None] * direction0
    env.reset(seed=922)
    requested = torch.ones((raw.num_envs, raw.action_manager.total_action_dim), device=raw.device)
    env.step(requested)
    first_limited_action = raw._previous_rate_limited_action.clone()
    expected_delta = torch.tensor(
        mdp.REALISTIC_JOINT_SPEED_LIMIT_RAD_S, device=raw.device
    ) * (raw.step_dt / mdp.REALISTIC_ACTION_SCALE_RAD)
    result = {
        "status": "target_conditioned_runtime_ok",
        "unique_centers": int(unique_centers),
        "unique_directions": int(unique_directions),
        "line_center_x_min_m": float(center0[:, 0].min()),
        "line_center_x_max_m": float(center0[:, 0].max()),
        "line_endpoint_x_min_m": float(torch.minimum(start_e[:, 0], end_e[:, 0]).min()),
        "line_endpoint_x_max_m": float(torch.maximum(start_e[:, 0], end_e[:, 0]).max()),
        "max_endpoint_radius_m": float(endpoint_radius.max()),
        "max_endpoint_xyz_radius_m": float(endpoint_xyz_radius.max()),
        "target_hold_span_m": hold_span,
        "target_motion_after_hold_min_m": float(target_motion.min()),
        "target_line_perpendicular_error_max_m": float(torch.linalg.norm(perpendicular, dim=1).max()),
        "base_second_step_motion_min_m": float(root_motion.min()),
        "first_limited_action_env0": first_limited_action[0].detach().cpu().tolist(),
        "expected_action_delta": expected_delta.detach().cpu().tolist(),
        "observation_manager": str(raw.observation_manager),
    }
    assert unique_centers >= 12, result
    assert unique_directions == 1, result
    assert torch.allclose(direction0, expected_direction[None, :].expand_as(direction0)), result
    assert -0.400001 <= result["line_center_x_min_m"] <= -0.30, result
    assert -0.400001 <= result["line_center_x_max_m"] <= -0.299999, result
    assert -0.400001 <= result["line_endpoint_x_min_m"] <= -0.30, result
    assert -0.400001 <= result["line_endpoint_x_max_m"] <= -0.299999, result
    assert result["max_endpoint_radius_m"] <= cfg.commands.ee_pose.reachable_radius_m + 1.0e-6, result
    assert hold_span <= 1.0e-5, result
    assert result["target_motion_after_hold_min_m"] > 1.0e-4, result
    assert result["target_line_perpendicular_error_max_m"] <= 1.0e-5, result
    assert result["base_second_step_motion_min_m"] > 1.0e-7, result
    assert torch.allclose(first_limited_action, expected_delta[None, :].expand_as(first_limited_action)), result
    assert "target_error_w" in result["observation_manager"], result
    print("target_conditioned_runtime_summary", json.dumps(result, sort_keys=True), flush=True)
    env.close()
except BaseException:
    traceback.print_exc()
    os._exit(1)
else:
    app.close()

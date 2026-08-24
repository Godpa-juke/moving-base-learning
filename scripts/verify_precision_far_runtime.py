#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import traceback

from isaaclab.app import AppLauncher

parser = AppLauncher.add_app_launcher_args
import argparse
argp = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(argp)
args = argp.parse_args()
args.headless = True
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import torch
    import isaaclab_tasks  # noqa: F401
    import marine_manipulator.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from isaaclab.utils.math import quat_apply
    from marine_manipulator.tasks.random_base_line import mdp

    task = "Marine-UR3-Random6DoFBase-WorldLinePrecisionFar50cm-Play-v0"
    cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=8, use_fabric=True)
    env = gym.make(task, cfg=cfg)
    raw = env.unwrapped
    env.reset(seed=920)
    robot = raw.scene["robot"]
    wrist = robot.body_names.index("wrist_3_link")
    offset = torch.tensor(mdp.TCP_OFFSET, device=raw.device).repeat(raw.num_envs, 1)
    command = raw.command_manager._terms["ee_pose"]

    def tcp_w():
        return robot.data.body_pos_w[:, wrist] + quat_apply(robot.data.body_quat_w[:, wrist], offset)

    def target_w():
        return raw.scene.env_origins + command.pose_command_b[:, :3]

    initial_tcp = tcp_w().clone()
    initial_error = torch.linalg.norm(initial_tcp - target_w(), dim=1)
    initial_base_pose = mdp.immediate_base_pose(raw).clone()
    initial_base_velocity = mdp.immediate_base_velocity(raw).clone()
    initial_target = target_w().clone()
    initial_root = robot.data.root_link_pos_w.clone()
    zeros = torch.zeros((raw.num_envs, raw.action_manager.total_action_dim), device=raw.device)
    target_samples = []
    base_samples = []
    first_step_error = None
    first_step_root = None
    second_step_root = None
    for step in range(1, 181):
        env.step(zeros)
        if step == 1:
            first_step_error = torch.linalg.norm(tcp_w() - target_w(), dim=1)
            first_step_root = robot.data.root_link_pos_w.clone()
        if step == 2:
            second_step_root = robot.data.root_link_pos_w.clone()
        if step <= 30:
            target_samples.append(target_w().clone())
        base_samples.append(mdp.immediate_base_pose(raw).clone())

    target_stack = torch.stack(target_samples)
    base_stack = torch.stack(base_samples)
    hold_span = float((target_stack[:, :, :3].amax(dim=0) - target_stack[:, :, :3].amin(dim=0)).abs().max())
    first_root_delta = torch.linalg.norm(first_step_root - initial_root, dim=1)
    second_root_delta = torch.linalg.norm(second_step_root - initial_root, dim=1)
    axis_spans = (base_stack.amax(dim=0) - base_stack.amin(dim=0)).mean(dim=0)
    result = {
        "status": "precision_far_runtime_ok",
        "initial_error_max_m": float(initial_error.max()),
        "first_step_error_max_m": float(first_step_error.max()),
        "initial_target_env0": initial_target[0].detach().cpu().tolist(),
        "initial_tcp_env0": initial_tcp[0].detach().cpu().tolist(),
        "initial_base_pose_abs_max": float(initial_base_pose.abs().max()),
        "initial_base_velocity_abs_max": float(initial_base_velocity.abs().max()),
        "first_step_root_delta_min_m": float(first_root_delta.min()),
        "second_step_root_delta_min_m": float(second_root_delta.min()),
        "target_hold_span_m": hold_span,
        "base_axis_spans": axis_spans.detach().cpu().tolist(),
        "joint_positions_env0": robot.data.joint_pos[0].detach().cpu().tolist(),
    }
    assert result["initial_error_max_m"] <= 0.01, result
    assert result["first_step_error_max_m"] <= 0.01, result
    assert result["initial_base_pose_abs_max"] <= 1.0e-7, result
    assert result["initial_base_velocity_abs_max"] > 1.0e-5, result
    assert result["second_step_root_delta_min_m"] > 1.0e-7, result
    assert hold_span <= 1.0e-5, result
    assert all(span > 1.0e-5 for span in result["base_axis_spans"]), result
    print("precision_far_runtime_summary", json.dumps(result, sort_keys=True), flush=True)
    env.close()
except BaseException:
    traceback.print_exc()
    os._exit(1)
else:
    app.close()

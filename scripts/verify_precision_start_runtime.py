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
    from isaaclab.utils.math import quat_apply
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from marine_manipulator.tasks.random_base_line import mdp

    task = "Marine-UR3-Random6DoFBase-WorldLinePrecisionStart-Play-v0"
    cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=8, use_fabric=True)
    env = gym.make(task, cfg=cfg)
    raw = env.unwrapped
    obs, _ = env.reset(seed=919)
    robot = raw.scene["robot"]
    wrist = robot.body_names.index("wrist_3_link")
    offset = torch.tensor(mdp.TCP_OFFSET, device=raw.device).repeat(raw.num_envs, 1)

    def tcp_w():
        return robot.data.body_pos_w[:, wrist] + quat_apply(robot.data.body_quat_w[:, wrist], offset)

    def target_w():
        term = raw.command_manager._terms["ee_pose"]
        return raw.scene.env_origins + term.pose_command_b[:, :3]

    initial_tcp = tcp_w().clone()
    initial_target = target_w().clone()
    initial_error = torch.linalg.norm(initial_tcp - initial_target, dim=1)
    initial_base = mdp.neutral_start_base_pose(raw).clone()
    target_y = [float(initial_target[0, 1] - raw.scene.env_origins[0, 1])]
    base_history = [initial_base]

    zeros = torch.zeros((raw.num_envs, 6), device=raw.device)
    snapshots = {}
    first_step_error = None
    first_step_tcp = None
    first_step_target = None
    early_step_errors = {}
    for step in range(1, 181):
        obs, _, _, _, _ = env.step(zeros)
        current_error = torch.linalg.norm(tcp_w() - target_w(), dim=1)
        if step <= 4:
            early_step_errors[str(step)] = {
                "max_m": float(current_error.max()),
                "mean_m": float(current_error.mean()),
                "tcp_env0": tcp_w()[0].detach().cpu().tolist(),
                "joint_pos_env0": robot.data.joint_pos[0].detach().cpu().tolist(),
            }
        if step == 1:
            first_step_tcp = tcp_w().clone()
            first_step_target = target_w().clone()
            first_step_error = torch.linalg.norm(first_step_tcp - first_step_target, dim=1)
        if step in (1, 15, 29, 30, 31, 90, 150, 180):
            y = float(target_w()[0, 1] - raw.scene.env_origins[0, 1])
            snapshots[str(step)] = y
        target_y.append(float(target_w()[0, 1] - raw.scene.env_origins[0, 1]))
        base_history.append(mdp.neutral_start_base_pose(raw).clone())

    base_stack = torch.stack(base_history)
    spans = (base_stack.amax(dim=(0, 1)) - base_stack.amin(dim=(0, 1))).detach().cpu().tolist()
    hold_samples = target_y[:31]
    hold_span = max(hold_samples) - min(hold_samples)
    result = {
        "status": "precision_runtime_ok",
        "event_terms": raw.event_manager.active_terms,
        "initial_error_max_m": float(initial_error.max()),
        "initial_error_mean_m": float(initial_error.mean()),
        "first_step_error_max_m": float(first_step_error.max()),
        "first_step_error_mean_m": float(first_step_error.mean()),
        "first_step_tcp_env0": first_step_tcp[0].detach().cpu().tolist(),
        "first_step_target_env0": first_step_target[0].detach().cpu().tolist(),
        "early_step_errors": early_step_errors,
        "initial_tcp_env0": initial_tcp[0].detach().cpu().tolist(),
        "initial_target_env0": initial_target[0].detach().cpu().tolist(),
        "initial_joint_positions_env0": robot.data.joint_pos[0].detach().cpu().tolist(),
        "initial_root_relative_env0": (robot.data.root_pos_w[0] - raw.scene.env_origins[0]).detach().cpu().tolist(),
        "initial_root_quat_env0": robot.data.root_quat_w[0].detach().cpu().tolist(),
        "default_root_state_env0": robot.data.default_root_state[0, :7].detach().cpu().tolist(),
        "reset_event_root_pose_env0": getattr(raw, "_precision_reset_root_pose", torch.empty((1, 7), device=raw.device))[0].detach().cpu().tolist(),
        "initial_base_abs_max": float(initial_base.abs().max()),
        "hold_target_y_minmax": [min(hold_samples), max(hold_samples)],
        "hold_target_y_span_m": hold_span,
        "target_y_snapshots": snapshots,
        "base_axis_spans": spans,
    }
    assert early_step_errors["2"]["max_m"] <= 0.01, result
    assert result["initial_base_abs_max"] <= 1.0e-7, result
    assert hold_span <= 1.0e-5, result
    expected_end_y = mdp.PRECISION_START_TCP_E[1] + 0.20
    assert snapshots["150"] >= expected_end_y - 1.0e-5, result
    assert all(span > 1.0e-5 for span in spans), result
    print("precision_runtime_summary", json.dumps(result, sort_keys=True), flush=True)
    env.close()
except BaseException:
    traceback.print_exc()
    os._exit(1)
else:
    app.close()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Marine-UR3-Random6DoFBase-WorldLine-Play-v0")
parser.add_argument("--seed", type=int, default=818)
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

    cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=4, use_fabric=True)
    cfg.seed = args.seed
    env = gym.make(args.task, cfg=cfg)
    raw = env.unwrapped
    robot = raw.scene["robot"]
    env.reset(seed=args.seed)

    state = getattr(raw, "_marine_random_base_motion")
    for key in ("amplitude", "frequency", "phase"):
        assert state[key].shape == (4, 6), (key, state[key].shape)
    assert bool(torch.all(state["amplitude"] > 0.0))
    assert bool(torch.all(state["frequency"] > 0.0))
    assert not bool(torch.allclose(state["phase"][0], state["phase"][1]))

    command = raw.command_manager._terms["ee_pose"]
    base_samples = []
    root_samples = []
    target_frame_error = []
    zero = torch.zeros(raw.action_manager.action.shape, device=raw.device)
    for _ in range(120):
        env.step(zero)
        pose = mdp.base_pose(raw).detach().clone()
        base_samples.append(pose)
        root_samples.append(robot.data.root_pos_w.detach().clone())
        expected_target = raw.scene.env_origins + command.pose_command_b[:, :3]
        target_frame_error.append(torch.max(torch.abs(command.pose_command_w[:, :3] - expected_target)))

    base_stack = torch.stack(base_samples)
    axis_span = torch.amax(base_stack, dim=(0, 1)) - torch.amin(base_stack, dim=(0, 1))
    assert bool(torch.all(axis_span > 1.0e-4)), axis_span
    max_target_frame_error = float(torch.max(torch.stack(target_frame_error)))
    if max_target_frame_error >= 1.0e-5:
        print(
            "target_frame_diagnostic",
            json.dumps(
                {
                    "max_error": max_target_frame_error,
                    "pose_command_b": command.pose_command_b[0, :3].detach().cpu().tolist(),
                    "pose_command_w": command.pose_command_w[0, :3].detach().cpu().tolist(),
                    "env_origin": raw.scene.env_origins[0].detach().cpu().tolist(),
                    "root_position": robot.data.root_pos_w[0].detach().cpu().tolist(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    assert max_target_frame_error < 1.0e-5

    initial_joint = robot.data.joint_pos.detach().clone()
    wrist_id = robot.body_names.index("wrist_3_link")
    offset = torch.tensor(mdp.TCP_OFFSET, device=raw.device).repeat(raw.num_envs, 1)
    initial_tcp = robot.data.body_pos_w[:, wrist_id] + quat_apply(robot.data.body_quat_w[:, wrist_id], offset)
    action = torch.full(raw.action_manager.action.shape, 0.4, device=raw.device)
    for _ in range(60):
        env.step(action)
    final_joint = robot.data.joint_pos.detach().clone()
    final_tcp = robot.data.body_pos_w[:, wrist_id] + quat_apply(robot.data.body_quat_w[:, wrist_id], offset)
    joint_delta = torch.mean(torch.abs(final_joint - initial_joint), dim=0)
    tcp_delta = torch.mean(torch.linalg.norm(final_tcp - initial_tcp, dim=1))
    assert bool(torch.all(joint_delta > 1.0e-3)), joint_delta
    assert float(tcp_delta) > 0.01, tcp_delta

    result = {
        "task": args.task,
        "seed": args.seed,
        "usd_path": cfg.scene.robot.spawn.usd_path,
        "body_names": list(robot.body_names),
        "joint_names": list(robot.joint_names),
        "amplitude_env0": state["amplitude"][0].detach().cpu().tolist(),
        "frequency_env0": state["frequency"][0].detach().cpu().tolist(),
        "phase_env0": state["phase"][0].detach().cpu().tolist(),
        "base_axis_span": axis_span.detach().cpu().tolist(),
        "max_world_target_frame_error": float(torch.max(torch.stack(target_frame_error)).cpu()),
        "mean_joint_delta": joint_delta.detach().cpu().tolist(),
        "mean_tcp_delta_m": float(tcp_delta.detach().cpu()),
        "status": "fresh_runtime_gates_ok",
    }
    print("fresh_runtime_summary", json.dumps(result, sort_keys=True), flush=True)
    env.close()
except BaseException:
    traceback.print_exc()
    os._exit(1)
else:
    app.close()

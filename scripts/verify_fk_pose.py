#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
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

    cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=1, use_fabric=True)
    env = gym.make(args.task, cfg=cfg)
    raw = env.unwrapped
    env.reset(seed=919)
    robot = raw.scene["robot"]
    q = torch.tensor(mdp.PRECISION_START_JOINT_POS, device=raw.device).unsqueeze(0)
    qd = torch.zeros_like(q)
    default = robot.data.default_root_state.clone()
    root_pose = default[:, :7].clone()
    root_pose[:, :3] += raw.scene.env_origins
    for _ in range(4):
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(default[:, 7:13])
        robot.write_joint_state_to_sim(q, qd)
        raw.scene.write_data_to_sim()
        raw.sim.step(render=False)
        raw.scene.update(raw.physics_dt)
    wrist = robot.body_names.index("wrist_3_link")
    offset = torch.tensor(mdp.TCP_OFFSET, device=raw.device).repeat(1, 1)
    tcp = robot.data.body_pos_w[:, wrist] + quat_apply(robot.data.body_quat_w[:, wrist], offset)
    result = {
        "task": args.task,
        "env_origin": raw.scene.env_origins[0].detach().cpu().tolist(),
        "root_link_pose": robot.data.root_link_pose_w[0].detach().cpu().tolist(),
        "joint_pos": robot.data.joint_pos[0].detach().cpu().tolist(),
        "tcp": tcp[0].detach().cpu().tolist(),
        "tcp_relative": (tcp[0] - raw.scene.env_origins[0]).detach().cpu().tolist(),
    }
    print("fk_pose_summary", json.dumps(result, sort_keys=True), flush=True)
    env.close()
except BaseException:
    traceback.print_exc()
    os._exit(1)
else:
    app.close()

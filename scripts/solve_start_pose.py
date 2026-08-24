#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--target", nargs=3, type=float, default=(0.32, -0.10, 0.28))
parser.add_argument("--iterations", type=int, default=120)
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

    task = "Marine-UR3-Random6DoFBase-WorldLine-Play-v0"
    cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=1, use_fabric=True)
    cfg.events.sample_base_motion = None
    cfg.events.apply_base_motion = None
    env = gym.make(task, cfg=cfg)
    raw = env.unwrapped
    env.reset(seed=818)
    robot = raw.scene["robot"]
    wrist = robot.body_names.index("wrist_3_link")
    target = torch.tensor(args.target, dtype=torch.float32, device=raw.device).unsqueeze(0)
    q = robot.data.joint_pos.detach().clone()
    qd = torch.zeros_like(q)
    offset = torch.tensor(mdp.TCP_OFFSET, dtype=torch.float32, device=raw.device).repeat(1, 1)
    jacobians = robot.root_physx_view.get_jacobians()
    jac_body = wrist - 1 if jacobians.shape[1] == len(robot.body_names) - 1 else wrist

    history = []
    for iteration in range(args.iterations):
        # Hold the candidate state for multiple physics steps so articulation
        # link transforms and joint-state buffers refer to the same q.
        for _ in range(3):
            robot.write_joint_state_to_sim(q, qd)
            raw.scene.write_data_to_sim()
            raw.sim.step(render=False)
            raw.scene.update(raw.physics_dt)
        position = robot.data.body_pos_w[:, wrist]
        quaternion = robot.data.body_quat_w[:, wrist]
        r = quat_apply(quaternion, offset)
        tcp = position + r
        error = target - tcp
        norm = float(torch.linalg.norm(error))
        history.append(norm)
        if norm < 0.001:
            break
        jac = robot.root_physx_view.get_jacobians()[0, jac_body, :, :]
        linear = jac[:3]
        angular = jac[3:]
        point_linear = linear + torch.cross(angular.T, r[0].repeat(angular.shape[1], 1), dim=1).T
        damping = 0.03
        dq = point_linear.T @ torch.linalg.solve(
            point_linear @ point_linear.T + damping * damping * torch.eye(3, device=raw.device),
            error[0],
        )
        dq = torch.clamp(dq, -0.12, 0.12)
        q = q + dq.unsqueeze(0)
        lower = robot.data.soft_joint_pos_limits[0, :, 0]
        upper = robot.data.soft_joint_pos_limits[0, :, 1]
        q = torch.clamp(q, lower.unsqueeze(0), upper.unsqueeze(0))

    result = {
        "status": "ik_ok" if history[-1] < 0.01 else "ik_not_converged",
        "target_w": target[0].detach().cpu().tolist(),
        "tcp_w": tcp[0].detach().cpu().tolist(),
        "error_m": history[-1],
        "iterations": len(history),
        "joint_names": list(robot.joint_names),
        "joint_positions": q[0].detach().cpu().tolist(),
        "jacobian_shape": list(jacobians.shape),
        "jacobian_body_index": jac_body,
    }
    print("precision_start_ik", json.dumps(result, sort_keys=True), flush=True)
    env.close()
    if result["status"] != "ik_ok":
        raise RuntimeError(result)
except BaseException:
    traceback.print_exc()
    os._exit(1)
else:
    app.close()

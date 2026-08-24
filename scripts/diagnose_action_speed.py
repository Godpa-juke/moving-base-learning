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

    task = "Marine-UR3-Random6DoFBase-WorldLineTargetConditioned-Play-v0"
    cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=1, use_fabric=True)
    env = gym.make(task, cfg=cfg)
    raw = env.unwrapped
    env.reset(seed=44)
    robot = raw.scene["robot"]
    arm_cfg = cfg.actions.arm_action
    arm_term = raw.action_manager._terms["arm_action"]

    def serial(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (tuple, list)):
            return [serial(x) for x in value]
        return str(value)

    actuators = {}
    for name, actuator in robot.actuators.items():
        actuators[name] = {
            key: serial(getattr(actuator, key, None))
            for key in (
                "joint_names",
                "velocity_limit",
                "velocity_limit_sim",
                "effort_limit",
                "effort_limit_sim",
                "stiffness",
                "damping",
            )
        }
    report = {
        "step_dt_s": float(raw.step_dt),
        "sim_dt_s": float(cfg.sim.dt),
        "decimation": int(cfg.decimation),
        "action_cfg_class": arm_cfg.class_type.__name__,
        "action_scale_cfg": serial(arm_cfg.scale),
        "action_use_default_offset": serial(getattr(arm_cfg, "use_default_offset", None)),
        "action_term_scale": serial(getattr(arm_term, "_scale", None)),
        "action_term_offset": serial(getattr(arm_term, "_offset", None)),
        "joint_names": list(robot.joint_names),
        "joint_velocity_limits": serial(robot.data.joint_vel_limits),
        "actuators": actuators,
    }
    print("action_speed_diagnostic", json.dumps(report, sort_keys=True), flush=True)
    env.close()
except BaseException:
    traceback.print_exc()
    os._exit(1)
else:
    app.close()

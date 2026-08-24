#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner
    import isaaclab_tasks  # noqa: F401
    import marine_manipulator.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

    task = "Marine-UR3-Random6DoFBase-WorldLinePrecisionStart-Play-v0"
    cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=1, use_fabric=True)
    cfg.seed = 44
    raw = gym.make(task, cfg=cfg)
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device or "cuda:0")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    command = env.unwrapped.command_manager._terms["ee_pose"]

    obs0 = env.get_observations().clone()
    with torch.inference_mode():
        action0 = policy(obs0).clone()
    command_b0 = command.pose_command_b.clone()
    command_w0 = command.pose_command_w.clone()

    command.line_center_e[:, 0] += 0.20
    command.pose_command_b[:, 0] += 0.20
    command.pose_command_w[:, 0] += 0.20
    obs1 = env.get_observations().clone()
    with torch.inference_mode():
        action1 = policy(obs1).clone()

    obs_delta = obs1["policy"][0] - obs0["policy"][0]
    action_delta = action1[0] - action0[0]
    changed = obs_delta.abs() > 1.0e-6
    report = {
        "command_b_0": command_b0[0].detach().cpu().tolist(),
        "command_b_plus20": command.pose_command_b[0].detach().cpu().tolist(),
        "command_w_0": command_w0[0].detach().cpu().tolist(),
        "command_w_plus20": command.pose_command_w[0].detach().cpu().tolist(),
        "obs_changed_indices": torch.nonzero(changed).flatten().detach().cpu().tolist(),
        "obs_delta_values": obs_delta[changed].detach().cpu().tolist(),
        "obs_delta_l2": float(torch.linalg.norm(obs_delta)),
        "action_0": action0[0].detach().cpu().tolist(),
        "action_plus20": action1[0].detach().cpu().tolist(),
        "action_delta_l2": float(torch.linalg.norm(action_delta)),
        "observation_manager": str(env.unwrapped.observation_manager),
    }
    print("target_observation_diagnostic", json.dumps(report, sort_keys=True), flush=True)
    env.close()
except BaseException:
    traceback.print_exc()
    os._exit(1)
else:
    app.close()

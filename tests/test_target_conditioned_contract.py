from __future__ import annotations

import math
from types import SimpleNamespace

import gymnasium as gym
import torch

import marine_manipulator.tasks  # noqa: F401
from marine_manipulator.tasks.random_base_line import mdp
from marine_manipulator.tasks.random_base_line.env_cfg import TargetConditionedWorldLineEnvCfg


def test_target_conditioned_task_is_registered():
    spec = gym.spec("Marine-UR3-Random6DoFBase-WorldLineTargetConditioned-v0")
    assert str(spec.entry_point).endswith("env:TargetConditionedRLEnv")


def test_each_episode_samples_a_reachable_random_line_in_the_negative_x_band():
    cfg = TargetConditionedWorldLineEnvCfg()
    command = cfg.commands.ee_pose
    assert command.class_type is mdp.ReachableRandomLineCommand
    assert command.ranges.pos_x == (-0.40, -0.30)
    assert command.ranges.pos_y == (-0.03, 0.03)
    assert command.ranges.pos_z == (0.18, 0.24)
    assert command.line_direction_e == (0.0, 1.0, 0.0)
    assert command.amplitude_range == (0.05, 0.07)

    # The complete bar, not just its center, remains inside the configured
    # conservative 0.50 m workspace radius. The Y extent combines center
    # randomization and half-line amplitude; the line never leaves the -X band.
    max_endpoint_distance = math.sqrt(
        max(abs(value) for value in command.ranges.pos_x) ** 2
        + (
            max(abs(value) for value in command.ranges.pos_y)
            + command.amplitude_range[1]
        )
        ** 2
        + max(abs(value) for value in command.ranges.pos_z) ** 2
    )
    assert max_endpoint_distance <= command.reachable_radius_m


def test_actor_observes_explicit_world_target_error():
    cfg = TargetConditionedWorldLineEnvCfg()
    assert cfg.observations.policy.target_error_w.func is mdp.target_error_w
    assert cfg.observations.policy.target_error_w.params["command_name"] == "ee_pose"
    assert cfg.observations.policy.target_error_w.params["body_offset"] == mdp.TCP_OFFSET
    assert cfg.observations.policy.tcp_position_w is None


def test_rendered_line_endpoints_follow_sampled_direction():
    command = SimpleNamespace(
        line_center_e=torch.tensor([[0.20, 0.10, 0.25]]),
        line_direction_e=torch.tensor([[1.0, 0.0, 0.0]]),
        line_amplitude=torch.tensor([0.10]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(env_origins=torch.tensor([[1.0, 2.0, 0.0]])),
        command_manager=SimpleNamespace(_terms={"ee_pose": command}),
    )
    start, end = mdp.line_endpoints_w(env, "ee_pose")
    assert torch.allclose(start, torch.tensor([[1.10, 2.10, 0.25]]))
    assert torch.allclose(end, torch.tensor([[1.30, 2.10, 0.25]]))


def test_realistic_joint_target_rate_limit():
    previous = torch.zeros((1, 6))
    requested = torch.ones((1, 6))
    limited = mdp.rate_limit_joint_action(
        previous,
        requested,
        max_speed_rad_s=torch.tensor([0.6, 0.6, 0.6, 0.8, 0.8, 0.8]),
        step_dt=1.0 / 30.0,
        action_scale=0.5,
    )
    expected = torch.tensor([[0.04, 0.04, 0.04, 0.053333333, 0.053333333, 0.053333333]])
    assert torch.allclose(limited, expected, atol=1.0e-7)


def test_reward_engineering_includes_realistic_speed_and_multiscale_tracking():
    cfg = TargetConditionedWorldLineEnvCfg()
    assert cfg.commands.ee_pose.hold_duration_s == 3.0
    assert cfg.rewards.position_coarse.params["std"] == 0.15
    assert cfg.rewards.position_medium.params["std"] == 0.05
    assert cfg.rewards.position_precision.params["std"] == 0.015
    assert cfg.rewards.tcp_speed_limit.func is mdp.tcp_speed_limit_penalty
    assert cfg.rewards.tcp_speed_limit.params["max_speed_m_s"] == 0.15


def test_base_moves_immediately_and_reward_tracks_same_command():
    cfg = TargetConditionedWorldLineEnvCfg()
    assert cfg.observations.policy.base_pose.func is mdp.normalized_immediate_base_pose
    assert cfg.observations.policy.base_velocity.func is mdp.normalized_immediate_base_velocity
    assert cfg.rewards.position.params["command_name"] == "ee_pose"
    assert cfg.rewards.position_coarse.params["command_name"] == "ee_pose"
    assert cfg.rewards.position_precision.params["command_name"] == "ee_pose"

from __future__ import annotations

import gymnasium as gym

import marine_manipulator.tasks  # noqa: F401
from marine_manipulator.tasks.random_base_line import mdp
from marine_manipulator.tasks.random_base_line.env_cfg import PrecisionStartWorldLineEnvCfg


TRAIN_ID = "Marine-UR3-Random6DoFBase-WorldLinePrecisionStart-v0"
PLAY_ID = "Marine-UR3-Random6DoFBase-WorldLinePrecisionStart-Play-v0"


def test_precision_start_task_ids_are_registered() -> None:
    assert gym.spec(TRAIN_ID) is not None
    assert gym.spec(PLAY_ID) is not None
    assert gym.spec(TRAIN_ID).entry_point.endswith("env:PrecisionStartRLEnv")
    assert gym.spec(PLAY_ID).entry_point.endswith("env:PrecisionStartRLEnv")


def test_line_starts_at_one_endpoint_holds_then_traverses() -> None:
    cfg = PrecisionStartWorldLineEnvCfg()
    assert cfg.commands.ee_pose.class_type is mdp.PrecisionStartLineCommand
    start_x, start_y, start_z = mdp.PRECISION_START_TCP_E
    assert cfg.commands.ee_pose.ranges.pos_x == (start_x, start_x)
    assert cfg.commands.ee_pose.ranges.pos_y == (start_y + 0.10, start_y + 0.10)
    assert cfg.commands.ee_pose.ranges.pos_z == (start_z, start_z)
    assert cfg.commands.ee_pose.amplitude_range == (0.10, 0.10)
    assert cfg.commands.ee_pose.hold_duration_s == 1.0
    assert cfg.commands.ee_pose.traverse_duration_s == 4.0


def test_reset_uses_verified_ik_pose_and_neutral_base_start() -> None:
    cfg = PrecisionStartWorldLineEnvCfg()
    assert cfg.events.reset_robot_joints.func is mdp.reset_joints_to_start_pose
    assert cfg.events.sample_base_motion.func is mdp.sample_precision_base_motion
    assert cfg.events.apply_base_motion is None
    assert cfg.observations.policy.base_pose.func is mdp.normalized_neutral_start_base_pose
    assert cfg.observations.policy.base_velocity.func is mdp.normalized_neutral_start_base_velocity
    assert len(mdp.PRECISION_START_JOINT_POS) == 6


def test_precision_reward_keeps_broad_pursuit_and_adds_one_cm_terms() -> None:
    cfg = PrecisionStartWorldLineEnvCfg()
    assert cfg.rewards.position.weight == -0.08
    assert cfg.rewards.position_fine.params["std"] == 0.06
    assert cfg.rewards.position_precision.params["std"] == 0.015
    assert cfg.rewards.position_precision.weight > 0.0
    assert cfg.rewards.success_1cm.params["threshold"] == 0.01
    assert cfg.rewards.success_1cm.weight > 0.0
    assert not hasattr(cfg.rewards, "joint_acceleration")

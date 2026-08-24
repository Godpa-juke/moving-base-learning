from __future__ import annotations

import gymnasium as gym

import marine_manipulator.tasks  # noqa: F401
from marine_manipulator.tasks.random_base_line import mdp
from marine_manipulator.tasks.random_base_line.env_cfg import PrecisionFarImmediateBaseEnvCfg

TRAIN_ID = "Marine-UR3-Random6DoFBase-WorldLinePrecisionFar50cm-v0"
PLAY_ID = "Marine-UR3-Random6DoFBase-WorldLinePrecisionFar50cm-Play-v0"


def test_far_precision_task_ids_are_distinct_and_registered() -> None:
    assert gym.spec(TRAIN_ID).entry_point.endswith("env:PrecisionFarImmediateBaseRLEnv")
    assert gym.spec(PLAY_ID).entry_point.endswith("env:PrecisionFarImmediateBaseRLEnv")


def test_line_is_shifted_half_meter_opposite_world_red_x_arrow() -> None:
    cfg = PrecisionFarImmediateBaseEnvCfg()
    old_x, old_y, old_z = mdp.PRECISION_START_TCP_E
    far_x, far_y, far_z = mdp.PRECISION_FAR_START_TCP_E
    assert abs(far_x - (old_x - 0.50)) < 1.0e-9
    assert (far_y, far_z) == (old_y, old_z)
    assert cfg.commands.ee_pose.ranges.pos_x == (far_x, far_x)
    assert cfg.commands.ee_pose.ranges.pos_y == (far_y + 0.10, far_y + 0.10)
    assert cfg.commands.ee_pose.ranges.pos_z == (far_z, far_z)


def test_far_start_uses_new_ik_and_immediate_neutral_start_motion() -> None:
    cfg = PrecisionFarImmediateBaseEnvCfg()
    assert cfg.events.reset_robot_joints.func is mdp.reset_joints_to_far_start_pose
    assert cfg.observations.policy.base_pose.func is mdp.normalized_immediate_base_pose
    assert cfg.observations.policy.base_velocity.func is mdp.normalized_immediate_base_velocity
    assert len(mdp.PRECISION_FAR_START_JOINT_POS) == 6


def test_target_still_holds_one_second() -> None:
    cfg = PrecisionFarImmediateBaseEnvCfg()
    assert cfg.commands.ee_pose.hold_duration_s == 1.0
    assert cfg.commands.ee_pose.traverse_duration_s == 4.0

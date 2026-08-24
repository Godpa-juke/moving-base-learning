from __future__ import annotations

import gymnasium as gym

import marine_manipulator.tasks  # noqa: F401
from marine_manipulator.tasks.random_base_line import mdp
from marine_manipulator.tasks.random_base_line.env_cfg import RandomBaseWorldLineEnvCfg


TRAIN_ID = "Marine-UR3-Random6DoFBase-WorldLine-v0"
PLAY_ID = "Marine-UR3-Random6DoFBase-WorldLine-Play-v0"


def test_fresh_task_ids_are_registered() -> None:
    assert gym.spec(TRAIN_ID) is not None
    assert gym.spec(PLAY_ID) is not None


def test_task_uses_stock_ur3_and_exact_cylinder_tip_tcp() -> None:
    cfg = RandomBaseWorldLineEnvCfg()
    assert cfg.scene.robot.spawn.usd_path.endswith("/ur3.usd")
    assert "welding_tip" not in cfg.scene.robot.spawn.usd_path
    assert cfg.commands.ee_pose.body_name == "wrist_3_link"
    assert tuple(cfg.commands.ee_pose.body_offset) == (0.0, 0.0, 0.12)
    assert cfg.commands.ee_pose.class_type is mdp.WorldHorizontalLineCommand


def test_all_six_base_axes_are_randomized_at_every_reset() -> None:
    cfg = RandomBaseWorldLineEnvCfg()
    assert cfg.events.sample_base_motion.func is mdp.sample_base_motion
    assert cfg.events.sample_base_motion.mode == "reset"
    assert cfg.events.apply_base_motion.func is mdp.apply_base_motion
    assert cfg.events.apply_base_motion.mode == "interval"
    assert cfg.events.apply_base_motion.interval_range_s == (0.0, 0.0)
    assert tuple(mdp.AXES) == ("x", "y", "z", "roll", "pitch", "yaw")
    assert all(mdp.MOTION_RANGES[axis].amplitude[0] > 0.0 for axis in mdp.AXES)


def test_initial_reward_is_broad_pursuit_without_smoothness_confounds() -> None:
    cfg = RandomBaseWorldLineEnvCfg()
    assert cfg.rewards.position.weight == -0.08
    assert cfg.rewards.position_fine.weight == 0.35
    assert cfg.rewards.position_fine.params["std"] == 0.06
    assert cfg.rewards.orientation.weight == -0.02
    assert cfg.rewards.action_rate.weight == -0.00005
    assert cfg.rewards.joint_vel.weight == -0.00005
    assert not hasattr(cfg.rewards, "joint_acceleration")
    assert not hasattr(cfg.rewards, "success_5cm")


def test_policy_observes_normalized_random_base_state_and_world_tcp() -> None:
    cfg = RandomBaseWorldLineEnvCfg()
    assert cfg.observations.policy.base_pose.func is mdp.normalized_base_pose
    assert cfg.observations.policy.base_velocity.func is mdp.normalized_base_velocity
    assert cfg.observations.policy.tcp_position_w.func is mdp.tcp_position_w


def test_target_conditioned_line_is_spawned_30_to_40_cm_on_negative_world_x() -> None:
    from marine_manipulator.tasks.random_base_line.env_cfg import TargetConditionedWorldLineEnvCfg

    cfg = TargetConditionedWorldLineEnvCfg()
    assert tuple(cfg.commands.ee_pose.ranges.pos_x) == (-0.40, -0.30)
    assert tuple(cfg.commands.ee_pose.line_direction_e) == (0.0, 1.0, 0.0)


def test_target_conditioned_training_defaults_to_4096_parallel_envs() -> None:
    from marine_manipulator.tasks.random_base_line.env_cfg import TargetConditionedWorldLineEnvCfg

    cfg = TargetConditionedWorldLineEnvCfg()
    assert cfg.scene.num_envs == 4096
